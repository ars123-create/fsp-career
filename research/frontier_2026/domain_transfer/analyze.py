"""Fixed domain-transfer analysis. No simulator, optimizer, training or tuning.

The new family of 89 endpoints is separate from all previous studies. Exact
case-design validation precedes inference. No failed case is silently dropped.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from . import io
from .design import materialize
from .statistics import (weighted_mean_interval,weighted_cvar,
                         stratified_tail_bootstrap,bootstrap_difference)
from ..statistics import empirical_cvar,cluster_mean_interval

HERE=Path(__file__).resolve().parent
METRICS=('sum_regret','best_of_slate_regret','ordinal_regret')


def validate_study_plan(plan):
    if plan.get('stage') not in {'DOMAIN_TRANSFER','HARNESS_SMOKE'}:
        raise ValueError('Unknown fixed-study stage')
    expected=materialize(smoke=plan['stage']=='HARNESS_SMOKE')
    role=plan.get('study_role')
    if role not in {'QUALITY','TIMING'} or plan!=expected[0 if role=='QUALITY' else 1]:
        raise ValueError('Plan differs from fixed source-bound materialized design')
    return plan['analysis_contract']


def bad_flags(row):
    flags=[]
    if row['status']!='PASS':flags.append('execution_failure')
    if row['constraint_status']!='PASS':flags.append('constraint_failure_or_incomplete')
    for key in ('solver_fallback','restricted_terminal_search'):
        if row.get(key) is not False:flags.append(key if row.get(key) is True else key+'_unverified')
    if row.get('calibration',{}).get('status')=='FAIL':flags.append('calibration_failure')
    return flags


def collect_main(verified, contract):
    methods,families=contract['methods'],contract['families']
    strata=contract['strata'];count=len(strata[0]['seeds'])
    if any(len(s['seeds'])!=count for s in strata):raise ValueError('Unequal root counts unsupported by this fixed bootstrap')
    index={seed:(j,k) for j,s in enumerate(strata) for k,seed in enumerate(s['seeds'])}
    if len(index)!=len(strata)*count:raise ValueError('Overlapping strata')
    fi={f:i for i,f in enumerate(families)};mi={m:i for i,m in enumerate(methods)}
    values=np.full((len(strata),count,len(families),len(methods),3),np.nan)
    arm_roles={a['name']:a['role'] for a in verified.protocol['plan']['arms']}
    coverage=np.zeros(values.shape[:-1],dtype=bool)
    flags={m:Counter() for m in methods};guard_counts=Counter()
    for row in verified.rows:
        if arm_roles[row['arm']]!='MAIN':continue
        s,c=index[row['seed']];f,m=fi[row['family']],mi[row['method']]
        if coverage[s,c,f,m]:raise ValueError('Duplicate main cell')
        coverage[s,c,f,m]=True
        flags[row['method']].update(bad_flags(row));guard_counts[row['method']]+=row['guard_applications']
        if row['status']=='PASS':
            values[s,c,f,m]=[row['regret'],row['auxiliary']['best_of_slate_regret'],1-row['oracle_overlap_tie_aware']]
    if not coverage.all():raise ValueError('Missing declared main cells')
    return values,{m:dict(v) for m,v in flags.items()},dict(guard_counts)


def descriptive_sample(rows):
    good=[r for r in rows if r['status']=='PASS']
    metrics={}
    for key in METRICS:
        sample=[r['regret'] if key=='sum_regret' else r['auxiliary']['best_of_slate_regret']
                if key=='best_of_slate_regret' else 1-r['oracle_overlap_tie_aware'] for r in good]
        metrics[key]={'mean':float(np.mean(sample)) if sample else None,
                      'cvar95':float(empirical_cvar(sample)) if sample else None}
    steps=[x for r in good for x in r['step_seconds']]
    return {'cases':len(rows),'successful':len(good),'failed':len(rows)-len(good),
            'scope':'ALL_PLANNED_CASES' if len(good)==len(rows) else 'SUCCESSFUL_SUBSET_ONLY',
            'flags':dict(Counter(f for r in rows for f in bad_flags(r))),
            'metrics':metrics,'step_p95_seconds':float(np.quantile(steps,.95)) if steps else None,
            'construction_p95_seconds':float(np.quantile([r['construction_seconds'] for r in good],.95)) if good else None,
            'total_p95_seconds':float(np.quantile([r['total_seconds'] for r in good],.95)) if good else None,
            'total_seconds_all_cases':float(sum(r['total_seconds'] for r in rows)),
            'guard_applications':sum(r['guard_applications'] for r in rows)}


def describe_arms(verified, role):
    return {a['name']:{m:descriptive_sample([r for r in verified.rows if r['arm']==a['name'] and r['method']==m])
                      for m in a['methods']} for a in verified.protocol['plan']['arms'] if a['role']==role}


def inference(values, flags, contract, *, bootstrap_progress=None):
    methods,families=contract['methods'],contract['families']
    w=np.array([s['weight_numerator']/s['weight_denominator'] for s in contract['strata']])
    s,c,f,m,_=values.shape
    masses=np.broadcast_to((w/(c*f))[:,None,None],(s,c,f)).ravel()
    alpha,multiplicity=contract['alpha'],contract['hypotheses']
    comparisons={};summary={};candidates=contract['candidates'];baselines=contract['baselines']
    for j,name in enumerate(methods):
        metrics={}
        for k,metric in enumerate(METRICS):
            x=values[:,:,:,j,k]
            metrics[metric]=({'weighted_mean':float(np.sum(w*x.mean(axis=(1,2)))),
                              'weighted_cvar95':weighted_cvar(x,masses)} if np.isfinite(x).all() else None)
        summary[name]={'metrics':metrics,'flags':flags[name],
                       'scope':'ALL_PLANNED_MAIN_CASES' if all(v is not None for v in metrics.values()) else 'INCOMPLETE_NOT_IMPUTED'}
    tail_names=[name for name in candidates+baselines if not flags[name]]
    tail_indices=[methods.index(name) for name in tail_names]
    tails=None
    if any(name in candidates for name in tail_names) and any(name in baselines for name in tail_names):
        tails=stratified_tail_bootstrap(values[:,:,:,tail_indices,0],w,
             alpha=contract['cvar_alpha'],confidence_alpha=alpha,hypotheses=multiplicity,
             resamples=contract['bootstrap_resamples'],seed=contract['bootstrap_seed'],
             batch=contract['bootstrap_batch'],prefix_fraction=contract['bootstrap_prefix_fraction'],progress=bootstrap_progress)
        tails['methods']=tail_names
    for candidate in candidates:
        comparisons[candidate]={}
        for baseline in baselines:
            if flags[candidate] or flags[baseline]:
                comparisons[candidate][baseline]={'status':'NOT_AVAILABLE_OPERATIONAL_FAILURE','all_quality_gates_pass':False}
                continue
            delta=values[:,:,:,methods.index(candidate),0]-values[:,:,:,methods.index(baseline),0]
            overall=weighted_mean_interval(delta.mean(axis=2),w,alpha=alpha,hypotheses=multiplicity)
            by_family={name:weighted_mean_interval(delta[:,:,j],w,alpha=alpha,hypotheses=multiplicity) for j,name in enumerate(families)}
            by_stratum={st['name']:cluster_mean_interval(delta[j].mean(axis=1),alpha=alpha,hypotheses=multiplicity)
                         for j,st in enumerate(contract['strata'])}
            tail=bootstrap_difference(tails,tail_names.index(candidate),tail_names.index(baseline)) if tails else None
            gates={'overall':overall['upper_one_sided']<contract['mean_upper_strict_bound'],
                   'all_families':all(v['upper_one_sided']<contract['family_and_stratum_upper_strict_bound'] for v in by_family.values()),
                   'all_strata':all(v['upper_one_sided']<contract['family_and_stratum_upper_strict_bound'] for v in by_stratum.values()),
                   'weighted_cvar95':tail is not None and tail['upper_one_sided']<contract['cvar_upper_strict_bound']}
            comparisons[candidate][baseline]={'status':'ANALYZED','overall':overall,'families':by_family,
                  'strata':by_stratum,'weighted_cvar95':tail,'gates':gates,'all_quality_gates_pass':all(gates.values())}
    contrast=None
    if all(not flags[name] for name in candidates):
        delta=values[:,:,:,methods.index(candidates[0]),0]-values[:,:,:,methods.index(candidates[1]),0]
        contrast=weighted_mean_interval(delta.mean(axis=2),w,alpha=alpha,hypotheses=multiplicity)
    return summary,comparisons,contrast,tails


def decide(comparisons, contrast, timing, contract, all_quality_flags=None):
    details={};eligible=[]
    all_quality_flags=all_quality_flags or {}
    for name in contract['candidates']:
        quality=all(comparisons[name][b]['all_quality_gates_pass'] for b in contract['baselines'])
        operational=not any(all_quality_flags.get(m,{}) for m in [name]+contract['baselines'])
        measurements={}
        for d in contract['timing_dimensions']:
            group=timing[f'timing_n100_d{d}'][name]
            measurements[str(d)]={'step_p95_seconds':group['step_p95_seconds'],
                'pass':not group['flags'] and group['step_p95_seconds'] is not None
                       and group['step_p95_seconds']<contract['latency_step_p95_strict_bound']}
        engineering=all(v['pass'] for v in measurements.values())
        details[name]={'quality_pass':quality,'all_quality_cases_operational_pass':operational,
                       'engineering_pass':engineering,'timing':measurements,
                       'eligible':quality and operational and engineering}
        if quality and operational and engineering:eligible.append(name)
    if not eligible:
        result='NO_SUPPORTED_DOMAIN_REPLACEMENT';selected=None
    elif len(eligible)==1:
        result='SOLE_CONFIGURATION_PASSING_DECLARED_RULE';selected=eligible[0]
    elif contrast is None:
        raise ValueError('Eligible pair lacks contrast')
    elif contrast['two_sided'][1]<-.001:
        result='PRACTICAL_MEAN_ADVANTAGE_WITHIN_ELIGIBLE_PAIR';selected=contract['candidates'][0]
    elif contrast['two_sided'][0]>.001:
        result='PRACTICAL_MEAN_ADVANTAGE_WITHIN_ELIGIBLE_PAIR';selected=contract['candidates'][1]
    else:
        result='ELIGIBLE_PAIR_NOT_SEPARATED';selected=None
    return {'result':result,'selected':selected,'eligible':eligible,'candidates':details,
            'no_global_best_or_equivalence_claim':True,'application_changed':False}


def analyze(quality_path, timing_path, out=None):
    quality,timing=io.load_run(quality_path),io.load_run(timing_path)
    contract=validate_study_plan(quality.protocol['plan'])
    if quality.protocol['plan']['study_role']!='QUALITY' or timing.protocol['plan']['study_role']!='TIMING':
        raise ValueError('Expected quality then timing runs')
    if contract!=validate_study_plan(timing.protocol['plan']):raise ValueError('Different analysis contracts')
    for key in ('source_sha256','methods','versions','primary_reference'):
        if quality.protocol[key]!=timing.protocol[key]:raise ValueError('Quality/timing provenance differs: '+key)
    for path in (Path(__file__),HERE/'statistics.py',HERE/'design.py'):
        relative=str(path.relative_to(HERE.parents[2]))
        if quality.protocol['source_sha256'].get(relative)!=io.digest(path):raise ValueError('Unbound analysis source: '+relative)
    out=Path(out) if out else quality.directory/'analysis_domain_v1'
    out.mkdir(parents=True,exist_ok=False)
    values,flags,guards=collect_main(quality,contract)
    def progress(done,total):
        if done%1024==0 or done==total:print(json.dumps({'bootstrap_completed':done,'total':total}),flush=True)
    summary,comparisons,contrast,tails=inference(values,flags,contract,bootstrap_progress=progress)
    timing_summary=describe_arms(timing,'TIMING');landmarks=describe_arms(quality,'LANDMARK')
    all_quality_flags={m:dict(Counter(flag for row in quality.rows if row['method']==m for flag in bad_flags(row)))
                        for m in contract['methods']}
    decision=decide(comparisons,contrast,timing_summary,contract,all_quality_flags)
    extra_hashes={}
    if tails is not None:
        np.save(out/'bootstrap_cvars.npy',tails['distribution'],allow_pickle=False)
        extra_hashes['bootstrap_cvars.npy']=io.digest(out/'bootstrap_cvars.npy')
    report={'schema':'fsp-domain-analysis-result-v1','status':'ANALYZED_FIXED_DOMAIN_TRANSFER',
            'scope':'TECHNICAL_SMOKE_ONLY' if quality.protocol['plan']['stage']=='HARNESS_SMOKE' else 'SEPARATE_FIXED_SYNTHETIC_DOMAIN_STUDY',
            'utc':datetime.now(timezone.utc).isoformat(),'human_quality':'NOT_RUN',
            'input_sha256':{'quality':quality.hashes,'timing':timing.hashes},
            'source_sha256':{str(p.relative_to(HERE.parents[2])):io.digest(p) for p in (Path(__file__),HERE/'statistics.py',HERE/'io.py',HERE/'design.py')},
            'runtime':quality.protocol['versions'],'analysis_contract':contract,
            'quality_rows':len(quality.rows),'timing_rows':len(timing.rows),
            'source_files_verified':len(quality.protocol['source_sha256']),
            'weighted_main_metrics':summary,'guard_applications_main':guards,
            'all_quality_case_flags':all_quality_flags,
            'comparisons':comparisons,'candidate_contrast':contrast,'decision':decision,
            'landmarks_descriptive':landmarks,'timing':timing_summary,
            'bootstrap':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in tails.items() if k!='distribution'} if tails else None,
            'extra_files_sha256':extra_hashes}
    # Landmarks remain descriptive paired root estimates, never added to M=89.
    report['landmark_comparisons']={}
    for arm in quality.protocol['plan']['arms']:
        if arm['role']!='LANDMARK':continue
        lookup={(r['seed'],r['family'],r['method']):r for r in quality.rows if r['arm']==arm['name']}
        pairs={}
        for candidate in contract['candidates']:
            for baseline in contract['baselines']+contract['descriptive_controls']:
                rows=[lookup[(seed,family,name)] for seed in arm['seeds'] for family in arm['families'] for name in (candidate,baseline)]
                if any(bad_flags(r) for r in rows):pairs[candidate+' minus '+baseline]={'status':'NOT_AVAILABLE_OPERATIONAL_FAILURE'};continue
                delta=[np.mean([lookup[(seed,family,candidate)]['regret']-lookup[(seed,family,baseline)]['regret'] for family in arm['families']]) for seed in arm['seeds']]
                pairs[candidate+' minus '+baseline]=cluster_mean_interval(delta)
        report['landmark_comparisons'][arm['name']]=pairs
    with (out/'analysis.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    (out/'analysis.sha256').write_text(io.digest(out/'analysis.json')+'\n')
    print(json.dumps({'status':report['status'],'scope':report['scope'],'decision':decision['result'],'selected':decision['selected'],'out':str(out)}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quality',type=Path,required=True)
    parser.add_argument('--timing',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args();analyze(args.quality,args.timing,args.out)


if __name__=='__main__':main()

"""Literal analysis fixtures only. Never open development/heldout quality rows."""
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

import numpy as np
import pytest
import scipy
from scipy.stats import t

from research.frontier_2026.extensions import analyze_supplement as analysis


CANDIDATES = ['P18_expected_swap','P19_zero_min_mean_tiebreak',
              'P20_guarded_lookahead','F013_muc_gp_guard']
FINALISTS = ['mixed_kg_finer_quadrature','P15_easy']
CONTROLS = FINALISTS+['linear_evoi_3','gp_topk_guard']


def arm(name='tiny', methods=None, seeds=None, n=6):
    return {'name':name,'seeds':seeds or [101,102,103],
            'families':['linear','interaction'],
            'methods':methods or ['candidate','mixed_kg_finer_quadrature'],
            'n':n,'d':3,'variant':'base'}


def plan():
    return {'stage':'HARNESS_SMOKE','workers':1,'arms':[arm()]}


def row(a, seed, family, method, *, regret=.1, best=.03, overlap=.8, seconds=.01):
    metrics = {'regret':regret,'oracle_overlap_tie_aware':overlap,'oracle_overlap':overlap,
               'dominance_violations':0,'auxiliary':{'best_of_slate_regret':best,
                 'worst_of_slate_utility':.1,'mean_of_slate_utility':.4}}
    trace = [{'slot':i,'selection':list(range(min(5,a['n']))),'query_id':0 if i==0 else None,
              'seconds':seconds,**deepcopy(metrics)} for i in (0,1)]
    trace[0]['choice'] = 2
    return {'arm':a['name'],'seed':seed,'family':family,'method':method,
            'n':a['n'],'d':a['d'],'variant':a.get('variant','base'),
            'status':'PASS','constraint_status':'PASS','solver_fallback':False,
            'restricted_terminal_search':False,'construction_seconds':.002,
            'total_seconds':2*seconds+.002,'slots':1,'answers':0,'history':[[0,2]],
            'trace':trace,'calibration':{'status':'NOT_AVAILABLE'},**metrics}


def rows_for(p):
    return [row(a,s,f,m) for a in p['arms'] for s,f,m in product(a['seeds'],a['families'],a['methods'])]


def set_loss(r, regret):
    r['regret'] = regret
    for step in r['trace']:
        step['regret'] = regret


def set_seconds(r, value):
    for step in r['trace']:
        step['seconds'] = value
    r['total_seconds'] = 2*value+.002


def failed(r):
    return {k:deepcopy(v) for k,v in r.items() if k in ('arm','seed','family','method','n','d','variant','total_seconds')} | {
        'status':'FAIL','constraint_status':'NOT_COMPLETE','error_type':'TechnicalFixture',
        'trace':[],'error':'literal failure fixture'}


def write_run(directory, *, p=None, rows=None):
    directory.mkdir()
    p = deepcopy(plan() if p is None else p)
    rows = deepcopy(rows_for(p) if rows is None else rows)
    manifest = {}
    for relative, path in ((analysis.ANALYZER_REL,Path(analysis.__file__)),
                           (analysis.STATISTICS_REL,analysis.HERE.parent/'statistics.py')):
        destination = directory/'snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(path.read_bytes())
        manifest[relative] = analysis.digest(path)
    protocol = {'schema':'fsp-supplement-v1','plan':p,
                'expected_rows':len(analysis.expected_keys(analysis.plan_arms(p))),
                'methods':{m:{'technical_fixture':m} for a in p['arms'] for m in a['methods']},
                'source_sha256':manifest,'primary_holdout_quality_opened':False,
                'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                            'torch':'literal-torch-1','botorch':'literal-botorch-1','gpytorch':'literal-gpytorch-1',
                            'platform':platform.system(),'machine':platform.machine(),
                            'python_abi':sysconfig.get_config_var('SOABI')}}
    (directory/'protocol.json').write_text(json.dumps(protocol))
    (directory/'protocol.sha256').write_text(analysis.digest(directory/'protocol.json')+'\n')
    shard_rows = []
    arms = {a['name']:a for a in p['arms']}
    for worker in range(p['workers']):
        # Invalid literal rows stay in worker0 so coverage tests cannot silently
        # lose malformed inputs while constructing their fixture.
        selected = [r for r in rows if (
            worker == (arms[r['arm']]['seeds'].index(r['seed']) % p['workers'])
            if r['arm'] in arms and r['seed'] in arms[r['arm']]['seeds'] else worker==0)]
        raw = ''.join(json.dumps(r)+'\n' for r in selected)
        shard_rows.append(raw)
        (directory/f'rows-{worker:03d}.jsonl').write_text(raw)
        preflight = {'status':'PASS','versions':protocol['versions'],'methods':protocol['methods'],
                     'source_files_verified':len(manifest),'worlds_generated':0,
                     'workspace':str(directory.resolve()/'snapshot')}
        worker_completion = {'status':'COMPLETE','rows':len(selected),
                             'failures':sum(r['status']!='PASS' for r in selected),
                             'source_files_verified':len(manifest),
                             'rows_sha256':analysis.digest(directory/f'rows-{worker:03d}.jsonl')}
        (directory/f'preflight-{worker:03d}.json').write_text(json.dumps(preflight))
        (directory/f'completion-{worker:03d}.json').write_text(json.dumps(worker_completion))
    (directory/'rows.jsonl').write_text(''.join(shard_rows))
    completion = {'status':'COMPLETE','rows':protocol['expected_rows'],
                  'rows_sha256':analysis.digest(directory/'rows.jsonl'),
                  'scope':p['stage'],'quality_of_real_hiring':'NOT_RUN'}
    (directory/'completion.json').write_text(json.dumps(completion))
    return directory


def rewrite_metadata(directory, name, change, *, reseal=True):
    path = directory/name
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    if name=='protocol.json' and reseal:
        (directory/'protocol.sha256').write_text(analysis.digest(path))


def independent_fractional_cvar(values, alpha=Fraction(19,20)):
    ordered = sorted(map(lambda x:Fraction(str(x)),values),reverse=True)
    remain = (1-alpha)*len(ordered)
    denominator, total = remain, Fraction(0)
    for value in ordered:
        amount = min(remain,Fraction(1))
        total += amount*value
        remain -= amount
        if remain == 0:
            break
    return float(total/denominator)


def test_three_metric_stats_match_independent_fractional_tail_oracle():
    values = [i/100 for i in range(23)]
    actual = analysis._stats(values)
    assert actual['mean'] == pytest.approx(sum(values)/23,abs=1e-15)
    assert actual['cvar95'] == pytest.approx(independent_fractional_cvar(values),abs=2e-15)
    assert actual['cvar95'] != pytest.approx(np.mean(sorted(values)[-2:]),abs=1e-4)


def test_paired_ci_uses_seed_clusters_not_individual_families():
    p = plan(); a=p['arms'][0]; rows=rows_for(p)
    expected = []
    for index,seed in enumerate(a['seeds']):
        diffs = []
        for family_index,family in enumerate(a['families']):
            difference = -.003*(index+1)+.002*family_index
            diffs.append(difference)
            for r in rows:
                if r['seed']==seed and r['family']==family:
                    set_loss(r,.2+difference if r['method']=='candidate' else .2)
        expected.append(sum(diffs)/len(diffs))
    measured=analysis.paired_comparison(rows,a,'candidate',FINALISTS[0])
    metric=measured['metrics']['sum_regret']
    mean=sum(expected)/3
    sd=(sum((v-mean)**2 for v in expected)/2)**.5
    radius=float(t.ppf(.975,2))*sd/3**.5
    assert metric['clusters']==3 and metric['multiplicity']==1
    assert metric['two_sided']==pytest.approx([mean-radius,mean+radius],abs=1e-15)
    assert measured['confirmatory'] is False and measured['multiplicity_adjusted'] is False


def test_sealed_reader_report_and_no_winner(tmp_path):
    run=write_run(tmp_path/'run')
    verified=analysis.load_run(run)
    report=analysis.summarize(verified)
    entry=report['arms']['tiny']['methods']['candidate']
    assert entry['operational_eligible_on_this_arm']
    assert entry['metric_scope']=='ALL_PLANNED_CASES'
    assert entry['metrics']['sum_regret']['mean']==pytest.approx(.1)
    assert entry['metrics']['best_of_slate_regret']['mean']==pytest.approx(.03)
    assert entry['metrics']['ordinal_regret']['mean']==pytest.approx(.2)
    assert report['overall_winner'] is None and report['confirmatory_decision'] is None
    assert report['source_files_verified']==2


def test_failure_kept_as_explicit_successful_subset_without_ci(tmp_path):
    p=plan(); rows=rows_for(p); rows[0]=failed(rows[0])
    report=analysis.summarize(analysis.load_run(write_run(tmp_path/'run',p=p,rows=rows)))
    entry=report['arms']['tiny']['methods']['candidate']
    assert entry['failed_cases']==1 and entry['successful_cases']==5
    assert entry['metric_scope']=='SUCCESSFUL_SUBSET_ONLY'
    assert not entry['operational_eligible_on_this_arm']
    assert entry['family_means']['linear']['scope']=='SUCCESSFUL_SUBSET_ONLY'
    assert report['arms']['tiny']['comparisons'][0]['status']=='NOT_AVAILABLE'


@pytest.mark.parametrize('flag',['solver_fallback','restricted_terminal_search'])
def test_complete_research_flags_do_not_disappear_or_claim_operational_eligibility(tmp_path,flag):
    rows=rows_for(plan()); rows[0][flag]=True
    result=analysis.summarize(analysis.load_run(write_run(tmp_path/'run',rows=rows)))
    entry=result['arms']['tiny']['methods']['candidate']
    assert entry['metric_scope']=='ALL_PLANNED_CASES'
    assert not entry['operational_eligible_on_this_arm'] and entry['flag_counts'][flag]==1
    comparison=result['arms']['tiny']['comparisons'][0]
    assert comparison['status']=='AVAILABLE' and comparison['operational_flags'][flag]==1


def test_constraint_failure_has_descriptive_metrics_but_no_pair_ci(tmp_path):
    rows=rows_for(plan()); rows[0]['constraint_status']='FAIL'
    rows[0]['trace'][0]['dominance_violations']=1
    result=analysis.summarize(analysis.load_run(write_run(tmp_path/'run',rows=rows)))
    assert result['arms']['tiny']['comparisons'][0]['status']=='NOT_AVAILABLE'
    assert result['arms']['tiny']['methods']['candidate']['flag_counts']['constraint_failure']==1


@pytest.mark.parametrize('mode',['missing','duplicate','unexpected','wrong_setting'])
def test_exact_coverage_never_drops_rows(tmp_path,mode):
    rows=rows_for(plan())
    if mode=='missing':rows.pop()
    elif mode=='duplicate':rows.append(deepcopy(rows[0]))
    elif mode=='unexpected':rows[0]['seed']=999
    else:rows[0]['n']=7
    with pytest.raises(ValueError):analysis.load_run(write_run(tmp_path/'run',rows=rows))


@pytest.mark.parametrize('mode',['seal','rows_sha','completion_count','completion_status','snapshot','analyzer_binding','runtime','scope'])
def test_metadata_drift_rejected(tmp_path,mode):
    run=write_run(tmp_path/'run')
    if mode=='seal':rewrite_metadata(run,'protocol.json',lambda x:x.update(utc='changed'),reseal=False)
    elif mode=='rows_sha':rewrite_metadata(run,'completion.json',lambda x:x.update(rows_sha256='0'*64))
    elif mode=='completion_count':rewrite_metadata(run,'completion.json',lambda x:x.update(rows=0))
    elif mode=='completion_status':rewrite_metadata(run,'completion.json',lambda x:x.update(status='RUNNING'))
    elif mode=='snapshot':(run/'snapshot'/analysis.STATISTICS_REL).write_text('changed')
    elif mode=='analyzer_binding':
        path=run/'snapshot'/analysis.ANALYZER_REL; path.write_text('changed')
        rewrite_metadata(run,'protocol.json',lambda x:x['source_sha256'].update({analysis.ANALYZER_REL:analysis.digest(path)}))
    elif mode=='runtime':rewrite_metadata(run,'protocol.json',lambda x:x['versions'].update(numpy='changed'))
    else:rewrite_metadata(run,'protocol.json',lambda x:x.update(primary_holdout_quality_opened=True))
    with pytest.raises(ValueError):analysis.load_run(run)


@pytest.mark.parametrize('mode',['missing_flag','nonbool_flag','hidden_restricted','hidden_fallback','bad_loss','wrong_terminal','bad_cardinality','fourth_query','history_mismatch'])
def test_malformed_success_flags_metrics_and_contracts_rejected(tmp_path,mode):
    rows=rows_for(plan()); r=rows[0]
    if mode=='missing_flag':r.pop('solver_fallback')
    elif mode=='nonbool_flag':r['solver_fallback']=0
    elif mode=='hidden_restricted':r['trace'][0]['diagnostics']={'terminal_search_complete':False}
    elif mode=='hidden_fallback':r['trace'][0]['diagnostics']={'fallback_from_milp':True}
    elif mode=='bad_loss':r['regret']=float('nan')
    elif mode=='wrong_terminal':r['trace'][-1]['regret']=.99
    elif mode=='bad_cardinality':r['trace'][0]['selection']=[0,0,1,2,3]
    elif mode=='fourth_query':r['slots']=4
    else:r['history']=[[0,0]]
    with pytest.raises(ValueError):analysis.load_run(write_run(tmp_path/'run',rows=rows))


def test_empty_success_subset_has_null_metrics_and_retained_failure_count(tmp_path):
    rows=[failed(r) for r in rows_for(plan())]
    result=analysis.summarize(analysis.load_run(write_run(tmp_path/'run',rows=rows)))
    entry=result['arms']['tiny']['methods']['candidate']
    assert entry['failed_cases']==6 and entry['metrics']['sum_regret']['mean'] is None
    assert entry['timings']['step_p95_seconds_successful_cases'] is None


def test_output_exclusive_and_cli_no_quality_generation(tmp_path):
    run=write_run(tmp_path/'run')
    before=analysis.digest(run/'rows.jsonl')
    result=subprocess.run([sys.executable,'-m','research.frontier_2026.extensions.analyze_supplement',
                           '--run',str(run)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['status']=='ANALYZED_EXPLORATORY'
    assert analysis.digest(run/'rows.jsonl')==before
    assert (run/'analysis_supplement_v1'/'REPORT_RU.md').is_file()
    with pytest.raises(FileExistsError):analysis.analyze_run(run)


def trigger_plan():
    quality=arm('full_n18',CANDIDATES+CONTROLS,[1000,1001],18)
    latency=arm('new_n100_latency',CANDIDATES+FINALISTS,[2000,2001],100)
    gates={'development_quality_arm':quality['name'],'development_latency_arm':latency['name'],
           'validation_arm':'selection_validation_n18','validation_n':18,'validation_d':3,'validation_variant':'base',
           'validation_seed_start':3000,'validation_seed_end_inclusive':3003,
           'finalists':FINALISTS,'prospective_methods':CANDIDATES,'mean_improvement_margin':.001,
           'maximum_family_degradation':.005,'maximum_cvar95_degradation':.005,'cvar_alpha':.95,
           'latency_step_p95_strict_bound_seconds':.25,'multiple_passers':'ALL_PASSERS_IN_PROSPECTIVE_ORDER',
           'bad_flags':['execution_failure','constraint_failure','solver_fallback','restricted_terminal_search','missing_or_duplicate_case'],
           'no_post_development_tuning':True,'point_estimate_screen_only':True}
    follow={'always_validate_full_n18_new_methods':CANDIDATES,'selection_validation_controls':CONTROLS,
            'selection_validation_seeds':{'start':3000,'end_inclusive':3003},
            'selection_validation_families':quality['families'],
            'selection_validation_setting':{'n':18,'d':3,'variant':'base'},'machine_gates':gates,
            'future_confirmation_references':CONTROLS+['actual retained primary reference if different']}
    return {'stage':'DEVELOPMENT','workers':1,'arms':[quality,latency],'followup_rule':follow}


def trigger_fixture():
    p=trigger_plan(); g=p['followup_rule']['machine_gates']
    validation=arm(g['validation_arm'],CANDIDATES+CONTROLS,list(range(3000,3004)),18)
    rows=rows_for(p)+rows_for({'arms':[validation]})
    for r in rows:
        set_loss(r,.097 if r['method'] in CANDIDATES else .1)
    return p,rows,validation


def test_trigger_all_passers_order_and_always_validate_all_four():
    p,rows,_=trigger_fixture()
    result=analysis.validation_trigger(rows,p)
    assert result['triggered_methods']==CANDIDATES
    assert result['always_validate_methods']==CANDIDATES
    assert result['confirmatory'] is False and result['overall_winner'] is None
    for method in CANDIDATES:
        assert result['candidates'][method]['trigger']
        assert result['candidates'][method]['comparisons'][FINALISTS[0]]['mean_difference']==pytest.approx(-.003)


@pytest.mark.parametrize('scope',['full_n18','new_n100_latency','selection_validation_n18'])
@pytest.mark.parametrize('defect',['failure','constraint','fallback','restricted','missing','duplicate'])
def test_trigger_every_required_scope_retains_failures(scope,defect):
    p,rows,_=trigger_fixture()
    index=next(i for i,r in enumerate(rows) if r['arm']==scope and r['method']==CANDIDATES[0])
    r=rows[index]
    if defect=='failure':rows[index]=failed(r)
    elif defect=='constraint':r['constraint_status']='FAIL';r['trace'][0]['dominance_violations']=1
    elif defect=='fallback':r['solver_fallback']=True
    elif defect=='restricted':r['restricted_terminal_search']=True
    elif defect=='missing':rows.pop(index)
    else:rows.append(deepcopy(r))
    result=analysis.validation_trigger(rows,p)
    assert not result['candidates'][CANDIDATES[0]]['eligible']
    assert not result['candidates'][CANDIDATES[0]]['trigger']
    assert result['triggered_methods']==CANDIDATES[1:]
    assert result['always_validate_methods']==CANDIDATES


def test_trigger_references_must_be_complete_and_pass():
    p,rows,_=trigger_fixture()
    index=next(i for i,r in enumerate(rows) if r['arm']=='selection_validation_n18' and r['method']==FINALISTS[0])
    rows[index]=failed(rows[index])
    result=analysis.validation_trigger(rows,p)
    assert not result['triggered_methods']
    assert all(not r['eligible'] for r in result['candidates'].values())


def test_trigger_n100_strict_latency_and_construction_exclusion():
    p,rows,_=trigger_fixture()
    for r in rows:
        if r['arm']=='new_n100_latency' and r['method']==CANDIDATES[0]:set_seconds(r,.25)
        if r['method']==CANDIDATES[1]:r['construction_seconds']=1000.
    result=analysis.validation_trigger(rows,p)
    assert not result['candidates'][CANDIDATES[0]]['latency_pass']
    assert result['candidates'][CANDIDATES[1]]['trigger']


def test_trigger_requires_both_finalists_and_point_means_not_ci():
    p,rows,_=trigger_fixture()
    for r in rows:
        if r['arm']=='selection_validation_n18' and r['method']==FINALISTS[1]:set_loss(r,.097)
    result=analysis.validation_trigger(rows,p)
    assert not result['triggered_methods']
    assert all(not x['comparisons'][FINALISTS[1]]['mean_pass'] for x in result['candidates'].values())


def test_trigger_worst_family_gate_cannot_be_hidden_by_pooled_mean():
    p,rows,_=trigger_fixture()
    for r in rows:
        if r['arm']=='selection_validation_n18' and r['method']==CANDIDATES[0]:
            set_loss(r,.11 if r['family']=='linear' else .02)
    result=analysis.validation_trigger(rows,p)['candidates'][CANDIDATES[0]]
    assert result['comparisons'][FINALISTS[0]]['mean_pass']
    assert not result['comparisons'][FINALISTS[0]]['family_pass'] and not result['trigger']


def test_trigger_fractional_pooled_tail_separate_from_family_and_mean():
    p,rows,_=trigger_fixture()
    p['followup_rule']['machine_gates']['validation_seed_end_inclusive']=3011
    p['followup_rule']['selection_validation_seeds']['end_inclusive']=3011
    validation=arm('selection_validation_n18',CANDIDATES+CONTROLS,list(range(3000,3012)),18)
    rows=[r for r in rows if r['arm']!='selection_validation_n18']+rows_for({'arms':[validation]})
    candidate_values=[]
    for r in rows:
        if r['arm']=='selection_validation_n18' and r['method']==CANDIDATES[0]:
            value=.25 if r['seed']==3000 and r['family']=='linear' else .02
            set_loss(r,value);candidate_values.append(value)
    result=analysis.validation_trigger(rows,p)['candidates'][CANDIDATES[0]]
    comparison=result['comparisons'][FINALISTS[0]]
    assert comparison['mean_pass'] and comparison['family_pass']
    assert not comparison['cvar_pass'] and not result['trigger']
    assert comparison['pooled_fractional_cvar_difference']==pytest.approx(independent_fractional_cvar(candidate_values)-.1,abs=1e-14)


def test_trigger_missing_validation_keeps_always_include_order():
    p,rows,_=trigger_fixture()
    result=analysis.validation_trigger([r for r in rows if r['arm']!='selection_validation_n18'],p)
    assert result['always_validate_methods']==CANDIDATES and not result['triggered_methods']


def test_combined_sealed_runs_verify_configs_and_prespecified_coverage(tmp_path):
    p,rows,validation=trigger_fixture()
    dev=write_run(tmp_path/'dev',p=p,rows=[r for r in rows if r['arm']!=validation['name']])
    valplan={'stage':'SELECTION_VALIDATION','workers':1,'arms':[validation]}
    val=write_run(tmp_path/'val',p=valplan,rows=[r for r in rows if r['arm']==validation['name']])
    report=analysis.analyze_run(val,development=dev)
    assert report['selection_trigger']['triggered_methods']==CANDIDATES
    assert report['selection_trigger']['same_source_configs_and_checkpoints'].startswith('VERIFIED')
    rewrite_metadata(val,'protocol.json',lambda x:x['methods'][CANDIDATES[0]].update(changed=True))
    rewrite_metadata(val,'preflight-000.json',lambda x:x['methods'][CANDIDATES[0]].update(changed=True))
    with pytest.raises(ValueError,match='config drift'):
        analysis.analyze_run(val,out=tmp_path/'second',development=dev)


def test_real_frozen_plan_structure_only_no_rows_opened():
    # Read only the preregistered plan text. No output/evidence paths accessed.
    p=json.loads((analysis.HERE/'SUPPLEMENT_DEVELOPMENT_PROTOCOL.json').read_text())
    result=analysis.validation_trigger([],p)
    assert len(result['always_validate_methods'])==4
    assert result['validation_expected_cases_per_method']==64*14
    assert result['triggered_methods']==[]


def test_json_duplicate_object_keys_are_rejected():
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        analysis._json('{"status":"PASS","status":"FAIL"}')


@pytest.mark.parametrize('workers',[1,2])
def test_worker_provenance_matches_merged_bytes_and_plan(tmp_path,workers):
    p=plan();p['workers']=workers
    run=write_run(tmp_path/'run',p=p)
    verified=analysis.load_run(run)
    assert len(verified.rows)==12
    assert all(f'{prefix}-{w:03d}.{suffix}' in verified.hashes
               for w in range(workers) for prefix,suffix in (
                   ('preflight','json'),('completion','json'),('rows','jsonl')))
    assert analysis.summarize(verified)['worker_provenance_verified']==workers


@pytest.mark.parametrize('name',['preflight-000.json','completion-000.json','rows-000.jsonl'])
def test_missing_worker_provenance_rejected(tmp_path,name):
    run=write_run(tmp_path/'run');(run/name).unlink()
    with pytest.raises(ValueError,match='Missing worker provenance'):
        analysis.load_run(run)


@pytest.mark.parametrize('field,value',[
    ('status','FAIL'),('workspace','/literal/other/workspace'),('worlds_generated',1),
    ('worlds_generated',False),('source_files_verified',0),('source_files_verified',True),
    ('versions',{'numpy':'contradictory'}),('methods',{})])
def test_worker_preflight_mismatch_rejected(tmp_path,field,value):
    run=write_run(tmp_path/'run')
    rewrite_metadata(run,'preflight-000.json',lambda x:x.update({field:value}))
    with pytest.raises(ValueError,match='Worker preflight/completion mismatch'):
        analysis.load_run(run)


@pytest.mark.parametrize('field,value',[
    ('status','RUNNING'),('source_files_verified',0),('rows',0),('rows',True),
    ('failures',1),('failures',False),('rows_sha256','0'*64)])
def test_worker_completion_counts_status_digest_rejected(tmp_path,field,value):
    run=write_run(tmp_path/'run')
    rewrite_metadata(run,'completion-000.json',lambda x:x.update({field:value}))
    with pytest.raises(ValueError,match='Worker'):
        analysis.load_run(run)


def test_contradictory_shard_with_valid_own_digest_rejected_against_merge(tmp_path):
    run=write_run(tmp_path/'run')
    shard=run/'rows-000.jsonl'
    records=[json.loads(line) for line in shard.read_text().splitlines()]
    set_loss(records[0],.9)
    shard.write_text(''.join(json.dumps(r)+'\n' for r in records))
    rewrite_metadata(run,'completion-000.json',lambda x:x.update(rows_sha256=analysis.digest(shard)))
    with pytest.raises(ValueError,match='contradict merged'):
        analysis.load_run(run)


def test_reordered_valid_worker_rows_cannot_claim_original_merged_bytes(tmp_path):
    run=write_run(tmp_path/'run')
    shard=run/'rows-000.jsonl'
    lines=shard.read_text().splitlines(keepends=True)
    shard.write_text(''.join(reversed(lines)))
    rewrite_metadata(run,'completion-000.json',lambda x:x.update(rows_sha256=analysis.digest(shard)))
    with pytest.raises(ValueError,match='contradict merged'):
        analysis.load_run(run)


def test_wrong_worker_seed_assignment_rejected_even_with_resealed_merged(tmp_path):
    p=plan();p['workers']=2
    run=write_run(tmp_path/'run',p=p)
    raw=[(run/f'rows-{w:03d}.jsonl').read_text() for w in range(2)]
    for w in range(2):
        shard=run/f'rows-{w:03d}.jsonl';shard.write_text(raw[1-w])
        rewrite_metadata(run,f'completion-{w:03d}.json',
                         lambda x,s=shard:x.update(rows=len(s.read_text().splitlines()),rows_sha256=analysis.digest(s)))
    (run/'rows.jsonl').write_text(raw[1]+raw[0])
    rewrite_metadata(run,'completion.json',lambda x:x.update(rows_sha256=analysis.digest(run/'rows.jsonl')))
    with pytest.raises(ValueError,match='Wrong worker shard'):
        analysis.load_run(run)


@pytest.mark.parametrize('field',['torch','botorch','gpytorch','platform','machine','python_abi'])
def test_combined_requires_complete_runtime_equality_after_valid_individual_seals(tmp_path,field):
    p,rows,validation=trigger_fixture()
    dev=write_run(tmp_path/'dev',p=p,rows=[r for r in rows if r['arm']!=validation['name']])
    valplan={'stage':'SELECTION_VALIDATION','workers':1,'arms':[validation]}
    val=write_run(tmp_path/'val',p=valplan,rows=[r for r in rows if r['arm']==validation['name']])
    rewrite_metadata(val,'protocol.json',lambda x:x['versions'].update({field:'literal-different-version'}))
    rewrite_metadata(val,'preflight-000.json',lambda x:x['versions'].update({field:'literal-different-version'}))
    # Each run is internally coherent; their joint execution environments are not.
    analysis.load_run(dev);analysis.load_run(val)
    with pytest.raises(ValueError,match='Complete runtime versions drift'):
        analysis.analyze_run(val,development=dev)
    assert not (val/'analysis_supplement_v1').exists()


def test_missing_runtime_field_is_not_treated_as_equal_unknown(tmp_path):
    run=write_run(tmp_path/'run')
    rewrite_metadata(run,'protocol.json',lambda x:x['versions'].pop('torch'))
    rewrite_metadata(run,'preflight-000.json',lambda x:x['versions'].pop('torch'))
    with pytest.raises(ValueError,match='Complete captured runtime versions'):
        analysis.load_run(run)


def test_empty_worker_and_failed_case_are_verified_without_dropping(tmp_path):
    p=plan();p['workers']=2;p['arms'][0]['seeds']=[101]
    rows=rows_for(p);rows[0]=failed(rows[0])
    run=write_run(tmp_path/'run',p=p,rows=rows)
    verified=analysis.load_run(run)
    assert len(verified.rows)==4 and verified.rows[0]['status']=='FAIL'
    assert (run/'rows-001.jsonl').read_text()==''
    assert json.loads((run/'completion-000.json').read_text())['failures']==1

"""Materialize N and disjoint seeds before any new utility world is generated.

Implements literature/DOMAIN_TRANSFER_PLAN_RU.md, proposed by literature_map.
No simulator or method constructor is imported. This is an engineering target
distribution, not a fitted distribution of real hiring requests.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .io import BASE_METHODS

HERE=Path(__file__).resolve().parent
FAMILIES=['linear','geometric','power','bottleneck','saturation','interaction','choquet',
          'threshold','min_linear','max_linear','smooth_neural','answer_noise','query_skips','correlated_quantized']
METHODS=list(BASE_METHODS)
BANDS=((6,25),(26,60),(61,100))


def dump(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def materialize(*, smoke=False):
    count=2 if smoke else 1024
    families=['linear','correlated_quantized'] if smoke else FAMILIES
    design_seed=202610199090 if smoke else 202610129001
    rng=np.random.default_rng(design_seed)
    strata=[];main=[];cursor=202610199000 if smoke else 202610120000
    for d in (2,3):
        for lo,hi in BANDS:
            name=f'd{d}_n{lo}_{hi}';seeds=list(range(cursor,cursor+count));cursor+=count
            sizes=rng.integers(lo,hi+1,size=count)
            strata.append({'name':name,'d':d,'low':lo,'high':hi,'weight_numerator':hi-lo+1,
                           'weight_denominator':190,'seeds':seeds})
            for n in range(lo,hi+1):
                roots=[s for s,size in zip(seeds,sizes) if int(size)==n]
                if roots:
                    main.append({'name':f'main_n{n}_d{d}','role':'MAIN','stratum':name,'n':n,'d':d,
                                 'variant':'base','seeds':roots,'families':families,'methods':METHODS})
    landmarks=[];cursor=202610199020 if smoke else 202610127000
    landmark_count=2 if smoke else 64
    for n in (18,50,100):
        for d in (2,3):
            landmarks.append({'name':f'landmark_n{n}_d{d}','role':'LANDMARK','n':n,'d':d,
                              'variant':'base','seeds':list(range(cursor,cursor+landmark_count)),
                              'families':families,'methods':METHODS});cursor+=landmark_count
    timing=[];cursor=202610199040 if smoke else 202610128000
    timing_count=2 if smoke else 16
    for d in (2,3):
        timing.append({'name':f'timing_n100_d{d}','role':'TIMING','n':100,'d':d,'variant':'base',
                       'seeds':list(range(cursor,cursor+timing_count)),'families':families,'methods':METHODS});cursor+=timing_count
    contract={'schema':'fsp-domain-analysis-v1','alpha':.05,'hypotheses':4+4*len(families)+4*len(strata)+4+1,
              'target':'N uniform integers6..100; d uniform2/3; family uniform14; base variant',
              'strata':strata,'families':families,'methods':METHODS,
              'candidates':['mixed_kg_finer_quadrature','P15_easy'],
              'baselines':['linear_evoi_3_boundary_guard','gp_topk_guard'],
              'descriptive_controls':['pabbo_rl','equal_boundary_guard'],
              'mean_upper_strict_bound':-.001,'family_and_stratum_upper_strict_bound':.005,
              'cvar_upper_strict_bound':.005,'cvar_alpha':.95,
              'bootstrap_resamples':199 if smoke else 49999,'bootstrap_seed':202610199091 if smoke else 202610129002,'bootstrap_batch':64,
              'bootstrap_prefix_fraction':.125,
              'bootstrap_rule':'independent multinomial root counts within each stratum; same counts across methods/families; exact fractional weighted tail; basic upper bound',
              'mean_interval':'paired stratum means; weighted Welch-Satterthwaite; zero variance gives degenerate point interval',
              'contrast':'mixed_kg_finer_quadrature minus P15_easy; adjusted two-sided interval; practical margin .001',
              'latency_step_p95_strict_bound':.25,'timing_dimensions':[2,3],
              'decision':'zero eligible => no supported domain replacement; one => sole passing configuration; two => contrast margin or unresolved pair',
              'failures':'retain every case; any execution/contract/fallback/restricted/calibration failure makes method ineligible; do not impute or drop failed outcomes',
              'no_optional_stopping':True,'no_post_outcome_tuning':True,
              'global_alpha_across_previous_studies_claimed':False,
              'human_quality':'NOT_RUN'}
    quality={'stage':'HARNESS_SMOKE' if smoke else 'DOMAIN_TRANSFER','study_role':'QUALITY','workers':2,
             'arms':main+landmarks,'analysis_contract':contract,
             'materialization_seed':design_seed,'worlds_generated_during_design':0}
    timing_plan={'stage':'HARNESS_SMOKE' if smoke else 'DOMAIN_TRANSFER','study_role':'TIMING','workers':1,
                 'arms':timing,'analysis_contract':contract,'worlds_generated_during_design':0,
                 'launch_condition':'After all quality workers finish; no other research compute during timing'}
    return quality,timing_plan


def main():
    quality,timing=materialize()
    ledger=HERE.parent/'SEED_LEDGER.json'
    previous={'schema':'fsp-domain-seeds-v1','primary_seed_ledger_sha256':sha256(ledger.read_bytes()).hexdigest(),
              'previous_worlds':{'world_seed_values':[],
                'world_seed_ranges':[{'start':202610100000,'end_inclusive':202610114095}]},
              'note':'Conservatively excludes all supplemental observed/reserved roots, in addition to original ledger. Design/bootstrap roots are separate streams.',
              'reserved_harness_roots':[202610199990,202610199991],
              'reserved_quality_ranges':[[202610120000,202610126143],[202610127000,202610127383]],
              'reserved_timing_range':[202610128000,202610128031],
              'design_rng':202610129001,'bootstrap_rng':202610129002}
    dump(HERE/'DOMAIN_QUALITY_PLAN.json',quality)
    dump(HERE/'DOMAIN_TIMING_PLAN.json',timing)
    dump(HERE/'DOMAIN_ANALYSIS_PLAN.json',quality['analysis_contract'])
    dump(HERE/'DOMAIN_SEED_LEDGER.json',previous)
    w=np.array([s['weight_numerator']/s['weight_denominator'] for s in quality['analysis_contract']['strata']])
    za,zp=norm.ppf(1-.05/89),norm.ppf(.9)
    planning=[]
    for sd in (.01448538,.02749462):
        se=sd*np.sqrt(np.sum(w*w)/1024)
        planning.append({'proxy_sd':sd,'normal_n90_per_stratum':int(np.ceil(np.sum(w*w)*sd**2*(za+zp)**2/.001**2)),
                         'normal_approx_power_at_1024':float(norm.cdf(.001/se-za)),
                         'two_sided_mean_halfwidth':float(norm.ppf(1-.05/(2*89))*se)})
    result={'status':'MATERIALIZED_NOT_FROZEN','worlds_generated':0,'quality_worlds':sum(len(a['seeds'])*14 for a in quality['arms']),
            'timing_worlds':448,'method_cases':sum(len(a['seeds'])*len(a['families'])*len(a['methods']) for p in (quality,timing) for a in p['arms']),
            'main_execution_arms':sum(a['role']=='MAIN' for a in quality['arms']),
            'planning_scenarios':planning,'power_scope':'Normal planning proxies from disclosed old models, not a power guarantee for new guards, tails or family gates',
            'files':{p.name:sha256(p.read_bytes()).hexdigest() for p in HERE.glob('DOMAIN_*PLAN.json')}}
    dump(HERE/'design-receipt-v1.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

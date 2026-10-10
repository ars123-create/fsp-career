"""Complete decision branches and sampling design checked on literal data."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from .design import materialize,HERE
from .analyze import validate_study_plan,collect_main,inference,decide
from .benchmark import validate_plan


def literal():
    plan,timing_plan=materialize(smoke=True);contract=plan['analysis_contract']
    levels={'mixed_kg_finer_quadrature':.10,'P15_easy':.12,'linear_evoi_3_boundary_guard':.20,
            'gp_topk_guard':.23,'pabbo_rl':.30,'equal_boundary_guard':.32}
    rows=[]
    for arm in plan['arms']:
        for seed in arm['seeds']:
            for family in arm['families']:
                for method in arm['methods']:
                    rows.append({'arm':arm['name'],'seed':seed,'family':family,'method':method,'status':'PASS',
                       'constraint_status':'PASS','solver_fallback':False,'restricted_terminal_search':False,
                       'regret':levels[method],'auxiliary':{'best_of_slate_regret':.02},
                       'oracle_overlap_tie_aware':.8,'guard_applications':0,'calibration':{'status':'NOT_AVAILABLE'}})
    return SimpleNamespace(protocol={'plan':plan},rows=rows),contract


def timing(contract, *, seconds=.01):
    return {f'timing_n100_d{d}':{m:{'flags':{},'step_p95_seconds':seconds}
             for m in contract['methods']} for d in contract['timing_dimensions']}


def test_production_design_matches_saved_inputs_and_full_stratification():
    quality,latency=materialize()
    assert quality==json.loads((HERE/'DOMAIN_QUALITY_PLAN.json').read_text())
    assert latency==json.loads((HERE/'DOMAIN_TIMING_PLAN.json').read_text())
    assert validate_plan(quality)==548352 and validate_plan(latency)==2688
    assert len([a for a in quality['arms'] if a['role']=='MAIN'])==190
    contract=validate_study_plan(quality);validate_study_plan(latency)
    assert contract['hypotheses']==89
    strata={s['name']:s for s in contract['strata']}
    actual={name:set() for name in strata}
    for arm in quality['arms']:
        if arm['role']!='MAIN':continue
        s=strata[arm['stratum']]
        assert s['low']<=arm['n']<=s['high'] and s['d']==arm['d']
        actual[s['name']].update(arm['seeds'])
    for name,s in strata.items():assert actual[name]==set(s['seeds']) and len(actual[name])==1024
    assert sum(s['weight_numerator'] for s in strata.values())==190


@pytest.mark.parametrize('change',[
    lambda p:p['arms'][0].update(n=100),
    lambda p:p['arms'][0]['seeds'].reverse(),
    lambda p:p['analysis_contract'].update(hypotheses=1),
    lambda p:p['analysis_contract'].update(mean_upper_strict_bound=0),
    lambda p:p['analysis_contract']['strata'][0].update(weight_numerator=10),
    lambda p:p.update(workers=1),
])
def test_design_mutation_rejected_before_generation(change):
    p,_=materialize();change(p)
    with pytest.raises(ValueError):validate_study_plan(p)


def test_literal_complete_inference_and_all_decision_branches():
    verified,contract=literal();values,flags,guards=collect_main(verified,contract)
    summary,comparisons,contrast,tails=inference(values,flags,contract)
    assert summary['P15_easy']['metrics']['sum_regret']['weighted_mean']==pytest.approx(.12)
    assert all(x['all_quality_gates_pass'] for v in comparisons.values() for x in v.values())
    timed=timing(contract)
    decision=decide(comparisons,contrast,timed,contract)
    assert decision['selected']=='mixed_kg_finer_quadrature'
    unclear=dict(contrast,two_sided=[-.0005,.0005])
    assert decide(comparisons,unclear,timed,contract)['result']=='ELIGIBLE_PAIR_NOT_SEPARATED'
    reverse=dict(contrast,two_sided=[.01,.03])
    assert decide(comparisons,reverse,timed,contract)['selected']=='P15_easy'
    timed['timing_n100_d2']['mixed_kg_finer_quadrature']['step_p95_seconds']=.25
    assert decide(comparisons,contrast,timed,contract)['selected']=='P15_easy'
    timed['timing_n100_d3']['P15_easy']['step_p95_seconds']=.251
    assert decide(comparisons,contrast,timed,contract)['result']=='NO_SUPPORTED_DOMAIN_REPLACEMENT'
    assert decide(comparisons,contrast,timing(contract),contract,{'linear_evoi_3_boundary_guard':{'execution_failure':1}})['selected'] is None


def test_failed_challenger_does_not_remove_valid_other_candidate_tail():
    verified,contract=literal();values,flags,_=collect_main(verified,contract)
    flags['P15_easy']={'execution_failure':1};values[0,0,0,1,:]=np.nan
    _,comparisons,contrast,tails=inference(values,flags,contract)
    assert contrast is None and tails['methods']==['mixed_kg_finer_quadrature']+contract['baselines']
    assert comparisons['P15_easy'][contract['baselines'][0]]['status']=='NOT_AVAILABLE_OPERATIONAL_FAILURE'
    assert all(c['all_quality_gates_pass'] for c in comparisons['mixed_kg_finer_quadrature'].values())
    assert decide(comparisons,contrast,timing(contract),contract)['selected']=='mixed_kg_finer_quadrature'


def test_missing_main_root_is_rejected_instead_of_averaged_away():
    verified,contract=literal();verified.rows.pop(0)
    with pytest.raises(ValueError,match='Missing'):collect_main(verified,contract)

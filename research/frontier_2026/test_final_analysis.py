"""Constructed analysis-only fixtures. No real final data or generator imports."""
from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pytest
import scipy
from scipy.stats import t

from . import analyze_final as analysis
from .statistics import cvar_difference_interval


def engineering():
    return {"status":"PASS","n":100,"speed_order":["P15_easy","mixed_kg_finer_quadrature"],
        "step_budget_seconds":.25,"candidates":{name:{"status":"PASS","step_p95_seconds":speed,
            "execution_failures":0,"constraint_failures":0,"solver_fallback_cases":0,"restricted_terminal_cases":0}
            for name,speed in zip(analysis.CANDIDATES,[.239,.078])}}


def protocol():
    return {"stage":"FINAL_HELDOUT","not_final_test":False,"data_generated_during_freeze":False,
        "candidates":list(analysis.CANDIDATES),"confirmatory_baselines":list(analysis.BASELINES),
        "descriptive_controls":list(analysis.CONTROLS),"seeds":[801,802,803,804],
        "args":{"n":18,"d":3,"variant":"base","seeds":4,"families":list(analysis.FAMILIES),
                "methods":list(analysis.CANDIDATES+analysis.BASELINES+analysis.CONTROLS)},
        "statistical_plan":{"alpha":.05,"multiplicity":65,"cvar_alpha":.95,
            "bootstrap_resamples":49999,"bootstrap_seed":202610084005,"family_weighting":"1/14 each",
            "invalid_input_policy":analysis.INVALID_INPUT_POLICY,
            "no_optional_stopping":True,"no_failed_case_deletion":True,
            "candidate_contrast":{"enabled":True,"ordered_pair":list(analysis.CANDIDATES),
                "interval":"two-sided paired cluster t with Bonferroni","practical_margin":.001},
            "winner_rule":{"fallback_reference":"gp_topk_guard"},
            "descriptive_controls_not_in_confirmatory_claim_family":True,
            "practical_replacement_gates":{"mean_difference_upper_below":-.001,
                "each_family_difference_upper_below":.005,"cvar_difference_upper_below":.005}},
        "engineering_audit":engineering()}


def rows(p=None):
    p = protocol() if p is None else p
    methods = p["args"]["methods"]
    result = []
    for s,f,m in product(p["seeds"],analysis.FAMILIES,methods):
        loss = [.04,.05,.09,.1,.11,.12][methods.index(m)]+.0001*(s-p["seeds"][0])+.0002*analysis.FAMILIES.index(f)
        result.append({"seed":s,"family":f,"method":m,"n":18,"d":3,"variant":"base",
            "status":"PASS","constraint_status":"PASS","solver_fallback":False,
            "restricted_terminal_search":False,"regret":loss,"dominance_violations":0,
            "history":[],"slots":0,"answers":0,"regret_by_slot":[loss]*4,
            "construction_seconds":.001,"total_seconds":.002,
            "trace":[{"slot":0,"selection":[0,1,2,3,4],"query_id":None,"seconds":.001,
                      "regret":loss,"dominance_violations":0}],
            "calibration":{"status":"MODEL_PREDICTIVE","asked_fraction":0.,
                "metrics":{"categorical_expected_brier":.3,"categorical_expected_logloss":.7},
                "nonfinite_metric_counts":{"categorical_expected_brier":0,"categorical_expected_logloss":0}}})
    return result


def write_rows(path,items):
    with path.open("w") as stream:
        for row in items:
            stream.write(json.dumps(row)+"\n")


def scalar_cvar(values,alpha):
    ordered = sorted(map(float,values),reverse=True)
    left,total = (1-alpha)*len(ordered),0.
    for value in ordered:
        mass = min(max(left,0.),1.)
        total += mass*value
        left -= mass
    return total/((1-alpha)*len(ordered))


@pytest.mark.parametrize("alpha",[0.,.5,.65,.95])
def test_weighted_tail_bootstrap_matches_independent_explicit_resamples(alpha):
    x = np.random.default_rng(81902).uniform(size=(5,3,2))
    # Ties and an all-zero cluster exercise discrete atoms and zero multiplicity.
    x[0] = 0.
    x[1,0] = x[1,1]
    observed,sampled,metadata = analysis.bootstrap_cvars(x,alpha=alpha,resamples=333,seed=771,batch=17)
    draws = np.random.default_rng(771).integers(0,len(x),size=(333,len(x)))
    independent = np.array([[scalar_cvar(x[draw,:,m].ravel(),alpha) for m in range(2)] for draw in draws])
    np.testing.assert_allclose(sampled,independent,atol=2e-14,rtol=0)
    np.testing.assert_allclose(observed,[scalar_cvar(x[:,:,m].flat,alpha) for m in range(2)],atol=1e-14)
    assert metadata["largest_sorted_prefix"] <= 15
    _,other,_ = analysis.bootstrap_cvars(x,alpha=alpha,resamples=333,seed=771,batch=128)
    np.testing.assert_array_equal(sampled,other)


def test_basic_upper_interval_matches_independent_scipy_nonconstant_oracle():
    x = np.random.default_rng(81904).uniform(size=(6,4,2))
    observed,samples,_ = analysis.bootstrap_cvars(x,resamples=999,seed=analysis.BOOTSTRAP_SEED)
    actual = analysis.tail_interval(observed,samples,0,1)
    expected = cvar_difference_interval(x[:,:,0],x[:,:,1],alpha=.95,hypotheses=65,
                                       resamples=999,seed=analysis.BOOTSTRAP_SEED)
    assert actual["difference"] == pytest.approx(expected["difference"],abs=2e-14)
    assert actual["upper_one_sided"] == pytest.approx(expected["upper_one_sided"],abs=2e-14)
    assert actual["bootstrap_standard_error"] == pytest.approx(expected["bootstrap_standard_error"],abs=2e-14)


def test_adaptive_prefix_reaches_tail_when_top_clusters_are_absent():
    values = np.arange(30,0,-1.)
    ids = np.repeat(np.arange(10),3)
    counts = np.zeros((2,10),int)
    counts[0,-1],counts[1,-2] = 10,10
    actual,prefix = analysis._weighted_upper_tail(values,ids,counts,.95)
    assert prefix == 30
    np.testing.assert_allclose(actual,[3.,6.])


@pytest.mark.parametrize("defect",["duplicate","missing","failed","constraint","nan","fallback","nested_fallback","cardinality","history","unplanned","configuration"])
def test_streaming_rejects_every_confirmatory_defect_without_dropping(tmp_path,defect):
    p,items = protocol(),rows()
    if defect == "duplicate": items.append(deepcopy(items[0]))
    elif defect == "missing": items.pop(0)
    elif defect == "failed": items[0]["status"] = "FAIL"
    elif defect == "constraint": items[0]["constraint_status"] = "FAIL"
    elif defect == "nan": items[0]["regret"] = float("nan")
    elif defect == "fallback": items[0]["solver_fallback"] = True
    elif defect == "nested_fallback": items[0]["trace"][0]["diagnostics"] = {"chance_postcheck_fallback":{"action":"baseline"}}
    elif defect == "cardinality": items[0]["trace"][0]["selection"] = [0,0,1,2,3]
    elif defect == "history": items[0]["history"] = [[0,2]]
    elif defect == "unplanned": items[0]["seed"] = -1
    elif defect == "configuration": items[0]["variant"] = "other"
    path = tmp_path/"rows.jsonl"
    write_rows(path,items)
    result = analysis.read_rows(path,p)
    assert result["issues"]["total"] > 0
    assert result["rows"] == len(items)
    assert result["rows_sha256"] == analysis.file_sha256(path)


def test_descriptive_failure_and_nonfinite_calibration_are_explicit_not_replacements(tmp_path):
    p,items = protocol(),rows()
    items[4]["status"] = "FAIL"  # PABBO, not a confirmatory control.
    items[0]["calibration"]["metrics"]["categorical_expected_logloss"] = None
    items[0]["calibration"]["nonfinite_metric_counts"]["categorical_expected_logloss"] = 2
    path = tmp_path/"rows.jsonl"
    write_rows(path,items)
    data = analysis.read_rows(path,p)
    assert data["issues"]["total"] == 0
    assert data["descriptive_issues"]["total"] == 1
    report = analysis.descriptive_report(data,p["args"]["methods"])
    assert report["pabbo_rl"]["mean_regret"] is None
    metric = report[analysis.CANDIDATES[0]]["calibration"]["metrics"]["categorical_expected_logloss"]
    assert metric["status"] == "PARTIAL" and metric["all_case_mean"] is None
    assert metric["valid_cases"] == 55 and metric["nonfinite_queries"] == 2
    json.dumps(report,allow_nan=False)


def test_decision_rules_no_single_and_two_pass_and_no_equivalence():
    a,b = analysis.CANDIDATES
    weak = {"two_sided":[-.0005,.0004]}
    assert analysis.choose_candidate({a:False,b:False},weak,engineering())["selected"] is None
    assert analysis.choose_candidate({a:True,b:False},weak,engineering())["selected"] == a
    tied = analysis.choose_candidate({a:True,b:True},weak,engineering())
    assert tied["selected"] == b and not tied["quality_best_established_among_both_finalists"]
    assert tied["equivalence_established"] is False
    assert analysis.choose_candidate({a:True,b:True},{"two_sided":[-.02,-.002]},engineering())["selected"] == a
    assert analysis.choose_candidate({a:True,b:True},{"two_sided":[.002,.02]},engineering())["selected"] == b
    slow = engineering()
    slow["candidates"][a]["step_p95_seconds"] = .25
    assert analysis.choose_candidate({a:True,b:False},weak,slow)["selected"] is None
    exactly_tied = engineering()
    for name in (a,b): exactly_tied["candidates"][name]["step_p95_seconds"] = .1
    assert analysis.choose_candidate({a:True,b:True},weak,exactly_tied)["selected"] == b
    boundary = analysis.choose_candidate({a:True,b:True},{"two_sided":[-.01,-.001]},engineering())
    assert boundary["candidate_with_practical_pairwise_advantage"] is None


def test_strict_gates_and_cluster_family_count_independent_mean_formula(monkeypatch):
    p = protocol()
    x = np.zeros((4,14,6))
    shift = np.array([-.002,-.001,0.,.001])
    x[:,:,0] = .04+shift[:,None]
    x[:,:,1] = .05+shift[:,None]
    x[:,:,2:] = .10
    # Keep the statistical formulas but bound this unit test's resamples.
    original = analysis.bootstrap_cvars
    monkeypatch.setattr(analysis,"bootstrap_cvars",lambda cube,**kw:original(cube,resamples=199,**kw))
    result = analysis.confirmatory_analysis(x,p)
    row = result["comparisons"][analysis.CANDIDATES[0]][analysis.BASELINES[0]]["overall"]
    difference = x[:,:,0].mean(axis=1)-.1
    expected = difference.mean()+t.ppf(1-.05/65,3)*difference.std(ddof=1)/np.sqrt(4)
    assert row["upper_one_sided"] == pytest.approx(expected,abs=1e-14)
    assert sum(result["hypothesis_count_breakdown"].values()) == 65
    assert len(result["comparisons"][analysis.CANDIDATES[0]][analysis.BASELINES[0]]["families"]) == 14
    assert result["candidate_contrast"]["two_sided"][1] < -.001


def test_exact_mean_margin_does_not_pass_a_strict_gate(monkeypatch):
    x = np.zeros((4,14,6))
    x[:,:,2:4] = .001
    original = analysis.bootstrap_cvars
    monkeypatch.setattr(analysis,"bootstrap_cvars",lambda cube,**kw:original(cube,resamples=199,**kw))
    # This tests the decision boundary, independently of decimal accumulation
    # in the statistic. Summing fourteen float .001 values is not an exact-real
    # equality oracle; the separate nonconstant test checks the statistic.
    interval = analysis.stats.cluster_mean_interval
    def exact_boundary(values,**kwargs):
        result = interval(values,**kwargs)
        result["upper_one_sided"] = -.001
        return result
    monkeypatch.setattr(analysis.stats,"cluster_mean_interval",exact_boundary)
    result = analysis.confirmatory_analysis(x,protocol())
    for candidate in analysis.CANDIDATES:
        for row in result["comparisons"][candidate].values():
            assert row["overall"]["upper_one_sided"] == -.001
            assert row["overall"]["passed"] is False
    assert result["decision"]["selected"] is None
    assert result["decision"]["fallback_reference"] == "gp_topk_guard"


def make_completed_fixture(tmp_path):
    run = tmp_path/"fake-final"
    snapshot = run/"snapshot"
    target = snapshot/"research/frontier_2026"
    target.mkdir(parents=True)
    (snapshot/"research/__init__.py").write_text("")
    (target/"__init__.py").write_text("")
    for source in (Path(analysis.__file__),Path(analysis.stats.__file__)):
        shutil.copyfile(source,target/source.name)
    manifest = {str(path.relative_to(snapshot)):analysis.file_sha256(path) for path in snapshot.rglob("*.py")}
    p = protocol()
    eng = tmp_path/"fake-engineering"
    eng.mkdir()
    for name in ("protocol.json","summary.json","completion.json","rows.jsonl"):
        (eng/name).write_text('{}\n')
    p["engineering_audit"].update(source_directory=str(eng),source_files_sha256={name:analysis.file_sha256(eng/name) for name in ("protocol.json","summary.json","completion.json","rows.jsonl")})
    p.update(source_sha256=manifest,methods={name:{"fixture":True} for name in p["args"]["methods"]},
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch="fixture")
    p["args"]["out"] = str(run)
    (run/"protocol.json").write_text(json.dumps(p))
    (run/"protocol.sha256").write_text(analysis.file_sha256(run/"protocol.json")+'  protocol.json\n')
    (run/"snapshot-manifest.json").write_text(json.dumps(manifest))
    data = rows(p)
    write_rows(run/"rows.jsonl",data)
    completion = {"status":"COMPLETE","rows":len(data),"snapshot_files_changed_while_running":[],
        "executed_workspace":str(snapshot),"rows_sha256":analysis.file_sha256(run/"rows.jsonl")}
    (run/"completion.json").write_text(json.dumps(completion))
    worker = {"status":"PASS","before_first_task":True,"methods":p["methods"],
        "source_hashes_verified":len(manifest),"versions":{k:p[k] for k in ("python","numpy","scipy","torch")},
        "executed_workspace":str(snapshot)}
    (run/"worker-config-verification.json").write_text(json.dumps(worker))
    return run


def run_cli(run,out=None):
    args = [sys.executable,"-m","research.frontier_2026.analyze_final","--run",str(run)]
    if out:
        args += ["--out",str(out)]
    return subprocess.run(args,cwd=run/"snapshot",env={**os.environ,"PYTHONPATH":str(run/"snapshot")},
                          text=True,capture_output=True,timeout=60)


def test_completed_fake_snapshot_cli_and_refuse_overwrite(tmp_path):
    run = make_completed_fixture(tmp_path)
    result = run_cli(run)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((run/"analysis/analysis.json").read_text())
    assert report["status"] == "COMPLETE" and report["generated_cases"] is False
    assert report["confirmatory"]["decision"]["selected"] == analysis.CANDIDATES[0]
    assert report["confirmatory"]["bootstrap"]["resamples"] == 49999
    assert report["confirmatory"]["bootstrap"]["approximate_peak_working_bytes"] < 1024**3
    before = analysis.file_sha256(run/"analysis/analysis.json")
    again = run_cli(run)
    assert again.returncode != 0
    assert analysis.file_sha256(run/"analysis/analysis.json") == before


@pytest.mark.parametrize("tamper",["protocol","source","rows","completion","worker"])
def test_invalid_artifact_never_creates_confirmatory_conclusion(tmp_path,tamper):
    run = make_completed_fixture(tmp_path)
    if tamper == "protocol":
        (run/"protocol.json").write_text((run/"protocol.json").read_text()+" ")
    elif tamper == "source":
        path = run/"snapshot/research/frontier_2026/statistics.py"
        path.write_text(path.read_text()+"\n# changed after freeze\n")
    elif tamper == "rows":
        with (run/"rows.jsonl").open("a") as stream: stream.write("\n")
    elif tamper == "completion":
        value = json.loads((run/"completion.json").read_text())
        value["status"] = "RUNNING"
        (run/"completion.json").write_text(json.dumps(value))
    else:
        value = json.loads((run/"worker-config-verification.json").read_text())
        value["before_first_task"] = False
        (run/"worker-config-verification.json").write_text(json.dumps(value))
    result = run_cli(run)
    assert result.returncode == 2
    report = json.loads((run/"analysis/analysis.json").read_text())
    assert report["status"] == "INVALID" and report["decision"]["selected"] is None
    assert "confirmatory" not in report


@pytest.mark.parametrize("change",["multiplicity","tail_seed","mean_margin","engineering","contrast","budget","speed_order","fallback"])
def test_protocol_gates_cannot_be_relaxed_after_results(change):
    p = protocol()
    if change == "multiplicity": p["statistical_plan"]["multiplicity"] = 4
    elif change == "tail_seed": p["statistical_plan"]["bootstrap_seed"] += 1
    elif change == "mean_margin": p["statistical_plan"]["practical_replacement_gates"]["mean_difference_upper_below"] = 0
    elif change == "engineering": p["engineering_audit"]["candidates"][analysis.CANDIDATES[0]]["solver_fallback_cases"] = 1
    elif change == "contrast": p["statistical_plan"]["candidate_contrast"]["ordered_pair"].reverse()
    elif change == "budget": p["engineering_audit"]["step_budget_seconds"] = .5
    elif change == "speed_order": p["engineering_audit"]["speed_order"].reverse()
    else: p["statistical_plan"]["winner_rule"]["fallback_reference"] = "equal"
    with pytest.raises(analysis.InvalidRun): analysis.validate_protocol(p)

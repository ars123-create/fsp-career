"""Analyze a completed frozen final run; never generate or refit a test case.

Primary reuse: local statistics.py and SciPy's Student-t distribution. SciPy
1.18.1 bootstrap resampling source (_bootstrap_resample) was read: each resample
draws C cluster indices using rng.integers(0,C,size=(B,C)). The implementation
below uses exactly those draws, but computes empirical CVaR from cluster counts
on a sorted finite loss table. It is algebraically the same basic paired
bootstrap, not a tail approximation or a new statistical method. Independent
tests compare it to SciPy and scalar mass integration. NumPy/SciPy retain their
upstream licenses; no upstream implementation code is copied here.

CLI must run from this run's captured snapshot. Only NEW output directories are
written. Inputs are streamed/hashes checked; no make_world, registry, model fit
or application imports occur. See protocol.statistical_plan for scope.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from . import statistics as stats

CANDIDATES = ("mixed_kg_finer_quadrature", "P15_easy")
BASELINES = ("linear_evoi_3", "gp_topk_guard")
CONTROLS = ("pabbo_rl", "equal")
FAMILIES = ("linear", "geometric", "power", "bottleneck", "saturation", "interaction",
            "choquet", "threshold", "min_linear", "max_linear", "smooth_neural",
            "answer_noise", "query_skips", "correlated_quantized")
MULTIPLICITY, RESAMPLES, BOOTSTRAP_SEED = 65, 49999, 202610084005
ALPHA, CVAR_ALPHA = .05, .95
MEAN_MARGIN, FAMILY_MARGIN, TAIL_MARGIN, CONTRAST_MARGIN = -.001, .005, .005, .001
STEP_BUDGET = .25
INVALID_INPUT_POLICY = "Any confirmatory candidate or baseline failure, fallback, missing or duplicate row invalidates the entire confirmatory analysis; descriptive controls retain explicitly incomplete summaries"


class InvalidRun(ValueError):
    """A failed validation prevents any confirmatory conclusion."""


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value):
    raise InvalidRun(f"Nonfinite JSON constant: {value}")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_constant)


def _finite_number(value, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int, np.number)) or not np.isfinite(value):
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def validate_protocol(protocol):
    """Accept this prespecified experiment only; never infer gates from results."""
    if (protocol.get("stage") != "FINAL_HELDOUT" or protocol.get("not_final_test") is not False
            or protocol.get("data_generated_during_freeze") is not False):
        raise InvalidRun("Expected a separately frozen final-heldout protocol")
    if tuple(protocol.get("candidates", ())) != CANDIDATES:
        raise InvalidRun("Unexpected frozen candidate order")
    if tuple(protocol.get("confirmatory_baselines", ())) != BASELINES or tuple(protocol.get("descriptive_controls", ())) != CONTROLS:
        raise InvalidRun("Unexpected baseline/control definitions")
    args = protocol["args"]
    if tuple(args.get("families", ())) != FAMILIES or tuple(args.get("methods", ())) != CANDIDATES+BASELINES+CONTROLS:
        raise InvalidRun("Expected all 14 equally weighted families and all six methods")
    if (args.get("n"), args.get("d"), args.get("variant")) != (18, 3, "base"):
        raise InvalidRun("Unexpected final test configuration")
    seeds = protocol.get("seeds", [])
    if (len(seeds) < 2 or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or args.get("seeds") != len(seeds)):
        raise InvalidRun("Expected at least two unique, complete seed clusters")
    plan = protocol["statistical_plan"]
    fixed = {"alpha": ALPHA, "multiplicity": MULTIPLICITY, "cvar_alpha": CVAR_ALPHA,
             "bootstrap_resamples": RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
             "invalid_input_policy": INVALID_INPUT_POLICY,
             "family_weighting": "1/14 each", "no_optional_stopping": True,
             "no_failed_case_deletion": True,
             "descriptive_controls_not_in_confirmatory_claim_family": True}
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise InvalidRun("Frozen statistical settings do not match the 65-comparison plan")
    gates = {"mean_difference_upper_below": MEAN_MARGIN,
             "each_family_difference_upper_below": FAMILY_MARGIN,
             "cvar_difference_upper_below": TAIL_MARGIN}
    if plan.get("practical_replacement_gates") != gates:
        raise InvalidRun("Unexpected practical replacement gates")
    contrast = plan.get("candidate_contrast", {})
    if (contrast.get("enabled") is not True or tuple(contrast.get("ordered_pair",())) != CANDIDATES
            or contrast.get("practical_margin") != CONTRAST_MARGIN
            or contrast.get("interval") != "two-sided paired cluster t with Bonferroni"):
        raise InvalidRun("Candidate contrast does not match the prefrozen paired two-sided test")
    if plan.get("winner_rule",{}).get("fallback_reference") != "gp_topk_guard":
        raise InvalidRun("Unexpected no-replacement research reference")
    audit = protocol.get("engineering_audit", {})
    if audit.get("status") != "PASS" or audit.get("n") != 100 or audit.get("step_budget_seconds") != STEP_BUDGET:
        raise InvalidRun("A passing prefrozen N100 engineering audit is required")
    for name in CANDIDATES:
        row = audit.get("candidates", {}).get(name, {})
        if row.get("status") != "PASS" or not _finite_number(row.get("step_p95_seconds"), minimum=0):
            raise InvalidRun(f"Missing finite prefrozen N100 timing for {name}")
        for key in ("execution_failures", "constraint_failures", "solver_fallback_cases", "restricted_terminal_cases"):
            if type(row.get(key)) is not int or row[key] != 0:
                raise InvalidRun(f"Unverified/failed prefrozen engineering contract: {name}/{key}")
    expected_order = sorted(CANDIDATES,key=lambda name:(audit["candidates"][name]["step_p95_seconds"],name))
    if audit.get("speed_order") != expected_order:
        raise InvalidRun("Frozen speed_order does not match declared timing/name tie rule")
    return protocol


def verify_artifacts(run):
    run = Path(run).resolve()
    protocol_path = run/"protocol.json"
    expected_digest = (run/"protocol.sha256").read_text().split()[0]
    if len(expected_digest) != 64 or file_sha256(protocol_path) != expected_digest:
        raise InvalidRun("Frozen protocol checksum mismatch")
    protocol = validate_protocol(_read_json(protocol_path))
    snapshot = run/"snapshot"
    if Path(__file__).resolve() != snapshot/"research/frontier_2026/analyze_final.py":
        raise InvalidRun("Run the analyzer from this run's frozen snapshot, not the editable workspace")
    if Path(stats.__file__).resolve() != snapshot/"research/frontier_2026/statistics.py":
        raise InvalidRun("statistics.py was imported outside the frozen snapshot")
    manifest = protocol.get("source_sha256", {})
    if not manifest or "research/frontier_2026/analyze_final.py" not in manifest:
        raise InvalidRun("Analysis source must be included in the prefrozen source manifest")
    if _read_json(run/"snapshot-manifest.json") != manifest:
        raise InvalidRun("Protocol and source manifest disagree")
    for relative, digest in manifest.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_sha256(snapshot/path) != digest:
            raise InvalidRun(f"Frozen source checksum mismatch: {relative}")
    completion = _read_json(run/"completion.json")
    if (completion.get("status") != "COMPLETE" or completion.get("snapshot_files_changed_while_running") != []
            or Path(completion.get("executed_workspace", "")).resolve() != snapshot):
        raise InvalidRun("Expected a completed unchanged snapshot execution")
    expected_rows = len(protocol["seeds"])*len(FAMILIES)*len(protocol["args"]["methods"])
    if completion.get("rows") != expected_rows:
        raise InvalidRun("Completion does not contain every planned row")
    worker = _read_json(run/"worker-config-verification.json")
    if (worker.get("status") != "PASS" or worker.get("before_first_task") is not True
            or worker.get("methods") != protocol["methods"]
            or worker.get("source_hashes_verified") != len(manifest)
            or Path(worker.get("executed_workspace", "")).resolve() != snapshot):
        raise InvalidRun("Worker did not verify frozen configurations before its first case")
    versions = {key:protocol[key] for key in ("python","numpy","scipy","torch")}
    if worker.get("versions") != versions:
        raise InvalidRun("Worker runtime versions differ from the frozen protocol")
    if any(versions[k] != v for k,v in {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}.items()):
        raise InvalidRun("Analysis NumPy/SciPy/Python versions differ from the frozen protocol")
    audit = protocol["engineering_audit"]
    source = Path(audit["source_directory"])
    if set(audit.get("source_files_sha256", {})) != {"protocol.json","summary.json","completion.json","rows.jsonl"}:
        raise InvalidRun("Prefrozen engineering provenance is incomplete")
    for name,digest in audit["source_files_sha256"].items():
        if file_sha256(source/name) != digest:
            raise InvalidRun(f"Prefrozen engineering evidence changed: {name}")
    return protocol, completion, expected_digest


def _weighted_upper_tail(sorted_values, cluster_ids, counts, alpha):
    """CVaR with integer bootstrap multiplicities; adaptive prefix is exact."""
    total = len(sorted_values)
    tail = (1-alpha)*total
    prefix = min(total, max(1, int(np.ceil(1.5*tail))+1))
    while True:
        weights = counts[:, cluster_ids[:prefix]]
        cumulative = np.cumsum(weights, axis=1)
        if np.all(cumulative[:, -1] >= tail):
            break
        if prefix == total:
            raise ArithmeticError("Bootstrap counts do not contain the planned probability mass")
        prefix = min(total, 2*prefix)
    cutoff = np.argmax(cumulative >= tail, axis=1)
    rows = np.arange(len(counts))
    weighted_sum = np.cumsum(weights*sorted_values[None, :prefix], axis=1)
    integral = weighted_sum[rows, cutoff]-(cumulative[rows, cutoff]-tail)*sorted_values[cutoff]
    return integral/tail, prefix


def bootstrap_cvars(cube, *, alpha=CVAR_ALPHA, resamples=RESAMPLES,
                    seed=BOOTSTRAP_SEED, batch=128, progress=None):
    """Paired cluster resamples for all methods, using bounded working arrays."""
    x = np.asarray(cube, dtype=float)
    if (x.ndim != 3 or x.shape[0] < 2 or min(x.shape[1:]) < 1 or not np.isfinite(x).all()
            or not 0 <= alpha < 1 or type(resamples) is not int or resamples < 100
            or type(batch) is not int or not 1 <= batch <= 1024):
        raise ValueError("Finite C×F×M data and valid bootstrap settings required")
    clusters, families, methods = x.shape
    observed = np.array([stats.empirical_cvar(x[:,:,m].ravel(),alpha) for m in range(methods)])
    table, ids = [], []
    for m in range(methods):
        flat = x[:,:,m].ravel()
        order = np.argsort(-flat,kind="stable")
        table.append(flat[order])
        ids.append(order//families)
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples,methods))
    largest_prefix = 0
    for start in range(0,resamples,batch):
        amount = min(batch,resamples-start)
        draws = rng.integers(0,clusters,size=(amount,clusters))
        offsets = np.arange(amount)[:,None]*clusters
        counts = np.bincount((draws+offsets).ravel(),minlength=amount*clusters).reshape(amount,clusters)
        for m in range(methods):
            samples[start:start+amount,m], prefix = _weighted_upper_tail(table[m],ids[m],counts,alpha)
            largest_prefix = max(largest_prefix,prefix)
        if progress is not None and (start == 0 or start+amount == resamples or (start//batch)%40 == 0):
            progress(start+amount,resamples)
    # Conservative approximate working-array bound, excluding small source cube.
    working_bytes = 8*(4*batch*clusters+4*batch*largest_prefix+resamples*methods)
    return observed,samples,{"resamples":resamples,"seed":seed,"batch":batch,
        "largest_sorted_prefix":largest_prefix,"full_sorted_length":clusters*families,
        "approximate_peak_working_bytes":working_bytes,
        "scope":"exact empirical weighted-tail algebra; paired seed-cluster basic bootstrap; approximate statistical coverage"}


def tail_interval(observed, samples, candidate, baseline):
    differences = samples[:,candidate]-samples[:,baseline]
    measured = float(observed[candidate]-observed[baseline])
    return {"difference":measured,
        "upper_one_sided":float(2*measured-np.quantile(differences,ALPHA/MULTIPLICITY,method="linear")),
        "bootstrap_standard_error":float(differences.std(ddof=1)),
        "cvar_alpha":CVAR_ALPHA,"confidence_alpha":ALPHA,"multiplicity":MULTIPLICITY,
        "resamples":len(samples),"bootstrap_seed":BOOTSTRAP_SEED,
        "adjusted_tail_probability":ALPHA/MULTIPLICITY,
        "expected_resamples_in_adjusted_tail":len(samples)*ALPHA/MULTIPLICITY,
        "quantile_method":"linear","interval_method":"basic one-sided upper, paired seed clusters"}


class Issues:
    def __init__(self):
        self.counts, self.examples = Counter(), []

    def add(self, kind, detail):
        self.counts[kind] += 1
        if len(self.examples) < 40:
            self.examples.append({"kind":kind,"detail":str(detail)})

    def report(self):
        return {"total":sum(self.counts.values()),"counts":dict(self.counts),"first_examples":self.examples}


def _decision_row(row, n):
    """Recheck finite values, slots/cardinality, trace consistency and fallbacks."""
    if row.get("status") != "PASS" or row.get("constraint_status") != "PASS":
        raise InvalidRun("execution_or_constraint_failure")
    for field in ("solver_fallback","restricted_terminal_search"):
        if row.get(field) is not False:
            raise InvalidRun(f"fallback_or_unverified_{field}")
    if not _finite_number(row.get("regret"),minimum=0,maximum=1+1e-12):
        raise InvalidRun("nonfinite_or_out_of_range_loss")
    history,trace = row.get("history"),row.get("trace")
    if (not isinstance(history,list) or not isinstance(trace,list) or not 0 <= len(history) <= 3
            or len(trace) != len(history)+1 or row.get("slots") != len(history)):
        raise InvalidRun("invalid_slot_history_trace")
    if any(not isinstance(h,list) or len(h)!=2 or type(h[0]) is not int or h[0]<0
           or type(h[1]) is not int or h[1] not in (0,1,2) for h in history):
        raise InvalidRun("invalid_response_history")
    if len({q for q,_ in history}) != len(history) or row.get("answers") != sum(r!=2 for _,r in history):
        raise InvalidRun("repeat_or_answer_count_mismatch")
    for slot,step in enumerate(trace):
        selected = step.get("selection",[])
        if (step.get("slot") != slot or len(selected) != min(5,n) or len(set(selected)) != len(selected)
                or any(type(i) is not int or not 0<=i<n for i in selected)):
            raise InvalidRun("invalid_cardinality_or_candidate_index")
        if (step.get("dominance_violations") != 0 or not _finite_number(step.get("regret"),minimum=0,maximum=1+1e-12)
                or not _finite_number(step.get("seconds"),minimum=0)):
            raise InvalidRun("nonfinite_or_constraint_failing_step")
        if slot < len(history):
            if (step.get("query_id"),step.get("choice")) != tuple(history[slot]):
                raise InvalidRun("trace_response_history_mismatch")
        elif step.get("query_id") is not None:
            raise InvalidRun("fourth_question_or_missing_response")
        diagnostic = step.get("diagnostics",{})
        if (diagnostic.get("fallback_from_milp") or diagnostic.get("chance_postcheck_fallback")
                or diagnostic.get("terminal_search_complete") is False):
            raise InvalidRun("nested_fallback_or_restricted_terminal")
    if (row.get("dominance_violations") != 0 or row["regret"] != trace[-1]["regret"]
            or row.get("regret_by_slot") != [trace[min(i,len(trace)-1)]["regret"] for i in range(4)]):
        raise InvalidRun("final_trace_metrics_mismatch")
    if not all(_finite_number(row.get(k),minimum=0) for k in ("construction_seconds","total_seconds")):
        raise InvalidRun("nonfinite_timing")
    return row["regret"]


def read_rows(path,protocol):
    """Stream large traces into a small loss cube; retain every defect count."""
    seeds,methods = protocol["seeds"],protocol["args"]["methods"]
    seed_idx,fam_idx,method_idx = ({value:i for i,value in enumerate(seq)} for seq in (seeds,FAMILIES,methods))
    shape = (len(seeds),len(FAMILIES),len(methods))
    losses = np.full(shape,np.nan)
    seen = np.zeros(shape,dtype=bool)
    curves = np.full((*shape,4),np.nan)
    timings = np.full((*shape,4),np.nan)
    construction = np.full(shape,np.nan)
    total_time = np.full(shape,np.nan)
    slots = np.full(shape,np.nan)
    issues, descriptive_issues = Issues(),Issues()
    statuses = {m:Counter() for m in methods}
    calibration = {m:{"statuses":Counter(),"metrics":{},"asked_fraction_sum":0.,"asked_fraction_cases":0} for m in methods}
    digest,count = sha256(),0
    with Path(path).open("rb") as stream:
        for line_number,raw in enumerate(stream,1):
            digest.update(raw)
            if not raw.strip():
                issues.add("blank_row",line_number)
                continue
            count += 1
            try:
                row = json.loads(raw,parse_constant=_reject_constant)
                s,f,m = row["seed"],row["family"],row["method"]
                if type(s) is not int or s not in seed_idx or f not in fam_idx or m not in method_idx:
                    raise InvalidRun("unplanned_row_identity")
                index = seed_idx[s],fam_idx[f],method_idx[m]
                if seen[index]:
                    issues.add("duplicate_row",f"line={line_number},key={(s,f,m)}")
                    continue
                seen[index] = True
                statuses[m][str(row.get("status","MISSING"))] += 1
                if (row.get("n"),row.get("d"),row.get("variant")) != (18,3,"base"):
                    raise InvalidRun("row_configuration_mismatch")
                target = issues if m in CANDIDATES+BASELINES else descriptive_issues
                try:
                    losses[index] = _decision_row(row,18)
                    curves[index] = row["regret_by_slot"]
                    for i,step in enumerate(row["trace"]):
                        timings[(*index,i)] = step["seconds"]
                    construction[index],total_time[index],slots[index] = row["construction_seconds"],row["total_seconds"],row["slots"]
                except (InvalidRun,KeyError,TypeError,ValueError) as exc:
                    target.add(str(exc),f"line={line_number},key={(s,f,m)}")
                cal = row.get("calibration",{})
                acc = calibration[m]
                acc["statuses"][str(cal.get("status","MISSING"))] += 1
                if _finite_number(cal.get("asked_fraction"),minimum=0,maximum=1):
                    acc["asked_fraction_sum"] += cal["asked_fraction"]
                    acc["asked_fraction_cases"] += 1
                for metric,value in cal.get("metrics",{}).items():
                    store = acc["metrics"].setdefault(metric,{"sum":0.,"valid_cases":0,"null_cases":0,
                        "nonfinite_queries":0,"family_sum":np.zeros(len(FAMILIES)),"family_count":np.zeros(len(FAMILIES),int)})
                    store["nonfinite_queries"] += int(cal.get("nonfinite_metric_counts",{}).get(metric,0))
                    if _finite_number(value):
                        store["sum"] += value
                        store["valid_cases"] += 1
                        store["family_sum"][fam_idx[f]] += value
                        store["family_count"][fam_idx[f]] += 1
                    else:
                        store["null_cases"] += 1
            except (InvalidRun,KeyError,TypeError,ValueError) as exc:
                issues.add("malformed_or_unplanned_row",f"line={line_number}: {exc}")
    for m,j in method_idx.items():
        missing = int((~seen[:,:,j]).sum())
        if missing:
            issues.add("missing_rows",f"method={m},missing={missing}")
    expected = int(np.prod(shape))
    if count != expected:
        issues.add("row_count_mismatch",f"observed={count},expected={expected}")
    return {"losses":losses,"curves":curves,"timings":timings,"construction":construction,
        "total_time":total_time,"slots":slots,"statuses":statuses,"calibration":calibration,
        "issues":issues.report(),"descriptive_issues":descriptive_issues.report(),
        "rows":count,"rows_sha256":digest.hexdigest(),"shape":shape}


def descriptive_report(data,methods):
    result = {}
    expected = data["shape"][0]*data["shape"][1]
    for j,method in enumerate(methods):
        loss = data["losses"][:,:,j]
        complete = bool(np.isfinite(loss).all())
        time = data["timings"][:,:,j,:]
        time = time[np.isfinite(time)]
        cal = data["calibration"][method]
        metrics = {}
        for key,value in cal["metrics"].items():
            count = value["valid_cases"]
            metrics[key] = {"status":"COMPLETE" if count==expected else "PARTIAL",
                "all_case_mean":value["sum"]/count if count==expected else None,
                "finite_case_mean_descriptive":value["sum"]/count if count else None,
                "valid_cases":count,"planned_cases":expected,"null_cases":value["null_cases"],
                "nonfinite_queries":value["nonfinite_queries"],
                "family_means_descriptive":{f:float(value["family_sum"][i]/n) if n else None
                                            for i,(f,n) in enumerate(zip(FAMILIES,value["family_count"]))},
                "scope":"Case means on fixed common pairs; missing/nonfinite cases never replaced by zero"}
        result[method] = {"status":"COMPLETE" if complete else "INVALID_OR_INCOMPLETE",
            "planned_cases":expected,"finite_valid_cases":int(np.isfinite(loss).sum()),
            "execution_status_counts":dict(data["statuses"][method]),
            "mean_regret":float(loss.mean()) if complete else None,
            "cvar95_regret":float(stats.empirical_cvar(loss.ravel(),CVAR_ALPHA)) if complete else None,
            "p95_regret":float(np.quantile(loss,.95)) if complete else None,
            "family_mean_regret":{f:float(loss[:,i].mean()) if np.isfinite(loss[:,i]).all() else None for i,f in enumerate(FAMILIES)},
            "mean_regret_by_slot":data["curves"][:,:,j,:].mean(axis=(0,1)).tolist() if complete else None,
            "mean_slots":float(data["slots"][:,:,j].mean()) if complete else None,
            "observed_n18_step_p95_seconds":float(np.quantile(time,.95)) if complete and len(time) else None,
            "observed_n18_construction_p95_seconds":float(np.quantile(data["construction"][:,:,j],.95)) if complete else None,
            "observed_n18_total_p95_seconds":float(np.quantile(data["total_time"][:,:,j],.95)) if complete else None,
            "calibration":{"status_counts":dict(cal["statuses"]),"metrics":metrics,
                "mean_asked_fraction_descriptive":cal["asked_fraction_sum"]/cal["asked_fraction_cases"] if cal["asked_fraction_cases"] else None,
                "asked_fraction_cases":cal["asked_fraction_cases"],
                "scope":"Synthetic response-prediction audit; no fitted/human calibration claim or confirmatory calibration threshold"},
            "confirmatory":method in CANDIDATES+BASELINES}
    return result


def choose_candidate(passed,contrast,engineering):
    """No observed final timing can influence the prefrozen speed tiebreak."""
    quality = [name for name in CANDIDATES if passed.get(name) is True]
    engineering_pass = {name:engineering["candidates"][name]["step_p95_seconds"] < STEP_BUDGET for name in CANDIDATES}
    practical_better = None
    low,high = contrast["two_sided"]
    if high < -CONTRAST_MARGIN:
        practical_better = CANDIDATES[0]
    elif low > CONTRAST_MARGIN:
        practical_better = CANDIDATES[1]
    eligible = [name for name in quality if engineering_pass[name]]
    selected,reason = None,"no_demonstrated_replacement"
    if len(eligible) == 1:
        selected,reason = eligible[0],"only_quality_and_engineering_eligible_candidate"
    elif len(eligible) == 2:
        if practical_better is not None:
            selected,reason = practical_better,"practical_quality_advantage_between_eligible_candidates"
        else:
            selected = next(name for name in engineering["speed_order"] if name in eligible)
            reason = "prefrozen_n100_speed_tiebreak_quality_best_not_established"
    elif quality:
        reason = "quality_gates_passed_but_n100_latency_budget_failed"
    return {"selected":selected,"reason":reason,"baseline_quality_pass_candidates":quality,
        "engineering_pass":engineering_pass,"quality_and_engineering_eligible":eligible,
        "candidate_with_practical_pairwise_advantage":practical_better,
        "quality_best_established_among_both_finalists":practical_better is not None,
        "equivalence_established":False,"n100_step_p95_budget_seconds":STEP_BUDGET,
        "fallback_reference":"gp_topk_guard" if selected is None else None,
        "application_changed":False,
        "scope":"Replacement supported only against the two specified baselines in this synthetic experiment; no universal/human optimum"}


def confirmatory_analysis(cube,protocol,*,progress=None):
    x = np.asarray(cube)
    if x.ndim != 3 or x.shape[1:] != (14,6) or not np.isfinite(x[:,:,:4]).all():
        raise InvalidRun("Complete finite confirmatory cube required; no row deletion permitted")
    observed,samples,boot = bootstrap_cvars(x[:,:,:4],progress=progress)
    comparisons,passed = {},{}
    for ci,candidate in enumerate(CANDIDATES):
        comparisons[candidate] = {}
        for bj,baseline in enumerate(BASELINES,2):
            difference = x[:,:,ci]-x[:,:,bj]
            overall = stats.cluster_mean_interval(difference.mean(axis=1),alpha=ALPHA,hypotheses=MULTIPLICITY)
            overall.update(threshold=MEAN_MARGIN,passed=bool(overall["upper_one_sided"]<MEAN_MARGIN))
            families = {}
            for fi,family in enumerate(FAMILIES):
                interval = stats.cluster_mean_interval(difference[:,fi],alpha=ALPHA,hypotheses=MULTIPLICITY)
                interval.update(threshold=FAMILY_MARGIN,passed=bool(interval["upper_one_sided"]<FAMILY_MARGIN))
                families[family] = interval
            tail = tail_interval(observed,samples,ci,bj)
            tail.update(threshold=TAIL_MARGIN,passed=bool(tail["upper_one_sided"]<TAIL_MARGIN))
            comparisons[candidate][baseline] = {"overall":overall,"families":families,"cvar95":tail,
                "all_gates_passed":bool(overall["passed"] and tail["passed"] and all(r["passed"] for r in families.values()))}
        passed[candidate] = all(row["all_gates_passed"] for row in comparisons[candidate].values())
    contrast = stats.cluster_mean_interval((x[:,:,0]-x[:,:,1]).mean(axis=1),alpha=ALPHA,hypotheses=MULTIPLICITY)
    contrast.update(candidate=CANDIDATES[0],baseline=CANDIDATES[1],decision_interval="two_sided",
                    practical_margin=CONTRAST_MARGIN)
    return {"status":"COMPLETE","multiplicity":MULTIPLICITY,
        "hypothesis_count_breakdown":{"candidate_baseline_overall":4,"candidate_baseline_family":56,
                                      "candidate_baseline_cvar":4,"candidate_contrast":1},
        "comparisons":comparisons,"candidate_contrast":contrast,"baseline_quality_pass":passed,
        "bootstrap":boot,"decision":choose_candidate(passed,contrast,protocol["engineering_audit"])}


def _markdown(report):
    if report["status"] != "COMPLETE":
        detail = {"error":report.get("error"),"validation":report.get("validation",{})}
        return "# Final analysis invalid\n\nNo confirmatory conclusion or replacement is authorized by these data. The research reference remains gp_topk_guard; the application is unchanged.\n\n```json\n"+json.dumps(detail,indent=2)+"\n```\n"
    decision = report["confirmatory"]["decision"]
    lines = ["# Frozen final analysis", "",f"Selected: **{decision['selected'] or 'none'}**.",
             f"Reason: `{decision['reason']}`.","", "| Candidate | Baseline | Overall upper | Worst family upper | CVaR95 upper | All gates |",
             "|---|---|---:|---:|---:|---|"]
    for candidate,baselines in report["confirmatory"]["comparisons"].items():
        for baseline,row in baselines.items():
            worst = max(v["upper_one_sided"] for v in row["families"].values())
            lines.append(f"| {candidate} | {baseline} | {row['overall']['upper_one_sided']:.8f} | {worst:.8f} | {row['cvar95']['upper_one_sided']:.8f} | {row['all_gates_passed']} |")
    interval = report["confirmatory"]["candidate_contrast"]["two_sided"]
    lines += ["",f"Candidate contrast ({CANDIDATES[0]} minus {CANDIDATES[1]}), Bonferroni two-sided CI: {interval}.",
              "", "A speed tiebreak does not establish quality equivalence. No demonstrated replacement is a valid outcome.",
              "", "Full per-family results, calibration availability/nonfinite counts, descriptive PABBO/equal controls, source hashes and limitations are in analysis.json.",
              "", "Scope: finite synthetic generator and tested alternatives. Approximate t/bootstrap coverage; no independent human validation or production-readiness claim."]
    return "\n".join(lines)+"\n"


def run_analysis(run,out,*,progress=None):
    run,out = Path(run).resolve(),Path(out).resolve()
    if not run.is_dir():
        raise InvalidRun("Run directory does not exist")
    if out == run or run/"snapshot" in out.parents or out == run/"snapshot":
        raise InvalidRun("Analysis outputs must not modify the captured source tree")
    out.mkdir(parents=True,exist_ok=False)
    report = {"utc":datetime.now(timezone.utc).isoformat(),"run":str(run),"status":"INVALID",
              "analysis_source_sha256":file_sha256(__file__),"generated_cases":False}
    try:
        protocol,completion,digest = verify_artifacts(run)
        before = (run/"rows.jsonl").stat()
        data = read_rows(run/"rows.jsonl",protocol)
        report.update(protocol_sha256=digest,rows_sha256=data["rows_sha256"],rows=data["rows"],
                      validation={"confirmatory":data["issues"],"descriptive":data["descriptive_issues"]},
                      results=descriptive_report(data,protocol["args"]["methods"]))
        expected_rows_hash = completion.get("rows_sha256")
        if expected_rows_hash != data["rows_sha256"]:
            raise InvalidRun("Completed rows checksum mismatch or missing checksum")
        if data["issues"]["total"]:
            raise InvalidRun("Incomplete/invalid confirmatory input; no failed cases were dropped")
        report["confirmatory"] = confirmatory_analysis(data["losses"],protocol,progress=progress)
        after = (run/"rows.jsonl").stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
            raise InvalidRun("Rows changed while analyzing")
        if file_sha256(run/"protocol.json") != digest:
            raise InvalidRun("Protocol changed while analyzing")
        report.update(status="COMPLETE",engineering_audit=protocol["engineering_audit"],
            limitations=["Known synthetic families and finite tested alternatives; no universal ranking optimum.",
                "Paired cluster t and basic bootstrap coverage are approximate; 49999 resamples are not 49999 independent clusters.",
                "All 65 confirmatory hypotheses use Bonferroni; family gates and CVaR are not selected after seeing results.",
                "Calibration is a descriptive synthetic response audit; absent/nonfinite forecasts are reported, not replaced.",
                "PABBO/equal are descriptive controls, not extra confirmatory superiority claims.",
                "The speed tiebreak uses only prefrozen N100 timings, never final observed timings.",
                "No human hiring validation, production/privacy/filter/grade guarantee follows from this analysis."])
    except (InvalidRun,ValueError,KeyError,TypeError,OSError) as exc:
        report["status"] = "INVALID"
        report["error"] = {"type":type(exc).__name__,"message":str(exc)}
        report.pop("confirmatory",None)
        report["decision"] = {"selected":None,"reason":"invalid_input_no_confirmatory_conclusion",
                              "fallback_reference":"gp_topk_guard","application_changed":False}
    (out/"analysis.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (out/"report.md").write_text(_markdown(report))
    checksums = {name:file_sha256(out/name) for name in ("analysis.json","report.md")}
    (out/"analysis.sha256.json").write_text(json.dumps(checksums,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    def progress(done,total):
        print(json.dumps({"bootstrap_resamples_complete":done,"total":total}),flush=True)
    try:
        result = run_analysis(args.run,args.out or args.run/"analysis",progress=progress)
    except (InvalidRun,FileExistsError,OSError) as exc:
        raise SystemExit(str(exc))
    print(json.dumps({"status":result["status"],"output":str((args.out or args.run/"analysis").resolve())}),flush=True)
    if result["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Fresh, paired, equal-budget preference benchmark. Synthetic, not hiring.

All adaptive policies use the SAME posterior ranker and SAME possible pairs.
Only the choice of the next question differs. Freeze before heldout; never
overwrite source results. Individual answers can hurt; publish all scenarios.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from .core import Belief, questions, response_probability, select_question, topk

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("matched_linear", "bounded_evidence", "correlated_profiles",
             "polarized_weights", "noisy_answers", "nonlinear_bottleneck",
             "nonlinear_geometric", "refusals")
IN_MODEL = {"matched_linear", "bounded_evidence", "correlated_profiles", "refusals"}
PRIMARY = ("matched_linear", "bounded_evidence", "correlated_profiles")
POLICIES = ("random", "uncertainty", "bald", "qeubo", "decision_evoi")
M, N, K, BUDGET = 9, 18, 5, 3
DEV_SEEDS = (61000, 61019)
HELDOUT_SEEDS = (73000, 73199)
FROZEN_FILES = ("core.py", "benchmark.py", "test_core.py", "PROOFS.md", "requirements.txt")


def hashes():
    return {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in FROZEN_FILES}


def make_case(seed, scenario):
    rng = np.random.default_rng(np.random.SeedSequence([seed, SCENARIOS.index(scenario), 3]))
    observed = rng.uniform(.5, .98, (N, 3))
    if scenario == "correlated_profiles":
        observed = np.clip(rng.normal(.75, .08, (N, 1)) + rng.normal(0, .06, (N, 3)), .5, .98)
    truth = observed.copy()
    if scenario == "bounded_evidence":
        observed = rng.uniform(.56, .92, (N, 3))
        width = rng.uniform(.01, .06, observed.shape)
        truth = observed + rng.uniform(-width, width)
    alpha = .15 if scenario == "polarized_weights" else 1.
    weights = rng.dirichlet(np.full(3, alpha))
    utility = lambda x: x @ weights
    if scenario == "nonlinear_bottleneck":
        utility = lambda x: .4 * x.mean(axis=1) + .6 * x.min(axis=1)
    elif scenario == "nonlinear_geometric":
        utility = lambda x: np.exp(np.log(x) @ weights)
    return {"observed": observed, "true_utility": utility(truth),
            "display_utility": utility(observed),
            "temperature": .14 if scenario == "noisy_answers" else .04,
            "lapse": .1 if scenario == "noisy_answers" else .02,
            "answer_uniforms": rng.random(BUDGET), "refusal_uniforms": rng.random(BUDGET),
            "refusal_rate": .3 if scenario == "refusals" else 0.}


def metrics(selected, truth):
    ideal = topk(truth, K)
    regret = max(0., float(truth[list(ideal)].sum() - truth[list(selected)].sum())) / K
    discount = 1/np.log2(np.arange(2, K+2))
    dcg = ((2**truth[list(selected)]-1)*discount).sum()
    idcg = ((2**truth[list(ideal)]-1)*discount).sum()
    return {"regret_per_candidate": regret, "ndcg_at_5": float(dcg/idcg),
            "oracle_overlap_at_5": len(set(selected) & set(ideal))/K}


def run_case(seed, scenario):
    case = make_case(seed, scenario)
    initial = Belief.linear(case["observed"], m=M)
    initial_options = questions(initial, K)
    rows, logs = [], []
    for policy in POLICIES:
        belief, asked, answered, seconds = initial, [], 0, 0.
        rng = np.random.default_rng(np.random.SeedSequence([seed, SCENARIOS.index(scenario), 919]))
        for budget in range(BUDGET+1):
            row = {"seed": seed, "scenario": scenario, "policy": policy, "budget": budget,
                   "answered": answered, "seconds": seconds, "in_model": scenario in IN_MODEL,
                   "selected": belief.ranking(K)}
            row.update(metrics(row["selected"], case["true_utility"]))
            rows.append(row)
            if budget == BUDGET:
                break
            start = perf_counter()
            options = initial_options if budget == 0 else questions(belief, K, asked=asked)
            question = select_question(options, policy, rng)
            seconds = perf_counter()-start
            a, b = question.pair
            likelihood = float(response_probability(case["display_utility"][a]-case["display_utility"][b],
                                                     case["temperature"], case["lapse"]))
            choice = 0 if case["answer_uniforms"][budget] < likelihood else 1
            if case["refusal_uniforms"][budget] < case["refusal_rate"]:
                choice = None
            before = metrics(belief.ranking(K), case["true_utility"])["regret_per_candidate"]
            belief = belief.observe(question.pair, choice)
            asked.append(question.pair)
            answered += int(choice is not None)
            after = metrics(belief.ranking(K), case["true_utility"])["regret_per_candidate"]
            logs.append({"seed": seed, "scenario": scenario, "policy": policy,
                         "step": budget+1, "pair": question.pair, "answer": choice,
                         "predicted_gain_per_candidate": question.expected_gain/K,
                         "realized_regret_change": after-before})
    return rows, logs


def summaries(rows):
    result = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for policy in POLICIES:
            result[scenario][policy] = {}
            for budget in range(BUDGET+1):
                group = [r for r in rows if (r["scenario"], r["policy"], r["budget"]) == (scenario, policy, budget)]
                result[scenario][policy][budget] = {"count": len(group), **{
                    key: float(np.mean([r[key] for r in group]))
                    for key in ("regret_per_candidate", "ndcg_at_5", "oracle_overlap_at_5", "answered")}}
    return result


def paired(rows, scenarios, baseline):
    indexed = {(r["scenario"], r["seed"], r["policy"]): r for r in rows
               if r["budget"] == BUDGET and r["scenario"] in scenarios}
    keys = sorted({(s, seed) for s, seed, _ in indexed})
    # Independent worlds; paired policies and stratified scenario sampling.
    diffs = {s: np.array([indexed[s, seed, "decision_evoi"]["regret_per_candidate"] -
                         indexed[s, seed, baseline]["regret_per_candidate"]
                         for scenario, seed in keys if scenario == s]) for s in scenarios}
    rng = np.random.default_rng(202610073)
    boots = np.mean([rng.choice(d, (3000, len(d)), replace=True).mean(axis=1)
                     for d in diffs.values()], axis=0)
    difference = float(np.mean([d.mean() for d in diffs.values()]))
    base_mean = float(np.mean([indexed[s, seed, baseline]["regret_per_candidate"] for s, seed in keys]))
    return {"difference": difference, "descriptive_95_percentile": [float(v) for v in np.quantile(boots, [.025, .975])],
            "relative_reduction": -difference/base_mean if base_mean else None, "worlds": len(keys)}


def compare(rows):
    comparisons = {"primary_aggregate": {p: paired(rows, PRIMARY, p) for p in POLICIES[:-1]}}
    for scenario in SCENARIOS:
        comparisons[scenario] = {p: paired(rows, (scenario,), p) for p in POLICIES[:-1]}
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_path = ROOT/"protocol.json"
    if args.freeze:
        protocol = {"date": "2026-10-07", "hashes": hashes(), "n": N, "d": 3, "k": K,
                    "max_budget": BUDGET, "prior": "Dirichlet(1), symmetrized Sobol",
                    "base_sobol_m": M, "posterior_particles": 6*2**M,
                    "response_model": {"temperature": .04, "lapse": .02},
                    "dev_seeds": DEV_SEEDS, "heldout_seeds": HELDOUT_SEEDS,
                    "scenarios": SCENARIOS, "primary_scenarios": PRIMARY, "policies": POLICIES,
                    "primary_endpoint": "Mean regret per candidate at budget=3, equal weight of three primary scenarios",
                    "gate": "At least 10% aggregate reduction vs random with paired upper 95% endpoint <0; no worse aggregate mean than BALD or qEUBO. This gate allows only consideration of experimental integration, not a hiring quality or winning claim.",
                    "failure_policy": "Publish all negative/OOD results, do not retune on heldout; report primary and every scenario. Bootstrap descriptive, no multiple-comparison guarantee.",
                    "scope": "Synthetic model validation; no humans, no calibrated skill intervals, no novelty claim."}
        with protocol_path.open("x") as f:
            json.dump(protocol, f, ensure_ascii=False, indent=2)
        print("FROZEN", sha256(protocol_path.read_bytes()).hexdigest())
        return
    if args.split == "heldout":
        protocol = json.loads(protocol_path.read_text())
        if protocol["hashes"] != hashes():
            raise SystemExit("Frozen source differs. Do not reuse the same heldout set.")
    output = args.output or ROOT/"results"/f"{args.split}_v3.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite a benchmark result")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    first, last = DEV_SEEDS if args.split == "dev" else HELDOUT_SEEDS
    rows, logs = [], []
    for scenario in SCENARIOS:
        for seed in range(first, last+1):
            case_rows, case_logs = run_case(seed, scenario)
            rows.extend(case_rows)
            logs.extend(case_logs)
        print(scenario, "complete", flush=True)
    payload = {"split": args.split, "date": "2026-10-07", "hashes": hashes(),
               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "seconds_total": perf_counter()-start, "summary": summaries(rows),
               "comparisons": compare(rows), "rows": rows, "questions": logs,
               "timing_limit": "Reported row seconds exclude shared root-pool computation and are not a latency benchmark",
               "limitations": "Synthetic. Query budget equal; refusal events shared. All policies use the same posterior ranker. No human validation."}
    with output.open("x") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload["comparisons"]["primary_aggregate"], indent=2))
    print(output)


if __name__ == "__main__":
    main()

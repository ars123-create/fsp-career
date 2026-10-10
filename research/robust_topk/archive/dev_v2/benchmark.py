"""Paired synthetic benchmark. Freeze source before the held-out run.

This tests a decision model, not hiring quality or item calibration.
Existing benchmark files are never overwritten by this script.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from .core import (WeightRegion, independent_envelope, question_options,
                   solve, topk, worst_case)

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("observed_points", "bounded_evidence", "polarized_preferences",
             "wrong_priority_constraint", "nonlinear_utility")
POLICIES = ("mean", "lower_mean", "evidence_v1", "weights_only", "independent_box", "joint")


def hashes():
    return {name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("core.py", "benchmark.py", "test_core.py", "PROOFS.md", "requirements.txt")}


def make_case(seed, scenario, n=18):
    rng = np.random.default_rng(np.random.SeedSequence([seed, SCENARIOS.index(scenario)]))
    # All candidates have already passed the same mandatory eligibility filter.
    observed = rng.uniform(.5, .98, (n, 3))
    width = rng.uniform(.01, .16, observed.shape)
    if scenario == "observed_points":
        width *= 0
    lo, hi = np.clip(observed - width, 0, 1), np.clip(observed + width, 0, 1)
    truth = rng.uniform(lo, hi)
    region = WeightRegion(3, lower=(.05,)*3, upper=(.9,)*3)
    alpha = .15 if scenario == "polarized_preferences" else 1.
    weights = .05 + .85 * rng.dirichlet(np.full(3, alpha))
    in_model = True
    if scenario == "wrong_priority_constraint":
        region = region.prefer(0, 1)
        weights = np.array([.1, .7, .2])
        in_model = False
    utilities = truth @ weights
    if scenario == "nonlinear_utility":
        utilities = truth.min(axis=1)  # complementary bottleneck, not additive
        in_model = False
    metadata = rng.uniform(size=(n, 3))  # synthetic families/freshness/FSP terms
    return rng, lo, hi, truth, weights, region, utilities, metadata, in_model


def metrics(selected, utilities, lo, hi, vertices, k, nominal):
    selected = tuple(sorted(selected, key=lambda i: (-nominal[i], i)))
    ideal = topk(utilities, k)
    regret = max(0., float(utilities[list(ideal)].sum() - utilities[list(selected)].sum()))
    discount = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(((2 ** utilities[list(selected)] - 1) * discount).sum())
    idcg = float(((2 ** utilities[list(ideal)] - 1) * discount).sum())
    bound = worst_case(lo, hi, vertices, selected).regret
    return {"model_regret": bound, "realized_regret": regret,
            "overlap_at_k": len(set(ideal) & set(selected)) / k,
            "ndcg_at_k": dcg / idcg if idcg else 1.,
            "certificate_violation": regret > bound + 1e-7,
            "selected": selected}


def run_case(seed, scenario, questions=True):
    rng, lo, hi, truth, w, region, utility, metadata, in_model = make_case(seed, scenario)
    k, vs = 5, region.vertices()
    middle = (lo + hi) / 2
    nominal = middle @ vs.mean(axis=0)
    base = {"mean": topk(nominal, k), "lower_mean": topk(lo @ vs.mean(axis=0), k),
            "evidence_v1": topk(.8 * middle.mean(axis=1) + metadata @ [.1, .05, .05], k)}
    rows = []
    solutions = {}
    for policy in POLICIES:
        start = perf_counter()
        status, gap = "not_applicable", None
        if policy in base:
            selected = base[policy]
        else:
            low, high, weights = lo, hi, vs
            if policy == "weights_only":
                low = high = middle
            elif policy == "independent_box":
                low, high = independent_envelope(lo, hi, vs)
                weights = [[1.]]
            sol = solve(low, high, weights, k, time_limit=2.)
            solutions[policy] = sol
            selected, status, gap = sol.selected, sol.status, sol.absolute_gap
        seconds = perf_counter() - start
        row = {"seed": seed, "scenario": scenario, "policy": policy, "in_model": in_model,
               "seconds": seconds, "status": status, "gap": gap}
        row.update(metrics(selected, utility, lo, hi, vs, k, nominal))
        rows.append(row)
    queries = []
    if questions:
        options = question_options(lo, hi, region, k, time_limit=2.)
        if options:
            for label, option in (("best_pair", options[0]),
                                  ("random_pair", options[int(rng.integers(len(options)))])):
                a, b = option["pair"]
                branch = 0 if w[a] >= w[b] else 1
                after = region.prefer(a, b) if branch == 0 else region.prefer(b, a)
                sol = option["branches"][branch]
                qs = {"seed": seed, "scenario": scenario, "policy": label, "pair": option["pair"],
                      "answer": branch, "worst_branch": option["worst_after"],
                      "all_branches_optimal": all(o["optimal"] for o in options),
                      "before_bound": solutions["joint"].witness.regret, "in_model": in_model,
                      "status": sol.status, "gap": sol.absolute_gap}
                qs.update(metrics(sol.selected, utility, lo, hi, after.vertices(), k, nominal))
                queries.append(qs)
    return rows, queries


def summaries(rows):
    result = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for policy in sorted({r["policy"] for r in rows}):
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            if not group:
                continue
            summary = {key: float(np.mean([r[key] for r in group]))
                       for key in ("model_regret", "realized_regret", "overlap_at_k", "ndcg_at_k")}
            summary["count"] = len(group)
            summary["violations"] = sum(r["certificate_violation"] for r in group)
            summary["nonoptimal"] = sum(r["status"] not in ("optimal", "not_applicable") for r in group)
            if "seconds" in group[0]:
                summary["seconds_p95"] = float(np.quantile([r["seconds"] for r in group], .95))
            result[scenario][policy] = summary
    return result


def paired_intervals(rows):
    # Resample independent markets, not candidate pairs. Within each scenario.
    rng = np.random.default_rng(202610071)
    out = {}
    for scenario in SCENARIOS:
        group = [r for r in rows if r["scenario"] == scenario]
        indexed = {(r["seed"], r["policy"]): r for r in group}
        seeds = sorted({r["seed"] for r in group})
        out[scenario] = {}
        for policy in POLICIES[:-1]:
            out[scenario][policy] = {}
            for metric in ("model_regret", "realized_regret", "ndcg_at_k"):
                diffs = np.array([indexed[s, "joint"][metric] - indexed[s, policy][metric] for s in seeds])
                boot = rng.choice(diffs, (2000, len(diffs)), replace=True).mean(axis=1)
                out[scenario][policy][metric] = {"joint_minus_baseline": float(diffs.mean()),
                    "descriptive_95_percentile": [float(x) for x in np.quantile(boot, [.025, .975])]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = ROOT / "protocol.json"
    if args.freeze:
        content = {"date": "2026-10-07", "hashes": hashes(), "dev_seeds": [30000, 30019],
                   "heldout_seeds": [50000, 50099], "scenarios": SCENARIOS,
                   "n": 18, "d": 3, "k": 5, "policies": POLICIES,
                   "go_no_go": "No in-model certificate violations; independent exhaustive tests pass. No mandatory improvement of average utility. Publish all scenarios/negative results.",
                   "limitations": "Synthetic utilities. No human labels, calibration, hiring outcome or scientific novelty claim."}
        with protocol.open("x") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print("FROZEN", protocol)
        return
    if args.split == "heldout":
        frozen = json.loads(protocol.read_text())
        if frozen["hashes"] != hashes():
            raise SystemExit("Frozen source changed; do not run/relabel the same heldout set.")
    seeds = range(30000, 30020) if args.split == "dev" else range(50000, 50100)
    output = args.output or ROOT / "results" / f"{args.split}_v2.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite existing benchmark.")
    rows, queries = [], []
    start = perf_counter()
    for scenario in SCENARIOS:
        for seed in seeds:
            case_rows, case_queries = run_case(seed, scenario)
            rows.extend(case_rows)
            queries.extend(case_queries)
        print(scenario, "completed", len(rows), "policy runs", flush=True)
    result = {"split": args.split, "date": "2026-10-07", "hashes": hashes(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "seconds": perf_counter() - start, "rows": rows, "queries": queries,
              "summary": summaries(rows), "query_summary": summaries(queries),
              "paired_intervals": paired_intervals(rows)}
    output.parent.mkdir(exist_ok=True)
    with output.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"path": str(output), "seconds": result["seconds"], "cases": len(rows) // len(POLICIES)}))


if __name__ == "__main__":
    main()

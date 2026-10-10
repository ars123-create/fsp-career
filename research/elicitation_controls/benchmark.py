"""Fresh frozen sensitivity audit of current app versus simpler input controls."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from .controls import run_controls, utility

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCENARIOS = ("linear", "rubric_grid", "polarized", "score_noise", "geometric", "answer_noise")
POLICIES = ("equal", "app_evoi", "fixed_swing", "direct_exact", "direct_round_5pp",
            "direct_sigma_005", "direct_sigma_015", "direct_sigma_030")
SEEDS = {"dev": (945000, 945007), "heldout": (946000, 946199)}
FROZEN = (
    "research/elicitation_controls/DESIGN.md", "research/elicitation_controls/controls.py",
    "research/elicitation_controls/benchmark.py", "research/elicitation_controls/test_controls.py",
    "research/elicitation_controls/report.py", "backend/app/preference_engine.py",
    "backend/app/preference_acceleration.py", "backend/app/preference_matching.py",
    "backend/app/ranking_ties.py", "backend/app/schemas.py", "VERSION.json",
)


def hashes():
    return {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in FROZEN}


def make_case(seed, scenario):
    streams = np.random.SeedSequence([seed, SCENARIOS.index(scenario), 94642]).spawn(5)
    profile_rng, weight_rng, score_rng, answer_rng, numeric_rng = [np.random.default_rng(s) for s in streams]
    if scenario == "rubric_grid":
        grid = np.array([v for v in product((.5, .75, 1.), repeat=3) if sum(v) >= 2.25])
        observed = grid[profile_rng.integers(len(grid), size=24)]
    else:
        rows = []
        while len(rows) < 24:
            batch = profile_rng.uniform(.5, 1., (24, 3))
            rows.extend(batch[batch.sum(axis=1) >= 2.25])
        observed = np.array(rows[:24])
    weights = weight_rng.dirichlet(np.full(3, .15 if scenario == "polarized" else 1.))
    truth = observed.copy()
    if scenario == "score_noise":
        truth = np.clip(truth + score_rng.normal(0., .08, truth.shape), 0., 1.)
    return {"seed": seed, "scenario": scenario, "observed": observed, "truth": truth,
            "weights": weights, "true_utility": utility(truth, weights, geometric=scenario == "geometric"),
            "temperature": .14 if scenario == "answer_noise" else .04,
            "lapse": .1 if scenario == "answer_noise" else .02,
            "uniforms": answer_rng.random(3), "noise_z": numeric_rng.normal(size=3)}


def summarize(cases):
    summaries = {}
    for index, scenario in enumerate(SCENARIOS):
        subset = [case["results"] for case in cases if case["scenario"] == scenario]
        rng = np.random.default_rng(np.random.SeedSequence([946777, index]))
        bootstrap = rng.integers(len(subset), size=(2000, len(subset)))
        app = np.array([r["app_evoi"]["regret_points"] for r in subset])
        equal = np.array([r["equal"]["regret_points"] for r in subset])
        methods = {}
        for policy in POLICIES:
            regret = np.array([r[policy]["regret_points"] for r in subset])
            difference = app - regret
            answers = [r[policy]["answers"] for r in subset if r[policy]["answers"] is not None]
            methods[policy] = {
                "mean_regret_points": float(regret.mean()),
                "mean_ndcg_at_5": float(np.mean([r[policy]["ndcg_at_5"] for r in subset])),
                "mean_weight_rmse": float(np.mean([r[policy]["weight_rmse"] for r in subset])),
                "mean_oracle_overlap": float(np.mean([r[policy]["oracle_overlap"] for r in subset])),
                "worse_than_equal_count": int(np.count_nonzero(regret > equal + 1e-10)),
                "mean_binary_answers": float(np.mean(answers)) if answers else None,
                "app_minus_control_points": float(difference.mean()),
                "app_minus_control_descriptive_ci95": np.quantile(difference[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                "app_better_count": int(np.count_nonzero(difference < -1e-10)),
                "app_worse_count": int(np.count_nonzero(difference > 1e-10)),
            }
        summaries[scenario] = {"tasks": len(subset), "methods": methods,
            "app_no_question_count": sum(r["app_evoi"]["answers"] == 0 for r in subset),
            "app_harmful_answer_count": sum(
                sum(metrics_regret(next_step, case) > metrics_regret(step, case) + 1e-10
                    for step, next_step in zip(case["results"]["app_evoi"]["trace"], case["results"]["app_evoi"]["trace"][1:]))
                for case in cases if case["scenario"] == scenario)}
    return summaries


def metrics_regret(step, case):
    from .controls import metrics
    return metrics(step["selected"], case["true_utility"])["regret_points"]


def jsonable(case):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in case.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SEEDS, default="dev")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    if args.freeze:
        protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes(),
                    "versions": versions, "seeds": SEEDS, "scenarios": SCENARIOS, "policies": POLICIES,
                    "n": 24, "d": 3, "k": 5, "max_binary_answers": 3,
                    "scope": "Sensitivity audit, not calibrated human behavior or equal cognitive effort. All scenarios reported separately."}
        with (HERE / "protocol.json").open("x") as file:
            json.dump(protocol, file, ensure_ascii=False, indent=2)
        print("Frozen", sha256((HERE / "protocol.json").read_bytes()).hexdigest())
        return
    if args.output is None:
        parser.error("--output required")
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_hash = None
    if args.split == "heldout":
        protocol = json.loads((HERE / "protocol.json").read_text())
        if protocol["hashes"] != hashes() or protocol["versions"] != versions:
            raise RuntimeError("frozen code or dependency version differs; do not run heldout")
        protocol_hash = sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    started = perf_counter()
    cases = []
    for scenario in SCENARIOS:
        lo, hi = SEEDS[args.split]
        for seed in range(lo, hi + 1):
            case = make_case(seed, scenario)
            case["results"] = run_controls(case)
            cases.append(jsonable(case))
        print(f"{scenario}: {hi-lo+1} tasks complete", flush=True)
    summary = summarize(cases)
    payload = {"split": args.split, "versions": versions, "hashes": hashes(),
               "protocol_sha256": protocol_hash, "elapsed_seconds": perf_counter() - started,
               "cases": cases, "summary": summary,
               "scope": "Synthetic only. Exact/direct inputs expose more information than binary questions; no measured human time or precision."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({s: {p: round(v["mean_regret_points"], 4) for p, v in d["methods"].items()}
                      for s, d in summary.items()}, ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

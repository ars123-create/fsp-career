"""Reproducible, paired synthetic benchmark. Run from the repository root."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from research.core import POLICIES, generate_market, run_policy


SCENARIOS = ("clean", "leakage", "misspecified_leakage", "shifted_items", "correlated_skills")


def summarize(rows):
    summary = {}
    rng = np.random.default_rng(20261007)
    for scenario in sorted({r["scenario"] for r in rows}):
        summary[scenario] = {}
        by_policy = {}
        for policy in POLICIES:
            selected = sorted([r for r in rows if r["scenario"] == scenario and r["policy"] == policy],
                              key=lambda r: r["seed"])
            if not selected:
                continue
            by_policy[policy] = selected
            summary[scenario][policy] = {
                key: float(np.mean([r[key] for r in selected]))
                for key in ("regret", "utility", "oracle_utility", "brier", "false_invites",
                            "invitations", "answers", "distinct_families", "seconds")
            }
        if "pair_voi_family" in by_policy:
            reference = by_policy["pair_voi_family"]
            comparisons = {}
            for policy, selected in by_policy.items():
                if policy == "pair_voi_family":
                    continue
                assert [r["seed"] for r in selected] == [r["seed"] for r in reference]
                # Positive = proposed method has lower regret.
                delta = np.array([a["regret"] - b["regret"] for a, b in zip(selected, reference)])
                means = np.mean(rng.choice(delta, size=(4000, len(delta)), replace=True), axis=1)
                comparisons[policy] = {
                    "regret_reduction": float(delta.mean()),
                    "descriptive_bootstrap_95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
                    "n_markets": len(delta),
                }
            summary[scenario]["paired_comparisons"] = comparisons
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--markets", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite experiment: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = args.markets or (20 if args.split == "dev" else 80)
    start = 0 if args.split == "dev" else 1000
    sources = [Path(__file__), Path("research/core.py"), Path("research/PROTOCOL.md")]
    metadata = {
        "status": "RUNNING", "split": args.split, "seeds": list(range(start, start + count)),
        "budget_extra_answers": 12, "scenarios": list(args.scenarios), "policies": list(args.policies),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "scipy"]},
        "data_type": "SYNTHETIC", "rows": [],
    }
    args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    begun = time.perf_counter()
    for scenario in args.scenarios:
        for seed in range(start, start + count):
            market = generate_market(seed, scenario)
            for policy in args.policies:
                t0 = time.perf_counter()
                result = run_policy(market, policy, seed)
                result.update(scenario=scenario, seed=seed, policy=policy,
                              seconds=time.perf_counter() - t0)
                metadata["rows"].append(result)
            if (seed - start + 1) % 10 == 0 or seed == start + count - 1:
                args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
                print(f"{scenario}: {seed-start+1}/{count}, elapsed {time.perf_counter()-begun:.1f}s", flush=True)
    metadata["status"] = "COMPLETE"
    metadata["elapsed_seconds"] = time.perf_counter() - begun
    metadata["summary"] = summarize(metadata["rows"])
    args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata["summary"], indent=2))


if __name__ == "__main__":
    main()

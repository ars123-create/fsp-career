"""Fixed equivalence/timing experiment; does not change test_v1 policies."""
from pathlib import Path
from itertools import combinations
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np

from research.core import Belief, assignment, generate_market, value_of_information
from research.fast_voi import SingleColumnCache, cached_value_of_information


def main():
    output = Path("research/results/cache_v1.json")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    sources = [Path(p) for p in [
        "research/core.py", "research/fast_voi.py", "research/benchmark_cache.py",
        "research/CACHE_PROTOCOL.md",
    ]]
    report = {
        "status": "RUNNING", "data_type": "SYNTHETIC COMPUTATIONAL CHECK",
        "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "rows": [],
    }
    output.write_text(json.dumps(report, indent=2))
    assignment(np.array([[1.0, -1.0]]))
    for n, m in [(8, 4), (40, 8), (120, 40)]:
        for seed in range(2000, 2020):
            market = generate_market(seed, "leakage", candidates=n, jobs=m)
            belief = Belief(n, 3, 3, leakage=0.3)
            for i in range(n):
                for k in range(3):
                    belief.update((i, k, 0), bool(market.responses[i, k, 0, 0]), additional=False)
            p = belief.mastery()
            w = market.weights(p)
            base = assignment(w)[0]
            singles = [(q,) for q in belief.available()]
            pairs = []
            for i in range(n):
                pool = [(i, k, f) for k in range(3) for f in [1, 2]]
                pairs.extend(tuple(pair) for pair in combinations(pool, 2))
            timings, values = {}, {}

            def measure_direct():
                t0 = time.perf_counter()
                a = np.array([value_of_information(q, belief, market, p, w, base) for q in singles])
                t1 = time.perf_counter()
                b = np.array([value_of_information(q, belief, market, p, w, base) for q in pairs])
                t2 = time.perf_counter()
                timings["direct_single"] = t1 - t0
                timings["direct_all"] = t2 - t0
                values["direct"] = np.r_[a, b]

            def measure_cached():
                t0 = time.perf_counter()
                cache = SingleColumnCache(w)
                t_cache = time.perf_counter()
                a = np.array([cached_value_of_information(q, belief, market, p, cache) for q in singles])
                t1 = time.perf_counter()
                b = np.array([cached_value_of_information(q, belief, market, p, cache) for q in pairs])
                t2 = time.perf_counter()
                timings["cache_build"] = t_cache - t0
                timings["cached_single"] = t1 - t0
                timings["cached_all"] = t2 - t0
                values["cached"] = np.r_[a, b]

            if seed % 2:
                measure_cached()
                measure_direct()
            else:
                measure_direct()
                measure_cached()
            error = float(np.max(np.abs(values["direct"] - values["cached"])))
            if error > 1e-9:
                raise AssertionError(f"Equivalence failed: {n}, {m}, {seed}, {error}")
            report["rows"].append({
                "candidates": n, "jobs": m, "seed": seed,
                "single_queries": len(singles), "pair_queries": len(pairs),
                "max_abs_error": error, **timings,
            })
        output.write_text(json.dumps(report, indent=2))
        print(f"Completed n={n}, m={m}", flush=True)
    report["summary"] = []
    for n, m in [(8, 4), (40, 8), (120, 40)]:
        rows = [r for r in report["rows"] if r["candidates"] == n and r["jobs"] == m]
        report["summary"].append({
            "candidates": n, "jobs": m, "n_states": len(rows),
            "max_abs_error": max(r["max_abs_error"] for r in rows),
            "median_single_speedup": float(np.median([r["direct_single"]/r["cached_single"] for r in rows])),
            "median_all_speedup": float(np.median([r["direct_all"]/r["cached_all"] for r in rows])),
            "median_cache_build_seconds": float(np.median([r["cache_build"] for r in rows])),
        })
    report["status"] = "COMPLETE"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

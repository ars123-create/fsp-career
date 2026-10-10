"""Paired component timing before integration; not an HTTP/SLA benchmark."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from app.preference_engine import Belief, questions, simplex_particles
from validation.load_capacity_100 import fixture_values
from validation.preference_acceleration.prototype import best_linear_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite component measurement")
    results = []
    weights = simplex_particles(3)
    probabilities = np.full(len(weights), 1/len(weights))
    for scenario in ("bank_v2", "varied"):
        _, raw = fixture_values(scenario)
        for n in (18, 100):
            values = raw[:n]/100
            excluded, seen = [], set()
            for a, b in combinations(range(n), 2):
                difference = values[a]-values[b]
                key = tuple(sorted((tuple(values[a]), tuple(values[b]))))
                if key in seen or not (np.any(difference > 0) and np.any(difference < 0)):
                    excluded.append((a, b))
                else:
                    seen.add(key)
            def reference():
                options = questions(Belief(weights @ values.T, probabilities), 5, asked=excluded)
                best = max(options, key=lambda q: q.expected_gain) if options else None
                return best if best and best.expected_gain > .0025 else None
            def accelerated():
                return best_linear_question(values, weights, probabilities, 5, asked=excluded, min_gain=.0025)
            reference()
            accelerated()
            samples = {"reference": [], "accelerated": []}
            for repeat in range(5):
                order = (("reference", reference), ("accelerated", accelerated))
                if repeat % 2:
                    order = tuple(reversed(order))
                outputs = {}
                for name, call in order:
                    start = perf_counter()
                    outputs[name] = call()
                    samples[name].append(1000*(perf_counter()-start))
                left, right = outputs["reference"], outputs["accelerated"]
                assert (left is None) == (right is None)
                if left:
                    assert left.pair == right.pair
                    assert abs(left.expected_gain-right.expected_gain) < 1e-12
            results.append({"scenario": scenario, "profiles": n, "allowed_pairs": len(seen),
                            "samples_ms": samples, "median_ms": {k: median(v) for k, v in samples.items()},
                            "reference_fallback": bool(right and right.used_reference)})
    result = {"status": "PASS", "scope": "5 paired local component timings; excludes DB/API/concurrency",
              "prototype_sha256": hashlib.sha256(Path(__file__).with_name("prototype.py").read_bytes()).hexdigest(),
              "results": results}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

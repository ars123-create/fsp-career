"""Independent read-only audit of published inputs, selections, metrics, hashes."""
from collections import Counter
from hashlib import sha256
import json
from math import fsum, isclose, log2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    data = json.loads((HERE / "results/heldout.json").read_text())
    assert data["protocol_sha256"] == sha256(protocol_bytes).hexdigest()
    for name, digest in protocol["hashes"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert data["hashes"] == protocol["hashes"]
    assert len(data["cases"]) == 1200
    seen, counts = set(), Counter()
    for case in data["cases"]:
        key = (case["scenario"], case["seed"])
        assert key not in seen and 946000 <= case["seed"] <= 946199
        seen.add(key)
        counts[case["scenario"]] += 1
        truth = case["true_utility"]
        ideal = sorted(range(24), key=lambda i: -truth[i])[:5]
        ideal_sum = fsum(truth[i] for i in ideal)
        for policy, result in case["results"].items():
            selected = result["selected"]
            assert len(selected) == len(set(selected)) == 5
            assert all(0 <= i < 24 for i in selected)
            weights = result["weights"]
            assert all(w >= 0 for w in weights) and isclose(fsum(weights), 1, abs_tol=1e-12)
            regret = max(0., (ideal_sum - fsum(truth[i] for i in selected)) * 20)
            assert isclose(result["regret_points"], regret, abs_tol=1e-11)
            dcg = fsum((2 ** truth[i] - 1) / log2(rank + 2) for rank, i in enumerate(selected))
            idcg = fsum((2 ** truth[i] - 1) / log2(rank + 2) for rank, i in enumerate(ideal))
            assert isclose(result["ndcg_at_5"], dcg / idcg, abs_tol=1e-12)
            scores = [round(fsum(x * w for x, w in zip(row, weights)), 12) for row in case["observed"]]
            assert selected == sorted(range(24), key=lambda i: -scores[i])[:5]
        if case["scenario"] not in ("score_noise", "geometric"):
            assert case["results"]["direct_exact"]["regret_points"] < 1e-10
        app = case["results"]["app_evoi"]
        assert 0 <= app["answers"] <= 3 and len(app["trace"]) == app["answers"] + 1
        assert app["trace"][0]["selected"] == case["results"]["equal"]["selected"]
    assert all(count == 200 for count in counts.values())
    assert set(counts) == set(protocol["scenarios"])
    for scenario, summary in data["summary"].items():
        subset = [c for c in data["cases"] if c["scenario"] == scenario]
        for policy, values in summary["methods"].items():
            mean = fsum(c["results"][policy]["regret_points"] for c in subset) / 200
            assert isclose(mean, values["mean_regret_points"], abs_tol=1e-12)
    # These optional checks operate only in the full project, not the small snapshot ZIP.
    historical = 0
    for branch in ("preference_topk", "robust_topk"):
        old_root = ROOT / "research" / branch
        if not (old_root / "protocol.json").exists():
            continue
        old = json.loads((old_root / "protocol.json").read_text())
        for name, digest in old["hashes"].items():
            assert sha256((old_root / name).read_bytes()).hexdigest() == digest, name
            historical += 1
    print(json.dumps({"status": "PASS", "frozen_files": len(protocol["hashes"]),
                      "tasks": len(seen), "selection_and_metric_checks": len(seen) * 8,
                      "historical_frozen_files_unchanged": historical,
                      "raw_result_sha256": sha256((HERE / "results/heldout.json").read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()

"""Constructed feature-only sequential smoke; no human/heldout quality data."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from research.frontier_2026.components.risk_decisions.tiny_full_query import Session


HERE = Path(__file__).resolve().parent
SOURCES = [HERE/f for f in ["tiny_full_query.py", "adapter.py", "decisions.py"]]
SOURCES += [HERE.parents[1]/f for f in ["models.py", "finite.py"]]


def hashes():
    return {str(p.relative_to(HERE.parents[3])): sha256(p.read_bytes()).hexdigest() for p in SOURCES}


def run():
    report = {"utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "constructed smoke, actual one-step risk formulas; no quality comparison",
              "feature_seed": 88113, "n": 8, "d": 3, "m": 2,
              "hashes_before": hashes(), "cases": []}
    x = np.random.default_rng(report["feature_seed"]).uniform(0,1,(8,3))
    report["values"] = x.tolist()
    for method in ["kl_dro", "cvar", "baseline_safe"]:
        session, history, calls = Session(x), [], []
        start = perf_counter()
        for slot in range(4):
            call_start = perf_counter()
            selected,q = session.recommend(tuple(history),method=method)
            calls.append({"history": list(history), "selected": selected, "query": q,
                          "seconds": perf_counter()-call_start,
                          "diagnostics": session.last_diagnostics})
            if q is None:
                break
            if slot == 3:
                raise AssertionError("unexpected fourth question")
            history.append((q,(0,1,2)[slot]))
        report["cases"].append({"method": method, "status": "PASS", "calls": calls,
                                "total_seconds": perf_counter()-start,
                                "hypotheses": len(session.prior.probabilities),
                                "queries": len(session.pairs)})
    report["hashes_after"] = hashes()
    report["source_hashes_unchanged"] = report["hashes_before"] == report["hashes_after"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    with args.output.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({"source_hashes_unchanged": report["source_hashes_unchanged"],
                      "cases": [{key:row[key] for key in ["method","status","total_seconds",
                                                          "hypotheses","queries"]}
                                | {"calls":len(row["calls"])} for row in report["cases"]]},indent=2))

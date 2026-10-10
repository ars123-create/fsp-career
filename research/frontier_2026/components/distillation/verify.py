"""Hash/test checks and optional fresh complete training replay for P20."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    output = HERE/"output"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    upstream = json.loads((HERE/"upstream/SOURCE.json").read_text())
    for entry in upstream["files"]:
        if sha256((HERE/"upstream"/entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit("Changed pinned PyTorch tutorial source/license")
    original = HERE/"artifact-v1"
    protocol = json.loads((original/"protocol.json").read_text())
    changed = [name for name, digest in protocol["source_sha256"].items()
               if sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    if changed:
        raise SystemExit(f"Training source drift: {changed}; original snapshot preserved")
    runs = []
    def run(name, command):
        before = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (output/f"{name}.log").write_text(completed.stdout+completed.stderr)
        runs.append({"name": name, "exit_code": completed.returncode,
                     "seconds": time.perf_counter()-before, "command": command})
        return completed.returncode == 0
    run("component_tests", [sys.executable, "-m", "pytest", str(HERE/"test_adapter.py"), "-q",
                            f"--junitxml={output/'tests.xml'}"])
    comparison = {"status": "NOT RUN"}
    if args.retrain:
        replay = output/"training-replay"
        success = run("training_replay", [sys.executable, "-m",
                                          "research.frontier_2026.components.distillation.train",
                                          "--output", str(replay)])
        if success:
            compared = {}
            for name in ("teacher-initial.npz", "teacher-aggregation.npz", "model-initial.npz", "model.npz"):
                with np.load(original/name, allow_pickle=False) as a, np.load(replay/name, allow_pickle=False) as b:
                    compared[name] = (a.files == b.files and all(np.array_equal(a[key], b[key]) for key in a.files))
            comparison = {"status": "PASS" if all(compared.values()) else "FAIL",
                          "arrays_identical": compared,
                          "scope": "Exact equality of saved numeric/string arrays in fresh process on this runtime"}
        else:
            comparison = {"status": "FAIL", "reason": "training replay command failed"}
    suites = list(ET.parse(output/"tests.xml").getroot().iter("testsuite"))
    files = [HERE/name for name in ("adapter.py", "train.py", "test_adapter.py", "verify.py",
                                    "README_RU.md", "REUSE_SOURCES.md", "DESIGN.md")]
    result = {
        "status": "PASS" if all(r["exit_code"] == 0 for r in runs) and comparison["status"] != "FAIL" else "FAIL",
        "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "tests": sum(int(s.get("tests", 0)) for s in suites),
        "failures": sum(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suites),
        "runs": runs, "reproducibility": comparison,
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in files},
        "checkpoint_sha256": sha256((original/"model.npz").read_bytes()).hexdigest(),
        "heldout": "NOT RUN", "scope": "P20 component engineering + optional independent offline replay; not hiring validation",
    }
    (output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "tests": result["tests"],
                      "reproducibility": comparison, "output": str(output)}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

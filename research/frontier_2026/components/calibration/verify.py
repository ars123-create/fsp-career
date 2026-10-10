"""Fresh-process calibration adapter checks and hashes, without fitting."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    output = HERE/"output"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    sources = [HERE/name for name in ("predict.py", "__init__.py", "test_predict.py", "smoke.py", "verify.py",
                                     "README_RU.md", "REUSE_SOURCES.md", "NOTICE.md")]
    sources += [ROOT/name for name in (
        "research/frontier_2026/finite.py", "research/frontier_2026/models.py",
        "research/frontier_2026/policies.py", "research/frontier_2026/constraints.py",
        "research/frontier_2026/components/constrained_gp/adapter.py",
        "research/frontier_2026/components/distillation/adapter.py",
        "research/frontier_2026/components/monne/adapter.py",
        "research/frontier_2026/components/legacy_controls/adapter.py",
        "research/frontier_2026/components/risk_decisions/adapter.py",
        "research/preference_topk/core.py", "research/nonlinear_topk/core.py")]
    hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sources}
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    commands = [
        ("tests", [sys.executable, "-m", "pytest", str(HERE/"test_predict.py"), "-q",
                   f"--junitxml={output/'tests.xml'}"]),
        ("smoke", [sys.executable, "-m", "research.frontier_2026.components.calibration.smoke",
                   "--output", str(output/"smoke.json")]),
    ]
    rows = []
    for name, command in commands:
        started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (output/f"{name}.log").write_text(process.stdout+process.stderr)
        rows.append({"name": name, "command": command, "exit_code": process.returncode,
                     "seconds": time.perf_counter()-started})
    changed = [str(p.relative_to(ROOT)) for p in sources
               if sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]]
    suites = list(ET.parse(output/"tests.xml").getroot().iter("testsuite"))
    result = {"status": "PASS" if not changed and all(r["exit_code"] == 0 for r in rows) else "FAIL",
              "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "tests": sum(int(s.get("tests", 0)) for s in suites),
              "failures": sum(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suites),
              "runs": rows, "source_sha256": hashes, "changed_during_verification": changed,
              "scope": "Actual model probability extraction and expected scoring, not fitted calibration",
              "final_benchmark": "NOT RUN"}
    (output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "tests": result["tests"], "failures": result["failures"],
                      "output": str(output)}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

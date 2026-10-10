"""Re-run bounded P06 checks in fresh processes, preserving earlier evidence."""
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE/"output"/stamp
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [HERE/name for name in (
        "adapter.py", "test_adapter.py", "smoke.py", "smoke_upstream.py", "verify.py",
        "README_RU.md", "DESIGN.md", "REUSE_SOURCES.md")]
    hashes = {str(p.relative_to(HERE)): sha256(p.read_bytes()).hexdigest() for p in source_paths}
    upstream = json.loads((HERE/"upstream/SOURCE.json").read_text())
    for entry in upstream["files"]:
        path = HERE/"upstream"/entry["path"]
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"Changed pinned upstream file: {entry['path']}")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    commands = [
        ("upstream", [sys.executable, str(HERE/"smoke_upstream.py"),
                      "--output", str(output/"upstream.json")]),
        ("tests", [sys.executable, "-m", "pytest", str(HERE/"test_adapter.py"), "-q",
                   f"--junitxml={output/'tests.xml'}"]),
        ("development_smoke", [sys.executable, "-m",
                                "research.frontier_2026.components.constrained_gp.smoke",
                                "--output", str(output/"smoke.json")]),
    ]
    runs = []
    for name, command in commands:
        started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (output/f"{name}.log").write_text(process.stdout+process.stderr)
        runs.append({"name": name, "exit_code": process.returncode,
                     "seconds": time.perf_counter()-started, "command": command,
                     "log": f"{name}.log"})
    for p in source_paths:
        if sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(HERE))]:
            raise SystemExit("Source changed during verification")
    xml = output/"tests.xml"
    suites = list(ET.parse(xml).getroot().iter("testsuite")) if xml.exists() else []
    result = {"status": "PASS" if all(r["exit_code"] == 0 for r in runs) else "FAIL",
              "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "runs": runs, "source_sha256": hashes,
              "upstream_commit": upstream["commit"], "upstream_hashed_files": len(upstream["files"]),
              "component_tests": sum(int(s.get("tests", 0)) for s in suites),
              "test_failures": sum(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suites),
              "scope": "Pinned original minimal example + approximate finite-pool adaptation; not full paper reproduction",
              "known_upstream_failure": "Original zero-comparison predict raises zero-size reduction ValueError",
              "heldout": "NOT RUN", "hiring_validation": "NOT RUN"}
    (output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "component_tests": result["component_tests"],
                      "test_failures": result["test_failures"], "output": str(output)}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

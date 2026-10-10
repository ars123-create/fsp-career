"""Package explicit project directories, omitting secrets, runtime state and caches."""
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=Path, default=ROOT/'output/submission-0.4.0-dev/FSP_Career_Submission_0.4.zip')
args = parser.parse_args()
TARGET = args.target.resolve()
TARGET.parent.mkdir(parents=True, exist_ok=True)
RELEASE = ROOT/'output/submission-0.4.0-dev'
VERSION = json.loads((ROOT/'VERSION.json').read_text())['version']
TOP_FILES = {"AGENTS.md", "README.md", "LICENSE", "THIRD_PARTY.md", "Dockerfile", "compose.yaml", "alembic.ini", ".env.example", ".gitignore", ".dockerignore", "VERSION.json"}
DIRECTORIES = ["backend", "frontend", "scripts", "authoring", "docs", "licenses", "research", "validation", "materials", "design", "output/validation", "output/submission-0.4.0-dev", "output/screenshots"]
CAPTURES = ["search-final", "search-final-detail", "assessment", "assessment-detail", "result", "invitation-before", "invitation-before-detail", "invitation-after", "invitation-after-detail"]
EXCLUDED = {"node_modules", "dist", "__pycache__", ".pytest_cache", ".git", ".DS_Store", ".component-smoke.mjs", ".hypothesis", ".ruff_cache"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected():
    result = {ROOT / f for f in TOP_FILES if (ROOT / f).is_file()}
    for directory in DIRECTORIES:
        for p in (ROOT / directory).rglob("*"):
            rel = p.relative_to(ROOT)
            if not p.is_file() or p.is_symlink() or any(n in EXCLUDED for n in rel.parts):
                continue
            if p.suffix in {".pyc", ".log", ".tsbuildinfo", ".zip"} or p.name.startswith(".env"):
                continue
            if p.name in {"MANIFEST.json", "SHA256SUMS.txt"}:
                continue
            result.add(p)
    result.update(ROOT / f"output/playwright/{n}.png" for n in CAPTURES)
    return sorted(result)


if __name__ == "__main__":
    if TARGET.exists():
        raise SystemExit("The release ZIP already exists. Choose a new version; do not silently replace it.")
    # Compare the exact configured local secrets in memory. Never emit their values.
    secrets = []
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(("FSP_SECRET_KEY=", "FSP_POSTGRES_PASSWORD=", "FSP_SMTP_PASSWORD=")):
                value = line.split("=", 1)[1].strip()
                if value:
                    secrets.append(value.encode())
    files = selected()
    manifest = []
    for p in files:
        data = p.read_bytes()
        if any(value in data for value in secrets):
            raise SystemExit(f"Refusing to package a configured secret in {p.relative_to(ROOT)}")
        manifest.append({"path": p.relative_to(ROOT).as_posix(), "size": len(data), "sha256": sha(data)})
    payload = json.dumps({"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "files": manifest}, ensure_ascii=False, indent=2).encode()
    manifest_path = TARGET.parent / "MANIFEST.json"
    manifest_path.write_bytes(payload)
    with ZipFile(TARGET, "x", ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, "fsp-career/" + p.relative_to(ROOT).as_posix())
        z.writestr("fsp-career/output/submission-0.4.0-dev/MANIFEST.json", payload)
    with ZipFile(TARGET) as z:
        assert z.testzip() is None
        for entry in manifest:
            assert sha(z.read("fsp-career/" + entry["path"])) == entry["sha256"]
        assert not any("/.env" in n and not n.endswith("/.env.example") for n in z.namelist())
        assert not any("/data/" in n or "/node_modules/" in n for n in z.namelist())
    checksum = sha(TARGET.read_bytes())
    (TARGET.parent / "SHA256SUMS.txt").write_text(f"{checksum}  {TARGET.name}\n")
    print(json.dumps({"status": "PASS", "file_count": len(manifest), "zip_bytes": TARGET.stat().st_size, "zip_sha256": checksum, "secrets_and_runtime_state": "excluded", "zip": str(TARGET)}, ensure_ascii=False))

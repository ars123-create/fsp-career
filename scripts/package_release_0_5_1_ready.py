"""Package the reviewed 0.5.1 release without runtime state or old release bundles."""
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path("output/submission-0.5.1-ready")
TOP = {"AGENTS.md", "README.md", "LICENSE", "THIRD_PARTY.md", "Dockerfile", "compose.yaml",
       "alembic.ini", ".env.example", ".gitignore", ".dockerignore", "VERSION.json"}
DIRECTORIES = ("backend", "frontend", "scripts", "authoring", "docs", "licenses", "research",
               "validation", "materials", "design", "output/validation", "output/screenshots",
               "output/expert-review-0.4.4")
ARTIFACTS = ("FSP_Career_Overview_0.5.1.mp4", "ТЕСТОВЫЕ_АККАУНТЫ.txt", "ИЗМЕНЕНИЯ_0.5.1.txt", "FSP_Career_Full_Flow_0.5.1.mp4", "FSP_Career_Pitch_0.5.1.pptx", "FSP_Career_Pitch_0.5.1.pdf", "FSP_Career_Documentation_0.5.1.docx", "FSP_Career_Documentation_0.5.1.pdf", "НАЧНИТЕ_ЗДЕСЬ.txt")
HISTORICAL_ARTIFACTS = ()
EXCLUDED = {"node_modules", "dist", "__pycache__", ".pytest_cache", ".git", ".DS_Store",
            ".hypothesis", ".ruff_cache", ".venv", "data"}
FORBIDDEN_SUFFIXES = {".pyc", ".log", ".original", ".tsbuildinfo", ".db", ".sqlite", ".sqlite3", ".pem", ".key"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected():
    paths = {ROOT / p for p in TOP}
    for directory in DIRECTORIES:
        for p in (ROOT / directory).rglob("*"):
            rel = p.relative_to(ROOT)
            if not p.is_file() or p.is_symlink() or any(s in EXCLUDED or s.startswith(".venv") for s in rel.parts):
                continue
            if p.suffix in FORBIDDEN_SUFFIXES or p.name.startswith(".env"):
                continue
            # Only the four explicitly prepared blind packets are included as nested ZIPs.
            if p.suffix == ".zip" and directory != "output/expert-review-0.4.4":
                continue
            paths.add(p)
    paths.update(ROOT / RELEASE / p for p in ARTIFACTS)
    paths.update(ROOT / "output/submission-0.5.0" / p for p in HISTORICAL_ARTIFACTS)
    paths.add(ROOT / "output/validation/source-v0.5.1.json")
    paths.add(ROOT / "output/elicitation-audit-0.4.2/FSP_Elicitation_Audit_0.4.2.zip")
    for name in ("search-final", "search-final-detail", "assessment", "assessment-detail", "result",
                 "invitation-before", "invitation-before-detail", "invitation-after", "invitation-after-detail"):
        paths.add(ROOT / f"output/playwright/{name}.png")
    return sorted(paths)


def package(target):
    if target.exists():
        raise FileExistsError("Choose a new archive path; existing releases are preserved")
    files = selected()
    version = json.loads((ROOT / "VERSION.json").read_text())
    snapshot = json.loads((ROOT / "output/validation/source-v0.5.1.json").read_text())
    combined = hashlib.sha256()
    for entry in sorted(snapshot["files"], key=lambda e: e["path"]):
        p = ROOT / entry["path"]
        assert p in files and sha(p.read_bytes()) == entry["sha256"], entry["path"]
        combined.update(entry["path"].encode() + b"\0" + p.read_bytes() + b"\0")
    assert combined.hexdigest() == version["application_source_sha256"]
    secrets = []
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(("FSP_SECRET_KEY=", "FSP_POSTGRES_PASSWORD=", "FSP_SMTP_PASSWORD=")):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    secrets.append(value.encode())
    manifest = []
    for p in files:
        data = p.read_bytes()
        if any(value in data for value in secrets) or re.search(rb"eyJ[\w-]{8,}\.eyJ[\w-]{16,}\.[\w-]{24,}", data):
            raise ValueError(f"Possible configured secret or bearer token in {p.relative_to(ROOT)}")
        if re.search(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]{64,}-----END [A-Z ]*PRIVATE KEY-----", data):
            raise ValueError(f"Private key material in {p.relative_to(ROOT)}")
        manifest.append({"path": p.relative_to(ROOT).as_posix(), "size": len(data), "sha256": sha(data)})
    payload = json.dumps({"version": version["version"], "created_utc": datetime.now(timezone.utc).isoformat(),
                          "application_source_sha256": combined.hexdigest(), "files": manifest},
                         ensure_ascii=False, indent=2).encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, "fsp-career/" + p.relative_to(ROOT).as_posix())
        z.writestr("fsp-career/" + str(RELEASE / "MANIFEST.json"), payload)
    with ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(manifest) + 1
        for item in manifest:
            assert sha(z.read("fsp-career/" + item["path"])) == item["sha256"]
        assert not any("/.env" in n and not n.endswith("/.env.example") for n in z.namelist())
        assert not any("/node_modules/" in n or "/data/" in n for n in z.namelist())
    (target.parent / "MANIFEST.json").write_bytes(payload)
    checksum = sha(target.read_bytes())
    (target.parent / "SHA256SUMS.txt").write_text(f"{checksum}  {target.name}\n")
    return {"status": "PASS", "zip": str(target), "files": len(manifest), "bytes": target.stat().st_size,
            "sha256": checksum, "application_files_verified": len(snapshot["files"]),
            "secrets_and_runtime_state": "excluded", "human_validation": "NOT RUN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.target.resolve()), ensure_ascii=False, indent=2))

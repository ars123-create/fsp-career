"""Independent SQL oracle and explicit wrong-rule controls; never human labels."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sqlite3

from .legacy_practical_bank import build_practical
from .retry_item import revise_retry_task

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
MODES = ("ignore_stage", "lost_response_rolls_back", "no_deduplication")


def initial_stock(task):
    return int(task["prompt"].split(": ", 1)[1].split(".", 1)[0])


def sql_reference(task):
    db = sqlite3.connect(":memory:")
    try:
        db.executescript("CREATE TABLE ledger(key TEXT PRIMARY KEY); CREATE TABLE inventory(value INTEGER);")
        db.execute("INSERT INTO inventory VALUES(?)", (initial_stock(task),))
        db.commit()
        for _, key, amount, stage in task["tables"][0]["rows"]:
            changed = db.execute("INSERT OR IGNORE INTO ledger VALUES(?)", (key,)).rowcount
            if changed:
                db.execute("UPDATE inventory SET value=value-?", (amount,))
            if stage == "before_commit":
                db.rollback()
            else:
                db.commit()
        return {"stock": db.execute("SELECT value FROM inventory").fetchone()[0],
                "writes": db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]}
    finally:
        db.close()


def wrong_rule(task, mode):
    if mode not in MODES:
        raise ValueError("Unknown wrong rule")
    stock, writes, seen = initial_stock(task), 0, set()
    for _, key, amount, stage in task["tables"][0]["rows"]:
        if mode == "lost_response_rolls_back" and stage != "ok":
            continue
        if mode == "no_deduplication" and stage == "before_commit":
            continue
        if mode != "no_deduplication" and key in seen:
            continue
        seen.add(key)
        stock -= amount
        writes += 1
    return {"stock": stock, "writes": writes}


def run(split, output):
    if output.exists():
        raise FileExistsError("Preserve prior results; choose a new output")
    seeds = range(948500, 948520) if split == "development" else range(949000, 949200)
    if split == "heldout":
        protocol = json.loads((DIRECTORY / "protocol.json").read_text())
        for name, digest in protocol["hashes"].items():
            assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == digest, name
    counts = {version: {mode: 0 for mode in MODES} for version in ("v2", "v3")}
    records = []
    for seed in seeds:
        old = next(q for q in build_practical("backend", "Middle", seed) if q["family"] == "api_retry_ledger")
        for version, task in (("v2", old), ("v3", revise_retry_task(old, seed))):
            oracle = sql_reference(task)
            expected = {k: int(v) for k, v in task["_correct"].items()}
            assert expected == oracle, (seed, version, expected, oracle)
            wrong = {mode: wrong_rule(task, mode) for mode in MODES}
            for mode, answer in wrong.items():
                counts[version][mode] += answer == oracle
            records.append({"seed": seed, "version": version, "task": task, "sql_reference": oracle,
                            "wrong_rules": wrong})
    result = {"split": split, "cases_per_version": len(seeds), "oracle_agreement": "PASS",
              "wrong_rules_accepted": counts, "human_discrimination": "NOT RUN",
              "grade_calibration": "NOT RUN", "python": platform.python_version(),
              "sqlite": sqlite3.sqlite_version, "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.split, args.output)

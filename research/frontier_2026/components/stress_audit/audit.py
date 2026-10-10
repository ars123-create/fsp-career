"""Bounded black-box audit of current research adapters, not quality evaluation.

Run only this module; do not discover pytest recursively through evidence trees.
Each method/fixture is a fresh child process, sequential, with a hard timeout.
Only X and explicit answer histories are passed into algorithms. No truth exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIXTURE_SEED = 202610084800
CASE_SECONDS = 10.
RUN_SECONDS = 300.
TIMING_SCOPE = "concurrent/noisy: parent validation may run; includes process/import costs; NOT SLA"
CORE = ("particle_kg", "particle_membership", "particle_fractional", "p06", "p20",
        "gp_guard", "pabbo_published", "risk_kl_dro", "risk_cvar", "risk_mmr", "risk_baseline_safe")
EXTRAS = ("particle_latent_noise", "particle_answerability", "p20_random", "p20_teacher",
          "linear_evoi", "pabbo_rl", "risk_edgecut")


def fixtures():
    """Declared independent engineering shapes; no respondent utility or label."""
    t50 = np.linspace(0., 1., 50)
    t95 = np.linspace(0., .25, 95)
    t100 = np.linspace(.02, .98, 100)
    result = [
        ("singleton_1x1", [[.4]], "single action, no query"),
        ("all_equal_5x3", np.full((5, 3), .5), "all duplicate, k equals N"),
        ("duplicates_6x2", [[.1, .9], [.9, .1], [.5, .5], [.5, .5], [.8, .8], [.2, .2]],
         "duplicate profiles plus strict dominance and tradeoffs"),
        ("chain_50x1", t50[:, None], "dense total dominance, one skill"),
        ("chain_50x3", np.column_stack((t50, t50*t50, np.sqrt(t50))),
         "dense total dominance, three skills"),
        ("mostly_chain_100x2", np.vstack((np.column_stack((t95, t95)),
                                         [[.5, .95], [.6, .85], [.7, .75], [.8, .65], [.9, .55]])),
         "100 candidates, dense dominance, five mutually incomparable maxima"),
        ("antichain_100x3", np.column_stack((t100, 1-t100,
            np.random.default_rng(FIXTURE_SEED+6).uniform(.1, .9, 100))),
         "4950 tradeoff pairs, dense query pool; intentionally OOD for P20"),
        ("cycle_6x3", [[1., .2, .2], [.2, 1., .2], [.2, .2, 1.],
                        [.5, .5, .5], [.15, .15, .15], [.8, .8, .8]],
         "three mutually incomparable arms allow strict cyclic reported preferences"),
        ("five_tradeoffs_5x2", np.column_stack((np.linspace(.1, .9, 5), np.linspace(.9, .1, 5))),
         "nonempty question pool but k=N means no decision value"),
    ]
    return {name: {"values": np.asarray(x, dtype=float).tolist(), "seed": FIXTURE_SEED+i,
                   "description": description, "n": len(x), "d": np.asarray(x).shape[1],
                   "uses_random_values": name == "antichain_100x3"}
            for i, (name, x, description) in enumerate(result)}


def grid():
    """80 declared cells, interleaved so every shape is reached early."""
    rows = []
    for fixture in ("singleton_1x1", "all_equal_5x3", "duplicates_6x2", "cycle_6x3"):
        rows.extend((method, fixture) for method in CORE)
    for fixture in ("mostly_chain_100x2", "antichain_100x3"):
        rows.extend((method, fixture) for method in CORE[:7])
    for fixture in ("chain_50x1", "chain_50x3"):
        rows.extend((method, fixture) for method in ("particle_kg", "p06", "p20", "gp_guard", "risk_mmr"))
    rows.extend((method, "five_tradeoffs_5x2") for method in
                ("particle_kg", "p06", "p20", "pabbo_published", "risk_cvar"))
    rows.extend((method, "cycle_6x3") for method in EXTRAS)
    # No result-dependent scheduling. Prioritize broad dimension coverage first.
    names = list(fixtures())
    queues = {f: [row for row in rows if row[1] == f] for f in names}
    ordered = []
    while any(queues.values()):
        for name in names:
            if queues[name]:
                ordered.append(queues[name].pop(0))
    return ordered


def canonical_pairs(x):
    """Independent contract check: key identity is vectors, not candidate IDs."""
    keys = sorted(set(map(tuple, x)))
    return [(a, b) for i, a in enumerate(keys) for b in keys[i+1:]
            if any(ai > bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))]


def factory(method, x):
    if method.startswith("particle_"):
        from ...policies import Config, ParticleSession
        return ParticleSession(x, Config(policy=method.removeprefix("particle_")))
    if method == "p06":
        from ..constrained_gp.adapter import Session
        return Session(x)
    if method.startswith("p20"):
        from ..distillation.adapter import Config, Session
        modes = {"p20": "learned_lookahead", "p20_random": "random_lookahead", "p20_teacher": "teacher"}
        return Session(x, Config(mode=modes[method]))
    if method.startswith("risk_"):
        from ..risk_decisions.adapter import Session
        return Session(x)
    from ..legacy_controls.adapter import make_session
    return make_session({"gp_guard": "gp_topk_guard", "linear_evoi": "linear_evoi_3"}.get(method, method), x)


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def source_paths():
    paths = set((ROOT/"research/frontier_2026").glob("*.py"))
    for folder in ("constrained_gp", "distillation", "legacy_controls", "risk_decisions", "stress_audit"):
        paths.update((ROOT/"research/frontier_2026/components"/folder).glob("*.py"))
    for folder in ("preference_topk", "nonlinear_topk", "pabbo_topk"):
        paths.update((ROOT/"research"/folder).glob("*.py"))
    paths.update((ROOT/"research/pabbo_topk/upstream").rglob("*.py"))
    paths.update((ROOT/"research/pabbo_topk/upstream").rglob("*.json"))
    paths.update(ROOT/p for p in (
        "research/frontier_2026/components/distillation/artifact-v1/model.npz",
        "research/pabbo_topk/upstream/results/PABBO/PABBO_GP6D/ckpt.tar",
        "research/pabbo_topk/checkpoints/rl-0600.pt"))
    return sorted(paths)


def hashes():
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in source_paths()}


def worker(args):
    protocol = json.loads(Path(args.protocol).read_text())
    fixture = protocol["fixtures"][args.fixture]
    x = np.array(fixture["values"], dtype=float)
    stream = Path(args.progress).open("w", buffering=1)
    started = time.perf_counter()
    results, exclusions, outputs = [], [], {}

    def emit(event):
        stream.write(json.dumps(safe_json({"elapsed_seconds": time.perf_counter()-started, **event}),
                                allow_nan=False)+"\n")
        stream.flush()

    def invoke(session, history):
        if args.method.startswith("risk_"):
            return session.recommend(history, method=args.method.removeprefix("risk_"), k=5)
        return session.recommend(history, k=5)

    def inspect_result(result, values, session, history):
        selected, query = result
        selected = tuple(selected)
        assert len(selected) == len(set(selected)) == min(5, len(values)), "wrong top-k cardinality"
        assert all(isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
                   and 0 <= i < len(values) for i in selected), "invalid candidate index"
        if query is not None:
            assert isinstance(query, (int, np.integer)) and not isinstance(query, (bool, np.bool_)), "invalid query type"
            assert 0 <= query < len(session.pairs), "query outside canonical pool"
            assert query not in {q for q, _ in history}, "repeated query"
        if len(history) == 3 or len(values) <= 5 or not session.pairs:
            assert query is None, "question after exhausted budget/single action/no questions"
        outside = set(range(len(values)))-set(selected)
        violations = [(int(i), int(j)) for i in selected for j in outside
                      if np.all(values[j] >= values[i]) and np.any(values[j] > values[i])]
        assert not violations, f"selected dominated candidate while dominator excluded: {violations[:8]}"
        return {"selected": tuple(map(int, selected)), "query": None if query is None else int(query),
                "semantic_selection": sorted(tuple(values[i]) for i in selected)}

    def check(name, operation, *, history=None, expected_rejection=False):
        emit({"event": "start", "check": name, "history": history})
        before = time.perf_counter()
        entry = {"check": name, "history": history}
        try:
            detail = operation()
            if expected_rejection:
                raise AssertionError("invalid history was accepted")
            entry.update(status="PASS", detail=detail)
        except (ValueError, TypeError) as exc:
            if expected_rejection:
                entry.update(status="PASS", rejected_type=type(exc).__name__, message=str(exc))
            else:
                entry.update(status="EXPLICIT_MODEL_LIMIT" if type(exc).__name__ == "VersionSpaceEmpty" else "FAIL",
                             error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        except Exception as exc:
            explicit = type(exc).__name__ in {"SamplingError", "PosteriorDegeneracyError", "VersionSpaceEmpty"}
            entry.update(status="EXPLICIT_MODEL_LIMIT" if explicit else "FAIL",
                         error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            if hasattr(exc, "diagnostics"):
                entry["diagnostics"] = exc.diagnostics
        entry["seconds"] = time.perf_counter()-before
        results.append(entry)
        emit({"event": "result", **entry})
        return entry

    emit({"event": "start", "check": "construct", "method": args.method, "fixture": args.fixture})
    try:
        session = factory(args.method, x.copy())
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(1)
        config = getattr(session, "config", None)
        emit({"event": "constructed", "pair_count": len(session.pairs),
              "config": asdict(config) if is_dataclass(config) else None,
              "sampling_diagnostics": getattr(session, "sampling_diagnostics", None)})
    except Exception as exc:
        explicit = type(exc).__name__ in {"SamplingError", "PosteriorDegeneracyError", "VersionSpaceEmpty"}
        emit({"event": "result", "check": "construct", "status": "EXPLICIT_MODEL_LIMIT" if explicit else "FAIL",
              "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
              "diagnostics": getattr(exc, "diagnostics", None)})
        return

    def pair_contract():
        keys = [(tuple(x[a]), tuple(x[b])) for a, b in session.pairs]
        assert keys == canonical_pairs(x), "canonical pairs differ from independent feature contract"
        return {"pairs": len(keys), "unique_vectors": len(set(map(tuple, x)))}
    check("canonical_pair_contract", pair_contract)

    def recommend(name, history):
        def operation():
            detail = inspect_result(invoke(session, history), x, session, history)
            outputs[name] = detail
            if name.startswith("skip_") and "cold" in outputs:
                assert detail["selected"] == outputs["cold"]["selected"], "neutral skip changed terminal selection"
            if name == "cold" and "skip_3" in outputs:
                assert detail["selected"] == outputs["skip_3"]["selected"], "three neutral skips changed terminal selection"
            if name == "cold_repeat" and "cold" in outputs:
                assert detail == outputs["cold"], "repeated identical API call changed output"
            if name == "history_reverse" and "mixed_three" in outputs:
                assert detail == outputs["mixed_three"], "history order changed posterior decision"
            diagnostics = getattr(session, "last_diagnostics", {})
            return {**detail, "diagnostics": diagnostics}
        return check(name, operation, history=history)

    count = len(session.pairs)
    # Cheap terminal checks run first, preserving evidence even if cold lookahead times out.
    if count >= 3:
        recommend("skip_3", ((0, 2), (1, 2), (2, 2)))
        mixed = ((0, 0), (1, 2), (2, 1))
        recommend("mixed_three", mixed)
        recommend("history_reverse", tuple(reversed(mixed)))
        if args.fixture == "cycle_6x3":
            history = []
            for preferred, other in ((0, 1), (1, 2), (2, 0)):
                q = next(q for q, pair in enumerate(session.pairs) if set(pair) == {preferred, other})
                history.append((q, 0 if session.pairs[q][0] == preferred else 1))
            recommend("contradictory_three_cycle", tuple(history))
    else:
        exclusions.append("Three distinct skips/mixed history/cycle unavailable: fewer than three canonical questions")
    recommend("cold", ())
    recommend("cold_repeat", ())
    if count:
        recommend("alternative_left", ((0, 0),))
        recommend("alternative_right", ((0, 1),))
        recommend("skip_1", ((0, 2),))
        if count >= 2:
            recommend("skip_2", ((0, 2), (1, 2)))
        for name, history in (
                ("reject_repeat", ((0, 0), (0, 1))),
                ("reject_bool_outcome", ((0, True),)),
                ("reject_four_slots", tuple((q % count, 2) for q in range(4))),
                ("reject_out_of_range", ((count, 0),))):
            check(name, lambda h=history: invoke(session, h), history=history, expected_rejection=True)
    else:
        exclusions.append("Alternative/skip/repeated-query histories undefined on empty canonical pair pool")
        check("reject_query_on_empty_pool", lambda: invoke(session, ((0, 0),)),
              history=((0, 0),), expected_rejection=True)

    def permutation():
        order = np.random.default_rng(fixture["seed"]).permutation(len(x))
        moved = factory(args.method, x[order])
        detail = inspect_result(invoke(moved, ()), x[order], moved, ())
        baseline = outputs.get("cold")
        emit({"event": "permutation_witness", "permutation": order.tolist(),
              "before": baseline, "after": detail})
        assert baseline is not None, "cold failed; permutation not comparable"
        assert detail["semantic_selection"] == baseline["semantic_selection"], \
            "candidate permutation changed selected feature multiset (possible tie sensitivity; inspect witness)"
        assert detail["query"] == baseline["query"], \
            "candidate permutation changed canonical next-query ID (possible numerical/tie sensitivity)"
        return {"permutation": order.tolist(), **detail}
    check("candidate_permutation", permutation)
    loaded = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            # Torch publishes virtual module __file__ values such as '_ops.py'.
            # They are not project sources and must not invalidate API evidence.
            if path.suffix == ".py" and path.is_relative_to(ROOT) and path.is_file():
                loaded[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    emit({"event": "done", "exclusions": exclusions, "loaded_source_sha256": loaded,
          "checks": len(results), "counts": dict(Counter(r["status"] for r in results))})
    stream.close()


def run():
    output = HERE/"output"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "version": 1, "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent X/history-only engineering stress audit, no hidden truth, not quality/final evaluation",
        "timing_scope": TIMING_SCOPE, "case_seconds": CASE_SECONDS, "run_seconds": RUN_SECONDS,
        "fixtures": fixtures(), "grid": grid(), "source_sha256": hashes(),
        "methods": list(CORE+EXTRAS), "parameters": "existing adapter defaults, no tuning; risk method passed by name",
        "exclusions": ["Not the full N×d×history×method Cartesian product",
                       "MoNNE refits excluded: this budget targets particle/GP/distillation/legacy/risk adapters",
                       "Risk N100 solver stress excluded from this bounded grid; no claim of all-shape support",
                       "P20 N50/100 inputs outside its small component training distribution; contract only",
                       "No algorithm quality, human hiring, calibration or SLA conclusions"],
        "environment": {"python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "botorch", "gpytorch")},
            "child_threads": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}},
    }
    protocol_path = output/"protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2)+"\n")
    env = dict(os.environ, **protocol["environment"]["child_threads"])
    results = []
    began = time.perf_counter()
    for i, (method, fixture) in enumerate(protocol["grid"]):
        remaining = RUN_SECONDS-(time.perf_counter()-began)
        row = {"index": i, "method": method, "fixture": fixture}
        if remaining <= 0:
            row.update(status="NOT RUN", reason="global predeclared harness time cap reached")
            results.append(row)
            continue
        progress = output/f"{i:03d}-{method}-{fixture}.jsonl"
        command = [sys.executable, "-m", "research.frontier_2026.components.stress_audit.audit", "--worker",
                   "--protocol", str(protocol_path), "--method", method, "--fixture", fixture, "--progress", str(progress)]
        before = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                       timeout=min(CASE_SECONDS, remaining))
            row.update(exit_code=completed.returncode, stderr=completed.stderr, stdout=completed.stdout)
        except subprocess.TimeoutExpired as exc:
            row.update(timeout=True, stdout=str(exc.stdout or ""), stderr=str(exc.stderr or ""))
        row["seconds"] = time.perf_counter()-before
        events = [json.loads(line) for line in progress.read_text().splitlines()] if progress.exists() else []
        entries = [e for e in events if e.get("event") == "result"]
        done = next((e for e in events if e.get("event") == "done"), None)
        row.update(progress=str(progress.relative_to(ROOT)), checks=len(entries),
                   counts=dict(Counter(e["status"] for e in entries)), last_event=events[-1] if events else None,
                   failures=[e for e in entries if e["status"] != "PASS"])
        if any(e["status"] == "FAIL" for e in entries) or (not row.get("timeout") and row.get("exit_code") != 0):
            row["status"] = "FAIL"
        elif row.get("timeout"):
            row["status"] = "TIMEOUT"
        elif any(e["status"] == "EXPLICIT_MODEL_LIMIT" for e in entries):
            row["status"] = "EXPLICIT_MODEL_LIMIT"
        elif done:
            row["status"] = "PASS"
        else:
            row.update(status="FAIL", reason="worker exited without terminal evidence")
        results.append(row)
        (output/"results.partial.json").write_text(json.dumps(safe_json(results), indent=2, allow_nan=False)+"\n")
        print(json.dumps({"index": i, "method": method, "fixture": fixture, "status": row["status"],
                          "checks": row["checks"], "seconds_concurrent_noisy": row["seconds"]}), flush=True)
    after = hashes()
    drift = {name: {"before": digest, "after": after.get(name)} for name, digest in protocol["source_sha256"].items()
             if after.get(name) != digest}
    summary = {"status": "PASS" if all(r["status"] == "PASS" for r in results) and not drift else "PARTIAL",
               "utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
               "counts": dict(Counter(r["status"] for r in results)), "cells": len(results),
               "checks": sum(r.get("checks", 0) for r in results), "source_drift": drift,
               "seconds_concurrent_noisy": time.perf_counter()-began, "timing_scope": TIMING_SCOPE,
               "results": results, "final_benchmark": "NOT RUN by this audit", "source_sha256_after": after}
    (output/"verification.json").write_text(json.dumps(safe_json(summary), indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"results", "source_sha256_after"}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    for name in ("protocol", "method", "fixture", "progress"):
        parser.add_argument("--"+name)
    args = parser.parse_args()
    worker(args) if args.worker else run()


if __name__ == "__main__":
    main()

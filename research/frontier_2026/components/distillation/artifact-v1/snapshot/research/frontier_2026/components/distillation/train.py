"""Frozen bounded teacher/student experiment; no pre-existing pilot data read.

PyTorch training follows the official nn/optimizer tutorial pattern, BSD3,
copyright PyTorch contributors; source/license retained in upstream/. The
finite-state data, labels, features, and checker are project adaptations.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from .adapter import Config, FEATURE_NAMES, HERE, Session, Student, action_values
from ...finite import plan


TRAIN_SEEDS = tuple(range(202610082000, 202610082032))
AGGREGATION_SEEDS = tuple(range(202610082032, 202610082048))
DEV_SEEDS = tuple(range(202610082060, 202610082076))
TINY_SEEDS = tuple(range(202610082090, 202610082094))
BASE = Config()
MODES = ("teacher", "greedy", "learned_lookahead", "random_lookahead",
         "student", "learned_kg", "random_kg", "guarded_kg")


def pool(seed, *, tiny=False):
    rng = np.random.default_rng(seed)
    index = seed % 100
    n, d = (6, 2) if tiny else ((6, 10, 14, 18)[index % 4], 2+index % 2)
    values = rng.uniform(.05, .95, (n, d))
    if index % 2 == 0:
        values[:, 1] = np.clip(1-values[:, 0]+rng.normal(0, .05, n), 0, 1)
    return values, rng


def observe(session, history, query, rng, k=5):
    model = session.model(k)
    probability = model.likelihood[query] @ model.posterior(history)
    probability /= probability.sum()
    return int(rng.choice(3, p=probability))


def labelled_state(session, history, *, task_seed, phase, state_id):
    model = session.model()
    history, available = session.available(history)
    roots = session.teacher_roots(history)
    if len(history) == 3 or not available or model.k == len(session.values):
        return None
    before = time.perf_counter()
    values, trace = action_values(model, history, roots, depth=3-len(history), beam=BASE.continuation_beam)
    teacher_seconds = time.perf_counter()-before
    ids, all_features = session.features(history)
    positions = {q: i for i, q in enumerate(ids)}
    features = all_features[[positions[q] for q in roots]]
    centered = values-values.mean()
    current = model.terminal(model.posterior(history))
    best = int(np.argmax(values))
    query = roots[best] if values[best] > current else None
    metadata = {
        "state_id": state_id, "phase": phase, "task_seed_metadata_only": task_seed,
        "values": session.values.tolist(), "history": list(history), "remaining_depth": 3-len(history),
        "available_count": len(available), "teacher_roots": list(roots),
        "teacher_pool_coverage": len(roots)/len(available), "teacher_action_values": values.tolist(),
        "teacher_best_query": query, "teacher_best_value": float(max(current, values[best])),
        "terminal_current": current, "teacher_seconds": teacher_seconds,
        "global_exhaustive": len(roots) == len(available) and trace["continuation_exhaustive"],
        **trace,
    }
    return features, centered, metadata


def collect(seeds, *, phase, checkpoint=None):
    rows, targets, metadata, state_ids = [], [], [], []
    mode = "teacher" if checkpoint is None else "learned_lookahead"
    cfg = replace(BASE, mode=mode, checkpoint=str(checkpoint or BASE.checkpoint))
    for task_index, seed in enumerate(seeds):
        values, rng = pool(seed)
        session = Session(values, cfg)
        history = ()
        for step in range(3):
            labelled = labelled_state(session, history, task_seed=seed, phase=phase,
                                      state_id=f"{phase}:{seed}:{step}")
            if labelled is None:
                break
            x, y, info = labelled
            rows.extend(x)
            targets.extend(y)
            state_ids.extend([info["state_id"]]*len(y))
            if checkpoint is None:
                query = info["teacher_best_query"]
            else:
                _, query = session.recommend(history)
            info["behavior_query"] = query
            if query is not None:
                outcome = observe(session, history, query, rng)
                info["behavior_outcome"] = outcome
            metadata.append(info)
            if query is None:
                break
            history += ((query, outcome),)
        if task_index % 8 == 7:
            print(json.dumps({"phase": phase, "tasks_done": task_index+1,
                              "states": len(metadata), "labels": len(rows)}), flush=True)
    return np.asarray(rows), np.asarray(targets), metadata, np.asarray(state_ids)


def save_data(directory, name, data):
    x, y, metadata, state_ids = data
    np.savez_compressed(directory/f"{name}.npz", features=x, target_centered_value=y, state_ids=state_ids)
    (directory/f"{name}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in metadata))


def fit(x, y, destination, *, seed):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    target_scale = max(float(y.std()), 1e-6)
    tx = torch.tensor((x-mean)/scale, dtype=torch.float32)
    ty = torch.tensor(y/target_scale, dtype=torch.float32)
    started = time.perf_counter()
    losses = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), 32), torch.nn.ReLU(),
                                      torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).float()
        optimizer = torch.optim.Adam(network.parameters(), lr=.002, weight_decay=.0001)
        for step in range(400):
            prediction = network(tx).flatten()
            loss = torch.nn.functional.mse_loss(prediction, ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 25 == 0 or step == 399:
                losses.append({"step": step, "standardized_train_mse": float(loss.detach())})
        network.eval()
        with torch.no_grad():
            train_prediction = network(tx).flatten().numpy()*target_scale
        arrays = {"mean": mean, "scale": scale, "target_scale": np.array(target_scale), "format": np.array(1)}
        for index, layer in enumerate((network[0], network[2], network[4])):
            arrays[f"w{index}"] = layer.weight.detach().numpy().copy()
            arrays[f"b{index}"] = layer.bias.detach().numpy().copy()
        np.savez_compressed(destination, **arrays)
    portable = Student(destination).predict(x)
    disagreement = float(np.max(np.abs(portable-train_prediction)))
    if disagreement > 1e-6:
        raise RuntimeError("NumPy inference disagrees with trained Torch network")
    return {"seed": seed, "steps": 400, "architecture": [len(FEATURE_NAMES), 32, 32, 1],
            "optimizer": "Adam", "learning_rate": .002, "weight_decay": .0001,
            "training_rows": len(y), "training_rmse": float(np.sqrt(np.mean((portable-y)**2))),
            "target_scale": target_scale, "losses": losses, "seconds": time.perf_counter()-started,
            "numpy_torch_max_abs_difference": disagreement,
            "checkpoint_sha256": sha256(destination.read_bytes()).hexdigest()}


def evaluate(checkpoint, destination):
    rows = []
    labels = []
    for seed in DEV_SEEDS:
        values, rng = pool(seed)
        reference = Session(values, replace(BASE, mode="teacher"))
        sessions = {mode: Session(values, replace(BASE, mode=mode, checkpoint=str(checkpoint))) for mode in MODES}
        history = ()
        for step in range(3):
            labelled = labelled_state(reference, history, task_seed=seed, phase="dev",
                                      state_id=f"dev:{seed}:{step}")
            if labelled is None:
                break
            x, y, info = labelled
            labels.append(info)
            model = reference.model()
            p = model.posterior(history)
            _, available = reference.available(history)
            kg = model.one_step_values(p, available)
            best_kg = float(max(model.terminal(p), kg.max()))
            kg_map = dict(zip(available, map(float, kg)))
            action_map = dict(zip(info["teacher_roots"], info["teacher_action_values"]))
            decisions = {}
            for mode, session in sessions.items():
                _, query = session.recommend(history)
                decisions[mode] = (query, dict(session.last_diagnostics))
            missing = sorted({q for q, _ in decisions.values() if q is not None and q not in action_map})
            if missing:
                extra, _ = action_values(model, history, missing, depth=3-len(history), beam=BASE.continuation_beam)
                action_map.update(zip(missing, map(float, extra)))
            for mode, (query, diagnostic) in decisions.items():
                current = info["terminal_current"]
                value = current if query is None else action_map[query]
                shortlist = diagnostic.get("shortlist", [])
                teacher_query = info["teacher_best_query"]
                rows.append({
                    "state_id": info["state_id"], "task_seed_metadata_only": seed,
                    "profiles": len(values), "remaining_depth": 3-step, "mode": mode, "query": query,
                    "approximate_teacher_signed_gap": info["teacher_best_value"]-value,
                    "checker_action_value": value,
                    "global_one_step_loss": best_kg-(current if query is None else kg_map[query]),
                    "teacher_best_in_shortlist": teacher_query is None or teacher_query in shortlist,
                    "teacher_pool_coverage": info["teacher_pool_coverage"],
                    "shortlist_coverage": diagnostic.get("pool_coverage", 0.),
                    "recommend_seconds": diagnostic["recommend_seconds"],
                    "diagnostics": diagnostic,
                })
            # All methods are evaluated at identical states from independent random
            # behavior, not at states chosen to favor teacher or current student.
            query = int(rng.choice(available))
            history += ((query, observe(reference, history, query, rng)),)
        print(json.dumps({"phase": "dev", "task": seed, "rows": len(rows)}), flush=True)
    (destination/"dev-rows.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    (destination/"dev-teacher-labels.jsonl").write_text("".join(json.dumps(row)+"\n" for row in labels))
    result = {"scope": "Independent development states only; signed gap to restricted approximate teacher",
              "tasks": len(DEV_SEEDS), "states": len(labels), "tuned_after_dev": False,
              "heldout": "NOT RUN", "modes": {}}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        early = [r for r in subset if r["remaining_depth"] > 1]
        gaps = np.array([r["approximate_teacher_signed_gap"] for r in subset])
        times = np.array([r["recommend_seconds"] for r in subset])
        result["modes"][mode] = {
            "states": len(subset), "mean_signed_teacher_gap": float(gaps.mean()),
            "max_signed_teacher_gap": float(gaps.max()),
            "states_better_than_restricted_teacher": int(np.sum(gaps < -1e-12)),
            "mean_global_one_step_loss": float(np.mean([r["global_one_step_loss"] for r in subset])),
            "early_teacher_best_shortlist_recall": float(np.mean([r["teacher_best_in_shortlist"] for r in early])),
            "early_mean_shortlist_coverage": float(np.mean([r["shortlist_coverage"] for r in early])),
            "median_recommend_seconds": float(np.median(times)),
            "p95_recommend_seconds": float(np.quantile(times, .95)),
        }
    (destination/"dev-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def tiny_control(destination):
    rows = []
    for seed in TINY_SEEDS:
        values, _ = pool(seed, tiny=True)
        session = Session(values, replace(BASE, mode="teacher"))
        model = session.model()
        roots = tuple(range(len(session.pairs)))
        if not roots:
            rows.append({"seed": seed, "status": "NO TRADEOFFS"})
            continue
        values, trace = action_values(model, (), roots, depth=3, beam=None)
        expected = max(model.terminal(model.prior), float(values.max()))
        oracle = plan(model, budget=3, beam=None)
        error = abs(expected-oracle.expected_total)
        rows.append({"seed": seed, "roots": len(roots), "value": expected,
                     "common_exhaustive_value": oracle.expected_total, "absolute_error": error,
                     "status": "PASS" if error < 1e-12 else "FAIL", **trace,
                     "scope": "Floating enumeration of all finite queries; not real arithmetic/Lean certified"})
    (destination/"tiny-exhaustive.json").write_text(json.dumps(rows, indent=2)+"\n")
    if any(r["status"] == "FAIL" for r in rows):
        raise RuntimeError("tiny exhaustive control failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE/"artifact-v1")
    destination = parser.parse_args().output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)  # training CLI process only
    repository = HERE.parents[3]
    source_paths = [HERE/"adapter.py", HERE/"train.py", HERE/"DESIGN.md",
                    repository/"research/frontier_2026/finite.py",
                    repository/"research/frontier_2026/models.py",
                    repository/"research/preference_topk/core.py"]
    hashes = {}
    for path in source_paths:
        name = str(path.relative_to(repository))
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        copy = destination/"snapshot"/name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(BASE),
        "train_seeds": TRAIN_SEEDS, "aggregation_seeds": AGGREGATION_SEEDS,
        "dev_seeds": DEV_SEEDS, "tiny_seeds": TINY_SEEDS,
        "feature_names": FEATURE_NAMES, "source_sha256": hashes,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "phase": "Frozen development; not heldout", "existing_pilot_data_used": False,
    }
    (destination/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    first = collect(TRAIN_SEEDS, phase="train_initial")
    save_data(destination, "teacher-initial", first)
    first_fit = fit(first[0], first[1], destination/"model-initial.npz", seed=202610082050)
    (destination/"fit-initial.json").write_text(json.dumps(first_fit, indent=2)+"\n")
    additional = collect(AGGREGATION_SEEDS, phase="train_learner_visited", checkpoint=destination/"model-initial.npz")
    save_data(destination, "teacher-aggregation", additional)
    x, y = np.r_[first[0], additional[0]], np.r_[first[1], additional[1]]
    final_fit = fit(x, y, destination/"model.npz", seed=202610082051)
    (destination/"fit-final.json").write_text(json.dumps(final_fit, indent=2)+"\n")
    tiny_control(destination)
    evaluation = evaluate(destination/"model.npz", destination)
    changed = [str(p.relative_to(repository)) for p in source_paths
               if sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(repository))]]
    manifest = {str(p.relative_to(destination)): sha256(p.read_bytes()).hexdigest()
                for p in destination.rglob("*") if p.is_file()}
    result = {"status": "PASS" if not changed else "SOURCE CHANGED DURING RUN",
              "changed_source_paths": changed, "trained_rows": len(y),
              "train_states": len(first[2])+len(additional[2]), "dev_states": evaluation["states"],
              "artifact_sha256": manifest, "heldout": "NOT RUN"}
    (destination/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "trained_rows": len(y),
                      "train_states": result["train_states"], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()

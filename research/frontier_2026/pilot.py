"""Exploratory pilot only: logged development, never a final winner test."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
import scipy
import torch

from .policies import Config, ParticleSession
from .components.monne.adapter import Config as MoNNEConfig, MoNNE
from .simulation import FAMILIES, evaluate, make_world, response

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "linear_kg_small_quadrature": Config(prior="linear"),
    "P01_mixed_kg": Config(),
    "mixed_no_questions": Config(policy="none"),
    "mixed_random": Config(policy="random"),
    "mixed_response_entropy": Config(policy="entropy"),
    "mixed_bald": Config(policy="bald"),
    "mixed_eubo": Config(policy="eubo"),
    "P02_mixed_depth3_beam4": Config(horizon=3, beam=4),
    "P04_lattice_kg": Config(prior="lattice"),
    "P05_choquet_kg": Config(prior="choquet"),
    "P07_membership": Config(policy="membership"),
    "P12_latent_noise": Config(policy="latent_noise"),
    "P13_fractional": Config(policy="fractional"),
    "P14_answerability": Config(policy="answerability"),
    "P15_easy": Config(policy="easy"),
    "P16_set_entropy": Config(policy="set_entropy"),
    "P18_thompson_boundary_kg": Config(policy="thompson"),
    "P19_class_robust": Config(policy="class_robust"),
}
NEURAL = ("P03_monne_top5", "monne_ieubo")


def run(world, name):
    started = perf_counter()
    history, trace = [], []
    try:
        if name in NEURAL:
            algorithm = MoNNE(world.values, MoNNEConfig(ensemble_size=8))
            acquisition = "top5" if name == "P03_monne_top5" else "ieubo"
            recommend = lambda: algorithm.recommend(tuple(history), acquisition=acquisition)
        else:
            algorithm = ParticleSession(world.values, CONFIGS[name])
            recommend = lambda: algorithm.recommend(tuple(history))
        construction_seconds = perf_counter()-started
        for slot in range(4):
            step = perf_counter()
            selected, question = recommend()
            metrics = evaluate(world, selected)
            row = {"slot": slot, "selection": list(selected), "query_id": question,
                   "seconds": perf_counter()-step, **metrics}
            diagnostics = getattr(algorithm, "last_diagnostics", None)
            if diagnostics:
                row["diagnostics"] = dict(diagnostics)
            trace.append(row)
            if slot == 3 or question is None:
                break
            if question in {q for q, _ in history}:
                raise ValueError("policy repeated a question")
            pair = algorithm.pairs[question]
            choice, p, skip = response(world, pair, slot)
            row.update({"pair": list(pair), "choice": choice,
                        "true_probability_left": p, "true_skip_probability": skip})
            history.append((question, choice))
        return {"method": name, "seed": world.seed, "family": world.family,
                "status": "PASS", "construction_seconds": construction_seconds,
                "total_seconds": perf_counter()-started, "slots": len(history),
                "answers": sum(r != 2 for _, r in history), "trace": trace,
                **{key: trace[-1][key] for key in ("regret", "oracle_overlap", "dominance_violations")}}
    except Exception as exc:
        return {"method": name, "seed": world.seed, "family": world.family,
                "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "trace": trace,
                "total_seconds": perf_counter()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=202610081000)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--methods", nargs="+", choices=tuple(CONFIGS)+NEURAL,
                        default=list(CONFIGS))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing overwrite of development evidence")
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    source_files = [ROOT/x for x in ("finite.py", "models.py", "policies.py", "simulation.py",
                                    "pilot.py", "components/monne/adapter.py")]
    protocol = {"stage": "EXPLORATORY DEVELOPMENT ONLY", "utc": datetime.now(timezone.utc).isoformat(),
                "seeds": list(range(args.seed_start, args.seed_start+args.seeds)),
                "families": args.families, "methods": args.methods,
                "configs": {name: asdict(CONFIGS[name]) if name in CONFIGS else asdict(MoNNEConfig())
                            for name in args.methods},
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                                  for p in source_files},
                "not_final_test": True, "common_random_numbers": "same turn-level draws across methods",
                "limitations": "Synthetic, known family definitions, parameter exploration allowed here; no human validation"}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+"\n")
    rows = []
    with (args.out/'rows.jsonl').open('w') as stream:
        for seed in protocol['seeds']:
            for family in args.families:
                world = make_world(seed, family)
                for name in args.methods:
                    result = run(world, name)
                    stream.write(json.dumps(result)+"\n")
                    stream.flush()
                    rows.append(result)
                print(json.dumps({"seed": seed, "family": family, "rows": len(rows),
                                  "failures": sum(r['status'] != 'PASS' for r in rows)}), flush=True)
    summary = {}
    for name in args.methods:
        method = [r for r in rows if r['method'] == name]
        good = [r for r in method if r['status'] == 'PASS']
        summary[name] = {"cases": len(method), "failures": len(method)-len(good),
                         "mean_regret_of_successful_cases": float(np.mean([r['regret'] for r in good])) if good else None,
                         "dominance_cases": sum(r['dominance_violations'] > 0 for r in good),
                         "mean_slots": float(np.mean([r['slots'] for r in good])) if good else None,
                         "step_p95_seconds": float(np.quantile([s['seconds'] for r in good for s in r['trace']], .95)) if good else None,
                         "mean_total_seconds": float(np.mean([r['total_seconds'] for r in method]))}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"stage": "DEVELOPMENT", "complete": True, "rows": len(rows), "summary": summary}, indent=2))


if __name__ == '__main__':
    main()

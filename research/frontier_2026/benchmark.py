"""Auditable sequential benchmark. Truth stays in the respondent/evaluator.

Every invocation writes a new immutable-by-convention evidence directory and
captures executable source/checkpoints BEFORE generating the first task.
Method errors are rows, never silently removed or replaced with a baseline.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import scipy
import torch

from .finite import tradeoff_pairs
from .registry import METHODS, create, description
from .simulation import FAMILIES, VARIANTS, evaluate, make_world, response, response_probabilities
from .components.calibration import common_query_ids, prediction_report, expected_scores

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n")


def source_files():
    # Narrow source traversal excludes Lean runtime, old results, private app
    # data, environments and recursive evidence snapshots.
    files = set(HERE.glob("*.py"))
    for name in ('SEED_LEDGER.json','STRESS_PROTOCOL.json','STRESS_PROTOCOL.sha256'):
        if (HERE/name).is_file():
            files.add(HERE/name)
    for path in (HERE/'components').rglob('*'):
        if path.is_file() and (path.suffix in {'.py', '.pt', '.npz'} or
                              path.name.startswith(('LICENSE', 'NOTICE'))):
            if not any(part in {'output', '__pycache__', '.venv', 'evidence'} for part in path.parts):
                files.add(path)
    relatives = (
        'research/__init__.py', 'research/preference_topk/__init__.py',
        'research/preference_topk/core.py', 'research/nonlinear_topk/__init__.py',
        'research/nonlinear_topk/core.py', 'research/pabbo_topk/__init__.py',
        'research/pabbo_topk/core.py', 'research/pabbo_topk/smoke_upstream.py',
        'research/pabbo_topk/upstream/policies/transformer.py',
        'research/pabbo_topk/checkpoints/rl-0600.pt',
        'research/pabbo_topk/upstream/results/PABBO/PABBO_GP6D/ckpt.tar',
        'research/pabbo_topk/LICENSE.txt', 'research/pabbo_topk/NOTICE.md',
        'research/frontier_2026/components/monne/upstream/LICENSE',
        'research/frontier_2026/NOTICE.md', 'LICENSE',
    )
    files.update(WORKSPACE/p for p in relatives if (WORKSPACE/p).is_file())
    return sorted(files)


def snapshot(out):
    manifest = {}
    for source in source_files():
        relative = source.relative_to(WORKSPACE)
        target = out/'snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha256(target.read_bytes()).hexdigest()
        if sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed while capturing: {source}")
        manifest[str(relative)] = digest
    write_json(out/'snapshot-manifest.json', manifest)
    return manifest


def calibration_audit(world, algorithm, history):
    """Evaluator obtains truth after the predictor has returned its forecasts."""
    started = perf_counter()
    ids = common_query_ids(world.values)
    if not ids:
        return {'status': 'NO_ADMISSIBLE_PAIRS', 'queries': 0, 'seconds': perf_counter()-started}
    report = prediction_report(algorithm, history, ids)
    result = {'status': report.status, 'source': report.source, 'reason': report.reason,
              'scope': report.scope, 'asked_fraction': report.asked_fraction,
              'queries': len(ids), 'query_ids': ids}
    if report.probabilities is not None:
        true = np.array([response_probabilities(world, algorithm.pairs[q]) for q in ids])
        scores = expected_scores(report.probabilities, true)
        valid_binary = scores['conditional_binary_valid']
        result['conditional_binary_undefined_queries'] = int((~valid_binary).sum())
        result['metrics'], result['nonfinite_metric_counts'] = {}, {}
        for key, values in scores.items():
            if key in {'conditional_binary_valid', 'conditional_binary_predicted_left',
                       'conditional_binary_true_left', 'true_answer_probability'}:
                continue
            sample = values[valid_binary] if key.startswith('conditional_binary') else values
            bad = int((~np.isfinite(sample)).sum())
            result['nonfinite_metric_counts'][key] = bad
            result['metrics'][key] = float(np.mean(sample)) if len(sample) and not bad else None
    result['seconds'] = perf_counter()-started
    return result


def run(world, name, *, factory=create):
    started = perf_counter()
    history, trace = [], []
    algorithm = None
    identity = {"method": name, "seed": world.seed, "family": world.family,
                "n": len(world.values), "d": world.values.shape[1],
                "variant": world.description.get('variant', 'base')}
    try:
        algorithm, recommend = factory(name, world.values.copy())
        construction_seconds = perf_counter()-started
        if tuple(algorithm.pairs) != tradeoff_pairs(world.values):
            raise ValueError("adapter did not preserve the canonical observable query pool")
        for slot in range(4):
            step = perf_counter()
            selected, question = recommend(tuple(history))
            elapsed = perf_counter()-step
            if question is not None:
                if (isinstance(question, (bool, np.bool_)) or
                        not isinstance(question, (int, np.integer)) or
                        not 0 <= question < len(algorithm.pairs)):
                    raise ValueError("invalid query id")
                question = int(question)
                if slot == 3 or question in {q for q, _ in history}:
                    raise ValueError("repeated or over-budget query")
            metrics = evaluate(world, selected)
            row = {"slot": slot, "selection": list(map(int, selected)), "query_id": question,
                   "seconds": elapsed, **metrics}
            diagnostics = getattr(algorithm, 'last_diagnostics', None)
            if diagnostics:
                row['diagnostics'] = jsonable(diagnostics)
            trace.append(row)
            if slot == 3 or question is None:
                break
            pair = algorithm.pairs[question]
            choice, probability, skip = response(world, pair, slot)
            row.update(pair=list(pair), choice=choice, true_probability_left=probability,
                       true_skip_probability=skip)
            history.append((question, choice))
        final = trace[-1]
        decision_seconds = perf_counter()-started
        try:
            calibration = calibration_audit(world, algorithm, tuple(history))
        except Exception as exc:
            # Primary decision error and diagnostic prediction error are
            # separate; neither becomes a fabricated finite calibration score.
            calibration = {'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc),
                           'traceback': traceback.format_exc()}
        approximate = any(s.get('diagnostics', {}).get('fallback_from_milp') for s in trace)
        restricted = any(s.get('diagnostics', {}).get('terminal_search_complete') is False for s in trace)
        return {**identity, "status": "PASS", "constraint_status":
                "FAIL" if any(s['dominance_violations'] for s in trace) else "PASS",
                "construction_seconds": construction_seconds, "total_seconds": decision_seconds,
                "calibration": calibration,
                "solver_fallback": bool(approximate), "restricted_terminal_search": restricted,
                "slots": len(history),
                "answers": sum(r != 2 for _, r in history), "history": history, "trace": trace,
                "regret_by_slot": [trace[min(i, len(trace)-1)]['regret'] for i in range(4)],
                **{key: final[key] for key in ('regret', 'oracle_overlap', 'oracle_overlap_tie_aware',
                                               'dominance_violations')}}
    except Exception as exc:
        return {**identity, "status": "FAIL", "constraint_status": "NOT_COMPLETE",
                "error_type": type(exc).__name__, "error": str(exc), "trace": trace,
                "diagnostics": jsonable(getattr(algorithm, 'last_diagnostics', None)),
                "exception_diagnostics": jsonable(getattr(exc, 'diagnostics', None)),
                "traceback": traceback.format_exc(), "total_seconds": perf_counter()-started}


def summarize(rows, methods):
    result = {}
    for name in methods:
        all_rows = [r for r in rows if r['method'] == name]
        good = [r for r in all_rows if r['status'] == 'PASS']
        losses = [r['regret'] for r in good]
        result[name] = {"cases": len(all_rows), "failures": len(all_rows)-len(good),
                        "constraint_failures": sum(r['constraint_status'] == 'FAIL' for r in good),
                        "solver_fallback_cases": sum(r['solver_fallback'] for r in good),
                        "restricted_terminal_cases": sum(r.get('restricted_terminal_search', False) for r in good),
                        "calibration_failures": sum(r.get('calibration', {}).get('status') == 'FAIL' for r in good),
                        "mean_regret_successful_only": float(np.mean(losses)) if good else None,
                        "p95_regret_successful_only": float(np.quantile(losses, .95)) if good else None,
                        "mean_slots": float(np.mean([r['slots'] for r in good])) if good else None,
                        "step_p95_seconds": float(np.quantile([s['seconds'] for r in good for s in r['trace']], .95)) if good else None,
                        "construction_p95_seconds": float(np.quantile([r['construction_seconds'] for r in good], .95)) if good else None,
                        "total_seconds": sum(r['total_seconds'] for r in all_rows),
                        "family_mean": {f: float(np.mean([r['regret'] for r in good if r['family'] == f]))
                                        for f in sorted({r['family'] for r in good})}}
    return result


class SummaryAccumulator:
    """Keep scalar samples for exact quantiles, never retain large row traces."""
    def __init__(self, methods):
        self.data = {m: dict(cases=0, failures=0, constraint_failures=0,
                            solver_fallback_cases=0, restricted_terminal_cases=0,
                            calibration_failures=0, losses=[], slots=[], steps=[],
                            construction=[], total_seconds=0., family={}) for m in methods}

    def add(self, row):
        d = self.data[row['method']]
        d['cases'] += 1
        d['total_seconds'] += row['total_seconds']
        if row['status'] != 'PASS':
            d['failures'] += 1
            return
        d['constraint_failures'] += row['constraint_status'] == 'FAIL'
        d['solver_fallback_cases'] += row['solver_fallback']
        d['restricted_terminal_cases'] += row.get('restricted_terminal_search', False)
        d['calibration_failures'] += row.get('calibration', {}).get('status') == 'FAIL'
        d['losses'].append(row['regret'])
        d['slots'].append(row['slots'])
        d['steps'].extend(s['seconds'] for s in row['trace'])
        d['construction'].append(row['construction_seconds'])
        d['family'].setdefault(row['family'], []).append(row['regret'])

    def result(self):
        answer = {}
        for method, d in self.data.items():
            answer[method] = {key: d[key] for key in ('cases', 'failures', 'constraint_failures',
                'solver_fallback_cases', 'restricted_terminal_cases', 'calibration_failures', 'total_seconds')}
            answer[method].update(
                mean_regret_successful_only=float(np.mean(d['losses'])) if d['losses'] else None,
                p95_regret_successful_only=float(np.quantile(d['losses'], .95)) if d['losses'] else None,
                mean_slots=float(np.mean(d['slots'])) if d['slots'] else None,
                step_p95_seconds=float(np.quantile(d['steps'], .95)) if d['steps'] else None,
                construction_p95_seconds=float(np.quantile(d['construction'], .95)) if d['construction'] else None,
                family_mean={f: float(np.mean(v)) for f, v in sorted(d['family'].items())})
        return answer


def seal_protocol(path, protocol):
    write_json(path, protocol)
    digest = sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.sha256').write_text(digest+'  '+path.name+'\n')
    return digest


def read_sealed_protocol(path):
    """Accidental-mutation check, not an externally timestamped commitment."""
    raw = path.read_bytes()
    expected = path.with_suffix('.sha256').read_text().split()[0]
    if sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen protocol checksum mismatch')
    protocol = json.loads(raw)
    args = protocol['args']
    seeds = protocol['seeds']
    if (type(args['seed_start']) is not int or args['seed_start'] < 0
            or type(args['seeds']) is not int or args['seeds'] < 1
            or seeds != list(range(args['seed_start'], args['seed_start']+args['seeds']))
            or any(type(s) is not int for s in seeds)):
        raise ValueError('Nonempty, unique, contiguous nonnegative seed namespace required')
    if any(not args[k] or len(set(args[k])) != len(args[k]) for k in ('methods', 'families')):
        raise ValueError('Nonempty unique method/family labels required')
    if not 1 <= args['n'] <= 100 or not 1 <= args['d'] <= 3:
        raise ValueError('Invalid N/D')
    if Path(args['out']).resolve() != path.parent.resolve():
        raise ValueError('Protocol output directory mismatch')
    if type(protocol.get('execution_workers', 1)) is not int or protocol.get('execution_workers', 1) not in (1, 2):
        raise ValueError('Only one or two deterministic workers are supported')
    return protocol


def execute(protocol, *, shard_index=0):
    args = argparse.Namespace(**protocol['args'])
    args.out = Path(args.out)
    if jsonable(read_sealed_protocol(args.out/'protocol.json')) != jsonable(protocol):
        raise RuntimeError('In-memory protocol differs from its sealed file')
    workers = protocol.get('execution_workers', 1)
    if type(shard_index) is not int or not 0 <= shard_index < workers:
        raise ValueError('Invalid worker index')
    suffix = f'.part-{shard_index:03d}' if workers > 1 else ''
    torch.set_num_threads(protocol['torch_threads'])
    mismatch = [relative for relative, digest in protocol['source_sha256'].items()
                if not (WORKSPACE/relative).is_file() or sha256((WORKSPACE/relative).read_bytes()).hexdigest() != digest]
    if mismatch:
        raise RuntimeError(f'Frozen sources changed before first task: {mismatch[:3]}')
    runtime_methods = {name: description(name) for name in args.methods}
    if jsonable(runtime_methods) != jsonable(protocol['methods']):
        raise RuntimeError('Worker configurations/checkpoint hashes differ from declared protocol')
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': scipy.__version__, 'torch': torch.__version__}
    if any(protocol[key] != value for key, value in versions.items()):
        raise RuntimeError('Runtime dependency versions differ from the frozen protocol')
    write_json(args.out/f'worker-config-verification{suffix}.json', {'status': 'PASS', 'methods': runtime_methods,
               'source_hashes_verified': len(protocol['source_sha256']), 'versions': versions,
               'before_first_task': True, 'executed_workspace': str(WORKSPACE), 'shard_index': shard_index})
    accumulator = SummaryAccumulator(args.methods)
    count, failures = 0, 0
    with (args.out/f'rows{suffix}.jsonl').open('x') as stream:
        for seed in protocol['seeds'][shard_index::workers]:
            for family in args.families:
                world = make_world(seed, family, n=args.n, d=args.d, variant=args.variant)
                # Cyclic order, also preserved in the output trace order.
                offset = (seed-args.seed_start+args.families.index(family)) % len(args.methods)
                order = args.methods[offset:]+args.methods[:offset]
                for name in order:
                    row = run(world, name)
                    stream.write(json.dumps(jsonable(row), allow_nan=False)+'\n')
                    stream.flush()
                    accumulator.add(row)
                    count += 1
                    failures += row['status'] != 'PASS'
                print(json.dumps({"seed": seed, "family": family, "rows": count,
                                  "failures": failures}), flush=True)
    write_json(args.out/f'summary{suffix}.json', accumulator.result())
    changed = [rel for rel, digest in protocol['source_sha256'].items()
               if not (WORKSPACE/rel).is_file() or sha256((WORKSPACE/rel).read_bytes()).hexdigest() != digest]
    write_json(args.out/f'completion{suffix}.json', {"status": "COMPLETE", "rows": count,
               "snapshot_files_changed_while_running": changed,
               "executed_workspace": str(WORKSPACE),
               "shard_index": shard_index, "workers": workers,
               "snapshot_scope": "Worker imports the captured source/checkpoints, not the editable workspace"})
    print(json.dumps({"complete": True, "rows": count, "changed_source_count": len(changed)}))


def launch_frozen(path):
    """Launch disjoint seed shards, then validate and merge every row once."""
    protocol = read_sealed_protocol(path)
    out, workers = path.parent, protocol.get('execution_workers', 1)
    args = protocol['args']
    environment = dict(os.environ, PYTHONPATH=str(out/'snapshot'))
    children, logs = [], []
    try:
        for index in range(workers):
            log = (out/f'worker-{index:03d}.log').open('x')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, '-m', 'research.frontier_2026.benchmark',
                '--execute-protocol', str(path), '--shard-index', str(index)],
                cwd=out/'snapshot', env=environment, stdout=log, stderr=subprocess.STDOUT))
        codes = [child.wait() for child in children]
    finally:
        for log in logs:
            log.close()
    if any(codes):
        write_json(out/'launcher-failure.json', {'status': 'FAIL', 'returncodes': codes,
                   'previous_output_retained': True})
        raise RuntimeError('A worker failed; no partial result is presented as complete')
    if workers == 1:
        return
    accumulator = SummaryAccumulator(args['methods'])
    seen = set()
    seed_set, family_set, method_set = map(set, (protocol['seeds'], args['families'], args['methods']))
    for index in range(workers):
        completion = json.loads((out/f'completion.part-{index:03d}.json').read_text())
        if completion['status'] != 'COMPLETE' or completion['snapshot_files_changed_while_running']:
            raise RuntimeError('Worker completion/source-integrity failure')
        verification = json.loads((out/f'worker-config-verification.part-{index:03d}.json').read_text())
        if (verification['status'] != 'PASS' or verification['before_first_task'] is not True
                or verification['methods'] != protocol['methods']
                or verification['source_hashes_verified'] != len(protocol['source_sha256'])
                or verification['versions'] != {key: protocol[key] for key in ('python','numpy','scipy','torch')}
                or Path(verification['executed_workspace']).resolve() != out/'snapshot'):
            raise RuntimeError('Worker configuration verification mismatch')
    digest = sha256()
    with (out/'rows.jsonl').open('x') as target:
        for index in range(workers):
            shard_count = 0
            with (out/f'rows.part-{index:03d}.jsonl').open() as source:
                for line in source:
                    row = json.loads(line)
                    key = row['seed'], row['family'], row['method']
                    if (key in seen or key[0] not in seed_set or key[1] not in family_set
                            or key[2] not in method_set or (key[0]-args['seed_start']) % workers != index):
                        raise RuntimeError('Duplicate, extraneous or wrongly sharded result')
                    seen.add(key)
                    accumulator.add(row)
                    target.write(line)
                    digest.update(line.encode())
                    shard_count += 1
            expected = len(protocol['seeds'][index::workers])*len(family_set)*len(method_set)
            if shard_count != expected:
                raise RuntimeError('Missing shard rows')
    expected = len(seed_set)*len(family_set)*len(method_set)
    if len(seen) != expected:
        raise RuntimeError('Missing complete-study rows')
    write_json(out/'summary.json', accumulator.result())
    write_json(out/'worker-config-verification.json', {'status': 'PASS', 'workers': workers,
               'before_first_task': True, 'methods': protocol['methods'],
               'executed_workspace': str(out/'snapshot'),
               'source_hashes_verified': len(protocol['source_sha256']),
               'versions': {key: protocol[key] for key in ('python', 'numpy', 'scipy', 'torch')},
               'parts': [f'worker-config-verification.part-{i:03d}.json' for i in range(workers)]})
    write_json(out/'completion.json', {'status': 'COMPLETE', 'rows': len(seen), 'workers': workers,
               'snapshot_files_changed_while_running': [], 'rows_sha256': digest.hexdigest(),
               'executed_workspace': str(out/'snapshot'),
               'timing_scope': 'Two concurrent workers; separate frozen N100 audit supplies engineering gate'})
    print(json.dumps({'complete': True, 'rows': len(seen), 'workers': workers}), flush=True)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] in ('--execute-protocol', '--launch-frozen'):
        path = Path(sys.argv[2]).resolve()
        if HERE != path.parent/'snapshot/research/frontier_2026':
            raise SystemExit('Worker must execute from its own captured snapshot')
        if sys.argv[1] == '--launch-frozen':
            return launch_frozen(path)
        worker = argparse.ArgumentParser()
        worker.add_argument('--shard-index', type=int, default=0)
        return execute(read_sealed_protocol(path), shard_index=worker.parse_args(sys.argv[3:]).shard_index)
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('DEVELOPMENT', 'SELECTION_VALIDATION', 'STRESS'), default='DEVELOPMENT')
    parser.add_argument('--seed-start', type=int, required=True)
    parser.add_argument('--seeds', type=int, required=True)
    parser.add_argument('--n', type=int, default=18)
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--variant', choices=VARIANTS, default='base')
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.seeds < 1 or args.seed_start < 0 or len(set(args.methods)) != len(args.methods) or len(set(args.families)) != len(args.families):
        raise SystemExit('Positive cluster count and unique method/family names required')
    if not 1 <= args.n <= 100 or not 1 <= args.d <= 3:
        raise SystemExit('Invalid N/D')
    if args.out.exists():
        raise SystemExit('Refusing to overwrite any prior experiment')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    manifest = snapshot(args.out)
    protocol = {"stage": args.stage, "utc": datetime.now(timezone.utc).isoformat(),
                "args": vars(args), "methods": {name: description(name) for name in args.methods},
                "seeds": list(range(args.seed_start, args.seed_start+args.seeds)),
                "independent_unit": "seed cluster spanning all utility families and methods",
                "source_sha256": manifest, "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                "execution_workers": args.workers,
                "argv": sys.argv, "not_final_test": True,
                "limitations": "Synthetic known families; method selection allowed only as labeled; no human validation"}
    seal_protocol(args.out/'protocol.json', protocol)
    environment = dict(os.environ, PYTHONPATH=str(args.out/'snapshot'))
    result = subprocess.run([sys.executable, '-m', 'research.frontier_2026.benchmark',
                             '--launch-frozen', str(args.out/'protocol.json')],
                            cwd=args.out/'snapshot', env=environment)
    if result.returncode:
        write_json(args.out/'launcher-failure.json', {"exit_code": result.returncode,
                   "status": "FAIL", "previous_output_retained": True})
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

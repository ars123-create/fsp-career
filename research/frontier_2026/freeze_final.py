"""Create a reviewable heldout protocol and source snapshot, without test data.

Execution is a separate command from this freeze. Fresh fixed-N tasks may not
be opened or regenerated here. Candidate selection uses earlier validation.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import torch

from .benchmark import snapshot, seal_protocol, HERE
from .registry import METHODS, description
from .simulation import FAMILIES
from .statistics import balanced_losses, prospective_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidates', nargs='+', choices=METHODS, required=True)
    parser.add_argument('--seed-start', type=int, default=202610090000)
    parser.add_argument('--minimum-clusters', type=int, default=512)
    parser.add_argument('--maximum-clusters', type=int, default=4096)
    parser.add_argument('--engineering-audit', type=Path, required=True)
    parser.add_argument('--source-replay', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.minimum_clusters <= args.maximum_clusters or args.seed_start < 0:
        raise SystemExit('Require 2 <= minimum <= maximum and nonnegative seed start')
    if args.candidates != ['mixed_kg_finer_quadrature', 'P15_easy']:
        raise SystemExit('This study freezes exactly the two validation-selected finalists in declared order')
    validation = args.validation.resolve()
    completion = json.loads((validation/'completion.json').read_text())
    if completion['status'] != 'COMPLETE' or completion['snapshot_files_changed_while_running']:
        raise SystemExit('Selection validation must finish before freeze')
    source_protocol = json.loads((validation/'protocol.json').read_text())
    if source_protocol['stage'] != 'SELECTION_VALIDATION':
        raise SystemExit('Expected a separately labeled selection-validation run')
    if source_protocol['args']['families'] != list(FAMILIES):
        raise SystemExit('Final family weighting must match the declared complete validation set')
    if any(source_protocol['args'][k] != v for k,v in {'n':18,'d':3,'variant':'base'}.items()):
        raise SystemExit('Selection validation must use N18/D3/base')
    if (not source_protocol['seeds'] or len(set(source_protocol['seeds'])) != len(source_protocol['seeds'])
            or len(source_protocol['seeds']) != source_protocol['args']['seeds']):
        raise SystemExit('Invalid selection seed inventory')
    for relative, digest in source_protocol['source_sha256'].items():
        if sha256((validation/'snapshot'/relative).read_bytes()).hexdigest() != digest:
            raise SystemExit('Selection snapshot hash mismatch')
    baselines = ['linear_evoi_3', 'gp_topk_guard']
    if set(args.candidates)&set(baselines):
        raise SystemExit('Candidate may not be its own comparison control')
    methods = list(dict.fromkeys(args.candidates+baselines+['pabbo_rl', 'equal']))
    rows = [json.loads(line) for line in (validation/'rows.jsonl').read_text().splitlines()]
    cube = balanced_losses(rows, methods, list(FAMILIES), source_protocol['seeds'])
    clusters = cube.mean(axis=1)
    # Two-sided descriptive CIs and one-sided gates use the same predeclared
    # number of confirmatory hypotheses, including family and tail checks.
    multiplicity = len(args.candidates)*len(baselines)*(2+len(FAMILIES))+(len(args.candidates) == 2)
    power = {}
    for candidate in args.candidates:
        for baseline in baselines:
            difference = clusters[:, methods.index(candidate)]-clusters[:, methods.index(baseline)]
            power[f'{candidate}__minus__{baseline}'] = prospective_clusters(
                float(difference.std(ddof=1)), hypotheses=multiplicity,
                minimum=args.minimum_clusters, maximum=args.maximum_clusters)
    nclusters = max(p['chosen_clusters'] for p in power.values())
    seeds = list(range(args.seed_start, args.seed_start+nclusters))
    ledger_path = HERE/'SEED_LEDGER.json'
    ledger = json.loads(ledger_path.read_text())
    used = set(ledger['world_seed_values']) | set(source_protocol['seeds'])
    if set(seeds)&used or any(args.seed_start <= r['end_inclusive'] and seeds[-1] >= r['start']
                             for r in ledger['world_seed_ranges']):
        raise SystemExit('Final seed namespace overlaps known earlier world seeds')
    engineering_path = args.engineering_audit.resolve()
    engineering_summary = json.loads((engineering_path/'summary.json').read_text())
    engineering_completion = json.loads((engineering_path/'completion.json').read_text())
    engineering_protocol = json.loads((engineering_path/'protocol.json').read_text())
    if (engineering_completion['status'] != 'COMPLETE'
            or engineering_completion['snapshot_files_changed_while_running']
            or engineering_protocol['args']['n'] != 100):
        raise SystemExit('Independent completed N100 engineering audit required')
    candidate_engineering = {}
    for m in args.candidates:
        e = engineering_summary[m]
        candidate_engineering[m] = {'status': 'PASS' if not any(e[k] for k in (
            'failures','constraint_failures','solver_fallback_cases','restricted_terminal_cases')) else 'FAIL',
            'step_p95_seconds': e['step_p95_seconds'],
            'construction_p95_seconds': e['construction_p95_seconds'],
            'execution_failures': e['failures'], 'constraint_failures': e['constraint_failures'],
            'solver_fallback_cases': e['solver_fallback_cases'],
            'restricted_terminal_cases': e['restricted_terminal_cases']}
    # Bind the engineering result to exactly the finalist implementation files.
    for filename in ('finite.py','models.py','policies.py','constraints.py','registry.py'):
        relative = 'research/frontier_2026/'+filename
        if engineering_protocol['source_sha256'][relative] != sha256((HERE/filename).read_bytes()).hexdigest():
            raise SystemExit('Finalist implementation differs from N100 engineering audit')
    replay_path = args.source_replay.resolve()
    replay = json.loads((replay_path/'summary.json').read_text())
    replay_protocol = json.loads((replay_path/'run/protocol.json').read_text())
    if (replay['status'] != 'PASS' or replay['regret_differences'] != 0
            or not replay['coverage_complete']
            or replay['rows'] != len(source_protocol['seeds'])*len(FAMILIES)*len(args.candidates)
            or set(replay_protocol['args']['methods']) != set(args.candidates)
            or replay_protocol['seeds'] != source_protocol['seeds']
            or replay_protocol['args']['families'] != list(FAMILIES)):
        raise SystemExit('Finalist source replay must preserve validation losses')
    for filename in ('finite.py','models.py','policies.py','constraints.py','registry.py','simulation.py'):
        relative = 'research/frontier_2026/'+filename
        if replay_protocol['source_sha256'][relative] != sha256((HERE/filename).read_bytes()).hexdigest():
            raise SystemExit('Finalist implementation differs from validation source replay')
    out = args.out.resolve()
    if out.exists():
        raise SystemExit('Refusing to overwrite a frozen protocol or any results')
    out.mkdir(parents=True)
    manifest = snapshot(out)
    torch.set_num_threads(2)
    run_args = {'out': str(out), 'stage': 'FINAL_HELDOUT', 'seed_start': args.seed_start,
                'seeds': nclusters, 'n': 18, 'd': 3, 'variant': 'base',
                'families': list(FAMILIES), 'methods': methods}
    protocol = {'stage': 'FINAL_HELDOUT', 'utc': datetime.now(timezone.utc).isoformat(),
                'args': run_args, 'methods': {m: description(m) for m in methods},
                'seeds': seeds, 'source_sha256': manifest, 'python': platform.python_version(),
                'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
                'torch_threads': 2, 'execution_workers': 2, 'not_final_test': False, 'data_generated_during_freeze': False,
                'independent_unit': 'seed cluster, all 14 families together with equal weights',
                'selection_source': str(validation),
                'selection_files_sha256': {name: sha256((validation/name).read_bytes()).hexdigest()
                                          for name in ('protocol.json', 'rows.jsonl', 'summary.json')},
                'candidates': args.candidates, 'confirmatory_baselines': baselines,
                'descriptive_controls': ['pabbo_rl', 'equal'], 'power': power,
                'seed_ledger_sha256': sha256(ledger_path.read_bytes()).hexdigest(),
                'seed_scope': ledger.get('scope', 'Known local world seeds; external pretraining inventory unavailable'),
                'source_replay': {'directory': str(replay_path),
                    'summary_sha256': sha256((replay_path/'summary.json').read_bytes()).hexdigest(),
                    'run_protocol_sha256': sha256((replay_path/'run/protocol.json').read_bytes()).hexdigest()},
                'engineering_audit': {'status': 'PASS', 'n': 100, 'source_directory': str(engineering_path),
                    'source_files_sha256': {name: sha256((engineering_path/name).read_bytes()).hexdigest()
                        for name in ('protocol.json','summary.json','completion.json','rows.jsonl')},
                    'candidates': candidate_engineering, 'step_budget_seconds': .25,
                    'speed_order': sorted(args.candidates, key=lambda m: (candidate_engineering[m]['step_p95_seconds'],m)),
                    'scope': 'Local p95 on 32 N100 synthetic tasks, Torch intraop threads=2; not a total CPU cap or production SLA'},
                'statistical_plan': {
                    'primary': 'normalized cardinal top5 regret after at most 3 slots',
                    'family_weighting': '1/14 each', 'alpha': .05, 'multiplicity': multiplicity,
                    'mean_intervals': 'paired seed-cluster Student t with Bonferroni; approximate coverage',
                    'tail_interval': 'paired seed-cluster basic bootstrap, Bonferroni upper bound; approximate coverage',
                    'cvar_alpha': .95, 'bootstrap_resamples': 49999, 'bootstrap_seed': 202610084005,
                    'candidate_eligible': 'no execution, hard-constraint or silent fallback failures',
                    'invalid_input_policy': 'Any confirmatory candidate or baseline failure, fallback, missing or duplicate row invalidates the entire confirmatory analysis; descriptive controls retain explicitly incomplete summaries',
                    'practical_replacement_gates': {'mean_difference_upper_below': -.001,
                        'each_family_difference_upper_below': .005, 'cvar_difference_upper_below': .005},
                    'gate_scope': 'every gate against BOTH specified baselines for each candidate',
                    'tail_power': 'NOT_ESTABLISHED; mean normal-power calculation does not imply tail power',
                    'candidate_contrast': {'enabled': len(args.candidates) == 2,
                        'ordered_pair': args.candidates, 'interval': 'two-sided paired cluster t with Bonferroni',
                        'practical_margin': .001},
                    'winner_rule': {'zero_pass': 'Practical replacement advantage NOT_ESTABLISHED; retain gp_topk_guard as research reference, application unchanged',
                        'fallback_reference': 'gp_topk_guard',
                        'one_pass': 'Recommend eligible candidate within synthetic benchmark scope',
                        'two_pass': 'If candidate contrast CI lies beyond +/- .001 select practically superior; otherwise use prefrozen N100 speed_order as engineering choice, not quality equivalence',
                        'engineering_required': 'N100 candidate status PASS and step_p95_seconds < .25'},
                    'no_optional_stopping': True, 'no_failed_case_deletion': True,
                    'descriptive_controls_not_in_confirmatory_claim_family': True},
                'latency_scope': 'Per-step local timings, Torch intraop threads=2, two final workers; separate N100 engineering audit. Other applications not stopped.',
                'hardware_provenance': json.loads((HERE/'evidence/hardware-provenance-v1.json').read_text()),
                'limitations': 'Finite tested alternatives and known synthetic generator. No universal or human hiring optimum.',
                'freeze_argv': sys.argv}
    digest = seal_protocol(out/'protocol.json', protocol)
    print(json.dumps({'status': 'FROZEN_NOT_RUN', 'clusters': nclusters,
                      'tasks': nclusters*len(FAMILIES), 'methods': len(methods),
                      'confirmatory_hypotheses': multiplicity, 'protocol_sha256': digest}))


if __name__ == '__main__':
    main()

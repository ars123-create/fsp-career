"""Strict descriptive analysis of sealed supplementary runs; no simulator import.

Reuse review, 2026-10-09: read PLAN_RU §§6/7/9, REQUIREMENTS, supplementary
runner/protocol, and audited statistics.py. Its empirical_cvar and
cluster_mean_interval were run separately on literal data before adaptation.
NumPy/SciPy remain existing BSD dependencies with preserved project notices;
no third-party algorithm source is copied here. CIs are exploratory, unadjusted
paired seed-cluster t intervals, never a confirmatory or human-quality claim.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import re

import numpy as np
import scipy

from ..statistics import cluster_mean_interval, empirical_cvar


HERE = Path(__file__).resolve().parent
ANALYZER_REL = 'research/frontier_2026/extensions/analyze_supplement.py'
STATISTICS_REL = 'research/frontier_2026/statistics.py'
CONTROL_NAMES = ('mixed_kg_finer_quadrature', 'nominal_kg_m2_tiny')
METRICS = ('sum_regret', 'best_of_slate_regret', 'ordinal_regret')
STAGES = {'DEVELOPMENT', 'SELECTION_VALIDATION', 'HARNESS_SMOKE'}
REQUIRED_VERSIONS = {'python','numpy','scipy','torch','botorch','gpytorch',
                     'platform','machine','python_abi'}


def digest(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(text):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON literal: {value}')
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid)


def _number(value, label, *, lower=0., upper=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < lower
            or (upper is not None and value > upper)):
        raise ValueError(f'Invalid {label}: {value!r}')
    return float(value)


def _integer(value, label, lower, upper=None):
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f'Invalid {label}: {value!r}')
    return value


def _labels(values, label):
    if (not isinstance(values, list) or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'Invalid unique {label}')
    return values


def plan_arms(plan):
    """Structural validation without importing method constructors or worlds."""
    if plan.get('stage') not in STAGES or type(plan.get('workers')) is not int or plan['workers'] not in (1,2):
        raise ValueError('Invalid supplementary stage/workers')
    if not isinstance(plan.get('arms'), list) or not plan['arms']:
        raise ValueError('Nonempty plan arms required')
    names, seeds_used, arms = set(), set(), {}
    for arm in plan['arms']:
        name = arm['name']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Distinct nonempty arm names required')
        names.add(name)
        _integer(arm['n'], 'n', 1, 100)
        _integer(arm['d'], 'd', 1, 3)
        _labels(arm['families'], 'families')
        _labels(arm['methods'], 'methods')
        seeds = arm['seeds']
        if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
                or seeds_used.intersection(seeds)):
            raise ValueError('Distinct nonoverlapping seed roots required')
        for seed in seeds:
            _integer(seed, 'seed', 0)
        seeds_used.update(seeds)
        if not isinstance(arm.get('variant','base'),str) or not arm.get('variant','base'):
            raise ValueError('Invalid variant')
        arms[name] = arm
    return arms


def expected_keys(arms):
    return {(a['name'], s, f, m) for a in arms.values()
            for s, f, m in product(a['seeds'], a['families'], a['methods'])}


def row_key(row):
    return row['arm'], row['seed'], row['family'], row['method']


def losses(row):
    return {'sum_regret': _number(row['regret'],'regret',upper=1.),
            'best_of_slate_regret': _number(row['auxiliary']['best_of_slate_regret'], 'best regret',upper=1.),
            'ordinal_regret': 1.-_number(row['oracle_overlap_tie_aware'],'tie-aware overlap',upper=1.)}


def validate_row(row, arm):
    """Reject malformed success records; retain explicit execution failures."""
    if (row['n'] != arm['n'] or type(row['n']) is not int
            or row['d'] != arm['d'] or type(row['d']) is not int
            or row['variant'] != arm.get('variant','base')):
        raise ValueError('World setting mismatch')
    if row.get('status') not in {'PASS','FAIL'}:
        raise ValueError('Unknown row execution status')
    _number(row['total_seconds'], 'total_seconds')
    trace = row.get('trace')
    if not isinstance(trace,list) or len(trace) > 4:
        raise ValueError('Invalid response trace')
    violations, questions = [], []
    for index, step in enumerate(trace):
        if step['slot'] != index or type(step['slot']) is not int:
            raise ValueError('Nonsequential trace slots')
        _number(step['seconds'], 'step seconds')
        selection = step['selection']
        if (not isinstance(selection,list) or len(selection) != min(5,arm['n'])
                or len(set(selection)) != len(selection)):
            raise ValueError('Invalid slate cardinality')
        for candidate in selection:
            _integer(candidate,'selected candidate',0,arm['n']-1)
        violations.append(_integer(step['dominance_violations'],'dominance violations',0))
        q = step['query_id']
        if q is not None:
            _integer(q,'query',0)
            if index == 3 or q in questions:
                raise ValueError('Repeated or over-budget query')
            questions.append(q)
    if row['status'] == 'FAIL':
        if row.get('constraint_status') != 'NOT_COMPLETE':
            raise ValueError('Failed row needs NOT_COMPLETE constraint status')
        return
    if not trace:
        raise ValueError('Success requires a nonempty trace')
    for flag in ('solver_fallback','restricted_terminal_search'):
        if type(row.get(flag)) is not bool:
            raise ValueError(f'Missing explicit boolean flag: {flag}')
    expected_constraint = 'FAIL' if any(violations) else 'PASS'
    if row.get('constraint_status') != expected_constraint:
        raise ValueError('Constraint flag disagrees with trace')
    if any(s.get('diagnostics',{}).get('fallback_from_milp',False) for s in trace) and not row['solver_fallback']:
        raise ValueError('Solver fallback hidden by row flag')
    if any(s.get('diagnostics',{}).get('terminal_search_complete') is False for s in trace) and not row['restricted_terminal_search']:
        raise ValueError('Restricted terminal search hidden by row flag')
    slots = _integer(row['slots'],'slots',0,3)
    if (len(trace) != slots+1 or len(questions) != slots or trace[-1]['query_id'] is not None
            or len(row.get('history',())) != slots):
        raise ValueError('Incomplete or inconsistent completed history')
    for index, (query, outcome) in enumerate(row['history']):
        _integer(query,'history query',0)
        _integer(outcome,'history outcome',0,2)
        if query != questions[index] or trace[index].get('choice') != outcome:
            raise ValueError('History disagrees with trace')
    if type(row.get('answers')) is not int or row['answers'] != sum(r != 2 for _,r in row['history']):
        raise ValueError('Answer count disagrees with history')
    _number(row['construction_seconds'],'construction seconds')
    measured, final = losses(row), losses(trace[-1])
    if any(abs(measured[m]-final[m]) > 1e-12 for m in METRICS):
        raise ValueError('Final metrics disagree with final trace')
    for step in trace:
        losses(step)


@dataclass(frozen=True)
class VerifiedRun:
    directory: Path
    protocol: dict
    completion: dict
    rows: tuple
    hashes: dict


def _verify_workers(directory, protocol, arms, expected, merged_digest):
    """Replay captured benchmark.merge provenance checks, without execution.

    Worker bytes concatenated in worker order must equal the sealed merged
    bytes. Each shard independently has exact planned coverage, valid settings,
    source/runtime/config preflight, failure counts and its own rows digest.
    """
    workers = protocol['plan']['workers']
    sources = len(protocol['source_sha256'])
    all_seen, combined_digest, hashes = set(), sha256(), {}
    for worker in range(workers):
        pre_name, completion_name, rows_name = (f'preflight-{worker:03d}.json',
                f'completion-{worker:03d}.json',f'rows-{worker:03d}.jsonl')
        for name in (pre_name,completion_name,rows_name):
            if not (directory/name).is_file():
                raise ValueError(f'Missing worker provenance file: {name}')
        metadata = {}
        for name in (pre_name,completion_name):
            raw = (directory/name).read_bytes()
            hashes[name] = sha256(raw).hexdigest()
            metadata[name] = _json(raw)
        preflight, completion = metadata[pre_name],metadata[completion_name]
        if (preflight.get('status') != 'PASS' or completion.get('status') != 'COMPLETE'
                or preflight.get('methods') != protocol['methods']
                or preflight.get('versions') != protocol['versions']
                or preflight.get('workspace') != str(directory/'snapshot')
                or type(preflight.get('worlds_generated')) is not int or preflight['worlds_generated'] != 0
                or type(preflight.get('source_files_verified')) is not int or preflight['source_files_verified'] != sources
                or type(completion.get('source_files_verified')) is not int or completion['source_files_verified'] != sources):
            raise ValueError(f'Worker preflight/completion mismatch: {worker}')
        worker_expected = {(a['name'],s,f,m) for a in arms.values()
                for s,f,m in product(a['seeds'][worker::workers],a['families'],a['methods'])}
        worker_seen, worker_digest, failures = set(),sha256(),0
        with (directory/rows_name).open('rb') as stream:
            for raw in stream:
                worker_digest.update(raw)
                combined_digest.update(raw)
                row = _json(raw)
                key = row_key(row)
                if key not in worker_expected or key in all_seen or type(row['seed']) is not int:
                    raise ValueError(f'Wrong worker shard, duplicate or unexpected worker row: {key}')
                validate_row(row,arms[row['arm']])
                worker_seen.add(key)
                all_seen.add(key)
                failures += row['status'] != 'PASS'
        if worker_seen != worker_expected:
            raise ValueError(f'Missing worker shard rows: {worker}')
        hashes[rows_name] = worker_digest.hexdigest()
        if (type(completion.get('rows')) is not int or completion['rows'] != len(worker_seen)
                or type(completion.get('failures')) is not int or completion['failures'] != failures
                or completion.get('rows_sha256') != hashes[rows_name]):
            raise ValueError(f'Worker row/failure count or digest mismatch: {worker}')
    if all_seen != expected or combined_digest.hexdigest() != merged_digest:
        raise ValueError('Worker shards contradict merged rows')
    return hashes


def load_run(directory):
    """Stream and seal-check one completed run, never dropping a case."""
    directory = Path(directory).resolve()
    paths = {name:directory/name for name in ('protocol.json','completion.json','rows.jsonl')}
    hashes = {name:digest(path) for name,path in paths.items() if name != 'rows.jsonl'}
    seal = (directory/'protocol.sha256').read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}',seal) or seal != hashes['protocol.json']:
        raise ValueError('Protocol seal mismatch')
    p, completion = (_json(paths[name].read_text()) for name in ('protocol.json','completion.json'))
    if p.get('schema') != 'fsp-supplement-v1':
        raise ValueError('Unsupported supplementary schema')
    arms = plan_arms(p['plan'])
    expected = expected_keys(arms)
    methods = {m for a in arms.values() for m in a['methods']}
    if p.get('expected_rows') != len(expected) or type(p.get('expected_rows')) is not int or set(p.get('methods',{})) != methods:
        raise ValueError('Protocol expected coverage/method metadata mismatch')
    if (completion.get('status') != 'COMPLETE' or completion.get('rows') != len(expected)
            or type(completion.get('rows')) is not int or completion.get('scope') != p['plan']['stage']):
        raise ValueError('Run is not complete with the planned coverage')
    if p.get('primary_holdout_quality_opened') is not False or completion.get('quality_of_real_hiring') != 'NOT_RUN':
        raise ValueError('Scope flags changed or missing')
    versions = p.get('versions')
    if (not isinstance(versions,dict) or not REQUIRED_VERSIONS <= versions.keys()
            or any(not isinstance(versions[k],str) or not versions[k] for k in REQUIRED_VERSIONS)):
        raise ValueError('Complete captured runtime versions required')
    runtime = {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    if any(versions.get(k) != v for k,v in runtime.items()):
        raise ValueError('Analysis runtime differs from captured numerical dependencies')
    manifest = p.get('source_sha256',{})
    if not isinstance(manifest,dict) or not manifest:
        raise ValueError('Nonempty source manifest required')
    snapshot = (directory/'snapshot').resolve()
    for relative, expected_hash in manifest.items():
        source = (snapshot/relative).resolve()
        if not source.is_relative_to(snapshot) or Path(relative).is_absolute():
            raise ValueError('Unsafe source-manifest path')
        if not re.fullmatch('[0-9a-f]{64}',expected_hash) or digest(source) != expected_hash:
            raise ValueError(f'Snapshot source drift: {relative}')
    for relative, current in ((ANALYZER_REL,Path(__file__)),(STATISTICS_REL,HERE.parent/'statistics.py')):
        if manifest.get(relative) != digest(current):
            raise ValueError(f'Analysis implementation not bound to frozen source: {relative}')
    seen, rows, row_digest = set(), [], sha256()
    with paths['rows.jsonl'].open('rb') as stream:
        for raw in stream:
            row_digest.update(raw)
            row = _json(raw)
            key = row_key(row)
            if key not in expected or key in seen or type(row['seed']) is not int:
                raise ValueError(f'Unexpected or duplicate row: {key}')
            validate_row(row,arms[row['arm']])
            seen.add(key)
            rows.append(row)
    if seen != expected:
        raise ValueError(f'Missing planned rows: {len(expected-seen)}')
    hashes['rows.jsonl'] = row_digest.hexdigest()
    if completion.get('rows_sha256') != hashes['rows.jsonl']:
        raise ValueError('Rows SHA256 mismatch')
    hashes.update(_verify_workers(directory,p,arms,expected,hashes['rows.jsonl']))
    for name in hashes:
        if digest(directory/name) != hashes[name]:
            raise ValueError(f'Input changed while reading: {name}')
    return VerifiedRun(directory,p,completion,tuple(rows),hashes)


def _stats(sample):
    x = np.asarray(sample,float)
    return {'count':len(x),'mean':float(x.mean()) if len(x) else None,
            'cvar95':float(empirical_cvar(x,.95)) if len(x) else None}


def _quantile(sample, q=.95):
    return float(np.quantile(sample,q)) if len(sample) else None


def bad_reasons(row):
    reasons = []
    if row.get('status') != 'PASS':
        reasons.append('execution_failure')
    if row.get('constraint_status') != 'PASS':
        reasons.append('constraint_failure')
    for flag in ('solver_fallback','restricted_terminal_search'):
        if row.get(flag) is True:
            reasons.append(flag)
        elif row.get(flag) is not False:
            reasons.append(flag+'_unverified')
    return reasons


def describe(rows, arm):
    expected_count = len(arm['seeds'])*len(arm['families'])
    if len(rows) != expected_count:
        raise ValueError('Descriptive group has incomplete coverage')
    good = [r for r in rows if r['status'] == 'PASS']
    scope = 'ALL_PLANNED_CASES' if len(good) == expected_count else 'SUCCESSFUL_SUBSET_ONLY'
    flags = Counter(reason for row in rows for reason in bad_reasons(row))
    values = {metric:[losses(r)[metric] for r in good] for metric in METRICS}
    family_means = {}
    for family in arm['families']:
        group = [r for r in good if r['family'] == family]
        family_means[family] = {'planned_cases':len(arm['seeds']),'successful_cases':len(group),
                               'scope':'ALL_PLANNED_CASES' if len(group)==len(arm['seeds']) else 'SUCCESSFUL_SUBSET_ONLY',
                               'means':{m:float(np.mean([losses(r)[m] for r in group])) if group else None for m in METRICS}}
    return {'planned_cases':expected_count,'successful_cases':len(good),'failed_cases':len(rows)-len(good),
            'flag_counts':dict(flags),'operational_eligible_on_this_arm':not flags,
            'metric_scope':scope,'metrics':{m:_stats(values[m]) for m in METRICS},
            'family_means':family_means,
            'slots':{'scope':scope,'mean':float(np.mean([r['slots'] for r in good])) if good else None,
                     'p95':_quantile([r['slots'] for r in good])},
            'timings':{'scope':scope,'p95_definition':'NumPy linear empirical quantile over individual recommend calls; construction excluded',
                       'step_p95_seconds_successful_cases':_quantile([s['seconds'] for r in good for s in r['trace']]),
                       'step_p95_seconds_all_observed_calls':_quantile([s['seconds'] for r in rows for s in r['trace']]),
                       'construction_p95_seconds_successful_cases':_quantile([r['construction_seconds'] for r in good]),
                       'total_seconds_all_cases':float(sum(r['total_seconds'] for r in rows))},
            'calibration_status_counts':dict(Counter(r.get('calibration',{}).get('status','NOT_RECORDED') for r in rows))}


def paired_comparison(rows, arm, candidate, control):
    selected = {row_key(r):r for r in rows if r['method'] in (candidate,control)}
    keys = {(arm['name'],s,f,m) for s,f,m in product(arm['seeds'],arm['families'],(candidate,control))}
    result = {'candidate':candidate,'control':control,'scope':'EXPLORATORY_UNADJUSTED_PAIRED_SEED_CLUSTER',
              'multiplicity_adjusted':False,'confidence_level':.95,'confirmatory':False}
    if set(selected) != keys or len(selected) != len([r for r in rows if r['method'] in (candidate,control)]):
        return {**result,'status':'NOT_AVAILABLE','reason':'missing_or_duplicate_case'}
    if any(r['status'] != 'PASS' or r['constraint_status'] != 'PASS' for r in selected.values()):
        return {**result,'status':'NOT_AVAILABLE','reason':'execution_or_constraint_failure; no successful-subset CI'}
    result['operational_flags'] = dict(Counter(reason for r in selected.values() for reason in bad_reasons(r)))
    if len(arm['seeds']) < 2:
        return {**result,'status':'NOT_AVAILABLE','reason':'fewer_than_two_seed_clusters'}
    estimates = {}
    for metric in METRICS:
        difference = [np.mean([losses(selected[(arm['name'],s,f,candidate)])[metric]
                              -losses(selected[(arm['name'],s,f,control)])[metric]
                              for f in arm['families']]) for s in arm['seeds']]
        estimates[metric] = cluster_mean_interval(np.asarray(difference),alpha=.05,hypotheses=1)
    return {**result,'status':'AVAILABLE','metrics':estimates,
            'interpretation':'Negative mean favors candidate on this endpoint; unadjusted exploratory intervals cannot choose a confirmatory winner'}


def summarize(verified):
    arms = plan_arms(verified.protocol['plan'])
    result = {'schema':'fsp-supplement-analysis-v1','scope':'EXPLORATORY_DESCRIPTIVE_ONLY',
              'overall_winner':None,'confirmatory_decision':None,'human_quality':'NOT_RUN',
              'run_directory':str(verified.directory),'input_sha256':verified.hashes,
              'source_sha256':{ANALYZER_REL:digest(__file__),STATISTICS_REL:digest(HERE.parent/'statistics.py')},
              'source_files_verified':len(verified.protocol['source_sha256']),
              'captured_versions':verified.protocol['versions'],
              'worker_provenance_verified':verified.protocol['plan']['workers'],
              'rows_verified':len(verified.rows),'stage':verified.protocol['plan']['stage'],
              'created_utc':datetime.now(timezone.utc).isoformat(),'arms':{}}
    for name,arm in arms.items():
        rows = [r for r in verified.rows if r['arm']==name]
        controls = [c for c in CONTROL_NAMES if c in arm['methods']]
        result['arms'][name] = {'n':arm['n'],'d':arm['d'],'variant':arm.get('variant','base'),
                               'seed_clusters':len(arm['seeds']),'families':arm['families'],
                               'methods':{m:describe([r for r in rows if r['method']==m],arm) for m in arm['methods']},
                               'comparisons':[paired_comparison(rows,arm,m,c) for c in controls for m in arm['methods'] if m!=c]}
    return result


def validation_trigger(rows, plan):
    """Pure prespecified point-estimate screen; NOT a confidence statement.

    Caller must verify seals and equality of frozen configs across dev/validation.
    Other research arms never erase an operational candidate's failed case.
    Every prospective method is scheduled for validation regardless of dev score.
    """
    arms = plan_arms(plan)
    follow = plan['followup_rule']
    gates = follow['machine_gates']
    prospective = _labels(gates['prospective_methods'],'prospective methods')
    finalists = _labels(gates['finalists'],'finalists')
    if len(finalists) != 2 or prospective != follow['always_validate_full_n18_new_methods']:
        raise ValueError('Always-validate rule and prospective order disagree')
    if (gates['multiple_passers'] != 'ALL_PASSERS_IN_PROSPECTIVE_ORDER'
            or gates['no_post_development_tuning'] is not True or gates['point_estimate_screen_only'] is not True):
        raise ValueError('Trigger cannot change selection/confirmation semantics')
    if set(gates['bad_flags']) != {'execution_failure','constraint_failure','solver_fallback',
                                  'restricted_terminal_search','missing_or_duplicate_case'}:
        raise ValueError('Eligibility flags changed')
    families = _labels(follow['selection_validation_families'],'validation families')
    controls = _labels(follow['selection_validation_controls'],'validation controls')
    if not set(finalists) <= set(controls):
        raise ValueError('Finalists absent from validation controls')
    start = _integer(gates['validation_seed_start'],'validation seed start',0)
    end = _integer(gates['validation_seed_end_inclusive'],'validation seed end',start)
    if follow['selection_validation_seeds'] != {'start':start,'end_inclusive':end}:
        raise ValueError('Validation seed specifications disagree')
    setting = follow['selection_validation_setting']
    if any(setting[key] != gates['validation_'+key] for key in ('n','d','variant')):
        raise ValueError('Validation setting specifications disagree')
    validation = {'name':gates['validation_arm'],'seeds':list(range(start,end+1)),
                  'families':families,'methods':prospective+controls,
                  'n':gates['validation_n'],'d':gates['validation_d'],'variant':gates['validation_variant']}
    if len(set(validation['methods'])) != len(validation['methods']):
        raise ValueError('Prospective and control methods must be distinct')
    scope_arms = [arms[gates['development_quality_arm']],arms[gates['development_latency_arm']],validation]
    if scope_arms[0]['n'] != 18 or scope_arms[1]['n'] != 100 or validation['n'] != 18 or validation['d'] != 3 or validation['variant'] != 'base':
        raise ValueError('Trigger settings differ from N18/N100 contract')
    for arm in scope_arms[:2]:
        if not set(prospective+finalists) <= set(arm['methods']):
            raise ValueError('Development scope lacks required candidate/finalists')
    numeric = {key:_number(gates[key],key) for key in ('mean_improvement_margin','maximum_family_degradation',
                'maximum_cvar95_degradation','cvar_alpha','latency_step_p95_strict_bound_seconds')}
    if not 0 < numeric['cvar_alpha'] < 1:
        raise ValueError('Invalid trigger tail fraction')
    table, duplicates = {}, set()
    wanted_arms = {a['name']:a for a in scope_arms}
    unexpected = []
    for row in rows:
        if row['arm'] not in wanted_arms:
            continue
        arm = wanted_arms[row['arm']]
        key = row_key(row)
        if (row['seed'] not in arm['seeds'] or row['family'] not in arm['families']
                or row['method'] not in arm['methods']):
            unexpected.append(key)
            continue
        if key in table:
            duplicates.add(key)
        else:
            table[key] = row
    def inspect(arm, method):
        reasons, found = Counter(), []
        for s,f in product(arm['seeds'],arm['families']):
            key = (arm['name'],s,f,method)
            if key in duplicates or key not in table:
                reasons['missing_or_duplicate_case'] += 1
                continue
            row = table[key]
            try:
                validate_row(row,arm)
            except (KeyError,ValueError,TypeError):
                reasons['malformed_case'] += 1
                continue
            reasons.update(bad_reasons(row))
            found.append(row)
        return reasons,found
    answer = {'scope':'SELECTION_POINT_ESTIMATES_ONLY','confirmatory':False,
              'always_validate_methods':prospective,'validation_expected_cases_per_method':len(validation['seeds'])*len(families),
              'prespecified_machine_gates':gates,
              'future_confirmation_references':follow.get('future_confirmation_references',[]),
              'same_source_configs_and_checkpoints':'CALLER_MUST_VERIFY_SEALS; not inferable from numeric rows',
              'overall_winner':None,'candidates':{},'triggered_methods':[],'unexpected_relevant_rows':unexpected}
    reference_results = {m:inspect(validation,m) for m in finalists}
    for method in prospective:
        reasons, candidate_rows = Counter(), {}
        for arm in scope_arms:
            flags, found = inspect(arm,method)
            reasons.update({f"{arm['name']}:{k}":v for k,v in flags.items()})
            candidate_rows[arm['name']] = found
        for reference,(flags,_) in reference_results.items():
            reasons.update({f'reference:{reference}:{k}':v for k,v in flags.items()})
        if unexpected:
            reasons['unexpected_relevant_rows'] = len(unexpected)
        detail = {'eligible':not reasons,'eligibility_reasons':dict(reasons),'trigger':False,'comparisons':{}}
        if not reasons:
            latency = candidate_rows[gates['development_latency_arm']]
            p95 = _quantile([s['seconds'] for r in latency for s in r['trace']])
            speed_pass = p95 is not None and p95 < numeric['latency_step_p95_strict_bound_seconds']
            detail.update(n100_step_p95_seconds=p95,latency_pass=bool(speed_pass))
            candidate = { (r['seed'],r['family']):r['regret'] for r in candidate_rows[validation['name']] }
            ca = np.array([[candidate[(s,f)] for f in families] for s in validation['seeds']])
            gates_pass = speed_pass
            for reference,(_,refrows) in reference_results.items():
                lookup = {(r['seed'],r['family']):r['regret'] for r in refrows}
                cb = np.array([[lookup[(s,f)] for f in families] for s in validation['seeds']])
                mean = float(np.mean(np.mean(ca-cb,axis=1)))
                family_differences = (ca-cb).mean(axis=0)
                tail = float(empirical_cvar(ca.ravel(),numeric['cvar_alpha'])-empirical_cvar(cb.ravel(),numeric['cvar_alpha']))
                comparison = {'mean_difference':mean,'mean_pass':mean <= -numeric['mean_improvement_margin'],
                              'family_mean_differences':dict(zip(families,map(float,family_differences))),
                              'worst_family_difference':float(max(family_differences)),
                              'family_pass':bool(max(family_differences) <= numeric['maximum_family_degradation']),
                              'pooled_fractional_cvar_difference':tail,'cvar_pass':tail <= numeric['maximum_cvar95_degradation']}
                detail['comparisons'][reference] = comparison
                gates_pass = gates_pass and comparison['mean_pass'] and comparison['family_pass'] and comparison['cvar_pass']
            detail['trigger'] = bool(gates_pass)
            if gates_pass:
                answer['triggered_methods'].append(method)
        answer['candidates'][method] = detail
    answer['result'] = 'SEPARATE_CONFIRMATION_REQUIRED' if answer['triggered_methods'] else 'NO_ADDITIONAL_CONFIRMATION_TRIGGER'
    return answer


def _markdown(report):
    lines = ['# Дополнительное исследовательское сравнение', '',
             'Описательные результаты; интервалы 95% без поправки на множественные сравнения. '
             'Общий победитель не определяется. Отказы остаются в eligibility.', '']
    def show(x):
        return '—' if x is None else f'{x:.6f}'
    for name,arm in report['arms'].items():
        lines += [f'## {name}', '', '| Метод | PASS / все | SUM mean / CVaR95 | BEST mean / CVaR95 | ORD mean / CVaR95 | p95 шага, s | Scope |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for method,entry in arm['methods'].items():
            fields = [' / '.join(show(entry['metrics'][metric][stat]) for stat in ('mean','cvar95')) for metric in METRICS]
            lines.append(f"| {method} | {entry['successful_cases']} / {entry['planned_cases']} | "+' | '.join(fields)
                         +f" | {show(entry['timings']['step_p95_seconds_successful_cases'])} | {entry['metric_scope']} |")
        lines += ['', 'Полные family means, flags, slots и paired seed-cluster intervals сохранены в analysis.json.', '']
    return '\n'.join(lines)+'\n'


def analyze_run(directory, *, out=None, development=None):
    directory = Path(directory).resolve()
    out = Path(out).resolve() if out is not None else directory/'analysis_supplement_v1'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite analysis: {out}')
    verified = load_run(directory)
    report = summarize(verified)
    if development is not None:
        dev = load_run(development)
        if dev.protocol['plan']['stage'] != 'DEVELOPMENT' or verified.protocol['plan']['stage'] != 'SELECTION_VALIDATION':
            raise ValueError('Combined trigger requires development and selection-validation runs')
        if dev.protocol['source_sha256'] != verified.protocol['source_sha256']:
            raise ValueError('Source/config drift between development and validation')
        if dev.protocol['versions'] != verified.protocol['versions']:
            raise ValueError('Complete runtime versions drift between development and validation')
        if any(dev.protocol['methods'].get(m) != desc for m,desc in verified.protocol['methods'].items()):
            raise ValueError('Method config drift between development and validation')
        follow = dev.protocol['plan']['followup_rule']
        g = follow['machine_gates']
        expected_validation = {'name':g['validation_arm'],
                'seeds':list(range(g['validation_seed_start'],g['validation_seed_end_inclusive']+1)),
                'families':follow['selection_validation_families'],
                'methods':g['prospective_methods']+follow['selection_validation_controls'],
                'n':g['validation_n'],'d':g['validation_d'],'variant':g['validation_variant']}
        if verified.protocol['plan']['arms'] != [expected_validation]:
            raise ValueError('Validation plan differs from always-include prespecified coverage')
        report['selection_trigger'] = validation_trigger(dev.rows+verified.rows,dev.protocol['plan'])
        report['selection_trigger']['same_source_configs_and_checkpoints'] = 'VERIFIED_EQUAL_SEALED_MANIFESTS_AND_METHOD_DESCRIPTIONS'
        report['selection_trigger']['same_complete_runtime_versions'] = True
        report['development_input_sha256'] = dev.hashes
    out.mkdir(parents=True,exist_ok=False)
    with (out/'analysis.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    with (out/'REPORT_RU.md').open('x') as stream:
        stream.write(_markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--development',type=Path,help='Optional sealed dev run for a selection-validation point screen')
    args = parser.parse_args()
    result = analyze_run(args.run,out=args.out,development=args.development)
    print(json.dumps({'status':'ANALYZED_EXPLORATORY','rows':result['rows_verified'],
                      'overall_winner':None,'run':str(args.run)}))


if __name__ == '__main__':
    main()

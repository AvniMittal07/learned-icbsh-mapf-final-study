"""Audit calibration, report censoring, and conditionally fit a fixed linear baseline.

Comparisons concern descendant-CBS expansion effort, not end-to-end solve time.
Two unfinished searches are never assigned a synthetic preference.
"""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np
from mapf.day3_protocol import ROOT, digest
from mapf.day3_calibration import restore
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.cardinal import classify_conflicts
from mapf.baselines import _compute_candidate_lookahead_tuple


CENSORED = {'TIME_CAP', 'EXPANSION_CAP'}


def preference(a, b):
    """+1 means a takes provably fewer pops; -1 means b; None is unknown/tied."""
    if a['status'] == b['status'] == 'SOLVED':
        if a['expanded'] == b['expanded']:
            return None
        return 1 if a['expanded'] < b['expanded'] else -1
    if a['status'] == 'SOLVED' and b['status'] in CENSORED and a['expanded'] < b['expanded']:
        return 1
    if b['status'] == 'SOLVED' and a['status'] in CENSORED and b['expanded'] < a['expanded']:
        return -1
    return None


def pairs(record, budget='high'):
    found = []
    for i, j in itertools.combinations(range(len(record['candidates'])), 2):
        a, b = (record['candidates'][k]['budgets'][budget] for k in (i, j))
        order = preference(a, b)
        if order is not None:
            found.append((i, j, order, 'fully_solved' if a['status'] == b['status'] == 'SOLVED' else 'solved_vs_censored'))
    return found


def decoded_key(candidate):
    return tuple(float(v) for v in candidate['analytical_key'])


def check_protocol(protocol):
    payload = dict(protocol)
    saved = payload.pop('protocol_sha256')
    if digest(payload) != saved:
        raise ValueError('Frozen protocol hash mismatch')
    families = protocol['families']
    ids = set(); inputs = {}; instances = {}
    for family in families:
        if family['family_id'] in ids:
            raise ValueError('Duplicate family')
        ids.add(family['family_id'])
        for instance in family['instances']:
            if instance['family_id'] != family['family_id'] or instance['split'] != family['split']:
                raise ValueError('Family split leakage')
            data = dict(instance); fingerprint = data.pop('instance_sha256')
            if digest(data) != fingerprint:
                raise ValueError('Instance hash mismatch')
            key = digest([instance['map_sha256'], instance['starts'], instance['goals']])
            if key in inputs or instance['id'] in instances:
                raise ValueError('Duplicate physical instance or identifier')
            inputs[key] = instance['split']; instances[instance['id']] = instance
    for instance in protocol['calibration_instances']:
        if instance['split'] == 'test' or instance != instances[instance['id']]:
            raise ValueError('Calibration selection leaked or was modified')
    return dict(families=len(families), instances=len(instances),
                families_by_split=dict(Counter(f['split'] for f in families)),
                instances_by_split=dict(Counter(i['split'] for i in instances.values())),
                cross_split_instance_overlap=0)


def quality_gate(records, thresholds, replay_mismatches=0):
    useful = [r for r in records if pairs(r)]
    metrics = {}
    for split in ('train', 'validation'):
        rows = [r for r in useful if r['instance']['split'] == split]
        metrics[f'{split}_informative_queries'] = len(rows)
        metrics[f'{split}_families'] = len({r['instance']['family_id'] for r in rows})
    metrics['maps_with_informative_queries'] = len({r['instance']['map_name'] for r in useful})
    metrics['informative_fraction'] = len(useful) / len(records) if records else 0
    metrics['replay_mismatches'] = replay_mismatches
    failed = []
    for key, minimum in thresholds.items():
        if key.startswith('min_') and metrics[key[4:]] < minimum:
            failed.append(dict(metric=key[4:], observed=metrics[key[4:]], required_minimum=minimum))
        elif key == 'replay_mismatches_allowed' and replay_mismatches > minimum:
            failed.append(dict(metric='replay_mismatches', observed=replay_mismatches, allowed=minimum))
    return dict(passed=not failed, metrics=metrics, failed_checks=failed, thresholds=thresholds)


def distribution(values):
    if not values:
        return None
    return dict(mean=float(np.mean(values)), median=float(np.median(values)), p95=float(np.percentile(values, 95)), total=sum(values))


def budget_summary(records, name):
    results = [c['budgets'][name] for r in records for c in r['candidates']]
    observed = [p for r in records for p in pairs(r, name)]
    by_split = {}
    by_map = {}
    for field, target in [('split', by_split), ('map_name', by_map)]:
        for value in sorted({r['instance'][field] for r in records}):
            subset = [r for r in records if r['instance'][field] == value]
            informative = [r for r in subset if pairs(r, name)]
            target[value] = dict(queries=len(subset), informative_queries=len(informative),
                informative_families=len({r['instance']['family_id'] for r in informative}),
                certified_pairs=sum(len(pairs(r, name)) for r in subset))
    return dict(statuses=dict(Counter(r['status'] for r in results)), candidate_rollouts=len(results),
        certified_pairs=len(observed), pair_types=dict(Counter(p[3] for p in observed)),
        informative_queries=sum(bool(pairs(r, name)) for r in records),
        fully_solved_queries=sum(all(c['budgets'][name]['status'] == 'SOLVED' for c in r['candidates']) for r in records),
        rollout_cpu_seconds=distribution([r['cpu_seconds'] for r in results]),
        rollout_wall_seconds=distribution([r['wall_seconds'] for r in results]),
        by_split=by_split, by_map=by_map)


def audit(records, protocol):
    split_audit = check_protocol(protocol)
    originals = [json.loads(s) for s in (ROOT/'states.jsonl').read_text().splitlines()]
    original_by_id = {r['query_id']: r for r in originals}
    if len(original_by_id) != len(originals) or len({r['query_id'] for r in records}) != len(records):
        raise ValueError('Duplicate query IDs')
    if {r['query_id'] for r in records} != set(original_by_id):
        raise ValueError('Calibration incomplete or unexpected queries')
    if len({r['state_sha256'] for r in records}) != len(records):
        raise ValueError('Duplicate states')
    h = hashlib.sha256()
    with zipfile.ZipFile(ROOT/'calibration_source.zip') as archive:
        for name in sorted(archive.namelist()):
            h.update(Path(name).name.encode()); h.update(archive.read(name))
    archived_digest = h.hexdigest()
    if archived_digest != json.loads((ROOT/'source.json').read_text())['sha256']:
        raise ValueError('Source archive hash mismatch')
    calibrated_instances = {i['id']: i for i in protocol['calibration_instances']}
    guard_pairs = guard_queries = guard_sampled_candidates = eligible_candidates = 0
    for record in records:
        if record['instance'] != calibrated_instances[record['instance']['id']]:
            raise ValueError('Instance outside frozen calibration plan')
        if record['protocol_sha256'] != protocol['protocol_sha256'] or record['source_sha256'] != archived_digest:
            raise ValueError('Mixed protocol/source revisions')
        if record['feature_names'] != FEATURE_NAMES or not record['features_replayed']:
            raise ValueError('Feature schema or initial replay missing')
        original = original_by_id[record['query_id']]
        for key in original:
            if key == 'candidates':
                stripped = [{k:v for k,v in c.items() if k != 'budgets'} for c in record[key]]
                if stripped != original[key]: raise ValueError('Candidate changed after harvest')
            elif record[key] != original[key]:
                raise ValueError(f'Harvest field changed: {key}')
        grid, starts, goals, node, ht, graph, chosen = restore(record)
        context = dict(paths=node.paths, goals=goals, G_CF=graph, mvc_size=graph.exact_mvc_size(),
                       pair_counts=node.pair_counts, grid=grid)
        features = FeatureExtractor(grid, goals).extract_all(chosen, context)
        if not np.isfinite(features).all(): raise ValueError('Non-finite features')
        np.testing.assert_allclose(features, [c['features'] for c in record['candidates']], rtol=1e-6, atol=1e-6)
        all_cc = classify_conflicts(grid, starts, goals, node.paths, node.constraints, h_tables=ht)
        tiers = {'CARDINAL':0, 'SEMI_CARDINAL':1, 'NON_CARDINAL':2}
        best = min(tiers[c.classification] for c in all_cc)
        eligible = [c for c in all_cc if tiers[c.classification] == best]
        if any(tiers[c.classification] != best for c in chosen):
            raise ValueError('Candidate outside allowed cardinality tier')
        keys = [_compute_candidate_lookahead_tuple(c, context) for c in eligible]
        top = max(k[:2] for k in keys)
        tied = [(c, k) for c,k in zip(eligible, keys) if k[:2] == top]
        tied.sort(key=lambda ck: (ck[1][4], ck[1][5]), reverse=True)
        shortlist = {c.conflict for c,_ in tied[:3]} if len(tied) > 1 else set()
        sampled_shortlist = {i for i,c in enumerate(chosen) if c.conflict in shortlist}
        gp = sum(i in sampled_shortlist and j in sampled_shortlist for i,j,_,_ in pairs(record))
        guard_pairs += gp; guard_queries += int(gp > 0)
        guard_sampled_candidates += len(sampled_shortlist); eligible_candidates += len(eligible)
        for c, cc in zip(record['candidates'], chosen):
            if decoded_key(c) != _compute_candidate_lookahead_tuple(cc, context):
                raise ValueError('Analytical key mismatch')
            for budget in protocol['rollout_budgets']:
                result = c['budgets'][budget['name']]
                if result['expansion_cap'] != budget['cap'] or result['wall_cap_seconds'] != budget['seconds']:
                    raise ValueError('Unequal or modified rollout budget')
                if result['status'] not in {'SOLVED','TIME_CAP','EXPANSION_CAP','INFEASIBLE','HORIZON_LIMIT'}:
                    raise ValueError('Unknown rollout status')
                if not 1 <= result['expanded'] <= budget['cap']:
                    raise ValueError('Invalid expansion count')
    return dict(splits=split_audit, replayed_queries=len(records), replay_mismatches=0,
                archived_source_sha256=archived_digest, source_archive_verified=True,
                guard_diagnostic=dict(certified_pairs_within_actual_guard_shortlist=guard_pairs,
                    queries_with_such_pairs=guard_queries, sampled_candidates_within_shortlist=guard_sampled_candidates,
                    all_eligible_candidates=eligible_candidates,
                    caveat='Diagnostic computed after labels; frozen quality gate uses all sampled tier-eligible pairs'))


def stability(records):
    output = {}
    for smaller, larger in [('low','medium'), ('medium','high'), ('low','high')]:
        comparable = mismatched = lost_solved = known_pairs = reversals = 0
        for record in records:
            for c in record['candidates']:
                a,b = c['budgets'][smaller], c['budgets'][larger]
                if a['status'] == b['status'] == 'SOLVED':
                    comparable += 1
                    mismatched += int((a['expanded'],a['solution_cost']) != (b['expanded'],b['solution_cost']))
                lost_solved += int(a['status'] == 'SOLVED' and b['status'] != 'SOLVED')
            a = {(i,j):v for i,j,v,_ in pairs(record, smaller)}
            b = {(i,j):v for i,j,v,_ in pairs(record, larger)}
            for key in a.keys() & b.keys():
                known_pairs += 1; reversals += int(a[key] != b[key])
        output[f'{smaller}_to_{larger}'] = dict(comparable_solved_candidates=comparable,
            deterministic_mismatches=mismatched, lost_solved_candidates=lost_solved,
            comparable_certified_pairs=known_pairs, reversed_preferences=reversals)
    # All forced splits cover the same constrained problem; solved costs must agree.
    cost_mismatches = []
    for r in records:
        costs = {v['solution_cost'] for c in r['candidates'] for v in c['budgets'].values() if v['status'] == 'SOLVED'}
        if len(costs) > 1: cost_mismatches.append(r['query_id'])
    return dict(comparisons=output, conflicting_solution_cost_queries=cost_mismatches,
                interpretation='Deterministic consistency check only; does not validate global-search oracle accuracy')


def evaluate(records, score_function):
    rows = []; by_family = defaultdict(list)
    for r in records:
        observed = pairs(r)
        if not observed: continue
        scores = score_function(r)
        wins = []
        for i,j,order,_ in observed:
            predicted = 0 if scores[i] == scores[j] else (1 if scores[i] > scores[j] else -1)
            wins.append(.5 if predicted == 0 else float(predicted == order))
        accuracy = statistics.mean(wins)
        rows.append(dict(query_id=r['query_id'], family_id=r['instance']['family_id'],
                         certified_pairs=len(wins), pair_accuracy=accuracy))
        by_family[r['instance']['family_id']].append(accuracy)
    return dict(queries=len(rows), families=len(by_family), certified_pairs=sum(r['certified_pairs'] for r in rows),
                query_macro_accuracy=statistics.mean(r['pair_accuracy'] for r in rows) if rows else None,
                family_macro_accuracy=statistics.mean(statistics.mean(v) for v in by_family.values()) if rows else None,
                per_query=rows, caveat='Only observable sampled pairs; censored selection bias remains; ties receive half credit')


def train_if_allowed(records, gate, protocol, output_root=None):
    if not gate['passed']:
        return dict(trained=False, reason='Frozen label-quality gate failed; no model fit or model selection performed')
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    train = [r for r in records if r['instance']['split'] == 'train']
    scaler = StandardScaler().fit(np.array([c['features'] for r in train for c in r['candidates']]))
    X=[]; y=[]; weights=[]
    for r in train:
        observed = pairs(r)
        z = scaler.transform(np.array([c['features'] for c in r['candidates']]))
        for i,j,order,_ in observed:
            X.append(z[i]-z[j]); y.append(order); weights.append(1/len(observed))
    model = Ridge(alpha=10, fit_intercept=False).fit(np.array(X), np.array(y), sample_weight=np.array(weights))
    payload = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=model.coef_.tolist(), intercept=0.)
    directory = ROOT if output_root is None else Path(output_root)
    path = directory/'linear_pairwise.json'
    if path.exists(): raise FileExistsError('Do not overwrite a fitted model')
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    metadata = dict(feature_names=FEATURE_NAMES, feature_schema='recovery-v1-contextual-counts', backend='sklearn_ridge',
                    protocol_sha256=protocol['protocol_sha256'], source_sha256=records[0]['source_sha256'],
                    calibration_sha256=hashlib.sha256((directory/'calibration.jsonl').read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), alpha=10, fitted_split='train',
                    training_family_ids=sorted({r['instance']['family_id'] for r in train}),
                    training_query_ids=[r['query_id'] for r in train if pairs(r)],
                    target='higher score means fewer descendant-search expansions; per-query pair weights',
                    reference_policy=protocol.get('reference','descendant-CBS'),
                    validation_used_for_fit=False, test_used=False)
    Path(str(path)+'.metadata.json').write_text(json.dumps(metadata, indent=2))
    evaluations = {}
    for split in ('train', 'validation'):
        subset = [r for r in records if r['instance']['split'] == split]
        evaluations[split] = evaluate(subset, lambda r: model.predict(scaler.transform(np.array([c['features'] for c in r['candidates']]))))
    return dict(trained=True, model_path=str(path), training_pairs=len(y), evaluations=evaluations,
                note='Experimental offline baseline, not a validated speed or memory improvement')


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    records = [json.loads(s) for s in (ROOT/'calibration.jsonl').read_text().splitlines()]
    integrity = audit(records, protocol)
    consistency = stability(records)
    if consistency['conflicting_solution_cost_queries'] or any(
        c['deterministic_mismatches'] or c['reversed_preferences'] for c in consistency['comparisons'].values()):
        raise ValueError('Rollout consistency failed; inspect raw results before using labels')
    gate = quality_gate(records, protocol['quality_gate'])
    summary = dict(protocol_sha256=protocol['protocol_sha256'], integrity=integrity, consistency=consistency,
        queries=len(records), candidate_count=sum(len(r['candidates']) for r in records),
        sampled_candidate_pairs=sum(len(r['candidates'])*(len(r['candidates'])-1)//2 for r in records),
        budgets={b['name']:budget_summary(records,b['name']) for b in protocol['rollout_budgets']},
        quality_gate=gate, test_evaluated=False)
    summary['analytical_offline_diagnostics'] = {}
    for split in ('train','validation'):
        subset = [r for r in records if r['instance']['split'] == split]
        summary['analytical_offline_diagnostics'][split] = dict(
            residual=evaluate(subset,lambda r:[decoded_key(c) for c in r['candidates']]),
            earliest=evaluate(subset,lambda r:[-c['conflict']['time'] for c in r['candidates']]))
    summary['linear_baseline'] = train_if_allowed(records, gate, protocol)
    with (ROOT/'observed_pairs.jsonl').open('w') as stream:
        for record in records:
            for budget in protocol['rollout_budgets']:
                for i,j,order,evidence in pairs(record,budget['name']):
                    row=dict(query_id=record['query_id'], instance_id=record['instance']['id'],
                        family_id=record['instance']['family_id'], split=record['instance']['split'],
                        budget=budget['name'], candidate_a=i, candidate_b=j, preference_a_over_b=order,
                        evidence=evidence, target='fewer descendant-CBS expansions',
                        source_sha256=record['source_sha256'], protocol_sha256=protocol['protocol_sha256'])
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
    (ROOT/'quality.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k != 'analytical_offline_diagnostics'}, indent=2))


if __name__ == '__main__':
    main()

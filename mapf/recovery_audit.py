"""Reproducible, read-only audit of historical datasets and benchmark provenance."""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def audit():
    data_path = Path('data/cl_oracle_training.csv')
    rows = list(csv.DictReader(data_path.open(encoding='utf-8')))
    groups = defaultdict(list)
    for row in rows:
        groups[row['query_id']].append(row)
    issues = Counter()
    for qid, group in groups.items():
        if len({(r['map_name'], r['k'], r['scenario_id']) for r in group}) != 1:
            issues['query_spans_instances'] += 1
        if len({r['candidate_index'] for r in group}) != len(group):
            issues['duplicate_candidate_ids'] += 1
        tuples = [ast.literal_eval(r['y_lex_tuple']) for r in group]
        denom = max(1, len(group)-1)
        for r, item in zip(group, tuples):
            rank = sum(x < item for x in tuples) + (sum(x == item for x in tuples)-1)/2
            expected = rank/denom
            if abs(float(r['y_lex_rank_norm'])-expected) > 0.00011:
                issues['lex_rank_mismatch_rows'] += 1
            if abs(float(r['relevance'])-2*expected) > 0.00011:
                issues['relevance_mismatch_rows'] += 1
        if len(set(r['relevance'] for r in group)) < 2:
            issues['tied_relevance_queries'] += 1
    missing = [k for k in ['state_hash', 'paths', 'constraints', 'map_sha256',
                           'scenario_sha256', 'rollout_status', 'label_budget', 'source_commit']
               if k not in rows[0]]
    bench = list(csv.DictReader(open('results/phase3_benchmark.csv')))
    instance_key = lambda r: (r['map_name'], int(r['k']), int(r['scenario_id']))
    train_instances = {instance_key(r) for r in rows}
    test_instances = {instance_key(r) for r in bench if r['split']=='test'}
    family_train = {(r['map_name'], int(r['scenario_id'])) for r in rows}
    family_test = {(r['map_name'], int(r['scenario_id'])) for r in bench if r['split']=='test'}
    output = dict(dataset_rows=len(rows), decision_queries=len(groups),
                  source_revision='8f26ac31c372a8c4e5a6426fed4e74da997b9676',
                  sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  aliases_identical=data_path.read_bytes()==Path('data/mapf_ct_benchmark_dataset.csv').read_bytes(),
                  maps=dict(Counter(r['map_name'] for r in rows)), issues=dict(issues),
                  missing_replay_or_censoring_fields=missing,
                  benchmark_test_instances=len(test_instances),
                  overlap_train_and_benchmark_test=sorted(train_instances & test_instances),
                  family_overlap_count=len(family_train & family_test),
                  scenario_files_available=len(list(Path('data').rglob('*.scen'))),
                  disposition='Historical diagnostic data only; exact labels and held-out claims are not certified')
    output['nonfinite_numeric_cells'] = sum(
        1 for r in rows for k,v in r.items()
        if k not in ['query_id','map_name','priority_tier','y_lex_tuple'] and v
        and not math.isfinite(float(v)))
    destination = Path('results/recovery')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'dataset_audit.json').write_text(json.dumps(output, indent=2))
    # Candidate grouping only. These are NOT certified train/test splits without
    # source instance hashes; explicitly mark them unapproved.
    families = sorted(family_train)
    proposed = [{'map_name':m, 'scenario_id':s, 'group':f'{m}/scenario-{s}',
                 'proposed_partition':['train','train','train','validation','test'][
                     int(hashlib.sha256(f'{m}/{s}'.encode()).hexdigest(),16)%5]}
                for m,s in families]
    (destination/'split_proposal.json').write_text(json.dumps(
        dict(status='UNAPPROVED_REQUIRES_SCENARIO_HASHES', groups=proposed), indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    audit()

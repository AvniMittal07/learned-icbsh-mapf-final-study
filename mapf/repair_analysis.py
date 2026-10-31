"""Audit the guarded pilot, conditionally train, and validate without test access."""
from collections import Counter, defaultdict
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import zipfile
import numpy as np
from mapf.repair_protocol import ROOT, read_protocol
from mapf.day3_protocol import digest
from mapf.day3_analysis import check_protocol, quality_gate, pairs, budget_summary, stability, evaluate, decoded_key, train_if_allowed
from mapf.repair_pilot import replay
from mapf.features import FEATURE_NAMES
from mapf.validation import validate_solution, validate_constrained_paths


def records():
    return [json.loads(s) for s in (ROOT/'calibration.jsonl').read_text().splitlines()]


def audit(rows, protocol):
    parent=json.loads(Path('results/recovery/day3/protocol.json').read_text())
    check_protocol(parent)
    if parent['protocol_sha256']!=protocol['parent_split_protocol_sha256'] or protocol['quality_gate']!=parent['quality_gate']:
        raise ValueError('Changed split or relaxed gate')
    all_instances={i['id']:i for f in parent['families'] for i in f['instances']}
    old_families={i['family_id'] for i in parent['calibration_instances']}
    selected={i['id']:i for i in protocol['calibration_instances']}
    for i in selected.values():
        if i!=all_instances[i['id']] or i['split']=='test' or i['family_id'] in old_families:
            raise ValueError('Old calibration or test family reused')
    source=json.loads((ROOT/'source.json').read_text())['sha256'];h=hashlib.sha256()
    with zipfile.ZipFile(ROOT/'source.zip') as z:
        for name in sorted(z.namelist()):h.update(Path(name).name.encode());h.update(z.read(name))
    if h.hexdigest()!=source:raise ValueError('Source archive mismatch')
    originals=[json.loads(s) for s in (ROOT/'states.jsonl').read_text().splitlines()]
    original={r['query_id']:r for r in originals}
    if len(original)!=len(originals) or {r['query_id'] for r in rows}!=set(original) or len(rows)!=len(original):
        raise ValueError('Missing or duplicated query')
    if len({r['state_sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate saved state')
    solution_count=0; cbs_comparisons=[]
    for r in rows:
        if r['instance']!=selected[r['instance']['id']] or r['source_sha256']!=source or r['protocol_sha256']!=protocol['protocol_sha256']:
            raise ValueError('Mixed instance, source, or protocol')
        for key,value in original[r['query_id']].items():
            if key=='candidates':
                value2=[{k:v for k,v in c.items() if k not in ('budgets','cbs_comparison')} for c in r[key]]
                if value2!=value:raise ValueError('Candidate changed after collection')
            elif r[key]!=value:raise ValueError('State field changed after collection')
        if not r['features_replayed'] or not r['guard_replayed']:raise ValueError('Missing initial replay')
        grid,starts,goals,node,ht,graph,chosen=replay(r,protocol)
        for candidate in r['candidates']:
            for budget in protocol['rollout_budgets']:
                result=candidate['budgets'][budget['name']]
                if result['expansion_cap']!=budget['cap'] or result['wall_cap_seconds']!=budget['seconds']:
                    raise ValueError('Budget mismatch')
                if not 0<=result['expanded']<=budget['cap']:raise ValueError('Invalid expansion count')
                if result['status'] not in ('SOLVED','TIME_CAP','EXPANSION_CAP','INFEASIBLE','HORIZON_LIMIT'):
                    raise ValueError('Unknown status')
                if result['status']=='SOLVED':
                    if not result['forced_split_applied']:raise ValueError('Action not applied')
                    paths=[[tuple(p) for p in path] for path in result['solution_paths']]
                    validate_solution(grid,starts,goals,paths,result['solution_cost'])
                    validate_constrained_paths(grid,starts,goals,paths,node.constraints);solution_count+=1
            if 'cbs_comparison' in candidate:
                if r['instance']['split']!='train' or not r['instance']['family_id'].endswith('family-04') or not r['query_id'].endswith('guard-q0'):
                    raise ValueError('Unplanned CBS comparison')
                a,b=candidate['budgets']['high'],candidate['cbs_comparison']
                if a['status']==b['status']=='SOLVED' and a['solution_cost']!=b['solution_cost']:
                    raise ValueError('CBS and ICBS continuation disagree on optimal cost')
                cbs_comparisons.append(dict(query_id=r['query_id'],icbs_status=a['status'],cbs_status=b['status'],
                    icbs_expanded=a['expanded'],cbs_expanded=b['expanded'],icbs_cpu=a['cpu_seconds'],cbs_cpu=b['cpu_seconds']))
    return dict(queries_replayed=len(rows),guard_replay_mismatches=0,feature_replay_mismatches=0,
                solution_paths_independently_revalidated=solution_count,source_archive_verified=True,
                source_sha256=source,fresh_family_selection=True,quality_thresholds_unchanged=True,
                cbs_same_state_comparisons=cbs_comparisons,test_evaluated=False)


def analyze():
    protocol=read_protocol();rows=records();integrity=audit(rows,protocol);consistency=stability(rows)
    if consistency['conflicting_solution_cost_queries'] or any(c['deterministic_mismatches'] or c['reversed_preferences']
            for c in consistency['comparisons'].values()):raise ValueError('Deterministic consistency failed')
    summary=dict(protocol_sha256=protocol['protocol_sha256'],queries=len(rows),
        families=len({r['instance']['family_id'] for r in rows}),
        candidate_count=sum(len(r['candidates']) for r in rows),
        sampled_candidate_pairs=sum(len(r['candidates'])*(len(r['candidates'])-1)//2 for r in rows),
        integrity=integrity,consistency=consistency,
        budgets={b['name']:budget_summary(rows,b['name']) for b in protocol['rollout_budgets']},
        quality_gate=quality_gate(rows,protocol['quality_gate']),test_evaluated=False)
    summary['analytical_offline_diagnostics']={}
    for split in ('train','validation'):
        subset=[r for r in rows if r['instance']['split']==split]
        summary['analytical_offline_diagnostics'][split]=dict(
            residual=evaluate(subset,lambda r:[decoded_key(c) for c in r['candidates']]),
            earliest=evaluate(subset,lambda r:[-c['conflict']['time'] for c in r['candidates']]))
    with (ROOT/'observed_pairs.jsonl').open('w') as stream:
        for r in rows:
            for b in protocol['rollout_budgets']:
                for i,j,order,evidence in pairs(r,b['name']):
                    row=dict(query_id=r['query_id'],family_id=r['instance']['family_id'],split=r['instance']['split'],
                        candidate_a=i,candidate_b=j,preference_a_over_b=order,evidence=evidence,budget=b['name'],
                        target='fewer residual-ICBS-h descendant expansions',protocol_sha256=protocol['protocol_sha256'])
                    stream.write(json.dumps(row)+'\n')
    (ROOT/'quality.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='analytical_offline_diagnostics'},indent=2))


def train():
    protocol=read_protocol();rows=records()
    quality=json.loads((ROOT/'quality.json').read_text())
    if quality['protocol_sha256']!=protocol['protocol_sha256']:raise ValueError('Quality report protocol mismatch')
    # Reverify raw records before any fit, including the gate; never trust an
    # edited summary's boolean to authorize training on invalid data.
    audit(rows,protocol);gate=quality_gate(rows,protocol['quality_gate'])
    report=train_if_allowed(rows,gate,protocol,output_root=ROOT)
    report['gate']=gate;report['test_evaluated']=False
    (ROOT/'model_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


def validate():
    from mapf.recovery_benchmark import run_isolated
    protocol=read_protocol();report=json.loads((ROOT/'model_report.json').read_text())
    if not report['trained']:
        print('No model trained; model validation not run');return
    config=protocol['validation'];selected=[i for i in protocol['calibration_instances'] if i['split']=='validation']
    jobs=[(i,r,p) for i in selected for r in range(config['repetitions']) for p in config['policies']]
    random.Random(config['seed']).shuffle(jobs)
    manifest=dict(protocol_sha256=protocol['protocol_sha256'],purpose='validation only; no final-test claims',
        instances=selected,config=config,model_paths=dict(linear=str(ROOT/'linear_pairwise.json')),
        jobs=[dict(instance_id=i['id'],repeat=r,policy=p) for i,r,p in jobs])
    (ROOT/'validation_manifest.json').write_text(json.dumps(manifest,indent=2))
    with (ROOT/'validation_runs.jsonl').open('x') as stream:
        for instance,repeat,policy in jobs:
            spec=dict(instance=instance,policy=policy,seed=config['seed'],timeout_seconds=config['timeout_seconds'],
                      model_paths=manifest['model_paths'])
            row=run_isolated(spec)
            row.update(instance_id=instance['id'],family_id=instance['family_id'],policy=policy,repeat=repeat,
                       protocol_sha256=protocol['protocol_sha256'],split='validation')
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            print('validation',instance['id'],repeat,policy,row['status'],flush=True)


def summarize_validation():
    protocol=read_protocol();config=protocol['validation'];cap=config['timeout_seconds']
    rows=[json.loads(s) for s in (ROOT/'validation_runs.jsonl').read_text().splitlines()]
    ids={i['id'] for i in protocol['calibration_instances'] if i['split']=='validation'}
    expected=set(itertools.product(ids,range(config['repetitions']),config['policies']))
    keyed={(r['instance_id'],r['repeat'],r['policy']):r for r in rows}
    if set(keyed)!=expected or len(keyed)!=len(rows):raise ValueError('Validation incomplete or duplicated')
    if any(r['split']!='validation' or r['protocol_sha256']!=protocol['protocol_sha256'] for r in rows):
        raise ValueError('Unexpected validation provenance')
    errors=[r for r in rows if r['status'] not in ('SOLVED','TIMEOUT','HARD_TIMEOUT','MEMORY_LIMIT','MAX_EXPANSIONS')]
    if errors:raise ValueError(f'Validation errors require investigation: {errors}')
    for identifier in ids:
        solved=[r for r in rows if r['instance_id']==identifier and r['status']=='SOLVED']
        if len({r['cost'] for r in solved})>1 or any(not r.get('solution_validated') for r in solved):
            raise ValueError('Validation solution cost/path check failed')
    def charged(r):return min(r['end_to_end_wall_seconds'],cap) if r['status']=='SOLVED' else cap
    policies={}
    for policy in config['policies']:
        subset=[r for r in rows if r['policy']==policy]
        solved=[r for r in subset if r['status']=='SOLVED']
        policies[policy]=dict(runs=len(subset),solved=len(solved),statuses=dict(Counter(r['status'] for r in subset)),
            capped_mean_wall_seconds=statistics.mean(charged(r) for r in subset),
            solved_median_wall_seconds=statistics.median(r['end_to_end_wall_seconds'] for r in solved) if solved else None,
            median_peak_rss_bytes=statistics.median(r['peak_rss_bytes'] for r in subset),
            ml_calls_total=sum(r.get('ml_tiebreaks',0) for r in subset),
            analytical_shortcircuits_total=sum(r.get('analytical_shortcircuits',0) for r in subset))
    paired=[];family_rows=[]
    for identifier in sorted(ids):
        a=[keyed[(identifier,r,'linear')] for r in range(config['repetitions'])]
        b=[keyed[(identifier,r,'residual')] for r in range(config['repetitions'])]
        for x,y in zip(a,b):
            if x['status']==y['status']=='SOLVED':paired.append((x,y))
        family_rows.append(dict(instance_id=identifier,linear_mean_capped_wall=statistics.mean(map(charged,a)),
            residual_mean_capped_wall=statistics.mean(map(charged,b)),linear_solved=sum(r['status']=='SOLVED' for r in a),
            residual_solved=sum(r['status']=='SOLVED' for r in b)))
    ratio=policies['linear']['capped_mean_wall_seconds']/policies['residual']['capped_mean_wall_seconds']
    memory_ratio=statistics.median(a['peak_rss_bytes']/b['peak_rss_bytes'] for a,b in paired) if paired else None
    passes=policies['linear']['solved']>=policies['residual']['solved'] and ratio<=.95 and memory_ratio is not None and memory_ratio<=1.10
    summary=dict(policies=policies,paired_solved_linear_residual_runs=len(paired),linear_to_residual_capped_time_ratio=ratio,
        paired_solved_median_memory_ratio=memory_ratio,exploratory_promotion_gate_passed=passes,
        per_family=family_rows,test_evaluated=False,
        interpretation='Six validation families, two timing repeats each; repeats are not independent samples. Fixed random seed. No test-set speed or memory claim.')
    (ROOT/'validation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['analyze','train','validate','summarize-validation'])
    args=parser.parse_args();{'analyze':analyze,'train':train,'validate':validate,'summarize-validation':summarize_validation}[args.stage]()

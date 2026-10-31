"""Collect actual guarded actions and label with the deployed ICBS-h engine."""
from collections import Counter
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from mapf.repair_protocol import ROOT, read_protocol
from mapf.day3_protocol import digest
from mapf.day3_calibration import restore
from mapf.icbs_h import ICBShSolver
from mapf.grid import load_map
from mapf.baselines import guarded_shortlist, f_aware_residual_selector, _compute_candidate_lookahead_tuple
from mapf.cardinal import classify_conflicts
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.recovery_pilot import HarvestComplete, rollout, source_digest
from mapf.validation import validate_solution, validate_constrained_paths


def eligible_tier(classified):
    tiers={'CARDINAL':0,'SEMI_CARDINAL':1,'NON_CARDINAL':2}
    best=min(tiers[c.classification] for c in classified)
    return [c for c in classified if tiers[c.classification]==best]


def context_for(node,grid,goals,graph):
    return dict(paths=node.paths,goals=goals,G_CF=graph,mvc_size=graph.exact_mvc_size(),
                pair_counts=node.pair_counts,grid=grid)


def archive_source():
    fingerprint=source_digest()
    path=ROOT/'source.zip'
    if path.exists():
        if json.loads((ROOT/'source.json').read_text())['sha256']!=fingerprint:
            raise ValueError('Source changed during campaign; do not mix revisions')
    else:
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(Path('mapf').glob('*.py')):archive.write(p,p.as_posix())
        (ROOT/'source.json').write_text(json.dumps(dict(sha256=fingerprint)))
    return fingerprint


def harvest(protocol):
    code=archive_source();summaries=[]
    with (ROOT/'states.jsonl').open('x') as stream:
        for instance in protocol['calibration_instances']:
            if instance['split']=='test':raise ValueError('Test data remains sealed')
            grid=load_map(instance['map_path']);starts=list(map(tuple,instance['starts']));goals=list(map(tuple,instance['goals']))
            extractor=FeatureExtractor(grid,goals);captured=[];seen=set();counts=Counter()
            def capture(node,classified,graph):
                counts['decision_visits']+=1
                eligible=eligible_tier(classified)
                if len(eligible)==1:
                    counts['single_candidate']+=1;return
                context=context_for(node,grid,goals,graph)
                shortlist,unique=guarded_shortlist(eligible,context,protocol['top_k_guardrail'])
                if unique:
                    counts['analytical_shortcircuits']+=1;return
                counts['potential_ml_calls']+=1
                if len(shortlist)<2:raise ValueError('No ranking choice in collector shortlist')
                state=dict(paths=node.paths,constraints=[dict(vertices=sorted(c.vertex_constraints),edges=sorted(c.edge_constraints))
                    for c in node.constraints],g=node.g,h=node.h,depth=node.depth,bypass_steps=node.bypass_steps,
                    pair_counts=[[a,b,n] for (a,b),n in sorted(node.pair_counts.items())])
                fingerprint=digest([instance,state])
                if fingerprint in seen:return
                if len(captured)>=protocol['capture_limit']:raise HarvestComplete()
                seen.add(fingerprint);features=extractor.extract_all(shortlist,context)
                candidates=[]
                for cc,feature in zip(shortlist,features):
                    key=_compute_candidate_lookahead_tuple(cc,context)
                    key=[v if abs(v)!=float('inf') else ('+inf' if v>0 else '-inf') for v in key]
                    candidates.append(dict(conflict=vars(cc.conflict),features=feature.tolist(),analytical_key=key))
                captured.append(dict(query_id=f"{instance['id']}/guard-q{len(captured)}",instance=instance,state=state,
                    state_sha256=fingerprint,eligible_count=len(eligible),candidates=candidates,feature_names=FEATURE_NAMES,
                    feature_schema='recovery-v1-contextual-counts',protocol_sha256=protocol['protocol_sha256'],source_sha256=code))
            began=time.perf_counter()
            try:
                result=ICBShSolver(grid,conflict_selector=f_aware_residual_selector).solve(starts,goals,
                    timeout=protocol['harvest_seconds_per_instance'],max_expansions=protocol['harvest_expansion_cap'],on_node_callback=capture)
                status=result['status']
            except HarvestComplete:status='STATE_QUOTA'
            indices=sorted({0,len(captured)//2,len(captured)-1}) if captured else []
            for index in indices:stream.write(json.dumps(captured[index],allow_nan=False)+'\n')
            stream.flush()
            summary=dict(instance_id=instance['id'],status=status,captured=len(captured),selected=len(indices),
                         counters=dict(counts),wall_seconds=time.perf_counter()-began)
            summaries.append(summary);print('harvest',summary,flush=True)
    (ROOT/'harvest_summary.json').write_text(json.dumps(summaries,indent=2))


def icbs_rollout(node,cc,grid,starts,goals,ht,cap,seconds):
    began=time.perf_counter();cpu=time.process_time();progress={}
    result=ICBShSolver(grid,conflict_selector=f_aware_residual_selector).solve(starts,goals,timeout=seconds,
        max_expansions=cap,initial_node=node,forced_first_conflict=cc.conflict,shared_h_tables=ht,progress=progress)
    status={'SOLVED':'SOLVED','TIMEOUT':'TIME_CAP','MAX_EXPANSIONS':'EXPANSION_CAP',
            'UNSOLVABLE':'INFEASIBLE','NO_ROOT_PATH':'INFEASIBLE','HORIZON_LIMIT':'HORIZON_LIMIT'}[result['status']]
    saved=dict(status=status,expanded=progress['nodes_expanded'],expansion_cap=cap,wall_cap_seconds=seconds,
        forced_split_applied=progress['forced_split_applied'],astar_calls=result.get('astar_calls'),
        wall_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,
        reference_policy='residual-selector ICBS-h descendant continuation; root MVC recomputed')
    if status=='SOLVED':
        validate_solution(grid,starts,goals,result['paths'],result['cost'])
        validate_constrained_paths(grid,starts,goals,result['paths'],node.constraints)
        if not progress['forced_split_applied']:raise ValueError('Solved without forced action')
        saved.update(solution_cost=result['cost'],solution_paths=result['paths'],solution_validated=True)
    return saved


def replay(record,protocol):
    grid,starts,goals,node,ht,graph,chosen=restore(record)
    context=context_for(node,grid,goals,graph)
    eligible=eligible_tier(classify_conflicts(grid,starts,goals,node.paths,node.constraints,h_tables=ht))
    shortlist,unique=guarded_shortlist(eligible,context,protocol['top_k_guardrail'])
    if unique or [c.conflict for c in shortlist]!=[c.conflict for c in chosen]:
        raise ValueError('Saved actions do not match the actual deployed guard')
    features=FeatureExtractor(grid,goals).extract_all(chosen,context)
    if record['feature_names']!=FEATURE_NAMES or not np.isfinite(features).all():raise ValueError('Feature schema/values invalid')
    np.testing.assert_allclose(features,[c['features'] for c in record['candidates']],atol=1e-6,rtol=1e-6)
    return grid,starts,goals,node,ht,graph,chosen


def calibrate(protocol):
    code=archive_source()
    records=[json.loads(s) for s in (ROOT/'states.jsonl').read_text().splitlines()]
    out=ROOT/'calibration.jsonl'
    existing={json.loads(s)['query_id'] for s in out.read_text().splitlines()} if out.exists() else set()
    for n,record in enumerate(records):
        if record['query_id'] in existing:continue
        if record['source_sha256']!=code:raise ValueError('Mixed collection/label source')
        grid,starts,goals,node,ht,graph,chosen=replay(record,protocol)
        compare=record['instance']['family_id'].endswith('family-04') and record['query_id'].endswith('guard-q0')
        for candidate,cc in zip(record['candidates'],chosen):
            candidate['budgets']={b['name']:icbs_rollout(node,cc,grid,starts,goals,ht,b['cap'],b['seconds'])
                                  for b in protocol['rollout_budgets']}
            if compare:
                b=protocol['rollout_budgets'][-1]
                candidate['cbs_comparison']=rollout(node,cc,grid,starts,goals,ht,cap=b['cap'],seconds=b['seconds'])
        record['features_replayed']=True;record['guard_replayed']=True
        with out.open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        print('calibrated',n+1,'/',len(records),record['query_id'],[c['budgets']['high']['status'] for c in record['candidates']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['harvest','calibrate'])
    args=parser.parse_args();protocol=read_protocol()
    (harvest if args.stage=='harvest' else calibrate)(protocol)

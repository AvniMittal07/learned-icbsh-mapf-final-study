"""Label-free state sampling followed by independent, deeper rollout calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile
from mapf.day3_protocol import ROOT, digest
from mapf.grid import load_map
from mapf.icbs_h import ICBShSolver, ICBSNode
from mapf.low_level import ConstraintSet, compute_bfs_h_table
from mapf.cardinal import classify_conflicts
from mapf.conflicts import Conflict, detect_all_conflicts
from mapf.conflict_graph import compute_h4_and_graph
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.baselines import _compute_candidate_lookahead_tuple
from mapf.recovery_pilot import rollout, HarvestComplete, source_digest


def restore(record):
    instance,state=record['instance'],record['state']
    if digest([instance,state])!=record['state_sha256']:
        raise ValueError('State fingerprint mismatch')
    map_path=Path(instance['map_path'])
    if hashlib.sha256(map_path.read_bytes()).hexdigest()!=instance['map_sha256']:
        raise ValueError('Map fingerprint mismatch')
    grid=load_map(str(map_path))
    starts=[tuple(x) for x in instance['starts']];goals=[tuple(x) for x in instance['goals']]
    paths=[[tuple(x) for x in p] for p in state['paths']]
    constraints=[ConstraintSet(set(map(tuple,c['vertices'])),set(map(tuple,c['edges']))) for c in state['constraints']]
    node=ICBSNode(constraints,paths,state['g'],state['h'],depth=state['depth'],
                  pair_counts={(a,b):v for a,b,v in state['pair_counts']})
    ht=[compute_bfs_h_table(grid,g) for g in goals]
    classified=classify_conflicts(grid,starts,goals,paths,constraints,h_tables=ht)
    _,graph=compute_h4_and_graph(classified)
    index={cc.conflict:cc for cc in classified}
    chosen=[]
    for c in record['candidates']:
        data=dict(c['conflict']);data['loc1']=tuple(data['loc1'])
        if data['loc2'] is not None:data['loc2']=tuple(data['loc2'])
        chosen.append(index[Conflict(**data)])
    return grid,starts,goals,node,ht,graph,chosen


def harvest(protocol):
    path=ROOT/'states.jsonl'
    statuses=[];total=0
    with path.open('x') as stream:
        for instance in protocol['calibration_instances']:
            if instance['split']=='test':raise ValueError('Test data must remain sealed')
            grid=load_map(instance['map_path'])
            starts=[tuple(p) for p in instance['starts']];goals=[tuple(p) for p in instance['goals']]
            extractor=FeatureExtractor(grid,goals);captured=[]
            def capture(node,classified,graph):
                tiers={'CARDINAL':0,'SEMI_CARDINAL':1,'NON_CARDINAL':2}
                best=min(tiers[c.classification] for c in classified)
                eligible=[c for c in classified if tiers[c.classification]==best]
                if len(eligible)<2:return
                if len(captured)>=8:raise HarvestComplete()
                state=dict(paths=node.paths,constraints=[dict(vertices=sorted(c.vertex_constraints),
                    edges=sorted(c.edge_constraints)) for c in node.constraints],g=node.g,h=node.h,
                    depth=node.depth,pair_counts=[[a,b,n] for (a,b),n in sorted(node.pair_counts.items())])
                fingerprint=digest([instance,state])
                rng=random.Random(int(fingerprint[:12],16))
                chosen=eligible if len(eligible)<=4 else rng.sample(eligible,4)
                context=dict(paths=node.paths,goals=goals,G_CF=graph,mvc_size=graph.exact_mvc_size(),
                             pair_counts=node.pair_counts,grid=grid)
                features=extractor.extract_all(chosen,context)
                candidates=[]
                for cc,feature in zip(chosen,features):
                    key=_compute_candidate_lookahead_tuple(cc,context)
                    # Infinite analytical bounds are explicitly encoded, not
                    # converted into invented finite model targets.
                    key=[v if abs(v)!=float('inf') else ('+inf' if v>0 else '-inf') for v in key]
                    candidates.append(dict(conflict=vars(cc.conflict),features=feature.tolist(),analytical_key=key))
                captured.append(dict(query_id=f"{instance['id']}/q{len(captured)}",instance=instance,
                    state=state,state_sha256=fingerprint,eligible_count=len(eligible),candidates=candidates,
                    feature_names=FEATURE_NAMES,feature_schema='recovery-v1-contextual-counts',
                    protocol_sha256=protocol['protocol_sha256']))
            began=time.perf_counter()
            try:
                result=ICBShSolver(grid).solve(starts,goals,timeout=protocol['harvest_seconds_per_instance'],
                    max_expansions=protocol['harvest_expansion_cap'],on_node_callback=capture)
                status=result['status']
            except HarvestComplete:status='STATE_QUOTA'
            selected=[captured[i] for i in sorted({0,len(captured)-1})] if captured else []
            for record in selected:stream.write(json.dumps(record,allow_nan=False)+'\n')
            stream.flush();total+=len(selected)
            statuses.append(dict(instance_id=instance['id'],status=status,captured=len(captured),selected=len(selected),
                                 wall_seconds=time.perf_counter()-began))
            print('harvest',statuses[-1],flush=True)
    (ROOT/'harvest_summary.json').write_text(json.dumps(statuses,indent=2))
    return total


def calibrate(protocol):
    import numpy as np
    records=[json.loads(s) for s in (ROOT/'states.jsonl').read_text().splitlines()]
    out=ROOT/'calibration.jsonl'
    existing={json.loads(s)['query_id'] for s in out.read_text().splitlines()} if out.exists() else set()
    code_digest=source_digest()
    archive=ROOT/'calibration_source.zip'
    if archive.exists():
        previous=json.loads((ROOT/'source.json').read_text())['sha256']
        if previous!=code_digest:raise ValueError('Source changed; cannot mix calibration runs')
    else:
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(Path('mapf').glob('*.py')):z.write(p,p.as_posix())
        (ROOT/'source.json').write_text(json.dumps(dict(sha256=code_digest)))
    with out.open('a') as stream:
        for n,record in enumerate(records):
            if record['query_id'] in existing:continue
            grid,starts,goals,node,ht,graph,chosen=restore(record)
            context=dict(paths=node.paths,goals=goals,G_CF=graph,mvc_size=graph.exact_mvc_size(),pair_counts=node.pair_counts)
            features=FeatureExtractor(grid,goals).extract_all(chosen,context)
            np.testing.assert_allclose(features,[c['features'] for c in record['candidates']],atol=1e-6,rtol=1e-6)
            began=time.perf_counter()
            for c,cc in zip(record['candidates'],chosen):
                c['budgets']={}
                for budget in protocol['rollout_budgets']:
                    # Outside the harvesting solver's deadline; every candidate
                    # receives the same independent nominal time/expansion caps.
                    c['budgets'][budget['name']]=rollout(node,cc,grid,starts,goals,ht,
                                                       cap=budget['cap'],seconds=budget['seconds'])
            record['source_sha256']=code_digest
            record['features_replayed']=True
            record['label_wall_seconds']=time.perf_counter()-began
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print('calibrated',n+1,'/',len(records),record['query_id'],
                  [c['budgets']['high']['status'] for c in record['candidates']],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['harvest','calibrate'])
    args=parser.parse_args();protocol=json.loads((ROOT/'protocol.json').read_text())
    stored=protocol.pop('protocol_sha256');actual=digest(protocol);protocol['protocol_sha256']=stored
    if actual!=stored:raise ValueError('Frozen protocol was modified')
    if args.stage=='harvest':harvest(protocol)
    else:calibrate(protocol)


if __name__=='__main__':main()

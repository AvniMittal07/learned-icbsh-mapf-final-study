"""Small replayable pilot. Raw statuses are preserved; no synthetic timeout labels."""
import argparse
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import random
import time
import zipfile
from mapf.grid import load_map
from mapf.icbs_h import ICBShSolver
from mapf.low_level import get_path_cost, space_time_astar, compute_bfs_h_table
from mapf.cardinal import build_constraint_for_agent
from mapf.conflicts import detect_all_conflicts
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.baselines import _compute_candidate_lookahead_tuple
from mapf.search_limits import deadline_scope, SearchTimeout, HorizonExceeded
from mapf.validation import validate_solution


def rollout(node, chosen, grid, starts, goals, h_tables, cap=50, seconds=.25):
    """Forced split then plain CBS in the node's descendant problem.

    This does not resume the original full OPEN frontier. Count includes the
    forced root. Both exact infeasibility and censoring are separate outcomes.
    """
    began=time.perf_counter(); cpu=time.process_time()
    expanded=1; serial=itertools.count(); queue=[]
    result={}
    try:
        with deadline_scope(seconds):
            for agent,alt in [(chosen.conflict.agent_i,chosen.alt_path_i),
                              (chosen.conflict.agent_j,chosen.alt_path_j)]:
                if alt is None:
                    continue
                paths=list(node.paths);paths[agent]=alt
                constraints=list(node.constraints)
                constraints[agent]=build_constraint_for_agent(chosen.conflict,agent,constraints[agent])
                heapq.heappush(queue,(sum(map(get_path_cost,paths)),next(serial),paths,constraints))
            while queue and expanded < cap:
                cost,_,paths,constraints=heapq.heappop(queue)
                expanded+=1
                conflicts=detect_all_conflicts(paths)
                if not conflicts:
                    validate_solution(grid,starts,goals,paths,cost)
                    result=dict(status='SOLVED',solution_cost=cost)
                    break
                c=conflicts[0]
                for agent in (c.agent_i,c.agent_j):
                    child_cs=list(constraints)
                    child_cs[agent]=build_constraint_for_agent(c,agent,constraints[agent])
                    alt=space_time_astar(grid,starts[agent],goals[agent],child_cs[agent],h_table=h_tables[agent])
                    if alt is not None:
                        child_paths=list(paths);child_paths[agent]=alt
                        heapq.heappush(queue,(sum(map(get_path_cost,child_paths)),next(serial),child_paths,child_cs))
            if not result:
                result=dict(status='EXPANSION_CAP' if queue else 'INFEASIBLE')
    except SearchTimeout:
        result=dict(status='TIME_CAP')
    except HorizonExceeded:
        result=dict(status='HORIZON_LIMIT')
    return dict(result,expanded=expanded,expansion_cap=cap,wall_cap_seconds=seconds,
                wall_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,
                reference_policy='plain CBS earliest conflict; descendant problem')


class HarvestComplete(Exception):
    pass


def generate(output, per_instance=10, seed=701, agent_counts=(8,12,16)):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(Path('mapf').glob('*.py')):
            archive.write(p,p.as_posix())
    rng=random.Random(seed)
    manifest={'seed':seed,'purpose':'synthetic pilot only; not held-out benchmark evidence','instances':[]}
    statuses=[]
    with (output/'queries.jsonl').open('x') as stream:
        for name in ('random-32-32-20','room-32-32-4','warehouse-32-32-aisles'):
            map_path=Path('data')/(name+'.map');grid=load_map(str(map_path))
            map_hash=hashlib.sha256(map_path.read_bytes()).hexdigest()
            cells=[(r,c) for r in range(grid.height) for c in range(grid.width) if grid.is_passable(r,c)]
            for index,k in enumerate(agent_counts):
                starts,goals=rng.sample(cells,k),rng.sample(cells,k)
                instance=dict(id=f'{name}/pilot-{index}',map_path=str(map_path),map_sha256=map_hash,
                              starts=starts,goals=goals,seed=seed,scenario_family=f'synthetic-{name}-{index}')
                manifest['instances'].append(instance)
                extractor=FeatureExtractor(grid,goals)
                h_tables=[compute_bfs_h_table(grid,g) for g in goals]
                count=0
                def capture(node,classified,graph):
                    nonlocal count
                    tiers={'CARDINAL':0,'SEMI_CARDINAL':1,'NON_CARDINAL':2}
                    best=min(tiers[cc.classification] for cc in classified)
                    eligible=[cc for cc in classified if tiers[cc.classification]==best]
                    if len(eligible)<2:
                        return
                    if count>=per_instance:
                        raise HarvestComplete()
                    # Sample at most four eligible actions, recording omitted count.
                    selected=eligible if len(eligible)<=4 else rng.sample(eligible,4)
                    context=dict(paths=node.paths,goals=goals,G_CF=graph,mvc_size=graph.exact_mvc_size(),
                                 pair_counts=node.pair_counts,grid=grid)
                    features=extractor.extract_all(selected,context)
                    snapshot=dict(paths=node.paths, constraints=[dict(vertices=sorted(c.vertex_constraints),
                        edges=sorted(c.edge_constraints)) for c in node.constraints],g=node.g,h=node.h,
                        pair_counts=[[a,b,n] for (a,b),n in sorted(node.pair_counts.items())])
                    state_hash=hashlib.sha256(json.dumps([instance,snapshot],sort_keys=True).encode()).hexdigest()
                    record=dict(query_id=f"{instance['id']}/q{count}",instance=instance,state=snapshot,
                                state_sha256=state_hash,feature_schema='recovery-v1-contextual-counts',
                                feature_names=FEATURE_NAMES,eligible_count=len(eligible),candidates=[],
                                source_base='8f26ac3',source_changes_sha256=source_digest())
                    for cc,feature in zip(selected,features):
                        candidate=dict(conflict=vars(cc.conflict),features=feature.tolist(),
                            rollout=rollout(node,cc,grid,starts,goals,h_tables))
                        if count % 5 == 0:
                            candidate['deeper_rollout']=rollout(node,cc,grid,starts,goals,h_tables,cap=200,seconds=1.0)
                        record['candidates'].append(candidate)
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    count+=1
                began=time.perf_counter()
                try:
                    result=ICBShSolver(grid).solve(starts,goals,timeout=30,max_expansions=100,
                                                  on_node_callback=capture,shared_h_tables=h_tables)
                    status=result['status']
                except HarvestComplete:
                    status='QUERY_QUOTA'
                statuses.append(dict(instance_id=instance['id'],queries=count,status=status,
                                     wall_seconds=time.perf_counter()-began))
                print(statuses[-1],flush=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'harvest_status.json').write_text(json.dumps(statuses,indent=2))


def source_digest():
    h=hashlib.sha256()
    for p in sorted(Path('mapf').glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/recovery/pilot')
    p.add_argument('--queries-per-instance',type=int,default=10)
    p.add_argument('--agent-counts', nargs='+',type=int,default=[8,12,16])
    a=p.parse_args();generate(a.output,a.queries_per_instance,agent_counts=a.agent_counts)

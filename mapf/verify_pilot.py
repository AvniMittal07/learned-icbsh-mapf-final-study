"""Replay saved features and summarize observable oracle information."""
import argparse
import hashlib
from collections import Counter
import itertools
import json
from pathlib import Path
import statistics
import time
import zipfile
import numpy as np
from mapf.grid import load_map
from mapf.low_level import ConstraintSet
from mapf.conflicts import Conflict, detect_all_conflicts
from mapf.cardinal import classify_conflicts
from mapf.conflict_graph import compute_h4_and_graph
from mapf.features import FeatureExtractor


def verify(directory):
    directory=Path(directory)
    records=[json.loads(s) for s in (directory/'queries.jsonl').read_text().splitlines()]
    archived_digest=None
    if (directory/'source.zip').exists():
        h=hashlib.sha256()
        with zipfile.ZipFile(directory/'source.zip') as archive:
            for name in sorted(archive.namelist()):
                h.update(Path(name).name.encode());h.update(archive.read(name))
        archived_digest=h.hexdigest()
    statuses=Counter(); deeper_statuses=Counter(); costs=[]; replayed=0
    pair_total=pair_observed=informative=stable=changed=comparable_deeper=0
    bounded_pairs=bounded_queries=0
    for record in records:
        instance,state=record['instance'],record['state']
        map_path=Path(instance['map_path'].replace('\\','/'))
        assert hashlib.sha256(map_path.read_bytes()).hexdigest()==instance['map_sha256']
        assert hashlib.sha256(json.dumps([instance,state],sort_keys=True).encode()).hexdigest()==record['state_sha256']
        if archived_digest:
            assert archived_digest==record['source_changes_sha256']
        grid=load_map(str(map_path))
        starts=[tuple(p) for p in instance['starts']];goals=[tuple(p) for p in instance['goals']]
        paths=[[tuple(p) for p in path] for path in state['paths']]
        constraints=[ConstraintSet(set(map(tuple,c['vertices'])),set(map(tuple,c['edges'])))
                     for c in state['constraints']]
        classified=classify_conflicts(grid,starts,goals,paths,constraints,detect_all_conflicts(paths))
        _,graph=compute_h4_and_graph(classified)
        cc_map={cc.conflict:cc for cc in classified}
        selected=[]
        for candidate in record['candidates']:
            c=dict(candidate['conflict']);c['loc1']=tuple(c['loc1'])
            if c['loc2'] is not None:c['loc2']=tuple(c['loc2'])
            selected.append(cc_map[Conflict(**c)])
            statuses[candidate['rollout']['status']]+=1
            costs.append(candidate['rollout']['cpu_seconds'])
            if 'deeper_rollout' in candidate:
                deeper_statuses[candidate['deeper_rollout']['status']]+=1
        context=dict(paths=paths,goals=goals,G_CF=graph,mvc_size=graph.exact_mvc_size(),
                     pair_counts={(a,b):n for a,b,n in state['pair_counts']})
        features=FeatureExtractor(grid,goals).extract_all(selected,context)
        np.testing.assert_allclose(features,[c['features'] for c in record['candidates']],rtol=1e-6,atol=1e-6)
        replayed+=1
        useful=False
        bounded_useful=False
        for a,b in itertools.combinations(record['candidates'],2):
            pair_total+=1
            ra,rb=a['rollout'],b['rollout']
            # Conservative partial order: a solved search taking fewer pops
            # than an unfinished search has already used is strictly cheaper.
            # Never infer a preference between two censored searches.
            for solved,capped in ((ra,rb),(rb,ra)):
                if solved['status']=='SOLVED' and capped['status'] in ('EXPANSION_CAP','TIME_CAP') and solved['expanded']<capped['expanded']:
                    bounded_pairs+=1;bounded_useful=True
            if ra['status']==rb['status']=='SOLVED' and ra['expanded']!=rb['expanded']:
                pair_observed+=1;useful=True
                da,db=a.get('deeper_rollout',{}),b.get('deeper_rollout',{})
                if da.get('status')==db.get('status')=='SOLVED':
                    comparable_deeper+=1
                    if (ra['expanded']<rb['expanded'])==(da['expanded']<db['expanded']):stable+=1
                    else:changed+=1
        informative+=int(useful)
        bounded_queries+=int(useful or bounded_useful)
    summary=dict(queries=len(records),replayed_feature_queries=replayed,
                 source_archive_verified=archived_digest is not None,
                 unique_state_hashes=len({r['state_sha256'] for r in records}),
                 candidate_rollouts=sum(statuses.values()),rollout_statuses=dict(statuses),
                 deeper_statuses=dict(deeper_statuses),fully_observed_unequal_pairs=pair_observed,
                 sampled_candidate_pairs=pair_total,queries_with_observed_unequal_pair=informative,
                 solved_vs_censored_strict_preference_pairs=bounded_pairs,
                 queries_with_any_observed_or_bounded_preference=bounded_queries,
                 comparable_deeper_pairs=comparable_deeper,stable_pairs=stable,changed_pairs=changed,
                 mean_rollout_cpu_seconds=statistics.mean(costs) if costs else None,
                 p95_rollout_cpu_seconds=sorted(costs)[min(len(costs)-1,int(.95*len(costs)))] if costs else None,
                 limitations=['Synthetic pilot; not held-out performance evidence',
                     'Solved-only pair count is diagnostic, not a censoring-aware production target',
                     'Deeper agreement on solved pairs is not evidence about capped pairs'])
    (directory/'quality.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();verify(a.directory)

"""Freeze reproducible family-level splits and calibration rules before labeling."""
from collections import deque
import hashlib
import json
from pathlib import Path
import random
from mapf.grid import load_map

MAPS = ['random-32-32-20', 'room-32-32-4', 'warehouse-32-32-aisles']
ROOT = Path('results/recovery/day3')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def largest_component(grid):
    remaining = {(r,c) for r in range(grid.height) for c in range(grid.width) if grid.is_passable(r,c)}
    components = []
    while remaining:
        start = min(remaining); remaining.remove(start)
        component = {start}; queue = deque([start])
        while queue:
            for nxt in grid.get_neighbors(*queue.popleft()):
                if nxt in remaining:
                    remaining.remove(nxt);component.add(nxt);queue.append(nxt)
        components.append(sorted(component))
    return max(components, key=len)


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/'protocol.json'
    if path.exists():
        raise FileExistsError('Protocol already frozen; do not overwrite after observing labels')
    families=[]; calibration=[]
    for map_name in MAPS:
        map_path=Path('data')/(map_name+'.map')
        map_hash=hashlib.sha256(map_path.read_bytes()).hexdigest()
        cells=largest_component(load_map(str(map_path)))
        for index in range(30):
            seed=int(hashlib.sha256(f'MAPF-recovery-v1/{map_name}/{index}'.encode()).hexdigest()[:12],16)
            rng=random.Random(seed)
            starts,goals=rng.sample(cells,32),rng.sample(cells,32)
            split='train' if index<20 else ('validation' if index<24 else 'test')
            family_id=f'{map_name}/family-{index:02d}'
            instances=[]
            for k in [16,24,32]:
                instance=dict(id=f'{family_id}/k{k}',family_id=family_id,split=split,k=k,
                    map_name=map_name,map_path=map_path.as_posix(),map_sha256=map_hash,
                    starts=starts[:k],goals=goals[:k],seed=seed,
                    sampling='uniform independent unique starts/goals in largest component')
                instance['instance_sha256']=digest(instance)
                instances.append(instance)
            family=dict(family_id=family_id,split=split,instances=instances)
            families.append(family)
            # Four training + two validation families per map; labels never
            # inspect test families. Calibration selection uses no outcomes.
            if index in [0,1,2,3,20,21]:
                k=24 if index%2==0 else 32
                calibration.append(next(i for i in instances if i['k']==k))
    protocol=dict(version=1,families=families,calibration_instances=calibration,
        selection='Harvest at most 8 eligible states per selected instance; select first and last distinct state without inspecting labels',
        harvest_seconds_per_instance=6,harvest_expansion_cap=150,
        max_states_per_instance=2,max_candidates_per_state=4,
        rollout_budgets=[dict(name='low',cap=50,seconds=.5),dict(name='medium',cap=200,seconds=2),dict(name='high',cap=1000,seconds=6)],
        reference='forced split then deterministic plain CBS in saved descendant problem',
        quality_gate=dict(min_train_informative_queries=12,min_train_families=6,
            min_validation_informative_queries=6,min_validation_families=3,
            min_maps_with_informative_queries=2,min_informative_fraction=.4,
            replay_mismatches_allowed=0),
        model_if_gate_passes='fixed alpha=10 pairwise Ridge on train-only standardized feature differences; per-query pair weights',
        limitations=['Synthetic scenarios on known maps; not unseen-map generalization',
            'Offline oracle effort is not actual global ICBS-h runtime',
            'Budget stability after a deterministic rollout solves is an implementation check, not independent accuracy evidence'])
    protocol['protocol_sha256']=digest(protocol)
    path.write_text(json.dumps(protocol,indent=2))
    # Hashes identify the exact test inputs; no test paths/labels are evaluated.
    (ROOT/'sealed_test_manifest.json').write_text(json.dumps(dict(protocol_sha256=protocol['protocol_sha256'],
        instances=[i for f in families if f['split']=='test' for i in f['instances']]),indent=2))
    print(json.dumps(dict(families=len(families),instances=sum(len(f['instances']) for f in families),
        calibration_instances=len(calibration),sealed_test_instances=54,protocol_sha256=protocol['protocol_sha256']),indent=2))


if __name__=='__main__':
    freeze()

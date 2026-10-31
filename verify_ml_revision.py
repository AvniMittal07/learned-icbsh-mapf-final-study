"""Independent child-MVC audit and offline diagnostics; never solves test inputs.

Kept outside mapf so adding this post-collection verifier does not change the
archived generation/deployment source fingerprint. Included in package manifest.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np
from mapf.ml_revision import ROOT, OLD, readlines, protocol, replay, teacher_pairs, verify_branches, write_json
from mapf.ml_revision_policy import SavedLinear, teacher_key
from mapf.cardinal import build_constraint_for_agent, classify_conflicts
from mapf.day3_protocol import digest
from mapf.day3_analysis import pairs as continuation_pairs
from mapf.recovery_pilot import source_digest


@lru_cache(maxsize=16384)
def independent_cover(edges):
    # Exhaustive edge-endpoint branching, no solver MVC code or its caches.
    if not edges:return 0
    u,v=edges[0]
    return 1+min(independent_cover(tuple(e for e in edges if u not in e)),
                 independent_cover(tuple(e for e in edges if v not in e)))


def offline(records,pair_fn,models,teacher=False):
    result={}
    for name,model in models.items():
        fam=defaultdict(list);optimal=Counter();disagreements=0
        for r in records:
            keys=[tuple(float('inf') if x=='+inf' else -float('inf') if x=='-inf' else x for x in c['analytical_key']) for c in r['candidates']]
            scores=keys if model is None else model.predict(np.array([c['features'] for c in r['candidates']])).tolist()
            chosen=max(range(len(scores)),key=lambda i:scores[i])
            pairs=pair_fn(r);wins=[]
            for i,j,order,_ in pairs:
                pred=int(scores[i]>scores[j])-int(scores[i]<scores[j]);wins.append(.5 if pred==0 else float(pred==order))
            if wins:fam[r['instance']['family_id']].append(statistics.mean(wins))
            if teacher and all(c['teacher']['status']=='COMPLETE' for c in r['candidates']):
                labels=[teacher_key(c['teacher']) for c in r['candidates']]
                optimal['queries']+=1;optimal['chosen_teacher_maximum']+=labels[chosen]==max(labels)
                disagreements+=chosen!=0
        result[name]=dict(family_macro_pair_accuracy=statistics.mean(statistics.mean(v) for v in fam.values()) if fam else None,
                          informative_families=len(fam),teacher_choice_counts=dict(optimal),choices_differing_from_residual=disagreements)
    return result


def main():
    p=protocol();records=readlines(ROOT/'labels.jsonl');states=readlines(ROOT/'states.jsonl')
    assert {r['query_id'] for r in records}=={r['query_id'] for r in states}
    harvest=readlines(ROOT/'harvest.jsonl')
    assert {h['instance_id'] for h in harvest}=={i['id'] for i in p['instances']}
    assert len(harvest)==len(p['instances'])==48
    archived=json.loads((ROOT/'source.json').read_text())['sha256']
    assert archived==source_digest()
    with zipfile.ZipFile(ROOT/'source.zip') as z:
        fingerprint=hashlib.sha256()
        for name in sorted(z.namelist()):
            fingerprint.update(Path(name).name.encode());fingerprint.update(z.read(name))
        assert fingerprint.hexdigest()==archived
    checked=0;raw_mvc_max=0
    for n,r in enumerate(records):
        ctx,chosen=replay(r);verify_branches(r,ctx,chosen)
        for candidate,cc in zip(r['candidates'],chosen):
            for b in candidate['teacher']['branches']:
                if b['status']!='FEASIBLE':continue
                agent=b['agent'];path=list(map(tuple,b['replanned_path']))
                exact_alt=cc.alt_cost_i if agent==cc.conflict.agent_i else cc.alt_cost_j
                assert len(path)-1==exact_alt, 'Child path differs from parent classification optimum'
                paths=list(ctx['paths']);paths[agent]=path
                cs=list(ctx['per_agent_constraints']);cs[agent]=build_constraint_for_agent(cc.conflict,agent,cs[agent])
                classified=classify_conflicts(ctx['grid'],ctx['starts'],ctx['goals'],paths,cs,h_tables=ctx['h_tables'])
                edges=tuple(sorted({tuple(sorted((c.conflict.agent_i,c.conflict.agent_j))) for c in classified if c.classification=='CARDINAL'}))
                exact=independent_cover(edges)
                assert exact==b['raw_h'], 'Teacher MVC does not match independent cover'
                raw_mvc_max=max(raw_mvc_max,exact);checked+=1
        if n%40==0:print('independent child audit',n+1,'/',len(records),flush=True)
    # No nested variants cross family splits. New train 08-19; development 20-23.
    bysplit={s:{r['instance']['family_id'] for r in records if r['instance']['split']==s} for s in ('train','validation')}
    assert not bysplit['train'] & bysplit['validation']
    assert all(8<=int(f[-2:])<20 for f in bysplit['train'])
    assert all(20<=int(f[-2:])<24 for f in bysplit['validation'])
    old=readlines(OLD/'calibration.jsonl');models={'model_a':SavedLinear(ROOT/'model_a.json'),'model_b':SavedLinear(ROOT/'model_b.json')}
    for name,dataset in [('model_a',old),('model_b',records)]:
        meta=json.loads((ROOT/(name+'.json.metadata.json')).read_text())
        train=[r for r in dataset if r['instance']['split']=='train']
        assert meta['training_records_sha256']==digest(train)
        X=np.array([c['features'] for r in train for c in r['candidates']])
        np.testing.assert_allclose(models[name].mean,X.mean(axis=0),rtol=1e-12,atol=1e-12)
        assert meta['old_collection_gate_passed'] is False and meta['test_used'] is False
    report=dict(states=len(records),candidate_labels=sum(len(r['candidates']) for r in records),
        independently_recomputed_child_MVC_values=checked,child_path_costs_match_parent_optimal_replans=True,
        max_child_MVC=raw_mvc_max,feature_pool_and_path_checks_passed=True,source_archive_verified=True,
        family_split_overlap=0,training_scalers_verified=True,all_48_families_attempted=True,
        zero_query_families=[h['instance_id'] for h in harvest if not h['queries']],test_evaluated=False,
        offline_diagnostics={s:dict(
            continuation=offline([r for r in old if r['instance']['split']==s],continuation_pairs,{'analytical_within_old_pool':None,'model_a':models['model_a']}),
            teacher=offline([r for r in records if r['instance']['split']==s],teacher_pairs,{'residual':None,'model_b':models['model_b']},True)) for s in ('train','validation')})
    destination=ROOT/'independent_integrity.json'
    if destination.exists():
        assert json.loads(destination.read_text())==report, 'Replayed integrity report differs'
    else:write_json(destination,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

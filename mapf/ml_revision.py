"""Frozen, bounded exploratory ML campaign. Never evaluates test families."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import zipfile
import numpy as np
from mapf.day3_protocol import digest
from mapf.day3_calibration import restore
from mapf.day3_analysis import pairs as continuation_pairs
from mapf.repair_pilot import eligible_tier, context_for, replay as old_replay
from mapf.recovery_pilot import HarvestComplete, source_digest
from mapf.ml_revision_policy import candidate_pool, immediate_children, teacher_key, SavedLinear
from mapf.baselines import f_aware_residual_selector, _compute_candidate_lookahead_tuple
from mapf.cardinal import classify_conflicts, build_constraint_for_agent
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.icbs_h import ICBShSolver
from mapf.grid import load_map
from mapf.validation import validate_constrained_paths, validate_solution

ROOT=Path('results/recovery/ml_revision')
OLD=Path('results/recovery/repair_pilot')


def readlines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def write_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def append(path, data):
    with path.open('a') as stream:
        stream.write(json.dumps(data,allow_nan=False)+'\n')


def freeze():
    parent=json.loads(Path('results/recovery/day3/protocol.json').read_text())
    check=dict(parent); expected=check.pop('protocol_sha256')
    if digest(check)!=expected: raise ValueError('Parent protocol hash mismatch')
    # Interleave map families so a global cap cannot consume one map first.
    selected=[]
    for index in range(8,24):
        for map_index,map_name in enumerate(['random-32-32-20','room-32-32-4','warehouse-32-32-aisles']):
            family=next(f for f in parent['families'] if f['family_id']==f'{map_name}/family-{index:02d}')
            k=[16,24,32][(index-8+map_index)%3]
            selected.append(next(i for i in family['instances'] if i['k']==k))
    protocol=dict(parent_protocol_sha256=expected, instances=selected,
        audit_family_ids=[i['family_id'] for i in selected if int(i['family_id'][-2:])<14],
        training_indices=list(range(8,20)), validation_indices=list(range(20,24)),
        harvest_seconds=6, harvest_expansions=200, max_train_queries=8, max_validation_queries=4,
        state_selection='First distinct post-bypass decisions with at least two highest-tier candidates; no outcome filtering',
        audit_selection='First two saved queries of training families 08-13 on each map (up to 36)',
        audit_bulk_rule='Proceed only if at least one fully teacher-ranked audit query strictly prefers an action to the residual winner. Unknowns and ties do not establish this.',
        pool='residual winner first, then up to five others sorted by SHA256 of frozen seed plus sorted-key conflict JSON',
        pool_seed='ml-revision-pool-v1-71', teacher_candidate_seconds=1.,
        teacher_priority='lexicographic (min effective child f, max effective child f), higher preferred; equal keys tied',
        infeasibility='absent branch +inf; two absent branches NODE_INFEASIBLE excluded from pair labels; time limits unknown',
        models=dict(alpha=10,fit_intercept=False,standardization='train candidates only',pair_weight='1 / strict pairs in query'),
        old_collection_gate_passed=False, exploratory_fit_explicitly_approved=True,
        evaluation=dict(policies=['plain','residual','pool_random','teacher','model_a','model_b'],repetitions=2,
            timeout_seconds=8,memory_mb=4096,schedule_seed=71,random_seed=71,
            note='Same random seed across timing repetitions; development only, test remains sealed'),
        approved_phase_cpu_seconds=5400, test_evaluation_allowed=False)
    protocol['protocol_sha256']=digest(protocol)
    ROOT.mkdir(parents=True,exist_ok=True); write_json(ROOT/'protocol.json',protocol)
    print(json.dumps(dict(instances=len(selected),protocol_sha256=protocol['protocol_sha256'])),flush=True)


def protocol():
    p=json.loads((ROOT/'protocol.json').read_text()); c=dict(p); expected=c.pop('protocol_sha256')
    if digest(c)!=expected or any(i['split']=='test' for i in p['instances']):
        raise ValueError('Modified protocol or test input')
    return p


def archive_source():
    code=source_digest()
    if (ROOT/'source.json').exists():
        if json.loads((ROOT/'source.json').read_text())['sha256']!=code:
            raise ValueError('Code changed during collection')
    else:
        with zipfile.ZipFile(ROOT/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(Path('mapf').glob('*.py')):z.write(path,path.as_posix())
        write_json(ROOT/'source.json',dict(sha256=code))
    return code


def context(node,grid,starts,goals,graph,ht=None):
    ctx=context_for(node,grid,goals,graph)
    ctx.update(starts=starts,per_agent_constraints=node.constraints,effective_h=node.h)
    if ht is not None:ctx['h_tables']=ht
    return ctx


def harvest(p, instances):
    code=archive_source()
    done={r['instance_id'] for r in readlines(ROOT/'harvest.jsonl')}
    for ins in instances:
        if ins['id'] in done:continue
        if ins['split'] not in ('train','validation'):raise ValueError('Test is sealed')
        grid=load_map(ins['map_path']);starts=list(map(tuple,ins['starts']));goals=list(map(tuple,ins['goals']))
        extractor=FeatureExtractor(grid,goals);seen=set();captured=[]
        quota=p['max_train_queries' if ins['split']=='train' else 'max_validation_queries']
        def capture(node,classified,graph):
            eligible=eligible_tier(classified)
            if len(eligible)<2:return
            ctx=context(node,grid,starts,goals,graph);pool=candidate_pool(eligible,ctx)
            state=dict(paths=node.paths,constraints=[dict(vertices=sorted(c.vertex_constraints),edges=sorted(c.edge_constraints)) for c in node.constraints],
                       g=node.g,h=node.h,depth=node.depth,pair_counts=[[a,b,n] for (a,b),n in sorted(node.pair_counts.items())])
            sha=digest([ins,state])
            if sha in seen:return
            seen.add(sha);X=extractor.extract_all(pool,ctx)
            if not np.isfinite(X).all():raise ValueError('Invalid features')
            candidates=[]
            for cc,features in zip(pool,X):
                key=_compute_candidate_lookahead_tuple(cc,ctx)
                candidates.append(dict(conflict=vars(cc.conflict),features=features.tolist(),
                    analytical_key=['+inf' if v==float('inf') else '-inf' if v==-float('inf') else v for v in key]))
            captured.append(dict(query_id=f"{ins['id']}/pool-q{len(captured)}",instance=ins,state=state,state_sha256=sha,
                candidates=candidates,eligible_count=len(eligible),feature_names=FEATURE_NAMES,
                protocol_sha256=p['protocol_sha256'],source_sha256=code))
            if len(captured)>=quota:raise HarvestComplete()
        began=time.perf_counter()
        try:
            result=ICBShSolver(grid,conflict_selector=f_aware_residual_selector).solve(starts,goals,
                timeout=p['harvest_seconds'],max_expansions=p['harvest_expansions'],on_node_callback=capture)
            status=result['status']
        except HarvestComplete:status='STATE_QUOTA'
        for r in captured:append(ROOT/'states.jsonl',r)
        row=dict(instance_id=ins['id'],status=status,queries=len(captured),wall_seconds=time.perf_counter()-began)
        append(ROOT/'harvest.jsonl',row); print('harvest',row,flush=True)


def replay(record):
    grid,starts,goals,node,ht,graph,chosen=restore(record)
    validate_constrained_paths(grid,starts,goals,node.paths,node.constraints)
    if sum(len(path)-1 for path in node.paths)!=node.g:raise ValueError('Bad parent g')
    ctx=context(node,grid,starts,goals,graph,ht)
    eligible=eligible_tier(classify_conflicts(grid,starts,goals,node.paths,node.constraints,h_tables=ht))
    pool=candidate_pool(eligible,ctx)
    if [c.conflict for c in pool]!=[c.conflict for c in chosen]:raise ValueError('Pool mismatch')
    X=FeatureExtractor(grid,goals).extract_all(chosen,ctx)
    if record['feature_names']!=FEATURE_NAMES or not np.isfinite(X).all():raise ValueError('Bad features')
    np.testing.assert_allclose(X,[c['features'] for c in record['candidates']],rtol=1e-6,atol=1e-6)
    for c,cc in zip(record['candidates'],chosen):
        decoded=tuple(float('inf') if x=='+inf' else -float('inf') if x=='-inf' else x for x in c['analytical_key'])
        if decoded!=_compute_candidate_lookahead_tuple(cc,ctx):raise ValueError('Analytical key mismatch')
    return ctx, chosen


def verify_branches(record,ctx,chosen):
    count=0
    for candidate,cc in zip(record['candidates'],chosen):
        label=candidate['teacher']
        if label['status']=='COMPLETE' and len(label['branches'])!=2:raise ValueError('Incomplete complete label')
        for branch in label['branches']:
            if branch['status']!='FEASIBLE':continue
            agent=branch['agent'];paths=list(ctx['paths']);paths[agent]=list(map(tuple,branch['replanned_path']))
            cs=list(ctx['per_agent_constraints']);cs[agent]=build_constraint_for_agent(cc.conflict,agent,cs[agent])
            validate_constrained_paths(ctx['grid'],ctx['starts'],ctx['goals'],paths,cs)
            g=sum(len(path)-1 for path in paths)
            if branch['g']!=g or branch['effective_f']!=max(g+branch['raw_h'],record['state']['g']+record['state']['h']):
                raise ValueError('Child bound inconsistency')
            count+=1
        if label['status']=='COMPLETE':
            fs=[float('inf') if b['effective_f']=='+inf' else b['effective_f'] for b in label['branches']]
            if teacher_key(label)!=(min(fs),max(fs)):raise ValueError('Priority inconsistent with children')
    return count


def label_records(p, records):
    code=archive_source();done={r['query_id'] for r in readlines(ROOT/'labels.jsonl')}
    for r in records:
        if r['query_id'] in done:continue
        if r['source_sha256']!=code or r['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Mixed source/protocol')
        ctx,chosen=replay(r)
        for c,cc in zip(r['candidates'],chosen):c['teacher']=immediate_children(cc,ctx,p['teacher_candidate_seconds'])
        r['verified_child_paths']=verify_branches(r,ctx,chosen);r['features_and_pool_replayed']=True
        append(ROOT/'labels.jsonl',r)
        print('label',r['query_id'],[c['teacher']['status'] for c in r['candidates']],flush=True)


def teacher_pairs(r):
    if any(c['teacher']['status']=='NODE_INFEASIBLE' for c in r['candidates']):return []
    keys=[teacher_key(c['teacher']) for c in r['candidates']];pairs=[]
    for i,a in enumerate(keys):
        for j in range(i+1,len(keys)):
            b=keys[j]
            if a is not None and b is not None and a!=b:pairs.append((i,j,1 if a>b else -1,'complete_teacher_pair'))
    return pairs


def data_summary(records):
    complete=[r for r in records if all(c['teacher']['status']=='COMPLETE' for c in r['candidates'])]
    disagreements=[r['query_id'] for r in complete if max(teacher_key(c['teacher']) for c in r['candidates'])>teacher_key(r['candidates'][0]['teacher'])]
    useful=[r for r in records if teacher_pairs(r)]
    return dict(queries=len(records),families=len({r['instance']['family_id'] for r in records}),
        candidates=sum(len(r['candidates']) for r in records),status_counts=dict(Counter(c['teacher']['status'] for r in records for c in r['candidates'])),
        fully_ranked_queries=len(complete),fully_ranked_all_tied=sum(len({teacher_key(c['teacher']) for c in r['candidates']})==1 for r in complete),
        residual_strictly_worse_queries=disagreements,informative_queries=len(useful),
        informative_families=len({r['instance']['family_id'] for r in useful}),strict_pairs=sum(len(teacher_pairs(r)) for r in records),
        identical_feature_strict_pairs=sum(np.array_equal(r['candidates'][i]['features'],r['candidates'][j]['features']) for r in records for i,j,_,_ in teacher_pairs(r)),
        teacher_cpu_seconds=sum(c['teacher']['cpu_seconds'] for r in records for c in r['candidates']),
        verified_child_paths=sum(r['verified_child_paths'] for r in records))


def audit_stage(p):
    instances=[i for i in p['instances'] if i['family_id'] in p['audit_family_ids']]
    harvest(p,instances)
    records=[r for r in readlines(ROOT/'states.jsonl') if r['instance']['family_id'] in p['audit_family_ids'] and int(r['query_id'].split('pool-q')[1])<2]
    label_records(p,records)
    ids={r['query_id'] for r in records}
    summary=data_summary([r for r in readlines(ROOT/'labels.jsonl') if r['query_id'] in ids])
    summary['bulk_allowed']=bool(summary['residual_strictly_worse_queries'])
    summary['audit_query_ids']=sorted(ids);write_json(ROOT/'teacher_audit.json',summary);print(json.dumps(summary),flush=True)


def bulk(p):
    if not json.loads((ROOT/'teacher_audit.json').read_text())['bulk_allowed']:
        print('No demonstrated teacher advantage in frozen audit; bulk skipped',flush=True);return
    harvest(p,p['instances']);label_records(p,readlines(ROOT/'states.jsonl'))


def accuracy(records,pair_fn,model):
    families=defaultdict(list);queries=0;pairs_count=0
    for r in records:
        pairs=pair_fn(r)
        if not pairs:continue
        scores=model.predict(np.array([c['features'] for c in r['candidates']]))
        values=[]
        for i,j,order,_ in pairs:
            pred=int(scores[i]>scores[j])-int(scores[i]<scores[j]);values.append(.5 if pred==0 else float(pred==order))
        families[r['instance']['family_id']].append(statistics.mean(values));queries+=1;pairs_count+=len(pairs)
    return dict(queries=queries,families=len(families),pairs=pairs_count,
        family_macro_pair_accuracy=statistics.mean(statistics.mean(v) for v in families.values()) if families else None)


def fit(records,pair_fn,name,p,data_file):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    train=[r for r in records if r['instance']['split']=='train']
    informative=[r for r in train if pair_fn(r)]
    if len({r['instance']['family_id'] for r in informative})<2:
        return dict(trained=False,reason='Fewer than two informative training families')
    allX=np.array([c['features'] for r in train for c in r['candidates']],dtype=float)
    scaler=StandardScaler().fit(allX);X=[];y=[];weights=[]
    for r in informative:
        pairs=pair_fn(r);z=scaler.transform(np.array([c['features'] for c in r['candidates']]))
        for i,j,order,_ in pairs:X.append(z[i]-z[j]);y.append(order);weights.append(1/len(pairs))
    if not np.any(np.array(X)):
        return dict(trained=False,reason='All strict pairs have identical features')
    estimator=Ridge(alpha=10,fit_intercept=False).fit(np.array(X),np.array(y),sample_weight=np.array(weights))
    path=ROOT/(name+'.json')
    write_json(path,dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=estimator.coef_.tolist(),intercept=0.))
    metadata=dict(feature_names=FEATURE_NAMES,backend='sklearn_ridge',model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        exploratory=True,old_collection_gate_passed=False,exploratory_fit_explicitly_approved=True,
        protocol_sha256=p['protocol_sha256'],source_sha256=source_digest(),alpha=10,fit_intercept=False,
        dataset_sha256=hashlib.sha256(data_file.read_bytes()).hexdigest(),training_records_sha256=digest(train),
        training_family_ids=sorted({r['instance']['family_id'] for r in train}),
        informative_training_family_ids=sorted({r['instance']['family_id'] for r in informative}),
        training_query_ids=[r['query_id'] for r in informative],training_pairs=len(y),
        target='ICBS-h descendant expansion preference' if name=='model_a' else p['teacher_priority'],
        candidate_pool='old guarded top three' if name=='model_a' else p['pool'],
        validation_used_for_fit=False,test_used=False)
    write_json(Path(str(path)+'.metadata.json'),metadata)
    loaded=SavedLinear(path)
    expected=estimator.predict(scaler.transform(allX));actual=loaded.predict(allX)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    return dict(trained=True,model_path=path.as_posix(),training_pairs=len(y),informative_training_queries=len(informative),
        informative_training_families=len(metadata['informative_training_family_ids']),
        serialization_max_abs_error=float(np.max(np.abs(actual-expected))),
        evaluations={s:accuracy([r for r in records if r['instance']['split']==s],pair_fn,loaded) for s in ('train','validation')})


def train_a(p):
    records=readlines(OLD/'calibration.jsonl');old_protocol=json.loads((OLD/'protocol.json').read_text())
    for r in records:
        grid,starts,goals,node,_,_,_=old_replay(r,old_protocol)
        for c in r['candidates']:
            label=c['budgets']['high']
            if label['status']=='SOLVED':
                paths=[list(map(tuple,path)) for path in label['solution_paths']]
                validate_solution(grid,starts,goals,paths,label['solution_cost'])
                validate_constrained_paths(grid,starts,goals,paths,node.constraints)
    result=fit(records,continuation_pairs,'model_a',p,OLD/'calibration.jsonl')
    write_json(ROOT/'model_a_report.json',result);print(json.dumps(result),flush=True)


def train_b(p):
    records=readlines(ROOT/'labels.jsonl');state_index={r['query_id']:r for r in readlines(ROOT/'states.jsonl')}
    if len({r['query_id'] for r in records})!=len(records):raise ValueError('Duplicate labels')
    valid_instances={i['id']:i for i in p['instances']}
    for r in records:
        if r['instance']!=valid_instances[r['instance']['id']]:raise ValueError('Instance changed')
        raw=state_index[r['query_id']]
        for k in ('state','instance','state_sha256','source_sha256','protocol_sha256'):
            if r[k]!=raw[k]:raise ValueError('Saved state/provenance changed')
        if r['source_sha256']!=json.loads((ROOT/'source.json').read_text())['sha256']:raise ValueError('Mixed data sources')
        for a,b in zip(raw['candidates'],r['candidates']):
            if any(a[k]!=b[k] for k in a):raise ValueError('Candidate changed after labeling')
        ctx,chosen=replay(r);verify_branches(r,ctx,chosen)
    summary={s:data_summary([r for r in records if r['instance']['split']==s]) for s in ('train','validation')}
    write_json(ROOT/'data_quality.json',summary)
    result=fit(records,teacher_pairs,'model_b',p,ROOT/'labels.jsonl') if json.loads((ROOT/'teacher_audit.json').read_text())['bulk_allowed'] else dict(trained=False,reason='Frozen audit did not authorize bulk teacher experiment')
    write_json(ROOT/'model_b_report.json',result);print(json.dumps(dict(data=summary,model=result)),flush=True)


def evaluate(p):
    from mapf.recovery_benchmark import run_isolated
    config=p['evaluation'];policies=list(config['policies']);paths={}
    for name in ('model_a','model_b'):
        report=json.loads((ROOT/(name+'_report.json')).read_text())
        if report['trained']:paths[name]=report['model_path']
        else:policies.remove(name)
    jobs=[(i,repeat,policy) for i in p['instances'] if i['split']=='validation'
          for repeat in range(config['repetitions']) for policy in policies]
    random.Random(config['schedule_seed']).shuffle(jobs)
    schedule=[dict(instance_id=i['id'],repeat=r,policy=policy) for i,r,policy in jobs]
    if not (ROOT/'evaluation_schedule.json').exists():write_json(ROOT/'evaluation_schedule.json',schedule)
    elif json.loads((ROOT/'evaluation_schedule.json').read_text())!=schedule:raise ValueError('Evaluation schedule changed')
    done={(r['instance_id'],r['repeat'],r['policy']) for r in readlines(ROOT/'development.jsonl')}
    for ins,repeat,policy in jobs:
        if (ins['id'],repeat,policy) in done:continue
        spec=dict(instance=ins,policy=policy,seed=config['random_seed'],timeout_seconds=config['timeout_seconds'],model_paths=paths)
        row=run_isolated(spec,memory_mb=config['memory_mb'])
        row.update(instance_id=ins['id'],family_id=ins['family_id'],repeat=repeat,policy=policy,
                   protocol_sha256=p['protocol_sha256'],source_sha256=source_digest())
        append(ROOT/'development.jsonl',row)
        print('development',ins['id'],repeat,policy,row['status'],round(row['end_to_end_wall_seconds'],3),flush=True)


def summarize(p):
    rows=readlines(ROOT/'development.jsonl');schedule=json.loads((ROOT/'evaluation_schedule.json').read_text())
    expected={(s['instance_id'],s['repeat'],s['policy']) for s in schedule}
    actual={(r['instance_id'],r['repeat'],r['policy']) for r in rows}
    if expected!=actual or len(actual)!=len(rows):raise ValueError('Incomplete or duplicated evaluation')
    bad=[r for r in rows if r['status'] not in ('SOLVED','TIMEOUT','HARD_TIMEOUT','MEMORY_LIMIT')]
    if bad:raise ValueError('Invalid evaluation statuses: '+str([(r['policy'],r['status']) for r in bad]))
    costs=defaultdict(set)
    for r in rows:
        if r['status']=='SOLVED':
            if not r.get('solution_validated'):raise ValueError('Unvalidated result')
            costs[r['instance_id']].add(r['cost'])
    if any(len(v)>1 for v in costs.values()):raise ValueError('Solved-cost disagreement')
    by_policy={policy:[r for r in rows if r['policy']==policy] for policy in sorted({r['policy'] for r in rows})}
    cap=p['evaluation']['timeout_seconds']
    def capped(r):return min(cap,r['end_to_end_wall_seconds']) if r['status']=='SOLVED' else cap
    report={}
    for policy,runs in by_policy.items():
        groups=defaultdict(list)
        for r in runs:groups[r['family_id']].append(r)
        base=by_policy['residual'];basegroups=defaultdict(list)
        for r in base:basegroups[r['family_id']].append(r)
        ratios=[];deltas=[];joint=[]
        for family,group in groups.items():
            b=basegroups[family];bt=statistics.mean(capped(r) for r in b);pt=statistics.mean(capped(r) for r in group)
            ratios.append(pt/bt);deltas.append(pt-bt)
            if all(r['status']=='SOLVED' for r in group+b):joint.append((family,pt/bt))
        rng=np.random.default_rng(71);arr=np.log(np.array(ratios))
        boots=np.exp(arr[rng.integers(0,len(arr),(5000,len(arr)))].mean(axis=1))
        solved=[r for r in runs if r['status']=='SOLVED']
        stats=[r['selector_stats'] for r in runs if 'selector_stats' in r]
        report[policy]=dict(runs=len(runs),families=len(groups),solved_runs=len(solved),
            solved_families_both_repeats=sum(all(r['status']=='SOLVED' for r in g) for g in groups.values()),
            status_counts=dict(Counter(r['status'] for r in runs)),mean_capped_seconds=statistics.mean(capped(r) for r in runs),
            median_peak_memory_mb=statistics.median(r['peak_rss_bytes']/1024**2 for r in runs),
            mean_solved_expansions=statistics.mean(r['nodes_expanded'] for r in solved) if solved else None,
            geometric_family_capped_time_ratio_vs_residual=float(np.exp(arr.mean())),
            exploratory_bootstrap_95_interval=np.quantile(boots,[.025,.975]).tolist(),
            family_time_wins=sum(x<0 for x in deltas),family_time_losses=sum(x>0 for x in deltas),
            jointly_solved_families=len(joint),jointly_solved_time_ratios=joint,
            telemetry_runs=len(stats),selector_totals={k:sum(s[k] for s in stats) for k in stats[0]} if stats else {})
    write_json(ROOT/'development_summary.json',dict(policies=report,solved_cost_agreement=True,
        total_runs=len(rows),independent_families=12,test_evaluated=False,
        caveats=['Development scenarios already informed design; not confirmatory evidence',
                 '12 families, timing repeats not independent; bootstrap intervals descriptive',
                 '8-second censoring; both-timeout pairs tie at cap, not equal latent performance',
                 'Memory is process peak, includes interpreter and model loading; solved-only expansions selection biased',
                 'Hard-killed runs have no selector telemetry; totals are observed only']))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train-a','audit','bulk','train-b','evaluate','summarize'])
    stage=parser.parse_args().stage
    if stage=='freeze':freeze()
    else:{'train-a':train_a,'audit':audit_stage,'bulk':bulk,'train-b':train_b,'evaluate':evaluate,'summarize':summarize}[stage](protocol())

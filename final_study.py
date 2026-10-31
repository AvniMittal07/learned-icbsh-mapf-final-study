"""Frozen final test runner and family-level analysis. No fitting or tuning.

All six policies execute the unchanged mapf.benchmark_worker. The new monitor
adds launch/monitor timing diagnostics to the existing isolated-run method.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
import psutil

ROOT=Path('results/recovery/final_study')
ML=Path('results/recovery/ml_revision')
HARNESS_FILES=['final_study.py','final_budget.py','test_final_study.py']


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def readlines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
def write_new(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def append(path,obj):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()


def source_digest():
    h=hashlib.sha256()
    for path in sorted(Path('mapf').glob('*.py')):h.update(path.name.encode());h.update(path.read_bytes())
    return h.hexdigest()


def freeze():
    plan=json.loads((ML/'final_study_plan.json').read_text())
    sealed=json.loads(Path('results/recovery/day3/sealed_test_manifest.json').read_text())
    assert plan['instances']==sealed['instances'] and len(plan['instances'])==54
    assert source_digest()==plan['source_sha256'], 'Approved solver source changed'
    assert plan['policies']==['cbs','plain','residual','pool_random','model_a','model_b']
    assert plan['repetitions']==2 and plan['timeout_seconds']==15 and plan['total_runs']==648
    assert len({i['id'] for i in plan['instances']})==54
    assert len({i['family_id'] for i in plan['instances']})==18
    test_families={i['family_id'] for i in plan['instances']}
    for i in plan['instances']:
        c=dict(i);expected=c.pop('instance_sha256')
        assert digest(c)==expected and sha(i['map_path'])==i['map_sha256']
        assert i['split']=='test' and 24<=int(i['family_id'][-2:])<30
    models={name:(ML/(name+'.json')).as_posix() for name in ('model_a','model_b')}
    for name,path in models.items():
        assert sha(path)==plan['model_sha256'][name]
        metadata=json.loads(Path(path+'.metadata.json').read_text())
        assert not test_families & set(metadata['training_family_ids'])
        assert metadata['test_used'] is False and metadata['validation_used_for_fit'] is False
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=dict(plan,status='APPROVED_FROZEN_BEFORE_TEST',frozen_utc=utc(),model_paths=models,
        approved_plan_sha256=sha(ML/'final_study_plan.json'),sealed_manifest_sha256=sha('results/recovery/day3/sealed_test_manifest.json'),
        harness_sha256={name:sha(name) for name in HARNESS_FILES},
        model_metadata_sha256={name:sha(path+'.metadata.json') for name,path in models.items()},
        execution='serial; one pinned logical CPU per solver, inherited from final_budget; numerical threads one; standby prevention',
        primary_estimator='geometric mean of 18 family ratios; each family time first averages 3 agent-count variants x 2 timing repeats, failures assigned 15 seconds',
        bootstrap=dict(replicates=10000,seed=20261031,method='sample six families with replacement within each of three map strata'),
        no_fewer_solved_rule='Both solved-run count and count of instances solved in both repeats must be at least residual',
        memory_estimator='Within each family, ratio of mean peak process bytes over matched instances solved by both policies in both repeats; report median of family ratios, coverage and conditional bias',
        valid_statuses=['SOLVED','TIMEOUT','HARD_TIMEOUT','MEMORY_LIMIT'],
        timing_fault_rule='wall > cap + 1 second; preserve initial attempt and rerun all 12 runs for that instance once within remaining phase budget, otherwise report invalid timing')
    protocol['protocol_sha256']=digest(protocol)
    write_new(ROOT/'protocol.json',protocol)
    jobs=[dict(instance_id=i['id'],family_id=i['family_id'],map_name=i['map_name'],k=i['k'],repeat=r,policy=policy)
          for i in plan['instances'] for r in range(2) for policy in plan['policies']]
    random.Random(plan['schedule_seed']).shuffle(jobs)
    for index,job in enumerate(jobs):job['job_id']=f'job-{index:04d}';job['schedule_index']=index
    write_new(ROOT/'schedule.json',jobs)
    files=sorted(Path('mapf').glob('*.py'))+[Path(n) for n in HARNESS_FILES]
    with zipfile.ZipFile(ROOT/'frozen_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.as_posix())
    write_new(ROOT/'frozen_source_manifest.json',{p.as_posix():sha(p) for p in files})
    env=dict(python=sys.version,platform=sys.platform,logical_cpu_count=psutil.cpu_count(),
             controller_affinity=psutil.Process().cpu_affinity(),source_sha256=source_digest(),frozen_utc=protocol['frozen_utc'])
    write_new(ROOT/'environment.json',env)
    print(json.dumps(dict(protocol_sha256=protocol['protocol_sha256'],jobs=len(jobs),families=18,solver_source_unchanged=True)),flush=True)


def protocol(check_files=True):
    p=json.loads((ROOT/'protocol.json').read_text());c=dict(p);expected=c.pop('protocol_sha256')
    assert digest(c)==expected
    if check_files:
        assert source_digest()==p['source_sha256'], 'Solver files changed after freezing'
        for name,path in p['model_paths'].items():
            assert sha(path)==p['model_sha256'][name], 'Model changed after freezing'
            assert sha(path+'.metadata.json')==p['model_metadata_sha256'][name]
        assert all(sha(name)==expected for name,expected in p['harness_sha256'].items()), 'Study harness changed'
    return p


def run_isolated(spec,memory_mb=4096):
    with tempfile.TemporaryDirectory() as tmp:
        inp,out=Path(tmp)/'input.json',Path(tmp)/'output.json';inp.write_text(json.dumps(spec))
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        start_utc=utc();began=time.perf_counter()
        proc=subprocess.Popen([sys.executable,'-m','mapf.benchmark_worker',str(inp),str(out)],env=env,
                              stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        launched=time.perf_counter();monitor=psutil.Process(proc.pid)
        peak=0;cpu=0.;terminated=None;last_poll=launched;max_gap=0.;polls=0
        while proc.poll() is None:
            now=time.perf_counter();max_gap=max(max_gap,now-last_poll);last_poll=now;polls+=1
            try:
                info=monitor.memory_info();peak=max(peak,getattr(info,'peak_wset',info.rss))
                times=monitor.cpu_times();cpu=max(cpu,times.user+times.system)
            except psutil.NoSuchProcess:break
            if time.perf_counter()-began>=spec['timeout_seconds']:terminated='HARD_TIMEOUT'
            elif info.rss>memory_mb*1024**2:terminated='MEMORY_LIMIT'
            if terminated:proc.kill();break
            time.sleep(.01)
        before_reap=time.perf_counter();_,stderr=proc.communicate();finished=time.perf_counter()
        if terminated:result=dict(status=terminated,cost=None)
        elif proc.returncode or not out.exists():result=dict(status='PROCESS_ERROR',cost=None,detail=stderr.decode(errors='replace')[-2000:])
        else:result=json.loads(out.read_text())
        peak=max(peak,result.pop('process_peak_rss_bytes',0));cpu=max(cpu,result.pop('process_cpu_seconds',0))
        result.update(utc_started=start_utc,utc_finished=utc(),end_to_end_wall_seconds=finished-began,
            launch_seconds=launched-began,max_monitor_gap_seconds=max_gap,monitor_polls=polls,reap_seconds=finished-before_reap,
            process_cpu_seconds=cpu,peak_rss_bytes=peak,timeout_seconds=spec['timeout_seconds'],memory_limit_mb=memory_mb,
            peak_memory_method='Windows peak_wset' if os.name=='nt' else 'sampled RSS',
            cpu_measurement='worker total when completed; sampled lower bound when killed',
            cache_mode='cold isolated process',model_loading_included=True,
            timing_fault=finished-began>spec['timeout_seconds']+1)
        return result


def validate_result(row,instance):
    if row['status']!='SOLVED':return
    from mapf.grid import load_map,GridMap
    from mapf.validation import validate_solution
    grid=load_map(instance['map_path']) if 'map_path' in instance else GridMap(instance['width'],instance['height'])
    paths=[list(map(tuple,path)) for path in row['paths']]
    validate_solution(grid,list(map(tuple,instance['starts'])),list(map(tuple,instance['goals'])),paths,row['cost'])
    if not row.get('solution_validated'):raise ValueError('Worker did not validate its returned solution')
    row['controller_solution_validated']=True


def run_jobs(p,jobs,path,attempt):
    instances={i['id']:i for i in p['instances']};existing=readlines(path)
    assert len({r['job_id'] for r in existing})==len(existing)
    done={r['job_id'] for r in existing};source=p['source_sha256']
    for job in jobs:
        if job['job_id'] in done:continue
        if job['schedule_index']%36==0:protocol()
        for name,model in p['model_paths'].items():assert sha(model)==p['model_sha256'][name]
        ins=instances[job['instance_id']]
        spec=dict(instance=ins,policy=job['policy'],seed=p['random_seed'],timeout_seconds=p['timeout_seconds'],model_paths=p['model_paths'])
        row=run_isolated(spec,p['memory_mb'])
        row.update(job,attempt=attempt,protocol_sha256=p['protocol_sha256'],source_sha256=source,instance_sha256=ins['instance_sha256'])
        try:validate_result(row,ins)
        except Exception as exc:
            row['controller_validation_error']=f'{type(exc).__name__}: {exc}';append(path,row);raise
        append(path,row);done.add(job['job_id'])
        print(json.dumps(dict(completed=len(done),total=len(jobs),job_id=job['job_id'],instance=ins['id'],policy=job['policy'],
            repeat=job['repeat'],status=row['status'],wall=round(row['end_to_end_wall_seconds'],3),timing_fault=row['timing_fault'])),flush=True)
        if row['status'] not in p['valid_statuses']:
            raise ValueError('Unexpected worker failure; preserve results and investigate without test-driven tuning')
    protocol()


def run():
    p=protocol();jobs=json.loads((ROOT/'schedule.json').read_text())
    run_jobs(p,jobs,ROOT/'initial_runs.jsonl','initial')


def timing_plan():
    p=protocol();rows=readlines(ROOT/'initial_runs.jsonl');jobs=json.loads((ROOT/'schedule.json').read_text())
    if len(rows)!=len(jobs):raise ValueError('Initial matrix incomplete')
    affected={r['instance_id'] for r in rows if r['timing_fault']}
    replacements=[j for j in jobs if j['instance_id'] in affected]
    random.Random(p['schedule_seed']+1).shuffle(replacements)
    data=dict(affected_instance_ids=sorted(affected),fault_job_ids=[r['job_id'] for r in rows if r['timing_fault']],
        replacement_jobs=replacements,replacement_count=len(replacements),nominal_seconds=len(replacements)*p['timeout_seconds'],
        rule=p['timing_fault_rule'],created_utc=utc())
    write_new(ROOT/'timing_plan.json',data);print(json.dumps({k:v for k,v in data.items() if k!='replacement_jobs'}),flush=True)


def repair():
    p=protocol();plan=json.loads((ROOT/'timing_plan.json').read_text())
    run_jobs(p,plan['replacement_jobs'],ROOT/'replacement_runs.jsonl','replacement')


def analysis_rows(p):
    initial=readlines(ROOT/'initial_runs.jsonl');schedule=json.loads((ROOT/'schedule.json').read_text())
    assert len(initial)==len(schedule)==648 and len({r['job_id'] for r in initial})==648
    expected={j['job_id']:j for j in schedule}
    for r in initial:
        assert all(r[k]==expected[r['job_id']][k] for k in expected[r['job_id']])
    affected={r['instance_id'] for r in initial if r['timing_fault']}
    replacements=readlines(ROOT/'replacement_runs.jsonl');index={r['job_id']:r for r in replacements}
    assert len(index)==len(replacements)
    required={r['job_id'] for r in initial if r['instance_id'] in affected}
    complete_repair=set(index)==required
    rows=[index.get(r['job_id'],r) for r in initial]
    valid=complete_repair and not any(r['timing_fault'] for r in rows)
    return rows,dict(initial_faults=sum(r['timing_fault'] for r in initial),affected_instance_ids=sorted(affected),
        replacement_count=len(replacements),complete_block_repair=complete_repair,timing_valid=valid)


def capped(row,cap):return min(cap,row['end_to_end_wall_seconds']) if row['status']=='SOLVED' else cap


def family_ratios(rows,policy,baseline,cap):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['policy'] in (policy,baseline):grouped[r['family_id']][r['policy']].append(r)
    out=[]
    for family,group in sorted(grouped.items()):
        a,b=group[policy],group[baseline]
        keys=lambda rr:{(r['instance_id'],r['repeat']) for r in rr}
        assert keys(a)==keys(b)
        ta=statistics.mean(capped(r,cap) for r in a);tb=statistics.mean(capped(r,cap) for r in b)
        out.append(dict(family_id=family,map_name=a[0]['map_name'],policy_mean_capped_seconds=ta,
            baseline_mean_capped_seconds=tb,time_ratio=ta/tb,solved_runs=sum(r['status']=='SOLVED' for r in a),
            baseline_solved_runs=sum(r['status']=='SOLVED' for r in b)))
    return out


def stratified_interval(families,replicates=10000,seed=20261031):
    import numpy as np
    strata=defaultdict(list)
    for r in families:strata[r['map_name']].append(np.log(r['time_ratio']))
    rng=np.random.default_rng(seed);sums=np.zeros(replicates);count=0
    for name,values in sorted(strata.items()):
        v=np.asarray(values);count+=len(v)
        sums+=v[rng.integers(0,len(v),(replicates,len(v)))].sum(axis=1)
    return np.quantile(np.exp(sums/count),[.025,.975]).tolist()


def compare(rows,policy,base,p):
    import math
    cap=p['timeout_seconds'];families=family_ratios(rows,policy,base,cap)
    runs=[r for r in rows if r['policy']==policy];base_runs=[r for r in rows if r['policy']==base]
    by_ins=defaultdict(list);base_ins=defaultdict(list)
    for r in runs:by_ins[r['instance_id']].append(r)
    for r in base_runs:base_ins[r['instance_id']].append(r)
    solved=[r for r in runs if r['status']=='SOLVED'];memory_families=defaultdict(list);matched_rows=[]
    for ins,group in sorted(by_ins.items()):
        other=base_ins[ins]
        if all(r['status']=='SOLVED' for r in group+other):
            a=statistics.mean(r['peak_rss_bytes'] for r in group);b=statistics.mean(r['peak_rss_bytes'] for r in other)
            match=dict(instance_id=ins,family_id=group[0]['family_id'],map_name=group[0]['map_name'],
                policy_peak_bytes=a,baseline_peak_bytes=b,memory_ratio=a/b,
                policy_expansions=statistics.mean(r['nodes_expanded'] for r in group),
                baseline_expansions=statistics.mean(r['nodes_expanded'] for r in other),
                policy_seconds=statistics.mean(r['end_to_end_wall_seconds'] for r in group),
                baseline_seconds=statistics.mean(r['end_to_end_wall_seconds'] for r in other))
            matched_rows.append(match);memory_families[group[0]['family_id']].append(match)
    memory=[]
    for family,group in sorted(memory_families.items()):
        memory.append(dict(family_id=family,map_name=group[0]['map_name'],matched_instances=len(group),
            memory_ratio=statistics.mean(r['policy_peak_bytes'] for r in group)/statistics.mean(r['baseline_peak_bytes'] for r in group)))
    boot=p['bootstrap'];ci=stratified_interval(families,boot['replicates'],boot['seed'])
    solved_both=sum(all(r['status']=='SOLVED' for r in group) for group in by_ins.values())
    base_solved_both=sum(all(r['status']=='SOLVED' for r in group) for group in base_ins.values())
    no_fewer=len(solved)>=sum(r['status']=='SOLVED' for r in base_runs) and solved_both>=base_solved_both
    stats=[r['selector_stats'] for r in runs if 'selector_stats' in r]
    subgroups={}
    for field in ('map_name','k'):
        subgroups[field]={}
        for value in sorted({r[field] for r in runs}):
            sub=[r for r in runs if r[field]==value]
            subgroups[field][str(value)]=dict(runs=len(sub),solved_runs=sum(r['status']=='SOLVED' for r in sub),
                mean_capped_seconds=statistics.mean(capped(r,cap) for r in sub))
    return dict(runs=len(runs),families=len(families),instances=len(by_ins),solved_runs=len(solved),
        solved_instances_both_repeats=solved_both,solved_instances_any_repeat=sum(any(r['status']=='SOLVED' for r in g) for g in by_ins.values()),
        status_counts=dict(Counter(r['status'] for r in runs)),mean_capped_seconds=statistics.mean(capped(r,cap) for r in runs),
        arithmetic_capped_time_ratio=statistics.mean(capped(r,cap) for r in runs)/statistics.mean(capped(r,cap) for r in base_runs),
        geometric_family_time_ratio=math.exp(statistics.mean(math.log(f['time_ratio']) for f in families)),
        stratified_family_bootstrap_95_interval=ci,family_time_wins=sum(f['time_ratio']<1 for f in families),
        family_time_losses=sum(f['time_ratio']>1 for f in families),family_time_ties=sum(f['time_ratio']==1 for f in families),
        no_fewer_solved=no_fewer,speed_claim_rule_passed=no_fewer and ci[1]<1,
        median_all_run_peak_memory_mib=statistics.median(r['peak_rss_bytes']/1024**2 for r in runs),
        matched_memory_families=len(memory),matched_memory_instances=len(matched_rows),
        median_matched_family_memory_ratio=statistics.median(r['memory_ratio'] for r in memory) if memory else None,
        memory_family_rows=memory,matched_instance_rows=matched_rows,family_rows=families,subgroups=subgroups,
        telemetry_runs=len(stats),selector_totals={k:sum(s[k] for s in stats) for k in stats[0]} if stats else {})


def analyze():
    p=protocol();rows,timing=analysis_rows(p);instances={i['id']:i for i in p['instances']}
    costs=defaultdict(set);validation_errors=[]
    for r in rows:
        assert r['protocol_sha256']==p['protocol_sha256'] and r['source_sha256']==p['source_sha256']
        try:validate_result(r,instances[r['instance_id']])
        except Exception as exc:validation_errors.append(dict(job_id=r['job_id'],error=str(exc)))
        if r['status']=='SOLVED':costs[r['instance_id']].add(r['cost'])
    disagreements={k:sorted(v) for k,v in costs.items() if len(v)>1}
    statuses_valid=all(r['status'] in p['valid_statuses'] for r in rows)
    valid=timing['timing_valid'] and statuses_valid and not validation_errors and not disagreements
    policies={name:compare(rows,name,'residual',p) for name in p['policies']}
    primary=policies['model_b']
    summary=dict(protocol_sha256=p['protocol_sha256'],analysis_utc=utc(),matrix_complete=True,total_runs=len(rows),
        independent_families=18,instances=54,valid_for_inference=valid,timing_integrity=timing,
        returned_solutions_revalidated=sum(r['status']=='SOLVED' for r in rows),validation_errors=validation_errors,
        solved_cost_disagreements=disagreements,policies=policies,
        primary_speed_claim_supported=valid and primary['speed_claim_rule_passed'],
        inference_scope='Frozen Python implementations on synthetic held-out scenario families from three known maps; not optimized published solvers or unseen maps',
        limitations=['Two repeats share algorithm seed and are timing observations, not independent samples',
            '15-second censoring; two failures tie at cap despite unknown latent solve times',
            'Family bootstrap has only six independent families per map; interval is approximate',
            'Peak memory includes interpreter/imports; matched completion analysis is selected and conditional',
            'Secondary policy comparisons are exploratory, without multiplicity adjustment',
            'Hard-killed workers have missing telemetry, not zero inference cost'])
    write_new(ROOT/'summary.json',summary)
    with (ROOT/'analysis_rows.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    print(json.dumps(dict(valid=valid,primary_speed_claim_supported=summary['primary_speed_claim_supported'],
        policies={name:{k:s[k] for k in ('solved_runs','solved_instances_both_repeats','mean_capped_seconds',
            'geometric_family_time_ratio','stratified_family_bootstrap_95_interval','median_matched_family_memory_ratio')} for name,s in policies.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run','timing-plan','repair','analyze'])
    stage=parser.parse_args().stage
    {'freeze':freeze,'run':run,'timing-plan':timing_plan,'repair':repair,'analyze':analyze}[stage]()

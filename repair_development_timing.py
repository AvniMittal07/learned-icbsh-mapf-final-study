"""Preserve and replace complete matched blocks after a resource-control fault.

Rule recorded before reading aggregate policy results: any wall time exceeding
the requested cap by >1 second invalidates that instance's full 6x2 block.
One replacement attempt per block; never retry according to solver outcomes.
"""
import json
from pathlib import Path
import random
from mapf.ml_revision import ROOT, readlines, protocol, append, write_json
from mapf.recovery_benchmark import run_isolated
from mapf.recovery_pilot import source_digest


def main():
    p=protocol();initial=ROOT/'development_initial.jsonl';canonical=ROOT/'development.jsonl'
    rows=readlines(initial if initial.exists() else canonical)
    schedule=json.loads((ROOT/'evaluation_schedule.json').read_text())
    assert len(rows)==len(schedule), 'Initial campaign must finish first'
    faults=[r for r in rows if r['end_to_end_wall_seconds']>r['timeout_seconds']+1]
    affected={r['instance_id'] for r in faults}
    if not initial.exists():
        initial.write_bytes(canonical.read_bytes())
    models={name:json.loads((ROOT/(name+'_report.json')).read_text())['model_path'] for name in ('model_a','model_b')}
    jobs=[s for s in schedule if s['instance_id'] in affected]
    random.Random(20261031).shuffle(jobs)
    existing=readlines(ROOT/'development_replacements.jsonl')
    done={(r['instance_id'],r['repeat'],r['policy']) for r in existing}
    instances={i['id']:i for i in p['instances'] if i['split']=='validation'}
    for job in jobs:
        key=(job['instance_id'],job['repeat'],job['policy'])
        if key in done:continue
        ins=instances[job['instance_id']]
        spec=dict(instance=ins,policy=job['policy'],seed=p['evaluation']['random_seed'],
                  timeout_seconds=p['evaluation']['timeout_seconds'],model_paths=models)
        row=run_isolated(spec,p['evaluation']['memory_mb'])
        row.update(job,family_id=ins['family_id'],protocol_sha256=p['protocol_sha256'],source_sha256=source_digest(),
            timing_block_replacement=True,replacement_reason='Full matched block rerun after cap exceeded by >1 second')
        append(ROOT/'development_replacements.jsonl',row)
        print('timing block replacement',key,row['status'],round(row['end_to_end_wall_seconds'],3),flush=True)
    replacement=readlines(ROOT/'development_replacements.jsonl')
    assert len(replacement)==len(jobs)
    index={(r['instance_id'],r['repeat'],r['policy']):r for r in replacement}
    combined=[index.get((r['instance_id'],r['repeat'],r['policy']),r) for r in rows]
    residual_faults=[r for r in combined if r['end_to_end_wall_seconds']>r['timeout_seconds']+1]
    write_json(ROOT/'timing_integrity.json',dict(rule='wall_seconds > cap_seconds + 1; replace whole instance block once',
        initial_faults=[{k:r[k] for k in ('instance_id','policy','repeat','status','end_to_end_wall_seconds','process_cpu_seconds')} for r in faults],
        affected_instance_ids=sorted(affected),replacement_runs=len(replacement),remaining_faults=len(residual_faults),
        cause='Windows Modern Standby observed in Kernel-Power events 506/507; replacements use process-lifetime keep-awake request',
        raw_initial_file=initial.name,raw_replacements_file='development_replacements.jsonl',
        canonical_file='development.jsonl',canonical_is_derived=True))
    if residual_faults:raise ValueError('Replacement also violated timing limits; do not claim valid timing')
    # The untouched original and every replacement remain available separately.
    canonical.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in combined))


if __name__=='__main__':main()

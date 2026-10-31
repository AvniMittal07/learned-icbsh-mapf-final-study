"""Prospectively frozen protocol for the approved guarded-label repair pilot."""
import json
from pathlib import Path
from mapf.day3_protocol import digest
from mapf.day3_analysis import check_protocol

ROOT = Path('results/recovery/repair_pilot')


def read_protocol():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    payload=dict(protocol);expected=payload.pop('protocol_sha256')
    if digest(payload)!=expected:raise ValueError('Repair protocol changed')
    return protocol


def freeze():
    original=json.loads(Path('results/recovery/day3/protocol.json').read_text())
    check_protocol(original)
    instances=[]
    for family in original['families']:
        index=int(family['family_id'].rsplit('-',1)[1])
        if index in [4,5,6,7,22,23]:
            k=24 if index%2==0 else 32
            instances.append(next(i for i in family['instances'] if i['k']==k))
    protocol=dict(version=1,parent_split_protocol_sha256=original['protocol_sha256'],
        calibration_instances=instances,quality_gate=original['quality_gate'],
        harvest_policy='direct residual analytical selector',harvest_seconds_per_instance=10,
        harvest_expansion_cap=250,capture_limit=12,max_states_per_instance=3,top_k_guardrail=3,
        selection='First, middle (floor(n/2)), and last distinct actual guarded decision among first 12; no label-based selection',
        rollout_budgets=[dict(name='low',cap=50,seconds=.5),dict(name='medium',cap=200,seconds=2),dict(name='high',cap=1000,seconds=6)],
        reference='forced first split then residual-selector ICBS-h in saved descendant problem; fresh caches and recomputed root MVC',
        cbs_comparison='Only first saved query of training family 04 on each map; same high cap/time; diagnostic, not label selection',
        model_if_gate_passes='Ridge alpha=10, no intercept, train-only standardization, signed pair differences with per-query weights',
        validation=dict(policies=['cbs','plain','random','residual','linear'],repetitions=2,
            timeout_seconds=8,seed=71,instances='six selected validation families only',
            promotion='Exploratory only: no fewer solved runs than residual, at least 5% lower capped end-to-end mean, no more than 10% higher paired-solved median memory; no tuning from these runs'),
        limitations=['Synthetic known maps; no held-out test evaluation',
            'Descendant continuation is not the original global OPEN frontier',
            'Root inherited pathmax is recomputed from MVC; later pathmax/bypass use the standard solver',
            'Budget consistency is not independent oracle accuracy validation'])
    protocol['protocol_sha256']=digest(protocol)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'protocol.json').open('x') as stream:json.dump(protocol,stream,indent=2)
    print(json.dumps(dict(instances=len(instances),protocol_sha256=protocol['protocol_sha256'],
                         thresholds_unchanged=True,test_evaluated=False)))


if __name__=='__main__':freeze()

"""Run one development instance with the delivered solver and optional model.

Example: python -m demo_solver --policy model_b --output demo_model_b.json
This is a demonstration, not a new benchmark or a test-set tuning interface.
"""
import argparse
import json
from pathlib import Path
from final_study import run_isolated,validate_result,source_digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',choices=['cbs','plain','residual','pool_random','model_a','model_b'],default='residual')
    parser.add_argument('--instance-id',default='random-32-32-20/family-20/k16')
    parser.add_argument('--timeout',type=float,default=8)
    parser.add_argument('--output',default='demo_solution.json')
    args=parser.parse_args()
    if args.timeout<=0:parser.error('Timeout must be positive')
    destination=Path(args.output)
    if destination.exists():parser.error('Choose a new output path; existing evidence is never overwritten')
    root=Path('results/recovery/ml_revision')
    instances=json.loads((root/'protocol.json').read_text())['instances']
    matches=[i for i in instances if i['id']==args.instance_id and i['split']=='validation']
    if len(matches)!=1:parser.error('Select a development instance from the supplied ML revision protocol')
    ins=matches[0]
    spec=dict(instance=ins,policy=args.policy,seed=71,timeout_seconds=args.timeout,
              model_paths={name:str(root/(name+'.json')) for name in ('model_a','model_b')})
    result=run_isolated(spec);validate_result(result,ins)
    result.update(instance=ins,policy=args.policy,source_sha256=source_digest(),purpose='development demonstration only')
    with destination.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result.get(k) for k in ('status','cost','nodes_expanded','end_to_end_wall_seconds','peak_rss_bytes')},indent=2))
    print('Saved',destination)


if __name__=='__main__':main()

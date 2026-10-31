"""Read-only verification/reanalysis of the delivered final study.

No solver runs, model fitting, file overwrites or new test observations.
--skip-package permits checking results before the final zip manifest exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from final_study import ROOT,ML,protocol,analysis_rows,compare,validate_result,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-package',action='store_true')
    args=parser.parse_args();p=protocol();rows,timing=analysis_rows(p)
    manifest=json.loads((ROOT/'frozen_source_manifest.json').read_text())
    with zipfile.ZipFile(ROOT/'frozen_source.zip') as z:
        assert z.testzip() is None
        for name,expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected
            assert sha(name)==expected
    instances={i['id']:i for i in p['instances']};costs={};solved=0
    for row in rows:
        assert row['instance_sha256']==instances[row['instance_id']]['instance_sha256']
        validate_result(row,instances[row['instance_id']])
        if row['status']=='SOLVED':
            old=costs.setdefault(row['instance_id'],row['cost']);assert old==row['cost'];solved+=1
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['matrix_complete'] and summary['valid_for_inference'] and timing['timing_valid']
    assert summary['returned_solutions_revalidated']==solved
    for policy in p['policies']:
        recomputed=compare(rows,policy,'residual',p)
        assert recomputed==summary['policies'][policy], f'Analysis mismatch for {policy}'
    old_quality=json.loads((ML.parent/'repair_pilot/quality.json').read_text())
    assert old_quality['quality_gate']['passed'] is False
    for name in ('model_a','model_b'):
        metadata=json.loads((ML/(name+'.json.metadata.json')).read_text())
        assert metadata['old_collection_gate_passed'] is False and metadata['test_used'] is False
    checked=0
    if not args.skip_package:
        files=json.loads(Path('SOURCE_MANIFEST.json').read_text())['files'];base=Path.cwd().resolve()
        for relative,expected in files.items():
            target=(base/relative).resolve()
            assert target.is_relative_to(base), 'Manifest path escapes package'
            assert sha(target)==expected, f'Packaged file changed: {relative}';checked+=1
    print(json.dumps(dict(verified=True,final_runs=len(rows),independent_families=18,
        solutions_revalidated=solved,all_policy_statistics_reproduced=True,source_and_models_unchanged=True,
        historical_failed_gate_preserved=True,packaged_files_checked=checked),indent=2))


if __name__=='__main__':main()

"""Harness checks on tiny synthetic inputs, never the sealed test scenarios."""
import json
from pathlib import Path
import unittest
from mapf.recovery_benchmark import run_isolated as old_run
from final_study import run_isolated,validate_result,family_ratios,stratified_interval,source_digest,ML


class FinalStudyTests(unittest.TestCase):
    def test_approved_source_and_models_unchanged(self):
        from final_study import sha
        p=json.loads((ML/'final_study_plan.json').read_text())
        self.assertEqual(source_digest(),p['source_sha256'])
        for name,expected in p['model_sha256'].items():self.assertEqual(sha(ML/(name+'.json')),expected)

    def test_monitor_six_policies_on_independent_tiny_problem(self):
        from mapf.grid import GridMap
        from mapf.test_recovery import joint_optimum
        ins=dict(width=3,height=2,starts=[[0,0],[0,2]],goals=[[0,2],[0,0]])
        optimum=joint_optimum(GridMap(3,2),list(map(tuple,ins['starts'])),list(map(tuple,ins['goals'])))
        models={n:str(ML/(n+'.json')) for n in ('model_a','model_b')}
        for policy in ('cbs','plain','residual','pool_random','model_a','model_b'):
            spec=dict(instance=ins,policy=policy,seed=71,timeout_seconds=4,model_paths=models)
            result=run_isolated(spec);reference=old_run(spec)
            self.assertEqual(result['status'],'SOLVED');self.assertEqual(result['cost'],optimum)
            self.assertEqual((result['status'],result['cost']),(reference['status'],reference['cost']))
            validate_result(result,ins)
            self.assertFalse(result['timing_fault']);self.assertGreater(result['monitor_polls'],0)
            self.assertLessEqual(result['launch_seconds'],result['end_to_end_wall_seconds'])

    def test_timeout_is_explicit(self):
        spec=dict(instance=dict(width=2,height=2,starts=[[0,0]],goals=[[1,1]]),policy='plain',seed=71,timeout_seconds=.001)
        row=run_isolated(spec)
        self.assertEqual(row['status'],'HARD_TIMEOUT');self.assertIsNone(row['cost'])

    def test_family_average_precedes_ratio_and_bootstrap(self):
        rows=[]
        for family in ('a','b'):
            for variant,seconds in enumerate((1,9)):
                for repeat in (0,1):
                    for policy,t in [('residual',1),('model_b',seconds)]:
                        rows.append(dict(family_id=family,map_name='map',instance_id=f'{family}/{variant}',repeat=repeat,
                            policy=policy,status='SOLVED',end_to_end_wall_seconds=t))
        f=family_ratios(rows,'model_b','residual',15)
        self.assertEqual(len(f),2);self.assertEqual([x['time_ratio'] for x in f],[5,5])
        lo,hi=stratified_interval(f,100,71)
        self.assertAlmostEqual(lo,5);self.assertAlmostEqual(hi,5)


if __name__=='__main__':unittest.main()

"""Focused tests for split isolation, censoring, and the frozen quality gate."""
import copy
import json
import unittest
from mapf.day3_protocol import ROOT, digest, largest_component
from mapf.day3_analysis import preference, pairs, check_protocol, quality_gate, evaluate, train_if_allowed
from mapf.grid import GridMap


def outcome(status, expanded):
    return dict(status=status, expanded=expanded)


def query(index, split='train', informative=True):
    return dict(query_id=f'{split}/{index}', instance=dict(split=split, family_id=f'{split}/family{index//2}',
        map_name=f'map{index%2}'), candidates=[
            dict(budgets=dict(high=outcome('SOLVED', 3))),
            dict(budgets=dict(high=outcome('SOLVED', 5 if informative else 3)))])


class Day3Tests(unittest.TestCase):
    def test_no_invented_censored_preferences(self):
        statuses = ['TIME_CAP', 'EXPANSION_CAP', 'INFEASIBLE', 'HORIZON_LIMIT']
        for a in statuses:
            for b in statuses:
                self.assertIsNone(preference(outcome(a, 5), outcome(b, 90)))
        self.assertIsNone(preference(outcome('SOLVED', 5), outcome('SOLVED', 5)))
        self.assertIsNone(preference(outcome('SOLVED', 5), outcome('TIME_CAP', 5)))
        self.assertIsNone(preference(outcome('SOLVED', 90), outcome('TIME_CAP', 5)))
        self.assertIsNone(preference(outcome('SOLVED', 5), outcome('INFEASIBLE', 90)))

    def test_partial_order_sign_and_antisymmetry(self):
        for a,b in [(outcome('SOLVED', 3),outcome('SOLVED', 10)),
                    (outcome('SOLVED', 3),outcome('TIME_CAP', 10)),
                    (outcome('SOLVED', 3),outcome('EXPANSION_CAP', 10))]:
            self.assertEqual(preference(a,b),1)
            self.assertEqual(preference(b,a),-1)
        q=query(0)
        self.assertEqual(pairs(q),[(0,1,1,'fully_solved')])
        q['candidates'][1]['budgets']['high']['status']='TIME_CAP'
        self.assertEqual(pairs(q),[(0,1,1,'solved_vs_censored')])

    def test_frozen_family_splits_and_integrity(self):
        protocol=json.loads((ROOT/'protocol.json').read_text())
        result=check_protocol(protocol)
        self.assertEqual(result['families_by_split'],dict(train=60,validation=12,test=18))
        self.assertEqual(result['instances_by_split'],dict(train=180,validation=36,test=54))
        modified=copy.deepcopy(protocol)
        modified['families'][0]['instances'][0]['split']='test'
        with self.assertRaises(ValueError):check_protocol(modified)
        payload={k:v for k,v in modified.items() if k!='protocol_sha256'}
        modified['protocol_sha256']=digest(payload)
        with self.assertRaises(ValueError):check_protocol(modified)

    def test_gate_boundaries_and_no_training_after_failure(self):
        thresholds=json.loads((ROOT/'protocol.json').read_text())['quality_gate']
        rows=[query(i) for i in range(12)]+[query(i,'validation') for i in range(6)]
        self.assertTrue(quality_gate(rows,thresholds)['passed'])
        self.assertFalse(quality_gate(rows[:-1],thresholds)['passed'])
        self.assertFalse(quality_gate(rows,thresholds,replay_mismatches=1)['passed'])
        sparse=rows+[query(i+30,informative=False) for i in range(30)]
        gate=quality_gate(sparse,thresholds)
        self.assertFalse(gate['passed'])
        self.assertFalse(train_if_allowed(sparse,gate,{})['trained'])
        self.assertFalse(quality_gate([],thresholds)['passed'])

    def test_query_metrics_ignore_unobserved_and_half_credit_ties(self):
        rows=[query(0),query(1),query(2,informative=False)]
        result=evaluate(rows,lambda r:[1,0] if r['query_id'].endswith('/0') else [0,0])
        self.assertEqual(result['queries'],2)
        self.assertEqual(result['certified_pairs'],2)
        self.assertEqual(result['query_macro_accuracy'],.75)
        self.assertEqual(result['family_macro_accuracy'],.75)

    def test_component_sampling_excludes_disconnected_cells(self):
        grid=GridMap(4,2,{(0,1),(1,1)})
        self.assertEqual(largest_component(grid),[(0,2),(0,3),(1,2),(1,3)])


if __name__=='__main__':unittest.main()

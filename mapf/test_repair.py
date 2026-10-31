"""Continuation correctness and deployed guard regression tests."""
import copy
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from mapf.baselines import guarded_shortlist, GuardedLearnedSelector, f_aware_residual_selector
from mapf.grid import GridMap
from mapf.low_level import ConstraintSet, space_time_astar
from mapf.cardinal import classify_conflicts
from mapf.conflict_graph import compute_h4_and_graph
from mapf.icbs_h import ICBShSolver, ICBSNode
from mapf.conflicts import Conflict
from mapf.validation import validate_solution, validate_constrained_paths
from mapf.test_recovery import joint_optimum
from mapf.repair_pilot import eligible_tier, icbs_rollout


class RepairTests(unittest.TestCase):
    def test_guard_order_unique_and_model_input(self):
        candidates=[SimpleNamespace(name=n) for n in 'abcde']
        keys=dict(a=(0,1,0,0,9,0,0),b=(1,1,0,0,2,0,0),c=(1,1,0,0,3,0,0),
                  d=(1,1,0,0,3,0,0),e=(1,1,0,0,1,0,0))
        with patch('mapf.baselines._compute_candidate_lookahead_tuple',side_effect=lambda c,ctx:keys[c.name]):
            shortlist,unique=guarded_shortlist(candidates,{})
            self.assertEqual([c.name for c in shortlist],list('cdb'));self.assertFalse(unique)
            self.assertEqual([c.name for c in guarded_shortlist(candidates,{},None)[0]],list('cdbe'))
            seen=[]
            extractor=SimpleNamespace(goals=[(0,0)],extract_all=lambda cc,ctx:(seen.extend(c.name for c in cc) or np.eye(3)))
            selector=GuardedLearnedSelector(SimpleNamespace(predict=lambda X:np.array([0,1,0])))
            selector._get_extractor=lambda grid:extractor
            self.assertEqual(selector(candidates,{'grid':None}).name,'d')
            self.assertEqual(seen,list('cdb'));self.assertEqual(selector.ml_tiebreaks,1)
            keys['b']=(2,1,0,0,0,0,0)
            self.assertEqual(selector(candidates,{'grid':None}).name,'b')
            self.assertEqual(selector.analytical_shortcircuits,1)
            self.assertEqual(seen,list('cdb'))

    def test_forced_continuations_match_independent_optima(self):
        grid=GridMap(2,2);cells=[(0,0),(0,1),(1,0),(1,1)];comparisons=0
        for starts in itertools.permutations(cells,2):
            for goals in itertools.permutations(cells,2):
                cs=[ConstraintSet(),ConstraintSet()]
                paths=[space_time_astar(grid,s,g,c) for s,g,c in zip(starts,goals,cs)]
                classified=classify_conflicts(grid,list(starts),list(goals),paths,cs)
                if not classified:continue
                h,_=compute_h4_and_graph(classified)
                # Deliberately untrusted saved h is ignored on continuation.
                node=ICBSNode(cs,paths,sum(len(p)-1 for p in paths),h+100)
                original=copy.deepcopy(vars(node))
                for cc in eligible_tier(classified):
                    progress={}
                    result=ICBShSolver(grid,conflict_selector=f_aware_residual_selector).solve(list(starts),list(goals),
                        initial_node=node,forced_first_conflict=cc.conflict,progress=progress,timeout=3)
                    self.assertEqual(result['status'],'SOLVED')
                    self.assertEqual(result['cost'],joint_optimum(grid,starts,goals))
                    self.assertTrue(progress['forced_split_applied'])
                    self.assertEqual(progress['nodes_expanded'],result['nodes_expanded'])
                    validate_solution(grid,starts,goals,result['paths'],result['cost']);comparisons+=1
                self.assertEqual(node.paths,original['paths'])
                self.assertEqual([c.freeze_key() for c in node.constraints],[c.freeze_key() for c in original['constraints']])
        self.assertGreater(comparisons,20)

    def test_nonroot_continuation_matches_constrained_solve(self):
        grid=GridMap(3,3);starts=[(1,0),(1,2),(0,1)];goals=[(1,2),(1,0),(2,1)];saved=[]
        def capture(node,cc,graph):
            if node.depth>0:saved.append((copy.deepcopy(node),cc[0].conflict))
        ICBShSolver(grid,enable_bypass=False).solve(starts,goals,on_node_callback=capture)
        self.assertTrue(saved)
        for node,_ in saved[:5]:
            eligible=eligible_tier(classify_conflicts(grid,starts,goals,node.paths,node.constraints))
            expected=ICBShSolver(grid).solve(starts,goals,initial_constraints=node.constraints)
            actual=ICBShSolver(grid).solve(starts,goals,initial_node=node,forced_first_conflict=eligible[0].conflict)
            self.assertEqual(actual['status'],expected['status']);self.assertEqual(actual['cost'],expected['cost'])
            validate_constrained_paths(grid,starts,goals,actual['paths'],node.constraints)

    def test_invalid_saved_paths_and_goal_constraints_rejected(self):
        grid=GridMap(2,2);starts=[(0,0)];goals=[(0,1)]
        for paths,cs in [([[(0,0),(1,1),(0,1)]],[ConstraintSet()]),
                         ([[(0,0),(0,0),(0,1)]],[ConstraintSet()]),
                         ([[(0,0),(0,1)]],[ConstraintSet(vertex_constraints={(0,1,10)})]),
                         ([[(0,0),(0,1)]],[ConstraintSet(edge_constraints={(0,1,0,1,10)})])]:
            node=ICBSNode(cs,paths,len(paths[0])-1,0)
            with self.assertRaises(ValueError):ICBShSolver(grid).solve(starts,goals,initial_node=node)

    def test_forced_action_and_cap_are_explicit(self):
        grid=GridMap(3,3);starts=[(1,0),(1,2),(0,1)];goals=[(1,2),(1,0),(2,1)]
        paths=[[s,(1,1),g] for s,g in zip(starts,goals)];cs=[ConstraintSet() for _ in starts]
        cc=eligible_tier(classify_conflicts(grid,starts,goals,paths,cs))[0]
        node=ICBSNode(cs,paths,6,0)
        progress={}
        result=ICBShSolver(grid).solve(starts,goals,initial_node=node,forced_first_conflict=cc.conflict,
                                     max_expansions=1,progress=progress)
        self.assertEqual(result['status'],'MAX_EXPANSIONS');self.assertEqual(progress['nodes_expanded'],1)
        self.assertTrue(progress['forced_split_applied'])
        result=ICBShSolver(grid).solve(starts,goals,initial_node=node,forced_first_conflict=cc.conflict,
                                     timeout=0,progress=progress)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertEqual(progress,dict(nodes_expanded=0,forced_split_applied=False))
        with self.assertRaises(ValueError):ICBShSolver(grid).solve(starts,goals,initial_node=node,
            forced_first_conflict=Conflict(0,1,999,'VERTEX',(1,1)))


if __name__=='__main__':unittest.main()

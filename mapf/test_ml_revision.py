"""Teacher/solver equivalence, pool control, bounds, and tiny optimality checks."""
import copy
import heapq
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from mapf.grid import GridMap
from mapf.low_level import ConstraintSet, space_time_astar
from mapf.cardinal import classify_conflicts
from mapf.conflict_graph import compute_h4_and_graph
from mapf.icbs_h import ICBShSolver, ICBSNode
from mapf.conflicts import Conflict
from mapf.test_recovery import joint_optimum
from mapf.validation import validate_solution
from mapf.repair_pilot import eligible_tier
from mapf.ml_revision_policy import candidate_pool, immediate_children, teacher_key, RevisionSelector
from mapf.ml_revision import context, teacher_pairs


class RevisionTests(unittest.TestCase):
    def test_pool_always_contains_residual_winner_and_is_deterministic(self):
        cs=[SimpleNamespace(conflict=Conflict(i,i+1,i,'VERTEX',(0,0))) for i in range(10)]
        with patch('mapf.ml_revision_policy.f_aware_residual_selector',return_value=cs[-1]):
            pool=candidate_pool(cs,{})
            self.assertIs(pool[0],cs[-1]);self.assertEqual(len(pool),6)
            self.assertEqual(pool,candidate_pool(list(reversed(cs)),{}))
            self.assertEqual(len({c.conflict for c in pool}),6)

    def test_teacher_matches_actual_root_children(self):
        grid=GridMap(3,3);starts=[(1,0),(1,2),(0,1)];goals=[(1,2),(1,0),(2,1)]
        saved=[];pushed=[];original=heapq.heappush
        def capture(node,classified,graph):saved.append((copy.deepcopy(node),classified,graph))
        def push(heap,node):
            if isinstance(node,ICBSNode):pushed.append(copy.deepcopy(node))
            original(heap,node)
        with patch('mapf.icbs_h.heapq.heappush',side_effect=push):
            ICBShSolver(grid,enable_bypass=False).solve(starts,goals,max_expansions=1,on_node_callback=capture)
        node,classified,graph=saved[0];chosen=eligible_tier(classified)[0]
        label=immediate_children(chosen,context(node,grid,starts,goals,graph))
        children=[n for n in pushed if n.depth==1]
        self.assertEqual(label['status'],'COMPLETE');self.assertEqual(len(children),2)
        for branch,child in zip(label['branches'],children):
            self.assertEqual(branch['g'],child.g);self.assertEqual(branch['effective_f'],child.f)
            self.assertEqual(branch['replanned_path'],child.paths[branch['agent']])
            # Brute-force vertex cover independent of MVC implementation.
            ccs=classify_conflicts(grid,starts,goals,child.paths,child.constraints)
            edges={(c.conflict.agent_i,c.conflict.agent_j) for c in ccs if c.classification=='CARDINAL'}
            mvc=min(len(s) for n in range(len(starts)+1) for s in itertools.combinations(range(len(starts)),n)
                    if all(i in s or j in s for i,j in edges))
            self.assertEqual(branch['raw_h'],mvc)
        raised=context(node,grid,starts,goals,graph);raised['effective_h']+=20
        raised_label=immediate_children(chosen,raised)
        self.assertTrue(all(b['effective_f']>=node.f+20 for b in raised_label['branches']))

    def test_unknown_and_infeasibility_are_not_finite_penalties(self):
        grid=GridMap(2,2);starts=[(0,0),(0,0)];goals=[(0,1),(1,0)]
        paths=[[starts[i],goals[i]] for i in range(2)];cs=[ConstraintSet(),ConstraintSet()]
        classified=classify_conflicts(grid,starts,goals,paths,cs);h,graph=compute_h4_and_graph(classified)
        node=ICBSNode(cs,paths,2,h);ctx=context(node,grid,starts,goals,graph)
        label=immediate_children(classified[0],ctx)
        self.assertEqual(label['status'],'NODE_INFEASIBLE');self.assertIsNone(teacher_key(label))
        capped=immediate_children(classified[0],ctx,seconds=0)
        self.assertEqual(capped['status'],'TIME_CAP');self.assertIsNone(teacher_key(capped))
        r=dict(candidates=[dict(teacher=capped),dict(teacher=dict(status='COMPLETE',priority=[2,3]))])
        self.assertEqual(teacher_pairs(r),[])
        r['candidates'][0]['teacher']=dict(status='COMPLETE',priority=[2,'+inf'])
        self.assertEqual(teacher_pairs(r)[0][2],1)

    def test_revised_policies_preserve_tiny_independent_optima(self):
        grid=GridMap(2,2);cells=[(0,0),(0,1),(1,0),(1,1)]
        model=SimpleNamespace(predict=lambda X:np.arange(len(X),dtype=float))
        comparisons=0
        for starts in itertools.permutations(cells,2):
            for goals in itertools.permutations(cells,2):
                optimum=joint_optimum(grid,starts,goals)
                for mode in ('model_a','model_b','pool_random','teacher'):
                    selector=RevisionSelector(mode,model,71)
                    result=ICBShSolver(grid,conflict_selector=selector).solve(list(starts),list(goals),timeout=2)
                    self.assertEqual(result['status'],'SOLVED');self.assertEqual(result['cost'],optimum)
                    validate_solution(grid,starts,goals,result['paths'],result['cost']);comparisons+=1
        self.assertEqual(comparisons,576)

    def test_model_inference_has_no_child_teacher(self):
        grid=GridMap(3,3);starts=[(1,0),(1,2),(0,1)];goals=[(1,2),(1,0),(2,1)]
        paths=[[s,(1,1),g] for s,g in zip(starts,goals)];cs=[ConstraintSet() for _ in starts]
        classified=classify_conflicts(grid,starts,goals,paths,cs);h,graph=compute_h4_and_graph(classified)
        ctx=context(ICBSNode(cs,paths,6,h),grid,starts,goals,graph)
        selector=RevisionSelector('model_b',SimpleNamespace(predict=lambda X:np.arange(len(X),dtype=float)))
        with patch('mapf.ml_revision_policy.immediate_children',side_effect=AssertionError('Teacher leakage')):
            self.assertIn(selector(eligible_tier(classified),ctx),classified)
        self.assertEqual(selector.stats['ml_calls'],1)

    def test_three_agent_choices_match_independent_optima(self):
        grid=GridMap(3,2);cells=[(r,c) for r in range(2) for c in range(3)]
        starts=[(0,0),(0,2),(1,1)];calls=0
        model=SimpleNamespace(predict=lambda X:np.arange(len(X),dtype=float))
        for goals in itertools.permutations(cells,3):
            optimum=joint_optimum(grid,starts,goals)
            for mode in ('model_a','model_b','pool_random','teacher'):
                selector=RevisionSelector(mode,model,71)
                result=ICBShSolver(grid,conflict_selector=selector,enable_bypass=False).solve(starts,list(goals),timeout=3)
                self.assertEqual(result['status'],'SOLVED');self.assertEqual(result['cost'],optimum)
                validate_solution(grid,starts,goals,result['paths'],result['cost'])
                calls+=selector.stats['selector_calls']
        self.assertGreater(calls,0)


if __name__=='__main__':unittest.main()

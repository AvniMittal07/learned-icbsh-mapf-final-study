"""Independent optimality and regression checks for the recovery pipeline."""
import heapq
import itertools
import random
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from mapf.grid import GridMap
from mapf.low_level import ConstraintSet, space_time_astar
from mapf.search_limits import HorizonExceeded
from mapf.conflicts import detect_all_conflicts, detect_first_conflict, count_single_agent_conflicts
from mapf.cardinal import classify_conflicts
from mapf.conflict_graph import ConflictGraph, compute_h4_and_graph
from mapf.icbs_h import ICBShSolver
from mapf.high_level_cbs import PlainCBS
from mapf.validation import validate_solution
from mapf.baselines import GuardedLearnedSelector, f_aware_residual_selector


def joint_optimum(grid, starts, goals):
    """Independent Dijkstra; finish decisions allow temporary visits to goals."""
    k = len(starts)
    initial = (tuple(starts), 0)
    best, queue = {initial: 0}, [(0, initial)]
    while queue:
        cost, state = heapq.heappop(queue)
        if best[state] != cost:
            continue
        positions, done = state
        if done == (1 << k)-1:
            return cost
        successors = [((positions, done | (1 << i)), 0) for i in range(k)
                      if positions[i] == goals[i] and not done & (1 << i)]
        moves = [[p] if done & (1 << i) else [p]+grid.get_neighbors(*p)
                 for i,p in enumerate(positions)]
        for nxt in itertools.product(*moves):
            if len(set(nxt)) != k or any(nxt[i] == positions[j] and nxt[j] == positions[i]
                                         for i in range(k) for j in range(i+1,k)):
                continue
            successors.append(((nxt, done), k-done.bit_count()))
        for nxt, step in successors:
            value = cost+step
            if value < best.get(nxt, float('inf')):
                best[nxt] = value
                heapq.heappush(queue, (value,nxt))
    return None


class RecoveryTests(unittest.TestCase):
    def test_exhaustive_two_agent_optimality(self):
        grid = GridMap(2,2)
        placements = list(itertools.permutations([(0,0),(0,1),(1,0),(1,1)],2))
        for starts in placements:
            for goals in placements:
                expected = joint_optimum(grid, starts, goals)
                for solver in (PlainCBS(grid), ICBShSolver(grid),
                               ICBShSolver(grid, enable_bypass=False)):
                    result = solver.solve(list(starts),list(goals),timeout=3)
                    self.assertEqual(result['status'],'SOLVED')
                    self.assertEqual(result['cost'],expected)
                    validate_solution(grid,starts,goals,result['paths'],result['cost'])

    def test_seeded_three_agent_and_guarded_selection(self):
        rng = random.Random(711)
        grid = GridMap(3,2)
        cells = [(r,c) for r in range(2) for c in range(3)]
        class Score:
            def predict(self, X):
                return X[:,0] - X[:,-1]
        for _ in range(60):
            starts,goals = rng.sample(cells,3),rng.sample(cells,3)
            expected = joint_optimum(grid, starts, goals)
            for selector in (None, f_aware_residual_selector, GuardedLearnedSelector(Score())):
                result = ICBShSolver(grid,conflict_selector=selector).solve(starts,goals,timeout=3)
                self.assertEqual(result['status'],'SOLVED')
                self.assertEqual(result['cost'],expected)
                validate_solution(grid,starts,goals,result['paths'],result['cost'])

    def test_late_constraints_and_explicit_horizon(self):
        grid = GridMap(2,1)
        cs = ConstraintSet(edge_constraints={(0,0,0,1,t) for t in range(1,101)})
        self.assertEqual(len(space_time_astar(grid,(0,0),(0,1),cs))-1,101)
        with self.assertRaises(HorizonExceeded):
            space_time_astar(grid,(0,0),(0,1),cs,max_timesteps=2)
        cs = ConstraintSet(edge_constraints={(0,0,0,0,50)})
        path = space_time_astar(grid,(0,0),(0,0),cs)
        self.assertEqual(len(path)-1,50)
        self.assertNotEqual(path[49],path[50])

    def test_initial_constraints_not_truncated(self):
        grid = GridMap(2,1)
        cs = ConstraintSet(vertex_constraints={(0,1,150)})
        result = ICBShSolver(grid).solve([(0,0)],[(0,1)],initial_constraints=[cs])
        self.assertEqual(result['cost'],151)
        self.assertNotEqual(result['paths'][0][150],(0,1))

    def test_initial_collisions_and_invalid_instances(self):
        paths = [[(0,0)],[(0,0)]]
        self.assertEqual(detect_first_conflict(paths).time,0)
        self.assertEqual(len(detect_all_conflicts(paths)),1)
        for cls in (PlainCBS,ICBShSolver):
            self.assertEqual(cls(GridMap(2,1)).solve([(0,0),(0,0)],[(0,0),(0,1)])['status'],'UNSOLVABLE')
            with self.assertRaises(ValueError):
                cls(GridMap(2,1)).solve([(0,0)],[])
            self.assertEqual(cls(GridMap(2,1)).solve([],[])['cost'],0)
            self.assertEqual(cls(GridMap(2,1)).solve([(0,0)],[(0,1)],timeout=0)['status'],'TIMEOUT')

    def test_subset_and_child_context_counts(self):
        grid = GridMap(3,3)
        starts=[(1,0),(1,2),(0,1)]
        goals=[(1,2),(1,0),(2,1)]
        paths=[[s,(1,1),g] for s,g in zip(starts,goals)]
        conflicts=detect_all_conflicts(paths)
        subset=classify_conflicts(grid,starts,goals,paths,[],conflicts[:1])
        for cc in subset:
            for agent,alt,count in [(cc.conflict.agent_i,cc.alt_path_i,cc.alt_total_conflicts_i),
                                    (cc.conflict.agent_j,cc.alt_path_j,cc.alt_total_conflicts_j)]:
                copied=list(paths);copied[agent]=alt
                self.assertEqual(count,len(detect_all_conflicts(copied)))
        visited=[]
        def check(node,classified,graph):
            visited.append(node.depth)
            for cc in classified:
                for agent,alt,count in [(cc.conflict.agent_i,cc.alt_path_i,cc.alt_total_conflicts_i),
                                        (cc.conflict.agent_j,cc.alt_path_j,cc.alt_total_conflicts_j)]:
                    if alt is not None:
                        copied=list(node.paths);copied[agent]=alt
                        self.assertEqual(count,len(detect_all_conflicts(copied)))
        ICBShSolver(grid,enable_bypass=False).solve(starts,goals,on_node_callback=check)
        self.assertIn(1,visited)

    def test_mvc_all_five_vertex_graphs(self):
        edges=list(itertools.combinations(range(5),2))
        for mask in range(1<<len(edges)):
            selected=[e for i,e in enumerate(edges) if mask & (1<<i)]
            expected=min(bits.bit_count() for bits in range(1<<5)
                         if all(bits & (1<<u) or bits & (1<<v) for u,v in selected))
            self.assertEqual(ConflictGraph(selected).compute_mvc(),expected)

    def test_full_and_single_agent_detection(self):
        rng=random.Random(17)
        for _ in range(100):
            paths=[[rng.choice([(0,0),(0,1),(1,0)]) for _ in range(rng.randint(1,6))] for _ in range(3)]
            for i in range(3):
                all_conflicts=detect_all_conflicts(paths)
                expected=sum(c.agent_i==i or c.agent_j==i for c in all_conflicts)
                self.assertEqual(count_single_agent_conflicts(paths,i,paths[i],max(map(len,paths))),expected)

    def test_missing_model_is_explicit(self):
        from mapf.benchmark_worker import run
        result=run(dict(instance=dict(width=2,height=1,starts=[[0,0]],goals=[[0,1]]),
                        policy='linear',seed=1,timeout_seconds=5))
        self.assertEqual(result['status'],'MODEL_UNAVAILABLE')

    def test_invalid_ranker_fails(self):
        with self.assertRaises(ValueError):
            GuardedLearnedSelector(None,top_k_guardrail=0)

    def test_incremental_detector_matches_full(self):
        from mapf.fast_dataset_builder import detect_conflicts_for_agent
        old=[[(0,0),(0,1),(0,0)],[(0,2),(0,1),(0,2)],[(1,1)]]
        new=[[(0,0),(0,1),(0,0),(0,1)],old[1],old[2]]
        self.assertEqual(set(detect_conflicts_for_agent(new,0,detect_all_conflicts(old))),
                         set(detect_all_conflicts(new)))

    def test_residual_is_lower_bound_on_child_progress(self):
        from mapf.baselines import _compute_candidate_lookahead_tuple
        from mapf.cardinal import build_constraint_for_agent
        grid=GridMap(3,3)
        starts=[(1,0),(1,2),(0,1)];goals=[(1,2),(1,0),(2,1)]
        paths=[[s,(1,1),g] for s,g in zip(starts,goals)]
        cs=[ConstraintSet() for _ in starts]
        classified=classify_conflicts(grid,starts,goals,paths,cs)
        h,graph=compute_h4_and_graph(classified)
        for cc in classified:
            bound=_compute_candidate_lookahead_tuple(cc,dict(paths=paths,G_CF=graph,mvc_size=h))[0]
            for agent,alt in [(cc.conflict.agent_i,cc.alt_path_i),(cc.conflict.agent_j,cc.alt_path_j)]:
                if alt is None:continue
                child=list(paths);child[agent]=alt
                child_cs=list(cs);child_cs[agent]=build_constraint_for_agent(cc.conflict,agent,cs[agent])
                hc,_=compute_h4_and_graph(classify_conflicts(grid,starts,goals,child,child_cs))
                actual=sum(len(p)-1 for p in child)+hc-sum(len(p)-1 for p in paths)-h
                self.assertLessEqual(bound,actual)

    def test_undefined_correlation_and_target_direction(self):
        import pandas as pd
        from mapf.train_cl_ranker import compute_mean_query_spearman
        df=pd.DataFrame(dict(query_id=[1,2,1,2],relevance=[0,0,1,1],oracle_score=[1,1,0,0]))
        self.assertAlmostEqual(compute_mean_query_spearman(df,np.array([0,0,1,1])),1.)
        self.assertTrue(np.isnan(compute_mean_query_spearman(df,np.ones(4))))

    def test_native_model_roundtrip(self):
        from xgboost import XGBRanker
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from mapf.features import FEATURE_NAMES
        from mapf.benchmark_worker import run
        rng=np.random.default_rng(7);X=rng.normal(size=(16,len(FEATURE_NAMES)))
        y=np.tile(np.arange(4),4)
        # Tiny serialization smoke test only, not research model training.
        model=XGBRanker(objective='rank:pairwise',n_estimators=3,max_depth=2,n_jobs=1)
        model.fit(X,y,group=[4,4,4,4])
        linear=make_pipeline(StandardScaler(),Ridge()).fit(X,y)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json';model.save_model(path)
            loaded=XGBRanker(n_jobs=1);loaded.load_model(path)
            np.testing.assert_allclose(model.predict(X),loaded.predict(X),rtol=1e-6)
            for policy in ['xgb','linear']:
                if policy=='linear':
                    scaler,ridge=linear.steps[0][1],linear.steps[1][1]
                    path=Path(directory)/'linear.json'
                    path.write_text(json.dumps(dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                                                    coef=ridge.coef_.tolist(),intercept=float(ridge.intercept_))))
                    np.testing.assert_allclose(linear.predict(X),((X-scaler.mean_)/scaler.scale_)@ridge.coef_+ridge.intercept_)
                Path(str(path)+'.metadata.json').write_text(json.dumps(dict(feature_names=FEATURE_NAMES,backend=policy if policy=='linear' else 'xgboost')))
                spec=dict(instance=dict(width=3,height=3,starts=[[1,0],[1,2],[0,1]],goals=[[1,2],[1,0],[2,1]]),
                          policy=policy,model_paths={policy:str(path)},seed=7,timeout_seconds=5)
                result=run(spec)
                self.assertEqual(result['status'],'SOLVED')
                self.assertEqual(result['cost'],9)

    def test_isolated_runner_and_hard_timeout(self):
        from mapf.recovery_benchmark import run_isolated
        spec=dict(instance=dict(width=3,height=3,starts=[[1,0],[1,2],[0,1]],goals=[[1,2],[1,0],[2,1]]),
                  policy='plain',seed=7,timeout_seconds=8)
        a,b=run_isolated(spec),run_isolated(spec)
        self.assertEqual(a['status'],'SOLVED');self.assertEqual(b['status'],'SOLVED')
        self.assertEqual((a['cost'],a['nodes_expanded'],a['astar_calls']),
                         (b['cost'],b['nodes_expanded'],b['astar_calls']))
        self.assertGreater(a['peak_rss_bytes'],0)
        self.assertGreater(a['end_to_end_wall_seconds'],a['runtime'])
        spec['timeout_seconds']=.000001
        self.assertEqual(run_isolated(spec)['status'],'HARD_TIMEOUT')


if __name__ == '__main__':
    unittest.main()

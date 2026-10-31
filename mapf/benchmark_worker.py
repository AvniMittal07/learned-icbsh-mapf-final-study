"""One cold, isolated solver execution. Called by recovery_benchmark."""
import json
from pathlib import Path
import sys
import time


def run(spec):
    began = time.perf_counter()
    import numpy as np
    from mapf.grid import GridMap, load_map
    from mapf.high_level_cbs import PlainCBS
    from mapf.icbs_h import ICBShSolver
    from mapf.baselines import f_aware_residual_selector, degree_selector, GuardedLearnedSelector
    from mapf.features import FEATURE_NAMES
    from mapf.validation import validate_solution
    import random

    instance = spec['instance']
    grid = load_map(instance['map_path'].replace('\\','/')) if 'map_path' in instance else GridMap(
        instance['width'], instance['height'], {tuple(p) for p in instance.get('obstacles', [])})
    starts, goals = [tuple(p) for p in instance['starts']], [tuple(p) for p in instance['goals']]
    policy = spec['policy']
    rng = random.Random(spec['seed'])
    selector = {'plain': None, 'random': lambda c, _: rng.choice(c),
                'residual': f_aware_residual_selector, 'degree': degree_selector}.get(policy)
    if policy in ('linear', 'xgb'):
        model_path = Path(spec.get('model_paths', {}).get(policy, 'MISSING_MODEL'))
        metadata_path = Path(str(model_path) + '.metadata.json')
        if not model_path.is_file() or not metadata_path.is_file():
            return dict(status='MODEL_UNAVAILABLE', cost=None,
                        detail='Model and provenance metadata are required; no heuristic substitution')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('feature_names') != FEATURE_NAMES:
            raise ValueError('Model feature schema does not match inference schema')
        if policy == 'linear':
            data = json.loads(model_path.read_text())
            class Linear:
                def predict(self, X):
                    return ((X-np.array(data['mean']))/np.array(data['scale'])) @ np.array(data['coef']) + data['intercept']
            model = Linear()
        else:
            if metadata.get('backend') != 'xgboost':
                raise ValueError('Native XGBoost provenance required')
            from xgboost import XGBRanker
            model = XGBRanker(n_jobs=1)
            model.load_model(model_path)
            model.set_params(n_jobs=1)
        selector = GuardedLearnedSelector(model)
    elif policy in ('model_a', 'model_b', 'pool_random', 'teacher'):
        from mapf.ml_revision_policy import RevisionSelector, SavedLinear
        model = SavedLinear(spec['model_paths'][policy]) if policy in ('model_a', 'model_b') else None
        selector = RevisionSelector(policy, model, spec['seed'])
    elif policy not in ('cbs', 'plain', 'random', 'residual', 'degree'):
        raise ValueError(f'Unknown policy: {policy}')
    solver = PlainCBS(grid) if policy == 'cbs' else ICBShSolver(grid, conflict_selector=selector)
    remaining = max(0.0, spec['timeout_seconds'] - (time.perf_counter()-began))
    result = solver.solve(starts, goals, timeout=remaining)
    if hasattr(selector, 'stats'):
        result['selector_stats'] = selector.stats
    if result['status'] == 'SOLVED':
        validate_solution(grid, starts, goals, result['paths'], result['cost'])
        result['solution_validated'] = True
    result['worker_wall_seconds'] = time.perf_counter()-began
    result['runtime_environment'] = dict(python=sys.version.split()[0], numpy=np.__version__)
    return result


def main():
    import psutil
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(spec)
    except Exception as exc:
        result = dict(status='ERROR', cost=None, detail=f'{type(exc).__name__}: {exc}')
    info = psutil.Process().memory_info()
    result['process_peak_rss_bytes'] = getattr(info, 'peak_wset', info.rss)
    result['process_cpu_seconds'] = time.process_time()
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

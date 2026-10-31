"""Fair cold-process benchmark with common limits and no cross-solver caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import psutil


def run_isolated(spec, memory_mb=4096):
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp)/'input.json', Path(tmp)/'output.json'
        inp.write_text(json.dumps(spec))
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        began = time.perf_counter()
        proc = subprocess.Popen([sys.executable, '-m', 'mapf.benchmark_worker', str(inp), str(out)],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        monitor = psutil.Process(proc.pid)
        peak, cpu = 0, 0.0
        terminated = None
        while proc.poll() is None:
            try:
                info = monitor.memory_info()
                peak = max(peak, getattr(info, 'peak_wset', info.rss))
                times = monitor.cpu_times()
                cpu = max(cpu, times.user + times.system)
            except psutil.NoSuchProcess:
                break
            if time.perf_counter()-began >= spec['timeout_seconds']:
                terminated = 'HARD_TIMEOUT'
            elif info.rss > memory_mb * 1024**2:
                terminated = 'MEMORY_LIMIT'
            if terminated:
                proc.kill()
                break
            time.sleep(0.01)
        _, stderr = proc.communicate()
        wall = time.perf_counter()-began
        if terminated:
            result = dict(status=terminated, cost=None)
        elif proc.returncode or not out.exists():
            result = dict(status='PROCESS_ERROR', cost=None, detail=stderr.decode(errors='replace')[-2000:])
        else:
            result = json.loads(out.read_text())
        peak = max(peak, result.pop('process_peak_rss_bytes', 0))
        cpu = max(cpu, result.pop('process_cpu_seconds', 0))
        result.update(end_to_end_wall_seconds=wall, process_cpu_seconds=cpu,
                      peak_rss_bytes=peak, peak_memory_method='Windows peak_wset' if os.name=='nt' else 'sampled RSS',
                      timeout_seconds=spec['timeout_seconds'], memory_limit_mb=memory_mb,
                      cache_mode='cold isolated process', model_loading_included=True,
                      cpu_measurement='worker total when completed; sampled lower bound when killed')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output', default='results/recovery/pilot_benchmark.jsonl')
    parser.add_argument('--policies', nargs='+', default=['cbs', 'plain', 'random', 'residual'])
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--repetitions', type=int, default=1)
    args = parser.parse_args()
    if args.timeout <= 0 or args.repetitions < 1:
        parser.error('Positive timeout and repetitions required')
    raw = Path(args.manifest).read_bytes()
    manifest = json.loads(raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for instance in manifest['instances']:
            for repeat in range(args.repetitions):
                for policy in args.policies:
                    spec = dict(instance=instance, policy=policy, seed=manifest.get('seed', 71),
                                timeout_seconds=args.timeout, model_paths=manifest.get('model_paths', {}))
                    row = run_isolated(spec)
                    row.update(instance_id=instance['id'], policy=policy, repeat=repeat,
                               manifest_sha256=hashlib.sha256(raw).hexdigest())
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                    print(instance['id'], policy, row['status'], row.get('cost'))


if __name__ == '__main__':
    main()

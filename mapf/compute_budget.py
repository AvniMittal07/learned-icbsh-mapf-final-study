"""Serial, capped local experiment runner. Ledger uses conservative reservations.

Usage: python -m mapf.compute_budget --seconds 120 -- unittest mapf.test_recovery
Charged budget is the full wall reservation times allocated logical CPUs, even
if a run finishes early. This bounds child CPU consumption without relying on
polling short-lived descendants. Do not run two controllers on the same ledger.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    import psutil
    p = argparse.ArgumentParser()
    p.add_argument('--ledger', default='results/recovery/compute_ledger.jsonl')
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--cpus', type=int, default=2)
    p.add_argument('--phase', choices=['days1-2', 'day3', 'repair-pilot', 'ml-revision'], default='days1-2')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.seconds <= 0 or args.cpus < 1:
        p.error('Positive reservation and module command required')
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = ledger.with_suffix('.lock')
    with lock.open('x'):
        pass
    try:
        charged = sum(json.loads(s)['reserved_cpu_seconds'] for s in ledger.read_text().splitlines()
                      if s.strip()) if ledger.exists() else 0
        entries = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        phase_charged = sum(e['reserved_cpu_seconds'] for e in entries
                            if e.get('phase', 'days1-2') == args.phase)
        reserve = args.seconds * args.cpus
        if charged + reserve > 8 * 3600:
            raise RuntimeError('Approved 8 CPU-hour reservation cap would be exceeded')
        if args.phase in ('day3', 'repair-pilot') and phase_charged + reserve > 2 * 3600:
            raise RuntimeError(f'Approved {args.phase} two CPU-hour reservation cap would be exceeded')
        if args.phase == 'ml-revision' and phase_charged + reserve > 5400:
            raise RuntimeError('Approved ML revision 1.5 CPU-hour cap would be exceeded')
        entry = dict(command=command, reserved_cpu_seconds=reserve,
                     phase=args.phase,
                     allocated_logical_cpus=args.cpus, wall_cap_seconds=args.seconds)
        with ledger.open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        env = dict(os.environ)
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[var] = '1'
        env['MAPF_BUDGET_CPUS'] = str(args.cpus)
        start = time.perf_counter()
        proc = subprocess.Popen([sys.executable, '-m', 'mapf.budget_worker', *command], env=env)
        try:
            code = proc.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            proc.wait()
            code = 124
        print(json.dumps(dict(exit_code=code, wall_seconds=time.perf_counter()-start,
                              total_reserved_cpu_hours=(charged+reserve)/3600)))
        raise SystemExit(code)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

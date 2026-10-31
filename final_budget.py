"""Serial final-study budget with temporary Windows standby prevention.

Kept outside mapf to preserve the approved solver/source snapshot. Reservations
are append-only and count in the shared original eight-CPU-hour ledger.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--ledger',default='results/recovery/compute_ledger.jsonl')
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.seconds<=0:p.error('Positive reservation and module required')
    ledger=Path(args.ledger);lock=ledger.with_suffix('.lock');setter=None
    with lock.open('x'):pass
    try:
        entries=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
        total=sum(e['reserved_cpu_seconds'] for e in entries)
        phase=sum(e['reserved_cpu_seconds'] for e in entries if e.get('phase')=='final-study')
        if total+args.seconds>28800 or phase+args.seconds>10800:
            raise ValueError('Approved final-study or cumulative CPU cap would be exceeded')
        if os.name=='nt':
            import ctypes
            setter=ctypes.WinDLL('kernel32',use_last_error=True).SetThreadExecutionState
            setter.argtypes=[ctypes.c_uint];setter.restype=ctypes.c_uint
            if not setter(0x80000003):raise ctypes.WinError(ctypes.get_last_error())
        with ledger.open('a') as f:
            f.write(json.dumps(dict(command=command,phase='final-study',allocated_logical_cpus=1,
                reserved_cpu_seconds=args.seconds,wall_cap_seconds=args.seconds,
                standby_prevention='system and display request, released on controller exit'))+'\n')
        env=dict(os.environ,MAPF_BUDGET_CPUS='1')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
        start=time.perf_counter()
        proc=subprocess.Popen([sys.executable,'-m','mapf.budget_worker',*command],env=env)
        try:code=proc.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            try:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
            proc.kill();proc.wait();code=124
        print(json.dumps(dict(exit_code=code,wall_seconds=time.perf_counter()-start,
            phase_reserved_cpu_hours=(phase+args.seconds)/3600,total_reserved_cpu_hours=(total+args.seconds)/3600)),flush=True)
        raise SystemExit(code)
    finally:
        if setter is not None:setter(0x80000000)
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()

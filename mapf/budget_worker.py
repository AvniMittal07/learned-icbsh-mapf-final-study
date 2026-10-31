"""Worker fixes CPU affinity before numerical imports; descendants inherit it."""
import os
import runpy
import sys
import time
import psutil

process = psutil.Process()
cpus = process.cpu_affinity()[:int(os.environ['MAPF_BUDGET_CPUS'])]
process.cpu_affinity(cpus)
module = sys.argv[1]
sys.argv = sys.argv[1:]
started = time.process_time()
try:
    runpy.run_module(module, run_name='__main__', alter_sys=True)
finally:
    print(f'Worker CPU seconds (excluding subprocesses): {time.process_time()-started:.4f}; affinity={cpus}')

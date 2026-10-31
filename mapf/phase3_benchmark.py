"""Recovery entry point: explicit manifest, isolated caches, uniform limits.
Historical results remain in results/phase3_benchmark.csv; this runner never
writes there. See RECOVERY.md for the new command and metric definitions.
"""
from mapf.recovery_benchmark import main

if __name__ == '__main__':
    main()

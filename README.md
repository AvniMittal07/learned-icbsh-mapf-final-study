# Learned conflict prioritization for optimal MAPF

This repository contains a repaired Python CBS/ICBS-h research implementation, two actual trained linear conflict selectors, provenance-checked training data, and a frozen final evaluation. **Model B did not establish a speed improvement over residual selection in the frozen final test.**

The final study used 54 held-out instances from 18 scenario families on three known maps, six policies and two timing repeats: 648 analyzed runs. Model B versus residual selection was the primary comparison. Its family runtime ratio is **1.031**, with a map-stratified bootstrap 95% interval **[0.972, 1.103]**. Lower is faster. Solved runs: model B **74/108**, residual **72/108**. This is evidence about these Python implementations and this workload, not optimized published solvers or unseen maps.

Read [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md) for all results, uncertainty and limitations, and [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for the submission assessment. [The runtime figure](results/recovery/final_study/figures/final_runtime.png) shows the final family comparisons. Historical README claims are preserved separately in [README_LEGACY_UNVERIFIED.md](README_LEGACY_UNVERIFIED.md) and must not be used as current results.

## Run a demonstration

From the repository root, use Python 3.12 and the pinned dependencies:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-recovery.lock.txt
python -m demo_solver --policy residual --output demo_residual.json
python -m demo_solver --policy model_b --output demo_model_b.json
```

The demo defaults to a development instance and saves complete solution/status information. Residual selection remains the conservative demo default. `--policy` also accepts `cbs`, `plain`, `pool_random`, and `model_a`. A timeout is an unknown outcome, not proof of infeasibility. Output files are never overwritten.

For model A/B telemetry, use `selector_stats.ml_calls` and its timing fields. The older top-level `ml_tiebreaks`/`analytical_shortcircuits` counters belong to the historical selector and default to zero for the new selector; those defaults do not mean no model activity occurred.

## Verify without a new benchmark

```powershell
python -m unittest -v mapf.test_recovery mapf.test_day3 mapf.test_repair mapf.test_ml_revision test_final_study
python -m verify_submission
```

The verifier checks packaged file hashes, frozen solver/model hashes, saved solutions, cost agreement and exact reproduction of the published statistics. It does not refit models or execute test solvers. Test logs include 32 recovery/ML regression tests and four final-harness checks. These include independent small-instance optimality comparisons; finite tests are not a universal implementation proof.

## What is implemented

- Plain CBS with individual space-time A* and ordinary binary conflict splitting.
- The repository's ICBS-h variant: cardinal/semi-cardinal priority, bypassing, CG/MVC lower bounds, pathmax and best-first search.
- A cached residual analytical selector, a matched random-pool control, and two learned selectors. Model scores only choose eligible conflicts; they never prune search or replace the heuristic.
- Explicit deadlines, cold isolated benchmarks, native peak-process-memory measurement, independent path validation and append-only CPU accounting.

The ICBS-h name refers to this implementation. Meta-agent merging/restart, WDG and triplet reasoning are outside this study; it is not a complete reproduction of every published ICBS enhancement.

## Models and dataset

`results/recovery/ml_revision/model_a.json` and `model_b.json`, with their adjacent metadata, are the supported trained weights. Both use 24 features and fixed pairwise Ridge with alpha 10. Model A used 27 strict continuation-label comparisons from six informative training families. Model B used 1,277 strict child-teacher comparisons from 29 informative training families. Both scalers were fitted on training candidates only, and JSON prediction roundtrips matched exactly.

The new dataset contains 237 saved states and 1,064 complete candidate labels. All 2,128 feasible child bounds were verified. It is useful for this proxy teacher-imitation study; it is not a large general MAPF corpus or ground truth about future runtime. Read [DATASET_CARD.md](DATASET_CARD.md).

## Evidence and reproduction

`results/recovery/final_study/` holds the frozen protocol, schedule, source archive, initial attempts, any timing replacements, derived analysis rows, summary, logs and figures. Two repeats and three nested agent counts belong to each family; they are not independent scenarios. Unsolved runs receive the full 15-second cap in capped-time analysis. Memory comparisons on jointly solved instances are explicitly conditional.

The final test is now consumed. Future development must use a fresh holdout. For an independently budgeted rerun, `reproduce_final --output NEW_DIRECTORY freeze` and its later stages preserve the delivered results; use `final_budget` for serial experiments with temporary Windows standby prevention. Do not reset the ledger to bypass its limits. The supplied evaluation already reserved 2.849 CPU-hours in this phase and 6.082 cumulatively out of the approved eight-hour ceiling.

Full supported recovery instructions and historical phase details are in [RECOVERY.md](RECOVERY.md). The unverified legacy bulk dataset builder and historical all-data training workflow remain quarantined. The old failed labeling gate remains recorded; subsequent exploratory fitting was explicitly authorized and does not change that failed result.

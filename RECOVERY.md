> **Final study completed:** Read [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md), [README.md](README.md) and [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). The final test is now consumed. The sections below preserve the earlier recovery and development workflow; their phase-specific pending plans are historical.

# Recovery and exploratory ML revision — supported workflow

Base: `8f26ac31c372a8c4e5a6426fed4e74da997b9676`; local branch `recovery/days-1-2`.
This includes two trained exploratory models and a verified teacher dataset, not a final validated research claim.
Historical results and models remain unchanged and must not be used as evidence
for this repaired engine. No new speedup or memory-saving claim is made.

## Latest deliverable: ML revision

Read `ML_REVISION_REPORT.md` before interpreting the historical pilot sections below.
The old failed-gate results are preserved; exploratory fitting was separately
approved. `results/recovery/ml_revision/` contains two new fixed Ridge models:
`model_a.json` learns continuation-expansion preferences with the old guard;
`model_b.json` learns child-bound preferences with the new shared candidate pool.
Both require the adjacent `.metadata.json` file. The native historical model
files elsewhere in the repository are not these models.

Example from the repository root, after installing the locked dependencies:

```python
from mapf.grid import load_map
from mapf.icbs_h import ICBShSolver
from mapf.ml_revision_policy import SavedLinear, RevisionSelector

grid = load_map('data/random-32-32-20.map')
model = SavedLinear('results/recovery/ml_revision/model_b.json')
selector = RevisionSelector('model_b', model)
solver = ICBShSolver(grid, conflict_selector=selector)
# Supply valid, distinct starts/goals on this map:
# result = solver.solve(starts, goals, timeout=15)
# Inspect result['status']; a timeout is not an infeasibility proof.
```

The matched benchmark uses `mapf.recovery_benchmark.run_isolated` with policies
`model_a`, `model_b`, `pool_random` and `teacher`, plus the existing baselines.
`spec['model_paths']` maps each learned policy name to its model JSON path.
Missing or invalid new models fail explicitly. Model loading and online
feature construction count toward the end-to-end time limit.

`labels.jsonl` stores one query per line with its split, family, saved state,
features, every chosen candidate, and both teacher child records. Features are
saved before teacher labeling. Models are fitted only on training records;
teacher quantities must not be added to inference features. `states.jsonl`
contains the pre-label snapshots. Validation families have informed development;
test families remain sealed.

Run the independent child-bound audit through the budget controller:

```powershell
python -m mapf.compute_budget --phase ml-revision --cpus 1 --seconds 180 -- verify_ml_revision
```

It verifies source/protocol provenance, splits, feature/pool replay, child paths,
child optimal-replan costs, all recorded child MVCs using a separate exhaustive
routine, training scalers and model checksums. An existing matching audit report
is compared rather than overwritten. The six new regression tests are in
`mapf.test_ml_revision`; the full suite is listed below.

`development_initial.jsonl` and `development_replacements.jsonl` preserve raw
benchmark attempts. `development.jsonl` is a derived dataset after the disclosed
complete-block timing correction. Read `BENCHMARK_TIMING_AMENDMENT.json` and
`timing_integrity.json`; do not omit this deviation from the report.

For a separately budgeted reproduction, `reproduce_ml_revision` takes a new
output directory, preventing accidental overwrite of the delivered evidence:

```powershell
python -m mapf.compute_budget --phase ml-revision --cpus 1 --seconds 15 -- reproduce_ml_revision --output results/recovery/reproduction freeze
```

Then use the same `--output` with stages `train-a`, `audit`, `bulk`, `train-b`,
`evaluate`, `summarize`, each through a suitable remaining CPU reservation.
The audit controls whether bulk teacher collection proceeds. Do not reset the
compute ledger to bypass its limits; these commands are not approval for a new
experiment budget. Do not rerun the campaign into the supplied result directory.
`final_study_plan.json` is a proposal, not an executed test benchmark.

On Windows, prefer `python -m keep_awake_budget` in place of
`python -m mapf.compute_budget` for timed experiments. It accepts the same
arguments and briefly requests that Windows keep the system/display awake.
The request is released when the controller exits; no persistent power setting
is changed. Modern Standby otherwise invalidates short wall-clock limits even
when CPU consumption is small. The development correction retains this evidence.

## Environment

Use Python 3.12 and install `requirements-recovery.lock.txt` into a virtual
environment. The implementation session used an isolated dependency directory;
you do not need that directory layout on another machine.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-recovery.lock.txt
python -m mapf.compute_budget --seconds 120 --cpus 1 -- unittest -v mapf.test_recovery mapf.test_day3 mapf.test_repair mapf.test_ml_revision
python -m mapf.compute_budget --seconds 30 -- mapf.recovery_audit
```

The compute runner pins its worker to the requested logical CPUs (default two), fixes numerical
thread counts to one, and reserves `wall cap × allocated CPUs` against an
8-hour ledger. It charges the full reservation even when work finishes early.
This is conservative CPU-budget accounting, not actual CPU usage. Child
processes inherit the affinity. The wrapper is serial; a lock prevents two
controllers from spending the same ledger simultaneously. Stale locks after
abnormal controller termination require checking that no worker remains before
removal. Dependency installation and interactive file inspection are setup work,
not dataset/benchmark campaigns.

## What changed

- Low-level default horizon now derives from the latest vertex/edge constraint
  and reachable component size. Exhausting a smaller explicit horizon raises
  `HorizonExceeded`; it no longer silently claims infeasibility.
- Goal self-loop constraints, time-zero collisions, invalid endpoints, duplicate
  starts/goals, empty instances, and cooperative deadlines are handled.
- Contextual downstream conflict counts are recomputed at every CT child. A*
  paths are still memoized. Historical reuse of another node's marginal counts
  was invalid. Subset classification also uses the full node collision baseline.
- The unsafe earliest-only incremental detector is replaced by a full reference
  detector. Its former acceleration claim does not apply to this implementation.
- The selector receives actual graph MVC separately from pathmax-adjusted h.
  Invalid selector/model outputs fail explicitly. The degree heuristic is named
  degree selection; it is not disjoint splitting.
- `phase3_benchmark` delegates to the new manifest-based isolated runner. Every
  run has fresh caches, common resource limits, independent solution validation,
  and explicit errors. No missing model silently becomes an analytical policy.
- `train_cl_ranker` has corrected target/grouping diagnostics and requires the
  actual sklearn/XGBoost dependencies. Unverified legacy bulk generation and
  all-data training are quarantined pending the Day 3 split/model workflow.

## Replayable pilot and benchmark

```powershell
python -m mapf.compute_budget --seconds 360 -- mapf.recovery_pilot --output results/recovery/new_pilot --queries-per-instance 10 --agent-counts 20 28 36
python -m mapf.compute_budget --seconds 240 -- mapf.recovery_benchmark results/recovery/new_pilot/manifest.json --timeout 5 --output results/recovery/new_benchmark.jsonl
```

Choose a new output location for each campaign. The pilot never overwrites the
historical CSV. It records states/constraints, map hashes, source archive,
eligible/sampled actions, feature schema, explicit rollout outcomes and budgets.
Its starts/goals are seeded synthetic instances on the supplied maps, not the
missing original Moving AI scenario files. The default pilot is deliberately
small and does not produce a certified production dataset.

Oracle semantics: force a split at the saved CT node and run plain CBS on the
descendant problem. This is not a continuation of the original global OPEN list.
The expansion count includes the forced root. `SOLVED`, `INFEASIBLE`, and capped
outcomes are different; capped outcomes have no fabricated exact ranking label.
The parent collection deadline can shorten an individual rollout's nominal cap.
The optional deeper sample is a calibration diagnostic, not ground truth for
all queries. Source snapshots allow replay after further code changes.

Timing includes a fresh Python process, imports, map/model loading, preprocessing,
and solve. The solver's own runtime is also retained. Total memory is Windows
process peak working set, or sampled RSS on other systems; it is not tracemalloc.
For killed processes, CPU usage is a sampled lower bound. Memory thresholds are
monitored at 10 ms intervals, not enforced by an OS allocation quota. Hard wall
timeouts are enforced by the parent. Partial expansion counts can be unavailable
on cooperative interruption and are recorded as null, never a fictitious zero.

Model inference requires a model plus `.metadata.json` with the exact feature
schema and backend. Old unprovenanced artifacts return `MODEL_UNAVAILABLE`.
Linear inference uses JSON weights, not implicit pickle fallback. The native
XGBoost loader does not substitute the custom historical estimator.

## Day 3 calibration and gate

The approved calibration froze 90 synthetic scenario families (270 instances),
keeping nested agent-count variants together. Test data is unevaluated. The
split and label-budget protocol is in `results/recovery/day3/protocol.json`.

Collection and labeling now run separately. All candidates receive independent
budgets of 50 expansions/0.5 seconds, 200/2 seconds, and 1,000/6 seconds; a parent
solver's harvest deadline cannot shorten them. A saved source archive identifies
the actual generation code. Replaying all 35 saved queries found no feature
mismatch. Across 116 candidates, the deepest budget produced 28 solved rollouts,
88 expansion caps, and 22 unequal pair comparisons in 9 queries. The frozen
training gate failed. No new research model was fitted.

See `DAY_3_CALIBRATION_REPORT.md` for the gate, limitations, and next-step plan.
`observed_pairs.jsonl` contains only justified partial-order comparisons, never
invented ranks for two capped searches. It is a diagnostic dataset, not a
validated general-purpose training release.

To replay the included audit (no rollout regeneration):

```powershell
python -m mapf.compute_budget --phase day3 --cpus 1 --seconds 120 -- mapf.day3_analysis
```

`--phase day3` enforces an additional two-CPU-hour phase cap as well as the
original eight-hour total. All reservations are charged in full. Do not reset
or delete the included ledger to obtain more budget.

Generation commands, for reproducing the archived experiment in a separate
checkout with its archived source and a separately approved ledger, are:

```powershell
python -m mapf.day3_protocol
python -m mapf.day3_calibration harvest
python -m mapf.day3_calibration calibrate
```

Do not run those commands over the included artifacts. The frozen protocol and
harvest refuse overwrites. Calibration supports resuming only with its exact
source revision and complete JSONL records. Hardware timing may affect censoring;
exact serialized output is not expected to match across machines. Source added
after calibration (analysis/tests) is covered by the final source manifest.

Historical data remains separate. Current tests are small-instance evidence,
not a universal proof or a large-map performance study. A larger campaign and
final test evaluation require the next approval under the staged project plan.

## Approved guarded-label repair pilot

The 31 October pilot is separate from the Day 3 artifacts. See
`REPAIR_PILOT_REPORT.md` for its measured outcome and training/validation decision.
Its inputs, raw labels, tests, and any trained model are in
`results/recovery/repair_pilot/`.

The deployed selector and collector share `guarded_shortlist`. Collection uses
18 fresh training/validation families, selecting the first, middle, and last
distinct guarded decision among at most twelve captured states per family.
Every action in the selected top-three shortlist is labeled. The Day 3 quality
thresholds are unchanged; these data do not include the final test families.

`ICBShSolver.solve(initial_node=..., forced_first_conflict=...)` validates saved
paths and their individual optimal costs, reconstructs constraints and caches,
and forces the requested first split. It recomputes the root MVC rather than
trusting an inherited heuristic; ordinary ICBS-h bypass and pathmax apply
thereafter. This is a descendant continuation, not a restored global frontier.
The optional `progress` dictionary preserves a counted-pop lower bound when a
deadline interrupts search. Completed label solutions are saved and validated.

Audit the included experiment without regenerating labels or fitting weights:

```powershell
python -m mapf.compute_budget --phase repair-pilot --cpus 1 --seconds 120 -- mapf.repair_analysis analyze
```

The `repair-pilot` phase has its own two-CPU-hour cap, within the original total
eight-hour ledger. Read the remaining ledger before reproducing jobs. Audit
regeneration rewrites derived summaries only. Original generation commands
(`repair_protocol`, `repair_pilot harvest`, `repair_pilot calibrate`) require
separate output paths/checkouts and the exact archived source; they are not
commands to rerun over the supplied frozen artifacts.

Conditional training is fixed-alpha pairwise Ridge with train-only feature
standardization. `repair_analysis train` rechecks the data and gate before
fitting, and refuses to overwrite model weights. A fitted model is saved as
JSON weights plus metadata; old historical models are not substituted.

If trained, `repair_analysis validate` performs the predeclared six-family
validation comparison in fresh processes, with randomized run order, a common
eight-second end-to-end cap, two timing repeats, and a fixed random-policy seed.
`repair_analysis summarize-validation` checks completeness and solution costs.
Validation outcomes guide the next experiment; they are not final test results.

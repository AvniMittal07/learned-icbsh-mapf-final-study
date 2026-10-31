# Days 1–2: audit, repairs, and pilot findings

## Decision

The repaired core and measurement pipeline pass the checks run here. The
production dataset/model gate is **not passed**: the pilot has too few observed
ranking outcomes to justify scaling the same labeler unchanged. No new ML
speedup or memory-saving claim is supported.

Base source: GabaSatvik/learned-icbsh-mapf, branch phase3-cl-icbsh-final,
commit 8f26ac31c372a8c4e5a6426fed4e74da997b9676. Changes are local on
recovery/days-1-2. Nothing was pushed or published; no paid compute was used.

## Defects addressed

| Finding | Repair / disposition |
|---|---|
| Bounded A* exhaustion could be treated as infeasibility | Complete default horizon; explicit HorizonExceeded for incomplete capped searches; no hidden expansion cutoff. |
| Late edge constraints and goal self-loop constraints were not fully reflected | Included in horizon/goal-holding logic and regression tests. |
| Downstream counts reused from another CT node | Recompute contextual counts while retaining memoized optimal paths. |
| Subset classification used partial collision counts as the node baseline | Full-node baseline used for downstream deltas. |
| Incremental detector returned only earliest conflicts, unlike full detection | Correct full reference detector substituted. Prior speed claims withdrawn for this path. |
| Time-zero collisions and duplicate endpoints could be mishandled | Input checks and time-zero detection added. |
| Pathmax h was supplied as if it were actual graph MVC | Separate actual MVC context from effective search h. |
| Shared mutable caches and unequal benchmark limits | New subprocess-per-run runner, cold caches, uniform wall limits, external termination. |
| tracemalloc was presented as total memory | Process peak working-set measurement on Windows; sampled RSS fallback documented. |
| Missing linear model silently became a heuristic | Explicit MODEL_UNAVAILABLE; model/schema provenance required. |
| Degree selection was labeled disjoint splitting | Renamed degree selection; no claim to actual disjoint splitting. |
| Dependency absence silently changed estimator implementations | Recovery workflow uses native dependencies; custom fallback excluded. |
| Rank metric could evaluate a different target or assume contiguous queries | Evaluate explicit relevance by query; undefined correlation remains NaN. |

Historical bulk generation and all-data training are quarantined. Their raw
outputs remain available for diagnostics, but production use requires the
verified split/label workflow scheduled next. The current deliverable is not
a completed replacement for every historical entry point.

## Verification evidence

- **15 unittest groups pass.** See test_results.txt and mapf/test_recovery.py.
- **612 solver/reference comparisons:** all 144 two-agent placements on a 2×2
  grid across three solver settings, plus 60 seeded three-agent instances on a
  3×2 grid across three selectors. Costs match independent joint-state Dijkstra.
- All **1,024 simple graphs on five vertices** checked against brute-force MVC.
- Additional tests cover constraints, contextual counts, residual lower bounds,
  native XGBoost serialization, linear inference, process isolation, and hard
  deadlines. Model serialization uses tiny synthetic data, not a research model.
- Both original feature/optimality smoke tests pass (cost 20 on their fixture).
- All mapf Python files parse; git diff whitespace checks pass.
- A CI workflow is supplied but has not run remotely.

This is finite test evidence, not a proof of universal solver correctness.

## Existing dataset audit

The two dataset CSV files are byte-identical aliases, each containing **6,789
rows / 2,251 query IDs**. They should not be counted as two datasets.

- All inspected numeric cells are finite.
- Stored lexicographic ranks and relevance values agree with their stored tuples.
- **554 queries (24.6%) have tied relevance values.**
- Source state snapshots, constraints, map/scenario hashes, per-rollout statuses,
  budgets, and source revision fields are absent. Exact feature/label replay is
  therefore not possible from these CSVs alone.
- Original scenario files are absent from the source checkout.
- **20 of 60 benchmark instance IDs labeled test overlap the training CSV**;
  ten map/scenario families overlap. The training entry point fits final models
  on the whole CSV. The saved model lineage is not independently certified, and
  the artifacts do not establish a clean held-out evaluation.

Keep these records for diagnostic/heuristic-imitation work. Do not present them
as verified downstream-search ground truth or use the old test split as proof
of generalization. Details: dataset_audit.json and split_proposal.json. The latter
is deliberately unapproved until actual scenario-family identities are verified.

## New pilot data

Two small synthetic sampling pilots were run on the supplied map files, using
seeded starts/goals rather than the unavailable original scenario files.

| Pilot | Queries | Candidate rollouts | Queries with an observed unequal solved pair |
|---|---:|---:|---:|
| 8/12/16 agents | 23 | 54 | 8 |
| 20/28/36 agents | 106 | 374 | 13 |

The dense pilot is the replay-verified artifact:

- All **106 states** have unique hashes and reproduced features.
- Map/state hashes and the archived labeling source were verified.
- 374 candidate rollouts: **37 solved, 188 expansion-capped, 149 time-capped**.
- Only **13/106 queries (12.3%)** have fully observed unequal outcomes.
- The deeper subset contains 87 rollouts: 12 solved, 35 expansion-capped, and
  40 time-capped. Ten comparable solved pairs retained their ranking; this says
  nothing reliable about the still-censored pairs.
- Mean candidate rollout CPU time was approximately **0.169 seconds**.
- The dense collection process used **126.375 CPU seconds** and about **142.9
  wall seconds**, including harvesting and the deeper subset.
- No additional strict solved-versus-censored preferences were justified by the
  observed expansion-count lower bounds in this sample.

Files in pilot_dense include queries.jsonl, manifest.json, source.zip,
harvest_status.json, and quality.json. States and explicit failure/cap statuses
are retained; there are no fabricated exact timeout labels. The earlier easy
pilot is retained as exploratory data and does not have the source archive of
the dense run.

Oracle semantics are deliberately narrow: a forced split followed by plain CBS
on the node's descendant problem. This does not resume the full original OPEN
frontier and must not be called an exact value of the global ICBS-h decision.

## Benchmark runner smoke check

36 cold-process runs: nine dense synthetic instances × four policies × one
repetition, with the same five-second end-to-end cap and no expansion cap.

| Policy | Solved / 9 | Hard timeouts |
|---|---:|---:|
| CBS | 3 | 6 |
| Plain ICBS-h | 4 | 5 |
| Random within tier | 4 | 5 |
| Residual analytical | 5 | 4 |

All 16 solutions were independently validated and costs agreed on shared solved
instances. Process memory and CPU/wall time were recorded. These nine cases and
one repetition validate instrumentation, not comparative research claims. No
learned deployment model was evaluated. See pilot_benchmark.jsonl and
benchmark_smoke_summary.json.

## Compute and remaining work

The serial experiment ledger has conservatively charged **1.0 of the approved
8 CPU-hours** by reserving the entire wall cap × two CPUs for every job, including
early completions and failed launches. This is not actual CPU consumed; actual
experiment usage was much lower. Lightweight file inspection and dependency
setup are not included in that experiment ledger. No large campaign or paid
resource was started.

An account/permission change during the session made the first task-local
dependency installation unreadable. The pinned dependencies were restored in
a second task-local directory, and the final suite passed there. Global Python
was not modified. See the lockfile and environment record for reproduction.

## Recommended next phase: calibration before scaling

Request approval for **up to 2 further CPU-hours for Day 3 calibration, keeping cumulative experiments within the original 8-hour ceiling**,
not a full production run:

1. Freeze new scenario manifests and family-level train/validation/test partitions.
2. Choose 20–40 representative states across the map types and difficulty bands.
3. Compare 50/200/1,000-expansion oracle budgets with explicit time caps; replay
   each forced action under one fixed reference policy. Include original full
   frontier effects in the scientific limitation or evaluate them explicitly.
4. Measure ranking yield, censoring, action diversity, and cost per useful query.
   Preserve incomparability between two capped searches.
5. Test a progress-based low-cost target as a heuristic target, and compare it
   to deeper observations and the direct analytical selector.
6. Set the smallest useful dataset target and measured collection budget. Train
   a small linear baseline only if the calibration/split gates pass.

For perspective, the dense pilot produced 13 informative queries in 126.4 CPU
seconds: approximately 9.7 CPU-seconds per such query on this sample. Naively
extrapolating to 1,000 would be about 2.7 CPU-hours, but **that is not an approved
or reliable forecast**: coverage, selection bias, harder states, and reference
budget changes can dominate. Do not scale on that extrapolation alone.

If learning fails to add value, recommend the analytical method and complete
the study as an honest measured result. This preserves the deadline and
scientific value better than adding unvalidated architectural complexity.

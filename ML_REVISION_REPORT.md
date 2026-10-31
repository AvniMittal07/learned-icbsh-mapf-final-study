# ML revision: results and limits — 31 October 2026

**Both learned models are slower than residual selection in this development study: model A by 3.8% and model B by 9.0% on mean capped runtime, with the same solved count.** The approved revision is complete: two real fitted models, a verified teacher dataset, and a matched development comparison. The final test remains sealed. This is a usable research prototype; it is not yet evidence of superiority over existing MAPF algorithms generally.

## What was built

**Model A** uses the prior repaired ICBS-h descendant-continuation targets and original guarded top-three pool. It was explicitly authorized as exploratory despite the old gate failure: 27 strict pairs, 10 informative queries, 6 informative training families. The old failed-gate artifacts are unchanged.

**Model B** imitates a verified immediate-child teacher. Both models use the fixed train-only standardized pairwise Ridge fit, alpha 10, no intercept, and equal total pair weight per informative query. No neural architecture, hyperparameter or model sweep was performed. Both JSON exports exactly reproduced fitted predictions on their training candidates (maximum absolute error zero).

The teacher performs ordinary CBS branching, obtains optimal individual child paths, fully classifies child conflicts, and computes the cardinal-graph minimum vertex cover. Its preference is the lexicographic pair `(minimum effective child f, maximum effective child f)`, higher first, after the solver's pathmax rule. Ties remain ties. Proven infeasibility has an explicit absent-branch bound; time limits are unknown. These are **lookahead preferences, not ground-truth future runtime labels**.

The revised pool includes the residual selector's winner and up to five other conflicts chosen by a frozen conflict hash. It retains the highest cardinality tier and ordinary splitting, and uses the same function for collection and inference. Learned scores neither prune search branches nor replace the admissible search bound. Model B computes no child teacher online; all its 24 inputs are available from the existing parent/classification context.

Learning an expensive one-step oracle follows the approach of Huang, Koenig and Dilkina. Their teacher uses WDG; ours uses CG/MVC and a declared secondary bound key. This is an empirical adaptation, not a replication of their gains or a novel invention of learned conflict selection. [AAAI paper](https://taoanhuang.github.io/files/aaai21c.pdf)

## Is the new dataset useful?

**Yes, for this bounded teacher-imitation experiment. It is not a large or broadly representative MAPF training corpus.**

| Quantity | Training | Development validation |
|---|---:|---:|
| Families attempted | 36 | 12 |
| Families yielding eligible query states | 30 | 11 |
| Saved query states | 199 | 38 |
| Complete candidate labels | 908 | 156 |
| Informative queries | 182 | 34 |
| Informative families | 29 | 10 |
| Strict candidate comparisons | 1277 | 245 |

All 1064 labels completed under the common one-second per-candidate allowance. Every one of the 2128 feasible child records had its path, constraint compliance and cost checked; its MVC was recomputed using a separate exhaustive vertex-cover routine that branches on edge endpoints. Replayed features and candidate pools matched. Strict comparisons did not have identical feature vectors. All 48 planned families were attempted; 7 yielded no eligible decision and are recorded, not silently replaced.

The training-only audit found 15 of 30 decisions where the teacher strictly preferred a different action from residual selection, satisfying the frozen rule to proceed. The full sample was then collected in fixed order without filtering for favorable labels. Training used families 08–19; development used 20–23; final families 24–29 remain unevaluated. There is no family overlap between the new training and development labels.

Remaining weaknesses: only three known maps; synthetic starts/goals; at most the first eight eligible training decisions per family; correlated decisions from the same search; labels gathered under the residual policy rather than the learned policy; and proxy targets that may not reduce whole-search work. The 1,277 training pairs are **not 1,277 independent problems**. The earlier CSVs remain quarantined and were not mixed into these fits.

## Does imitation translate into speed or memory savings?

Model A's development family-macro pair accuracy is 61.1%, versus 80.6% for analytical ordering within its saved pool. Model B's corresponding teacher-pair accuracy is 85.3%, versus 87.5% for analytical ordering. Pair accuracy alone is not solver performance.

The complete-solver comparison used **12 development families, two repeats and six policies: 144 runs**, common eight-second end-to-end caps, fresh processes, one allocated logical CPU, one numerical-library thread and 4-GiB memory limits. Starts/goals and agent counts were fixed before labels. Algorithm random seed was identical across the two timing repeats. Model loading, feature construction, score computation and direct-teacher cost are included.

| Policy | Solved runs | Mean capped seconds | Median observed peak process MiB |
|---|---:|---:|---:|
| Plain ICBS-h | 12/24 | 4.908 | 35.17 |
| Residual analytical | 14/24 | 4.058 | 34.76 |
| Random in new pool | 12/24 | 4.628 | 37.57 |
| Direct child teacher | 10/24 | 5.070 | 34.85 |
| Model A: continuation / old guard | 14/24 | 4.213 | 38.80 |
| Model B: child teacher / new pool | 14/24 | 4.423 | 38.42 |

Failures receive the full eight-second cap. The time column does not estimate latent time-to-solution for failures. Memory includes the interpreter and imports; pooled medians can reflect different completion mixes. All returned solutions passed independent path validation, and solved costs agreed across policies and repetitions.

| Map | Residual solved | Model B solved | Residual capped s | Model B capped s |
|---|---:|---:|---:|---:|
| random-32-32-20 | 8/8 | 8/8 | 1.216 | 2.025 |
| room-32-32-4 | 2/8 | 2/8 | 6.761 | 7.043 |
| warehouse-32-32-aisles | 4/8 | 4/8 | 4.198 | 4.202 |

Map rows expose where the cap prevents a useful comparison: equal capped times for two failing policies do not mean equal underlying performance. Very easy cases can be dominated by process startup, while hard cases can be censored; neither should be hidden by reporting only the solved subset.

Model B / residual **arithmetic mean capped-time ratio: 1.090** (below one is better). Its geometric mean of the 12 paired family time ratios is **1.186**, with a descriptive family-bootstrap 95% interval **[1.039, 1.387]**. These are different summaries and should not be interchanged. Development data already informed the method; the interval is not confirmatory evidence.

Both learned policies lost on all seven jointly solved families; the other five families timed out under both learned policies and residual selection. Model B expanded an average 242.7 nodes versus 160.6 for residual on the same solved cases, about 51.2% more. Its observed feature extraction plus prediction took only 0.344 seconds across those 14 completed runs. This points to worse search choices as a major problem; switching to a larger network or merely accelerating the dot product is not supported as a remedy by these results.

Keep residual selection as the current practical default. The new dataset repairs provenance and labeling mechanics; it does not validate the teacher objective as a surrogate for total search work. Before a future learning redesign, demonstrate that an offline teacher's preferred choices reduce matched descendant/global search work often enough to matter, and separate candidate-pool effects from target effects. That is a new research experiment, not something established by the 85.3% imitation score.

For model B, the median paired process-memory ratio on the 7 families solved by both policies in both repeats is **1.105**. That is a conditional small-sample measurement, not a general memory-saving claim. Fewer expansions do not imply lower measured memory.

Detailed per-family timing, outcomes, model calls and feature/inference time are retained in the JSONL and summary. Hard-killed runs cannot return selector telemetry; totals only cover runs with recorded telemetry and must not be interpreted as zeros for killed runs. Two timing repetitions do not double the number of independent families. Changing both targets and candidate access between A and B also prevents attributing their difference to labels alone; the matched random-pool control helps assess the new learned selection.

## Timing fault and correction

One observed run exceeded the eight-second limit: 195.287 seconds wall time but 4.906 seconds sampled CPU; a second recorded 150.608 seconds wall time. Windows Kernel-Power events subsequently showed entry to and exit from Modern Standby. Before aggregate policy analysis, `BENCHMARK_TIMING_AMENDMENT.json` recorded a policy-independent rule: a wall-limit overshoot over one second triggers one full matched-instance-block replacement, preserving the original observations. Replacements used a process-lifetime keep-awake request, released on completion; persistent power settings were not changed. Power-event evidence is retained in `standby_events.json`.

There were 2 initial fault(s), affecting 2 instance block(s); 24 replacement runs were performed. No replacement violated the rule. `development_initial.jsonl` and `development_replacements.jsonl` contain raw attempts. `development.jsonl` is the explicitly derived 144-row analysis dataset. This deviation and residual environmental uncertainty must be disclosed in the submission.

## Verification and resource use

The final local suite passed **32 tests**. New tests cover teacher agreement with actual solver children, independent MVC bounds, unknown/infeasible distinctions, pool determinism, absence of online-teacher leakage, and **1,056 additional tiny-instance policy/optimum comparisons** (576 two-agent and 480 three-agent checks), alongside the previous recovery suite. Some three-agent tests disable bypass to exercise actual policy choices. Finite tests support the implementation; they are not a universal proof. Remote CI has not run.

The revision conservatively reserved **0.983 of its 1.5 approved CPU-hours**. Cumulative reservations are **3.233 of 8 CPU-hours**. Full reservations are charged even when a task finishes early; these are not measured CPU consumption. Worker logs and raw records preserve measured times, including failed initial verification attempts. Offline teacher labels consumed 43.72 CPU-seconds; that excludes harvesting, replay, fitting and evaluation.

## What still prevents an A+ claim

1. **Untouched performance evidence:** run the frozen final study before claiming a learned speedup. Preserve negative results and uncertainty. Do not change the model after seeing test outcomes.
2. **Correct comparison scope:** these results concern our repaired Python solver. They do not establish superiority over optimized published CBS/ICBS implementations, WDG-based systems or other MAPF solvers. Either add a genuinely matched external baseline in a separately approved scope, or narrow the claim explicitly.
3. **Explain the contribution:** provide the teacher equation, candidate rule, split/provenance audit, two actual model artifacts, analytical/random/teacher controls, and costs of learning/inference. Present this as an evaluated adaptation.
4. **Reproducible submission:** use the supplied package, raw results, environment lock and manifest. Clearly label the old claims as unverified history. Present limitations rather than hiding them behind model accuracy.

An honest negative final result can still support a strong engineering/research submission. An A+ grade or a positive speed/memory result cannot be promised.

## Next concrete step

`FINAL_ML_STUDY_PLAN.md` proposes 54 sealed instances × six policies × two repeats, with a 15-second cap: 648 runs. A three-CPU-hour ceiling would bring cumulative reservations to 6.233/8 hours. Model B versus residual is the frozen primary comparison; model A, CBS, ICBS-h and random-pool results are retained as controls. The expensive teacher remains a development control.

This final study **has not started** because your staged approval requirement reserves it for the next decision. No paid compute or remote publishing occurred in this revision.

## Artifacts

Within the package, `results/recovery/ml_revision/` contains the frozen protocol, source archive, raw states and labels, both model JSON files and provenance metadata, integrity results, all development attempts, timing amendment, derived analysis and compute/test logs. `RECOVERY.md` documents loading, verification and reproduction in a new directory. `SOURCE_MANIFEST.json` hashes each packaged file except itself; the zip is CRC-checked and checked against that manifest.

Protocol SHA-256: `c1cff0f89fb400bd95cb588be878fa5e37f5b79b6f68683eb0ab2b23f13be1d8`.

Generation/deployment source SHA-256: `3c638b8edee05ef3643edf29ffc68308b6c85a271e0edfb32e944c8992a0b8a2`.

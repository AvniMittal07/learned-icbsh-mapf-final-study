# Day 3: label calibration and decision

Date: 30 October 2026. Base revision: `8f26ac31c372a8c4e5a6426fed4e74da997b9676`.

**Decision: do not scale this labeling recipe or train a research model from this sample.** The repaired data pipeline passed its integrity checks, but the predeclared label-quality gate failed. Day 3 is complete; the overall research project is not complete.

We now have useful, replayable diagnostic data. We do **not** yet have a sufficiently informative training dataset, a validated learned policy, or evidence of lower runtime or memory than existing algorithms. More rows alone would not fix this.

## What was completed

- Froze 90 synthetic scenario families on three supplied maps. Their nested 16-, 24-, and 32-agent variants stay in the same partition: 180 training instances, 36 validation instances, and 54 test instances. There is no duplicate physical instance across partitions. The maps themselves are shared; this does not test unseen-map generalization.
- Selected 18 training/validation families before labeling and saved 35 decision states. The test scenarios were not solved, labeled, or used for model selection.
- Saved paths, constraints, candidate conflicts, 24-feature vectors, map hashes, state hashes, source code, and the frozen protocol.
- Separated collection from labeling. Every candidate received its own rollout budget, without inheriting the collection solver's remaining deadline.
- Ran 348 rollouts: the same 116 candidate actions at three budgets. Replayed all 35 feature records with zero mismatches.
- Added censoring, split-isolation, and training-gate tests. The complete local suite passed **21 tests**, including the earlier independent small-instance optimal-cost comparisons. CI configuration was updated but has not run remotely.

The states are the first and last distinct collected states from each selected family, capped at two. Candidates are sampled within the highest cardinality tier, capped at four. These are deliberately small calibration data, not a representative population sample.

## What deeper labels actually yielded

The oracle forces a candidate split and runs deterministic plain CBS in that saved node's descendant problem. Its target is the number of expansions needed to solve that problem. It does not resume the original global ICBS-h OPEN frontier, and it does not directly measure end-to-end ICBS-h runtime.

| Budget per candidate | Solved / 116 | Expansion-capped | Informative queries / 35 | Justified unequal pairs / 147 | Total rollout CPU seconds |
|---|---:|---:|---:|---:|---:|
| 50 expansions or 0.5 s | 13 | 103 | 4 | 5 | 13.08 |
| 200 expansions or 2 s | 21 | 95 | 7 | 15 | 51.84 |
| 1,000 expansions or 6 s | 28 | 88 | 9 | 22 | 248.91 |

All unfinished rollouts in this run hit the expansion cap, not the time cap. Increasing only wall time at the same expansion cap would not resolve them. Moving from medium to high cost approximately 4.8 times as much rollout CPU for two additional informative queries.

At the deepest budget:

- Random map: 5 informative queries out of 12.
- Room map: **0 out of 12**.
- Warehouse map: 4 out of 11.
- Training: 6 informative queries from 4 families; 13 unequal pairs.
- Validation: 3 informative queries from 2 families; 9 unequal pairs.

“Informative” means at least one strict comparison is supported by observed search effort. Two solved rollouts with different expansion counts can be compared. A solved rollout can also be compared with a capped rollout if it used strictly fewer expansions than the latter had already consumed. Two capped rollouts are never ranked. Equal observed counts are ties. At the high budget, all 22 strict comparisons were between solved rollouts.

The three medium-budget solved-versus-censored comparisons became fully observed at the high budget. Across budgets there were no reversed observable preferences, inconsistent solved expansion counts, or differing solved costs for the same saved problem. This checks deterministic implementation consistency; it does not prove that the oracle predicts the best global search decision.

## Training gate: failed without changing thresholds

These thresholds were frozen before labeling. Passing them would only permit a small fixed linear experiment, not establish final dataset adequacy.

| Requirement | Required | Observed | Result |
|---|---:|---:|---|
| Informative training queries | at least 12 | 6 | Fail |
| Informative training families | at least 6 | 4 | Fail |
| Informative validation queries | at least 6 | 3 | Fail |
| Informative validation families | at least 3 | 2 | Fail |
| Maps with informative queries | at least 2 | 2 | Pass |
| Informative fraction | at least 40% | 25.7% | Fail |
| Feature replay mismatches | 0 | 0 | Pass |

The conditional pairwise Ridge trainer is implemented, but the gate correctly prevented fitting. No new research weights were created. The existing tiny synthetic model round-trip test is a software test, not a trained research result.

The guarded selector introduces another issue: only **13 of the 22** high-budget observable pairs were between candidates in its actual top-three analytical shortlist. Those pairs cover eight queries. Only 56 of the 116 sampled candidates belonged to a shortlist where ML could be invoked. Labeling arbitrary same-tier actions therefore spends effort on decisions the deployed learner may never make. This is a post-run diagnostic, not a retroactive change to the gate.

## What the dataset is useful for

The release supports exact state/feature reconstruction, censoring analysis, regression tests, and comparisons of label-budget yield. `observed_pairs.jsonl` exports only supported partial-order comparisons; it does not turn incomplete outcomes into fake exact ranks. Its rows across budgets are correlated repetitions of the same decisions and must not be counted as independent examples.

The release is **not** sufficient to train and claim a general improvement. It undercovers room-map decisions, contains very few independent informative families, samples only some eligible actions, and uses a descendant-CBS proxy for an ICBS-h decision. Historical CSV labels remain separate because their provenance and semantics were not validated for the repaired engine.

## Compute accounting

Day 3 reserved **1,950 CPU-seconds = 0.542 CPU-hours**, including protocol creation, collection, labeling, analysis, and checks. This is below its approved 2-hour ceiling. Cumulative conservative reservations are **5,550 CPU-seconds = 1.542 CPU-hours**, below the original 8-hour ceiling.

Reservations charge the full wall cap times allocated logical CPUs, even when a job finishes early. The calibration worker itself reported 316.72 CPU-seconds and 334.18 wall-seconds. Its three rollout budgets account for 313.83 CPU-seconds. These are measured costs for this sample; they are not a guaranteed extrapolation to harder instances. Test subprocess CPU is covered by reservations, not by the parent worker's reported CPU counter.

No paid compute, remote publishing, or large generation campaign was used. Extra hardware is not the immediate priority; aligning the sampled decisions and labeling target is.

## Concrete next phase proposed for approval

Request: up to **2 additional CPU-hours** for a bounded label-repair and validation pilot, still within the original cumulative 8-hour ceiling. This is proposed work, not work already executed.

1. **Collect decisions the model actually sees.** Share the exact analytical plateau/top-three shortlist logic between the guarded selector and collector. Save every shortlisted action and invocation/short-circuit counters. Check that refactoring preserves choices and solution costs. Sample by family and decision state, not by successful label outcome.
2. **Pilot an oracle closer to the deployed search.** Add a resumable saved-node ICBS-h continuation with a forced first split, using the same bypass, heuristic, queue ordering, and low-level engine thereafter. Independently test path/constraint validity and small-instance optimal cost. Continue to describe it as a descendant proxy; this still does not restore the original global OPEN frontier.
3. **Freeze the revised protocol before new labels.** Use new families from the existing training/validation pool, for example training family indices 04–07 and validation indices 22–23 on each map. Keep test families sealed. Compare the cheaper proxy against the existing CBS oracle on a small fixed training subset. Measure informative yield, map coverage, actual guard coverage, cost, and censoring. Do not lower the present quality thresholds to obtain a pass.
4. **Train only if the revised gate passes.** Fit the fixed train-only standardized pairwise linear baseline; use family-separated validation to compare against direct analytical selection. Report the sample size and guarded-decision results. A passing small-sample gate permits an experiment, not a speedup claim. If it fails again, retain the analytical method as the implementation baseline and report the learned approach as an unsuccessful experiment.
5. **Prepare the final benchmark before opening test data.** Freeze policies, caps, repetitions, solver/source hashes, and analysis. Use isolated processes and equal limits; retain failures; validate returned paths/costs; report solved fraction, paired runtime and peak process memory. Final test execution will receive its own measured budget request under the staged approval plan.

A suitable budget allocation for this next pilot is 600 CPU-seconds for correctness/refactoring checks, 4,800 for capped collection/oracle comparisons, 1,200 for conditional training and validation, and 600 reserve. No tree ranker or triplet expansion is proposed before this gate passes. If the pilot remains uninformative, stop spending on labels and finish the benchmark/report with an honest negative ML result.

## What remains necessary for a strong submission

Correct code and reproducible data are necessary, but they do not establish a research contribution or guarantee an A+ grade. The unfinished essentials are a defensible learning target, enough independent training/validation coverage, and a final untouched benchmark with fair time and memory measurement. A positive claim must beat direct analytical selection as well as weaker baselines. Comparisons only against this Python implementation cannot establish superiority over optimized published MAPF systems.

## Evidence in the project package

- `results/recovery/day3/protocol.json`: frozen splits, budgets, thresholds, and identifiers.
- `states.jsonl`, `calibration.jsonl`: raw states/features and every rollout outcome.
- `observed_pairs.jsonl`: justified comparisons with candidate indices and provenance.
- `quality.json`: machine-readable integrity checks, yield, gate, and offline diagnostics.
- `calibration_source.zip`, `source.json`: exact labeling source and its digest.
- `full_tests.log`: complete local 21-test result.
- `results/recovery/compute_ledger.jsonl`: conservative cumulative reservations.
- `SOURCE_MANIFEST.json`: hashes for the final packaged files.

Protocol SHA-256: `0258fd32d45d4ba98b3077bd11f323a443bcfbae41f4578985a5f3124cd51988`.

Calibration source SHA-256: `23be6a912228e75fe0f8e1d5f986ab8a7952375d3f2a3036a7c2c4305f945250`.

# Guarded-label repair pilot — 31 October 2026

**The labeling repair still failed the frozen gate. No research model was trained; the remaining supported study is a fair analytical-baseline comparison plus a documented labeling limitation.**

This completes the approved two-CPU-hour repair pilot. The overall research project still needs its final untouched benchmark and final report. No paid compute or remote publishing was used.

## Engineering changes

- The deployed learned selector and collector share one analytical-plateau/top-three shortlist implementation. Regression tests verify stable tie order, analytical short-circuits, and actual model inputs.
- ICBS-h can continue a saved node with a forced first conflict. It checks legal paths, goal persistence constraints, individual optimal path costs, and eligible priority tier. Fresh caches are created and root MVC is recomputed rather than trusting an arbitrary supplied h. Bypass is suppressed only for the forced root; ordinary bypass/pathmax/queue logic resumes afterward.
- Interrupted continuations expose an expansion-count lower bound. Incomplete outcomes remain distinct from infeasibility. Completed label solutions are saved and independently validated, including the saved root constraints.
- The current local suite passed **26 tests**, including independent tiny-grid optimum checks and constrained non-root continuation checks. This is evidence for the tested cases, not a universal proof. Remote CI has not run.

## Dataset and split integrity

The protocol was frozen before new labeling. It selected training families 04–07 and validation families 22–23 on each supplied map: 18 new families, with 24 or 32 agents. Related agent-count variants remain grouped in the original 90-family split. No old calibration family was reused and no test outcome was inspected.

Collection follows direct residual analytical selection. It saves first/middle/last distinct guarded decisions among at most twelve captures per family, with every shortlisted action. The result is **53 decision queries**, **144 unique sampled candidate actions**, and **432 main rollouts**. Repeated budgets do not create additional independent examples. The map files are shared across splits; this is known-map scenario generalization.

All 53 states/features and guarded shortlists replayed with zero mismatches. The audit revalidated 182 saved solved trajectories across budgets. The source archive and map/state/protocol hashes passed. Raw censored outcomes are preserved. `observed_pairs.jsonl` exports only supported strict comparisons, never fabricated ranks between two unfinished searches.

## Label quality

| Expansion/time budget | Solved actions | Time-capped | Expansion-capped | Informative queries | Strict observed/bounded pairs | Rollout CPU seconds |
|---|---:|---:|---:|---:|---:|---:|
| 50 / 0.5 s | 36 | 108 | 0 | 5 | 10 | 56.75 |
| 200 / 2 s | 60 | 84 | 0 | 11 | 26 | 181.98 |
| 1000 / 6 s | 86 | 58 | 0 | 19 | 48 | 419.30 |

At the highest budget, 19/53 queries (35.8%) contain at least one useful comparison. Every sampled action belongs to the deployed guard. There are 129 within-query candidate pairs in total; unresolved/tied pairs are excluded from training targets. Of 31 fully solved queries, **13 gave identical expansion counts for every sampled action**. Resolving timeouts alone therefore does not guarantee useful ranking labels.

| Map | Queries | Informative queries | Informative families | Supported pairs |
|---|---:|---:|---:|---:|
| random-32-32-20 | 18 | 13 | 5 | 33 |
| room-32-32-4 | 18 | 2 | 2 | 5 |
| warehouse-32-32-aisles | 17 | 4 | 3 | 10 |

The quality thresholds are unchanged from Day 3:

| Criterion | Required | Observed |
|---|---:|---:|
| train informative queries | at least 12 | 10 |
| train families | at least 6 | 6 |
| validation informative queries | at least 6 | 9 |
| validation families | at least 3 | 4 |
| maps with informative queries | at least 2 | 3 |
| informative fraction | at least 0.4 | 0.358 |
| replay mismatches | at most 0 | 0 |

**Gate: FAIL.** Passing this gate would permit a small linear experiment; it would not certify that the dataset is large or representative enough for a general claim.

The cutoff remains the predeclared practical decision rule, not a statistical proof that learning is impossible. A failed data gate is not an empirical demonstration that a trained ML solver performs worse: no research model was trained or benchmarked in this pilot. The learned contribution remains unproven.

The target remains a proxy: fewer expansions under residual-selector ICBS-h after a forced action in a saved descendant problem. It does not restore the original global OPEN frontier, preserve the inherited root pathmax bound, or directly optimize measured wall time. Feature reconstruction and shared immutable distance-table setup occur outside the per-candidate rollout cap; fresh root validation/classification occur inside the ICBS-h cap.

For the fixed same-state training subset, ICBS-h solved 5/8 actions and CBS solved 0/8 under the same high cap/time. This very small diagnostic does not establish a general oracle or solver speedup. The old Day 3 and new pilot use different samples and collection policies, so their overall yields are not a controlled causal comparison. Deterministic solved counts/costs and observable preference stability are checked in `quality.json`; agreement is not proof of global-search oracle accuracy.

## Conditional linear model

No model was fitted because the unchanged gate failed.

## End-to-end validation

No learned-policy validation was run because training was not permitted by the gate.

## Compute and next approval

This pilot reserved **0.708 CPU-hours** of its approved 2-hour phase budget. Cumulative conservative reservations are **2.250 CPU-hours** of the original 8-hour ceiling. Full wall-cap-times-CPU reservations are charged even for early completion; these are not actual consumed CPU hours. Measured run/rollout timings are in the logs and raw JSONL.

The concrete next proposal is a final evaluation using the 54 untouched test instances, 4 frozen policies (cbs, plain, random, residual), three repeats, and a common 15-second cap. This is 648 runs and at most 9,720 nominal solver wall-seconds. Request **up to 3 additional CPU-hours**, including runner overhead, result analysis, uncertainty estimates clustered by the 18 scenario families, and the final report. With current reservations the cumulative envelope would be 5.250 CPU-hours, within the original ceiling. The exact proposed matrix is in `final_evaluation_plan.json`; it has not been executed.

Freeze the solver code and four policy configurations before opening test outcomes. Keep failed runs in all denominators and report solved fraction, paired runtime, and memory. Describe the inability to construct sufficient training data separately. This final baseline study will measure analytical policy differences; it will not establish whether a trained ML policy reduces runtime or memory. Broader claims against existing optimized MAPF systems require verified external baselines beyond these Python implementations. A strong submission needs this final evidence; an A+ grade or positive gain cannot be guaranteed.

## Reproduction and evidence

Start with `RECOVERY.md`. The package contains raw `states.jsonl`, `calibration.jsonl`, saved label solution paths, `observed_pairs.jsonl`, `quality.json`, source archives, tests, the cumulative ledger, and any conditional model/validation outputs. Historical artifacts and previous reports are preserved separately.

Protocol SHA-256: `5127e81046ac4361644524f6939daeb60d2922a571b784a0bad6dabac66f9fca`.

Generation source SHA-256: `391d72f8dd77d8bdff09bc8e3f08f592c1fc8e2636f47e494059f5b05c70dc4e`.

The final `SOURCE_MANIFEST.json` hashes every packaged file except itself. The archive is CRC-checked and its entries are verified against that manifest.

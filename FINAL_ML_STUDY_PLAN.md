# Proposed final study — approval required

Status: **not executed**. This supersedes the earlier baseline-only proposal. Both models and solver code are now concrete, reviewable artifacts in the ML revision package.

- Freeze model B as the primary learned policy; compare it with direct residual selection. Preserve model A as a secondary learned-policy result.
- Evaluate all 54 previously sealed instances: three maps, six new families per map, with nested 16/24/32-agent variants. These are **18 independent families**, not 54 independent scenario families.
- Run six policies: plain CBS, plain ICBS-h, residual selection, random in the revised pool, model A, and model B. The expensive direct teacher has already been evaluated as a development control.
- Two timing repetitions, 15-second end-to-end caps, 4-GiB memory limits, fresh processes, fixed algorithm seed 71, randomized schedule seed 20261031.
- Total: **648 runs**, at most **9,720 nominal solver seconds**. Request **3 additional CPU-hours**, including monitoring, analysis, checks and reporting. Cumulative reserved envelope: **6.233 / 8 hours**.
- Validate every solved path and compare solved costs. Report all failures, family-level paired capped runtime, solved fraction, matched memory, expansions and inference overhead. Use map-stratified family bootstrap intervals; repeats and nested agent variants are not independent samples.
- Preserve timing-control faults and any complete-block reruns. Do not silently remove outliers. If repairs cannot fit the approved budget, report the incompleteness.
- Do not tune models or code from test outcomes. Keep a negative result if the learned policies lose. A positive speed claim requires a time-ratio interval below 1 with no fewer solved instances; report effect sizes and limitations even if that rule fails.

The final deliverables would be raw test results, reproducible analysis, a concise final research report, an accurate project README and submission checklist. This tests our Python implementations. Claims against optimized published MAPF implementations would still require a matched external baseline.

Your staged approval requirement is why this study has not started. No paid compute, remote publishing or additional dataset generation is requested.

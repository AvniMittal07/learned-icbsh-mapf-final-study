# Submission assessment and checklist

**Result to defend:** Model B did not establish a speed improvement over residual selection in the frozen final test.

The project now supports a reproducible comparative study. It does not justify calling learned conflict selection a new invention, promising universal optimal-solver acceleration, or inferring memory savings from fewer nodes. I would not justify an A+ specifically for a new state-of-the-art MAPF algorithm from this evidence. The grade for an honest implementation/evaluation project depends on the supervisor's rubric and required novelty.

## Completed evidence

- [x] Supported solver workflow, bounded demo, explicit failures and independent path validation.
- [x] Separate documented historical data problems; no silent reuse of unverified legacy claims.
- [x] Two actual train-only models, metadata/checksums and serialization verification.
- [x] Verified teacher dataset with raw states, labels, generating source and family provenance.
- [x] Frozen models, source, policies, schedule and analysis before opening final test outcomes.
- [x] Complete 648-run matrix; all failures retained; timing replacements, if any, fully disclosed.
- [x] Independent solution checks and solved-cost comparison across policies/repeats.
- [x] Family-level paired time estimates and map-stratified bootstrap uncertainty.
- [x] Separate, conditional peak-memory measurements and honest missing-telemetry handling.
- [x] Raw attempts, derived results, analysis scripts, environment lock and verified package manifest.
- [x] Read-only result verifier and runnable development demo.

## Presentation requirements

1. State the primary comparison and its result in the abstract. Use the exact final numbers, not the development result or the best selected subset.
2. Explain why 54 nested instances are 18 independent scenario families; timing repetitions do not multiply the sample size.
3. Distinguish correct labels from useful targets, imitation accuracy from runtime, and expansions from memory.
4. Show analytical and random-pool controls, not just a comparison against plain CBS. Identify secondary comparisons as exploratory.
5. Acknowledge the 15-second cap, only three known maps, small independent sample, cold-process timing and conditional memory selection.
6. Explain the implementation's scope relative to published ICBS, CG/WDG and learned conflict-selection literature.
7. Keep the test set retired for future tuning; any redesigned method needs a new holdout.

## Not completed or not established

- [ ] A matched external optimized reference-solver comparison. The current result covers local Python implementations.
- [ ] Unseen-map generalization or a production-grade multi-robot system.
- [ ] A novel triplet method, WDG implementation, or proof that a larger neural model would improve results.
- [ ] Remote CI execution or GitHub publication; the approved scope excluded publishing.
- [ ] Supervisor confirmation that an honest negative/limited result satisfies the submission rubric.

These are scope limits, not hidden finished features. The approved local study/report work is complete. If a positive ML gain is mandatory, that requirement remains unmet unless the final primary criterion passes; changing that conclusion requires new research rather than different wording.

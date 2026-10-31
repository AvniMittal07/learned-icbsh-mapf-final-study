# Revised ML scope — proposal for approval

31 October 2026. Status: **planning and existing-data review only**. No new solver experiment, model fitting, or final-test evaluation was performed for this review. The previous failed gates and reports remain unchanged.

## Recommendation and correction

Keep a trained, evaluated conflict selector as a core project deliverable. My recommendation to finish with only analytical baselines was premature: we have not yet evaluated a newly trained research model. The 40% informative-query threshold was an agreed collection-efficiency decision rule, not a scientific test proving that ML cannot work. The minimum of 12 informative training queries was also a pilot rule, not a sample-size guarantee.

The proposed revision explicitly permits an **exploratory** model despite the old collection gate failing. It does not relabel that failure as a pass. Separate three questions:

1. **Integrity:** are labels, features, splits, and inference correct? This remains mandatory.
2. **Evidence strength:** how many independent families and informative decisions support the model? Report this honestly, including sparse coverage and uncertainty.
3. **Usefulness:** does the learned policy improve complete solver runs against an analytical comparator with matched candidate access? Only empirical evaluation can answer this.

The revised research question is: **Can an inexpensive learned conflict scorer approximate an expensive lookahead policy and improve the runtime of our Python ICBS-h solver while retaining its optimal-search mechanisms?** Memory is measured as a separate outcome; a reduction is not promised.

## What the existing evidence says

The latest pilot contains 53 queries and 144 candidate actions. Its high-budget outcomes yield 27 strict training pairs in 10 queries from six families, and 21 validation pairs in nine queries from four families. This is enough to fit an exploratory regularized model; it is weak support for general performance claims.

An additional read-only review found:

| Diagnostic on fully solved queries | Training | Validation |
|---|---:|---:|
| Queries | 21 | 10 |
| All candidate expansion counts tied | 11 | 2 |
| Analytical choice within saved pool worse than best recorded choice | 4 | 4 |
| Mean hindsight relative reduction in descendant expansions | 5.1% | 6.2% |

The last row averages per-query reductions over the fully solved subset. It is an optimistic retrospective diagnostic, **not** a global runtime bound, an expected model gain, or evidence about the capped queries. The comparator here is restricted to the saved shortlist; it can differ from the full residual selector. No strict recorded pair had identical saved feature vectors, but distinguishable features do not establish learnability.

This suggests two problems beyond model architecture: the current analytical restriction may leave little useful choice, and complete descendant searches are an expensive, censored source of labels. A larger neural network does not resolve either problem by itself. Evidence is in `ML_SCOPE_EVIDENCE.json`; the script only read existing JSONL data.

## Why a lookahead teacher is a defensible direction

Huang, Koenig, and Dilkina use an expensive one-step conflict oracle and a linear ranking model that imitates it. Their oracle scores the child lower bounds using WDG. Our proposed teacher would use this repository's cardinal-graph/MVC heuristic, so it is an **adaptation**, not a reproduction of their algorithm or reported gains. [AAAI paper](https://taoanhuang.github.io/files/aaai21c.pdf)

The distinction between raising child path costs and raising child f-values is also the subject of f-aware conflict prioritization. [Boyarski et al., AAAI 2021](https://pathfinding.ai/pdf/bflbhsk-aaai21-facfihfcbs.pdf)

Learning to imitate an expensive computation can have value. Predicting a cheap formula already available at inference would require beating that formula directly to justify ML. Accordingly, teacher agreement will be a diagnostic, and complete-solver time will remain the decisive outcome.

## Proposed work within one 1.5-CPU-hour approval

### 1. Fit a clearly labeled exploratory reference model

- Use only the latest pilot's training families and supported strict comparisons. Do not pool the previous CBS-oracle targets with ICBS-h continuation targets.
- Fit one fixed pairwise linear Ridge model with alpha 10, no intercept, train-only standardization, and equal total pair weight per informative query.
- Preserve `old_collection_gate_passed: false` and mark the artifact exploratory. Validate saved-weight inference against in-memory predictions.
- Use the existing guarded shortlist for this reference model. This supplies an actual trained-policy experiment even if the new teacher later proves unhelpful. It does not certify adequate data coverage.

### 2. Build a verified immediate-child teacher

- Reuse the repaired constraint, shortest-path, and full conflict-classification code. Keep the legacy bulk generator quarantined: its old arbitrary infeasibility penalties and unverified labels are not reusable evidence.
- For a candidate conflict, construct both ordinary CBS child branches, find their optimal individual paths, classify their current conflicts, and compute child MVC values. Do not run each descendant search to completion merely to obtain this label.
- Record raw child g, raw MVC h, effective child f after the solver's pathmax rule, branch feasibility, and teacher computation cost.
- Predeclare the teacher priority as the lexicographic pair **(minimum effective child f, maximum effective child f)**, higher preferred. This secondary key is our declared adaptation. Equal pairs remain ties. No composite weights or arbitrary tie labels are introduced.
- Keep proven infeasibility distinct from a time cap. One infeasible child is represented as an absent branch/extended bound, not `parent_f + 15`. Unknown branches cannot provide a complete teacher rank. A node proven infeasible is recorded as such rather than used to invent training preferences.
- The target is a lookahead heuristic preference, **not ground-truth future runtime or optimal conflict choice**. Child-derived teacher quantities do not become model input features.

### 3. Give the learned policy a meaningful, controlled choice

Keep the highest cardinality tier and the normal two-branch splitting rule. For the revised policy, remove the requirement that every selectable conflict tie on the maximum residual score.

Use a deterministic pool of at most six conflicts from that tier: always include the full residual selector's winner, then select up to five other conflicts using a fixed seed/hash rule established before labels. Use exactly the same pool rule during collection and inference. Keep the original top-three analytical-tie gate as the reference variant.

Including the analytical winner lets direct residual selection remain a fair comparator. Add random selection within the revised pool to test whether any gain is due to the learned scores rather than the pool alone.

The extra analytical gate is a performance restriction; it is not the mechanism that establishes CBS optimality. Preserve exact low-level search, admissible high-level bounds, best-first ordering, and exhaustive ordinary conflict splitting. Test the revised selectable choices against independent tiny-instance optima and validate all returned solutions. Model scores never replace the heuristic or prune branches. These checks are finite evidence, not a universal correctness proof. The CBS search mechanisms are described in the [Huang et al. background section](https://taoanhuang.github.io/files/aaai21c.pdf).

### 4. Collect a bounded, reproducible replacement sample

- Retain the original family partitions. The final test families 24–29 on each map remain sealed.
- Use training families 08–19 on each map: 36 families. Select one of the existing 16/24/32-agent variants by a deterministic balanced rule, fixed before new labels.
- Use development-validation families 20–23 on each map: 12 families. They have already informed development and must not be presented as untouched final evidence.
- Target at most eight query states per training family and four per validation family: at most 336 queries. Choose query positions without inspecting teacher labels; save every selected pool action.
- Begin with a fixed training-only teacher audit. Record complete-label rate, ties, teacher/analytical disagreements, feature differences, and labeling cost. A teacher agreeing with the cheap rule on every audited choice supplies no demonstrated reason to spend the full generation allowance; in that case skip bulk teacher generation and finish the exploratory-model evaluation.
- Otherwise collect up to the fixed sample or resource cap, whichever occurs first. Use a common per-candidate offline teacher cap, initially one second. Record incomplete labels explicitly. No outcome-based selection of only successful families.

Sample size and informative yield are reported as evidence limitations, not converted into a claim of adequacy at an arbitrary percentage. The technical prerequisites for fitting remain verified data, strict comparisons with differing features, and multiple training families. If no valid learning signal exists, do not manufacture labels or weights.

### 5. Train one teacher-imitation model and evaluate matched controls

Use the same fixed linear learning specification as the exploratory reference. Train only on new teacher-labeled training data; keep the two target datasets separate. No architecture search, tree-model sweep, triplet expansion, or GPU requirement is proposed in this phase.

Development comparison, when the teacher model is trainable:

| Policy | Purpose |
|---|---|
| Plain ICBS-h | Solver reference |
| Direct residual selector | Main inexpensive analytical comparator |
| Random within the new pool | Candidate-pool control |
| Direct verified teacher within the new pool | Teacher quality and cost |
| Exploratory continuation-trained model with old guard | Test the current learned formulation |
| Teacher-trained linear model with new pool | Test the revised ML formulation |

Use 12 development instances, two timing repetitions, common eight-second end-to-end caps, fresh subprocesses, deterministic randomized scheduling, and identical memory policy. Six policies give 144 runs, at most 1,152 nominal solver seconds. Teacher evaluation includes its online cost; the learned model receives no teacher computation at inference. If a teacher cannot complete within the run deadline, report the timeout, without silently substituting another algorithm.

Record solved fraction, capped paired runtime, memory, path validity/cost, expansions, model invocation counts, and feature/inference costs. Analyze families as the independent units; repetitions are timing observations. Keep zero-gain and negative results. Do not claim lower memory merely because nodes decrease.

## Deliverables and success criteria

The phase should produce an actual exploratory learned selector, a verified teacher/data pipeline, a second learned selector if that teacher yields valid training signal, matched development comparisons, and a candid recommendation for the final study.

**Engineering completion** means reproducible training, reloadable weights, valid solver behavior, and a complete report. **Scientific success** requires useful held-out behavior against strong controls; it cannot be guaranteed. A weak or losing trained model is still an interpretable experiment. No trained model and a failed data pipeline must be reported as a narrower result, not as evidence that ML is ineffective.

Freeze the final candidate and study design only after this phase. Opening or running the final test set requires the subsequent approval. A positive development result is not a final speedup claim. Attribution and novelty must be modest: this is an empirical adaptation and evaluation, not a new invention of learning conflict selection.

## Budget and approval requested

| Allocation | Maximum reserved CPU time |
|---|---:|
| Correctness checks, data/model integrity, and report | 0.20 h |
| Training-only teacher audit | 0.20 h |
| Bounded teacher-data collection | 0.70 h |
| Fixed model fitting and serialization checks | 0.05 h |
| Development evaluation | 0.35 h |
| **Total** | **1.50 h** |

Current conservative reservations are 2.25 CPU-hours. This approval would put the cumulative envelope at 3.75 hours, leaving 4.25 within the original eight-hour ceiling. No paid compute, GPU acquisition, remote publishing, or final-test execution is included. The collection target is a cap, not a guaranteed number of useful labels; measured throughput determines how much can be collected within it.

**Requested decision:** approve this 1.5-CPU-hour ML revision, or choose a smaller 0.5-hour exploratory-model evaluation using existing labels only. The latter establishes a trained-policy experiment more quickly but leaves the sparse-data and restricted-choice problems largely unresolved. Implementation and new experiments wait for this decision, as required by the staged approval plan.

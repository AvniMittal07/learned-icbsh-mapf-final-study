"""Generate final reports and a hash-verified source/data/model package."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import zipfile

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
args=parser.parse_args();repo=Path.cwd();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
root=repo/'results/recovery/final_study';ml=repo/'results/recovery/ml_revision'
load=lambda path:json.loads(path.read_text())
p=load(root/'protocol.json');s=load(root/'summary.json');q=load(ml/'data_quality.json');integrity=load(ml/'independent_integrity.json')
rows=[json.loads(line) for line in (root/'analysis_rows.jsonl').read_text().splitlines()]
initial=[json.loads(line) for line in (root/'initial_runs.jsonl').read_text().splitlines()]
replacement=[json.loads(line) for line in (root/'replacement_runs.jsonl').read_text().splitlines()] if (root/'replacement_runs.jsonl').exists() else []
ledger=[json.loads(line) for line in (repo/'results/recovery/compute_ledger.jsonl').read_text().splitlines()]
phase=sum(e['reserved_cpu_seconds'] for e in ledger if e.get('phase')=='final-study')
total=sum(e['reserved_cpu_seconds'] for e in ledger)
assert phase<=10800 and total<=28800 and s['total_runs']==648 and s['valid_for_inference']
policies=s['policies'];b=policies['model_b'];a=policies['model_a'];res=policies['residual']
ci=b['stratified_family_bootstrap_95_interval'];ratio=b['geometric_family_time_ratio']
claim=s['primary_speed_claim_supported']
if claim:
    verdict='Model B met the prespecified speed criterion against residual selection on these held-out scenario families.'
elif ratio<1:
    verdict='Model B has a faster point estimate, but the prespecified evidence requirements for a speedup were not met.'
else:
    verdict='Model B did not establish a speed improvement over residual selection in the frozen final test.'
names={'cbs':'Plain CBS','plain':'Plain ICBS-h','residual':'Residual analytical','pool_random':'Random in revised pool',
       'model_a':'Model A: continuation / old guard','model_b':'Model B: child teacher / revised pool'}
fmt=lambda x:'not available' if x is None else f'{x:.3f}'
table='\n'.join(f"| {names[name]} | {v['solved_runs']}/108 | {v['solved_instances_both_repeats']}/54 | {v['mean_capped_seconds']:.3f} | {v['geometric_family_time_ratio']:.3f} | [{v['stratified_family_bootstrap_95_interval'][0]:.3f}, {v['stratified_family_bootstrap_95_interval'][1]:.3f}] |" for name,v in policies.items())
memory_table='\n'.join(f"| {names[name]} | {v['median_all_run_peak_memory_mib']:.2f} | {fmt(v['median_matched_family_memory_ratio'])} | {v['matched_memory_instances']} / {v['matched_memory_families']} |" for name,v in policies.items())
map_table='\n'.join(f"| {name} | {res['subgroups']['map_name'][name]['solved_runs']}/36 | {b['subgroups']['map_name'][name]['solved_runs']}/36 | {res['subgroups']['map_name'][name]['mean_capped_seconds']:.3f} | {b['subgroups']['map_name'][name]['mean_capped_seconds']:.3f} |" for name in sorted(res['subgroups']['map_name']))
size_table='\n'.join(f"| {k} | {res['subgroups']['k'][k]['solved_runs']}/36 | {b['subgroups']['k'][k]['solved_runs']}/36 | {res['subgroups']['k'][k]['mean_capped_seconds']:.3f} | {b['subgroups']['k'][k]['mean_capped_seconds']:.3f} |" for k in ('16','24','32'))
faults=s['timing_integrity'];actual_cpu_lower_bound=sum(r['process_cpu_seconds'] for r in initial+replacement)
max_launch=max(r['launch_seconds'] for r in rows);max_gap=max(r['max_monitor_gap_seconds'] for r in rows)
telemetry_table='\n'.join(f"| {names[name]} | {v['telemetry_runs']}/108 | {v['selector_totals'].get('ml_calls',0)} | {v['selector_totals'].get('feature_seconds',0):.3f} | {v['selector_totals'].get('inference_seconds',0):.3f} |" for name,v in policies.items() if name in ('model_a','model_b'))
matched=b['matched_instance_rows']
exp_ratio=(statistics.mean(r['policy_expansions'] for r in matched)/statistics.mean(r['baseline_expansions'] for r in matched)) if matched else None
source=p['source_sha256'];models={name:load(ml/(name+'_report.json')) for name in ('model_a','model_b')}
preflight=(root/'preflight.log').read_text(encoding='utf-8-sig');verification=(root/'verification.log').read_text(encoding='utf-8-sig')
assert 'Ran 4 tests' in preflight and '\nOK\n' in preflight.replace('\r','') and '"verified": true' in verification
hardware=load(root/'hardware.json')
readme=f'''# Learned conflict prioritization for optimal MAPF

This repository contains a repaired Python CBS/ICBS-h research implementation, two actual trained linear conflict selectors, provenance-checked training data, and a frozen final evaluation. **{verdict}**

The final study used 54 held-out instances from 18 scenario families on three known maps, six policies and two timing repeats: 648 analyzed runs. Model B versus residual selection was the primary comparison. Its family runtime ratio is **{ratio:.3f}**, with a map-stratified bootstrap 95% interval **[{ci[0]:.3f}, {ci[1]:.3f}]**. Lower is faster. Solved runs: model B **{b['solved_runs']}/108**, residual **{res['solved_runs']}/108**. This is evidence about these Python implementations and this workload, not optimized published solvers or unseen maps.

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

The final test is now consumed. Future development must use a fresh holdout. For an independently budgeted rerun, `reproduce_final --output NEW_DIRECTORY freeze` and its later stages preserve the delivered results; use `final_budget` for serial experiments with temporary Windows standby prevention. Do not reset the ledger to bypass its limits. The supplied evaluation already reserved {phase/3600:.3f} CPU-hours in this phase and {total/3600:.3f} cumulatively out of the approved eight-hour ceiling.

Full supported recovery instructions and historical phase details are in [RECOVERY.md](RECOVERY.md). The unverified legacy bulk dataset builder and historical all-data training workflow remain quarantined. The old failed labeling gate remains recorded; subsequent exploratory fitting was explicitly authorized and does not change that failed result.
'''
report=f'''# Final research report: learned conflict prioritization for optimal MAPF

31 October 2026. Base repository revision: `8f26ac31c372a8c4e5a6426fed4e74da997b9676`.

## Main result

**{verdict}** The primary model B / residual geometric family runtime ratio is **{ratio:.3f}**, with a map-stratified bootstrap 95% interval **[{ci[0]:.3f}, {ci[1]:.3f}]**. Both the effect and solved coverage matter: model B solved **{b['solved_runs']}/108 runs** and **{b['solved_instances_both_repeats']}/54 instances in both repeats**, compared with residual's **{res['solved_runs']}/108** and **{res['solved_instances_both_repeats']}/54**. The prespecified primary speed criterion **{'passed' if claim else 'did not pass'}**.

The approved project deliverables are complete: repaired search code, a provenance-checked dataset, two fitted models, an untouched-until-frozen final comparison, reproducible analysis and this report. Completion does not imply a successful performance hypothesis. The result applies to these Python implementations on synthetic scenario families from three known maps. It is not evidence of superiority over optimized published MAPF systems or generalization to unseen maps.

## Research question and attribution

The question was whether a cheap learned scorer can select conflicts more effectively than the repository's analytical selector while leaving optimal search mechanisms intact. CBS searches over constraints and solves individual constrained path problems at its low level. The original algorithm and its optimality argument are described by [Sharon et al., Artificial Intelligence 2015](https://www.sciencedirect.com/science/article/pii/S0004370214001386).

Conflict prioritization and bypassing are established ICBS ideas, described by [Boyarski et al., IJCAI 2015](https://www.ijcai.org/Proceedings/15/Papers/110.pdf). Cardinal-conflict aggregation can supply an admissible high-level heuristic, as developed by [Felner et al., ICAPS 2018](https://www2.cs.sfu.ca/~hangma/pub/icaps18.pdf). Our ICBS-h label denotes the local cardinal-priority, bypass and CG/MVC variant; it does not reproduce all published improvements such as meta-agent merging/restart.

Imitating a more expensive conflict oracle follows [Huang, Koenig and Dilkina, AAAI 2021](https://taoanhuang.github.io/files/aaai21c.pdf). Their method uses WDG-based oracle information. Our teacher instead uses CG/MVC and a declared secondary child-bound key. The residual comparator is a cached, f-aware-inspired policy in this repository, not a verified reproduction of every detail or gain in the [f-aware conflict-prioritization paper](https://pathfinding.ai/pdf/bflbhsk-aaai21-facfihfcbs.pdf). The contribution here is an audited empirical adaptation and evaluation, not the invention of learning conflict selection.

## Method and invariants

Agents move or wait on a four-neighbor grid. Vertex collisions and edge swaps are forbidden; agents remain at their goals after their paths finish. The objective is sum of path costs. The repaired implementation uses exact constrained space-time A*, a constraint-derived complete default low-level horizon, explicit incomplete-horizon/time outcomes, full contextual conflict counting and independent solution validation.

All ICBS-h policies keep highest-tier cardinality priority, ordinary two-branch splitting, best-first ordering, CG/MVC lower bounds and pathmax. Model predictions only choose an eligible conflict. They never remove a necessary branch, supply a heuristic value or alter the optimization objective. These mechanisms follow the CBS correctness framework; finite regression tests and cost agreement provide implementation evidence, not a universal proof that every program execution is optimal.

For a conflict between agents i and j, the residual selector uses cached alternative individual path costs and the parent's conflict graph. For each agent a, let `dg_a` be its constrained path-cost increase and `df_a = dg_a + MVC(G without a) - MVC(G)`. Its exact lexicographic key is `(min df, min dg, max df, max dg, degree_sum, pair_frequency, -conflict_time)`, higher preferred. It makes no extra A* calls for this ranking step, but still costs computation.

Model A uses the original analytical-tie/top-three guard and prior repaired ICBS-h descendant-expansion comparisons. Model B uses at most six highest-tier candidates: the full residual winner first, followed by up to five other conflicts selected by a frozen hash rule. Collection and deployment call the same pool function. The matched random-pool policy tests whether the broadened candidate access alone is useful.

For each candidate c, the offline teacher constructs its two ordinary CBS children. For feasible child b, `F_b = max(g_b + MVC(G_b), F_parent)`. The frozen preference is `(min(F_left,F_right), max(F_left,F_right))`, higher first. Ties remain ties. Proven infeasibility is an absent branch with an extended bound; timeout is unknown; a two-branch infeasible node is not used to invent strict labels. The target is a one-step lower-bound preference, **not ground truth about eventual search runtime**.

Both models use fixed pairwise Ridge, alpha 10, no intercept, train-only standardization and equal total pair weight per informative query. Input consists of the existing 24 features, including parent geometry/context and alternative-path quantities already available from conflict classification. Full child teacher bounds are excluded from model inputs. Both saved JSON models reproduced their fitted training-candidate predictions exactly. No parameter, feature, model or candidate-pool change was made from final-test outcomes.

## Dataset and development history

The legacy CSV/model claims were not reused as evidence for the repaired solver. Earlier collection gates failed and remain recorded. Exploratory fitting despite the old gate failure was subsequently authorized explicitly; that did not turn the old failure into a pass.

Model A has 27 strict training comparisons from 10 informative decisions in six families. The replacement teacher dataset has **237 states, 1,064 complete candidate labels and 2,128 feasible child records**. Training comprises 199 states from 30 families, yielding **1,277 strict comparisons from 182 informative decisions in 29 families**. Development comprises 38 states from 11 families and 245 strict comparisons. All 48 planned train/development families were attempted; seven yielded no eligible captured decision and were not replaced based on outcomes.

All child paths, constraints and costs were checked, all saved features and candidate pools replayed, and every child MVC was recomputed using a separate exhaustive vertex-cover routine. The training and development families are disjoint. Final families 24–29 on each map were sealed until the approved solver/model configuration and final schedule were frozen.

This dataset is useful for reproducible proxy-teacher experiments. It is small and dependent: multiple decisions belong to the same family, the collector follows residual selection, only early eligible states are sampled, and all maps are known during development. A large count of candidate pairs does not create an equally large count of independent problems.

Development already cautioned against equating imitation with usefulness: model B's pair accuracy was 85.3%, versus 87.5% for analytical ordering on the same teacher pairs. Models A and B were respectively 3.8% and 9.0% slower than residual on development mean capped time, with the same solved count. These development results motivated honest final measurement, not test-driven redesign.

## Frozen final experiment

- **Inputs:** 54 instances from 18 held-out families, six families per map on `random-32-32-20`, `room-32-32-4`, and `warehouse-32-32-aisles`. Each family supplies nested 16/24/32-agent variants from seeded synthetic starts/goals in the largest connected map component. These are not claimed to be the original Moving AI scenario files.
- **Policies:** plain CBS, plain ICBS-h, residual, random in the revised pool, model A and model B. The expensive direct teacher was evaluated during development and is not in the final matrix.
- **Runs:** two timing repetitions per policy/instance, fixed algorithm seed 71, randomized schedule seed 20261031; **648 analyzed runs**. Repetitions share the algorithm seed and are not independent stochastic trials.
- **Resources:** 15-second end-to-end caps, 4-GiB process memory limit, fresh subprocesses, serial execution, CPU affinity 0 and one numerical-library thread. Model loading and online feature construction are included. Hardware: {hardware['processor']}, {hardware['ram_bytes']/1024**3:.1f} GiB usable physical RAM, {hardware['os']}; Python 3.12.14. Background system activity and thermal state are not fully controlled. Figure-only dependencies were installed in an isolated directory during the early part of the batch; solver dependencies were unchanged, but this setup could contribute background load. All scheduled results are retained. A cleaner replication should install every dependency before benchmarking.
- **Correctness:** the worker and controller independently validate returned paths; analysis revalidates them and compares costs. **{s['returned_solutions_revalidated']} returned solutions** passed, with no solved-cost disagreements.

For each run, capped time is its measured wall time if solved, capped at 15 seconds; otherwise it is 15 seconds. First average the six observations (three agent counts × two repeats) inside each family. The primary estimate is the geometric mean of the 18 model-B/residual family ratios. The 95% interval resamples six families with replacement separately within each map, using 10,000 replicates and fixed seed 20261031. It does not treat the 648 runs as independent samples.

A speed claim was permitted only if that interval lies below one and model B has no fewer solved runs or instances solved in both repeats. Model A and other policy comparisons are secondary and exploratory, with no multiplicity adjustment. Memory is a separate conditional analysis over matched instances solved by both compared policies in both repeats.

## Final results

| Policy | Solved runs | Instances solved twice | Mean capped s | Geometric family ratio | Family bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
{table}

Ratios use residual selection as the denominator; below one is better. Model B's arithmetic mean capped-time ratio is {b['arithmetic_capped_time_ratio']:.3f}, a different summary from the primary geometric family ratio. It won on {b['family_time_wins']} families, lost on {b['family_time_losses']}, and tied on {b['family_time_ties']}. Equal capped values can conceal different latent times when both policies fail.

Model A's secondary geometric family ratio is {a['geometric_family_time_ratio']:.3f}, with interval [{a['stratified_family_bootstrap_95_interval'][0]:.3f}, {a['stratified_family_bootstrap_95_interval'][1]:.3f}] and {a['solved_runs']}/108 solved runs. Its result is reported regardless of direction; it is not used to replace the prespecified primary model after viewing test outcomes.

| Map | Residual solved runs | Model B solved runs | Residual capped s | Model B capped s |
|---|---:|---:|---:|---:|
{map_table}

| Agents | Residual solved runs | Model B solved runs | Residual capped s | Model B capped s |
|---|---:|---:|---:|---:|
{size_table}

![Final runtime ratios and family averages](final_runtime.png)

### Memory and search work

| Policy | Median all-run peak MiB | Median matched family memory ratio | Matched instances / families |
|---|---:|---:|---:|
{memory_table}

The pooled peak column mixes completion outcomes and includes the Python interpreter and imported libraries. The paired ratio averages peak bytes over matched completion cases within each family, then reports the median family ratio. It is conditioned on shared solvability and cannot justify an unconditional memory-saving claim. Observed memory for killed workers may miss a final peak. Node counts cannot substitute for measured memory.

On {len(matched)} instances solved by both model B and residual in both repeats, the ratio of their mean expansions is **{fmt(exp_ratio)}**. This is a matched solved-subset diagnostic; it excludes censored searches and is not the primary endpoint.

| Learned policy | Runs returning telemetry | Model calls | Feature seconds | Prediction seconds |
|---|---:|---:|---:|---:|
{telemetry_table}

Telemetry sums include only workers that returned it. Missing data from killed workers are not zeros. Full wall time, rather than prediction time alone, determines whether learning is useful. Improving a cheap dot product cannot by itself repair poor conflict choices.

The new policies' authoritative telemetry is the `selector_stats` object. Legacy top-level `ml_tiebreaks` and `analytical_shortcircuits` fields are not populated by the new selector and default to zero; they are not used in this analysis and must not be interpreted as absence of ML activity. This instrumentation compatibility limitation does not change the selected conflicts or timed runs.

## Integrity, timing deviations and compute

There were **{faults['initial_faults']} initial timing faults**, affecting **{len(faults['affected_instance_ids'])} instance blocks**, and **{faults['replacement_count']} replacement runs**. The frozen rule was to preserve all original attempts and replace all 12 policy/repetition runs for an affected instance once when wall time exceeded the cap by more than one second, provided the remaining budget allowed it. Final timing validity: **{faults['timing_valid']}**. The largest measured launch delay was {max_launch:.3f} seconds; largest recorded monitor-loop gap was {max_gap:.3f} seconds.

The preceding development experiment suffered Modern Standby interruptions; its original and replacement data remain in that phase's folder. The final budget controller temporarily requests system/display wakefulness and releases that request on exit. Persistent power settings are not modified. All initial final-test attempts are in `initial_runs.jsonl`; any retries are separate; `analysis_rows.jsonl` is derived and explicitly identified as such.

The approved final phase reserved **{phase/3600:.3f} of 3 CPU-hours**. Cumulative conservative reservations are **{total/3600:.3f} of 8 CPU-hours**. Full wall-cap reservations are charged even if work ends early. Worker CPU measurements across raw final attempts sum to {actual_cpu_lower_bound/3600:.3f} hours, a lower bound when workers were killed and excluding controller/report work. It must not be confused with the reservation ledger.

The unchanged solver/model source was hash-checked before testing, during execution and afterward. Four new final-harness tests passed on tiny synthetic cases, including comparison with the prior monitor and an independent optimum. The previous 32 recovery/ML regression tests remain applicable to the identical solver source. The final statistics were recomputed by the read-only submission verifier. The archive is CRC-checked and every packaged file has a manifest hash. Remote CI has not run; no remote publishing occurred.

## Candid assessment and remaining work

**Useful dataset:** yes, for the stated one-step teacher-imitation question. **Real trained ML:** yes. **Supported final speed improvement:** {'yes, within the narrowly stated benchmark scope' if claim else 'no under the prespecified criterion'}. **General memory improvement or state-of-the-art superiority:** not established by this study. **Deployment readiness:** a bounded research prototype, not a production-certified multi-robot planner.

An A+ grade cannot be guaranteed without the supervisor's rubric. The strongest defensible contribution is the repaired, reproducible evaluation and its honest result. A claim that the method is inherently faster and smaller than existing MAPF algorithms would overstate the evidence. The literature already contains learned conflict selection, so novelty must be framed as an adaptation and empirical investigation.

For a subsequent research iteration, prioritize three questions: (1) do teacher-preferred choices reduce matched eventual search work, not just one-step bounds; (2) does learning add value beyond analytical ordering when candidate access is held fixed; and (3) how does the repaired Python engine compare with a correctly configured external reference implementation? Profile feature construction and search separately, collect decisions across more independent families/maps, and evaluate any redesigned method on a **new holdout**. The final test in this package is now consumed. A larger model or triplet feature set is not a justified automatic next step without evidence for those questions.

The current approved work is complete. The package supplies a working demo, both models, data, raw results, source snapshots, tests, analysis and submission guidance. Further experiments require a new design and budget; this report does not silently extend the current scope.

## Provenance

Frozen protocol SHA-256: `{p['protocol_sha256']}`.

Unchanged solver-source SHA-256: `{source}`.

Model A SHA-256: `{p['model_sha256']['model_a']}`.

Model B SHA-256: `{p['model_sha256']['model_b']}`.

See `SOURCE_MANIFEST.json`, `results/recovery/final_study/frozen_source_manifest.json`, the protocol/schedule, and the append-only compute ledger for file-level evidence.
'''
card='''# Dataset card — verified child-bound conflict preferences

## Intended use

Train and inspect conflict-ranking models that imitate the declared CG/MVC one-step teacher. These labels do not establish the best conflict for eventual runtime, global search expansions or memory. Do not describe them as ground-truth optimal conflict choices.

## Contents

- `results/recovery/ml_revision/states.jsonl`: 237 pre-label query snapshots, with grid/map hash, scenario and split, constraints, current paths, g/h, depth, historical pair counts, candidate conflicts and 24 features.
- `labels.jsonl`: the same queries plus labels for every candidate in the deterministic pool; 1,064 complete candidate labels and 2,128 feasible child records.
- `protocol.json`, `source.zip`, `source.json`: frozen selection/labeling rules and generating code provenance.
- `harvest.jsonl`: all 48 planned families, including seven with no captured eligible query. No replacement of families based on label quality.
- `independent_integrity.json` and `data_quality.json`: split, replay, path/bound and yield checks.
- `model_a.json` / `model_b.json` and `.metadata.json`: real fitted weights, feature names, training-family IDs, hashes and provenance. Model A uses the separate repaired continuation-label corpus, not these teacher targets.

Training uses families 08–19: 199 saved states from 30 observed families, 182 informative states from 29 families, 908 candidate labels and 1,277 strict comparisons. Development uses families 20–23: 38 states from 11 observed families, 156 candidate labels and 245 strict comparisons. The frozen final evaluation used families 24–29 and is separate from label generation. Those test inputs are now consumed; they cannot serve as a fresh holdout for a later redesign.

## Label semantics

Each conflict creates two ordinary CBS child branches. For a feasible child, the recorded fields include `g`, `raw_h` (actual child graph MVC), `effective_f` after pathmax, and the replanned path. The priority is the lexicographic pair `(min child effective_f, max child effective_f)`, higher preferred. Equal pairs are ties; they are not converted into arbitrary positive/negative labels. A proven absent branch is encoded with `+inf`; a time cap is unknown; two absent branches mark a dead node. The current collected corpus has no capped candidate labels.

The pool is at most six conflicts from the highest cardinality tier: residual winner first, then up to five others selected using the frozen conflict hash. The same function is used at inference. Child-bound label fields must never be added to the model's online feature vector.

## Quality and limitations

All saved feature vectors and pools replayed. Child paths/constraints/costs were checked, and all child MVCs were independently recomputed. Training scalers use only training candidates, and serialized model predictions matched fitted predictions exactly. Historical CSVs and old models were not mixed into these fits; the previous failed data gate remains in its original report.

The data cover only three known maps and seeded synthetic scenarios. Many rows come from related states of the same scenario family, and capture favors early eligible decisions along residual-policy searches. Family-level splitting prevents nested variants leaking across partitions, but does not establish unseen-map generalization. The corpus supports this small proxy-imitation experiment, not a large-data or deployment claim. The final research report shows whether the trained policy improved actual complete-solver runs.
'''
checklist=f'''# Submission assessment and checklist

**Result to defend:** {verdict}

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
'''
report_path=out/'FINAL_RESEARCH_REPORT.md';report_path.write_text(report,encoding='utf-8')
(repo/'FINAL_RESEARCH_REPORT.md').write_text(report.replace('(final_runtime.png)','(results/recovery/final_study/figures/final_runtime.png)'),encoding='utf-8')
for name,text in [('DATASET_CARD.md',card),('SUBMISSION_CHECKLIST.md',checklist)]:
    (out/name).write_text(text,encoding='utf-8');(repo/name).write_text(text,encoding='utf-8')
legacy=repo/'README_LEGACY_UNVERIFIED.md'
if not legacy.exists():shutil.copy2(repo/'README.md',legacy)
(repo/'README.md').write_text(readme,encoding='utf-8')
recovery=repo/'RECOVERY.md';old=recovery.read_text(encoding='utf-8')
notice='''> **Final study completed:** Read [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md), [README.md](README.md) and [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). The final test is now consumed. The sections below preserve the earlier recovery and development workflow; their phase-specific pending plans are historical.

'''
if not old.startswith('> **Final study completed:**'):recovery.write_text(notice+old,encoding='utf-8')
for name in ('final_runtime.png','final_runtime.svg'):shutil.copy2(root/'figures'/name,out/name)
shutil.copy2(repo/'README.md',out/'FINAL_QUICKSTART.md')
shutil.copy2(root/'summary.json',out/'FINAL_RESULTS.json')
environment=dict(solver_packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','xgboost','psutil')},
    plotting_packages={name:importlib.metadata.version(name) for name in ('matplotlib',)},hardware=hardware)
(root/'package_environment.json').write_text(json.dumps(environment,indent=2))
status=dict(status='APPROVED_STUDY_AND_REPORT_COMPLETED',primary_speed_claim_supported=claim,
    verdict=verdict,final_runs=648,independent_families=18,final_test_consumed=True,
    phase_reserved_cpu_hours=phase/3600,total_reserved_cpu_hours=total/3600,
    remote_published=False,protocol_sha256=p['protocol_sha256'])
(root/'completion.json').write_text(json.dumps(status,indent=2));(out/'FINAL_STUDY_STATUS.json').write_text(json.dumps(status,indent=2))
excluded={'.git','__pycache__','.venv','venv','.pytest_cache'}
files=sorted(f for f in repo.rglob('*') if f.is_file() and not any(part in excluded for part in f.relative_to(repo).parts)
    and f.suffix not in {'.pyc','.pyo'} and f.name not in {'.env','SOURCE_MANIFEST.json','compute_ledger.lock'})
manifest=dict(base_revision='8f26ac31c372a8c4e5a6426fed4e74da997b9676',status=status,
    files={f.relative_to(repo).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
mp=repo/'SOURCE_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2));files.append(mp)
destination=out/'learned-icbsh-final-study.zip'
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,'learned-icbsh-mapf/'+f.relative_to(repo).as_posix())
with zipfile.ZipFile(destination) as z:
    assert z.testzip() is None
    for name,expected in manifest['files'].items():assert hashlib.sha256(z.read('learned-icbsh-mapf/'+name)).hexdigest()==expected
archive_sha=hashlib.sha256(destination.read_bytes()).hexdigest()
(out/'learned-icbsh-final-study.sha256.txt').write_text(f'{archive_sha}  {destination.name}\n')
print(json.dumps(dict(verdict=verdict,archive=str(destination),files=len(files),bytes=destination.stat().st_size,
    sha256=archive_sha,verified=True,phase_cpu_hours=phase/3600,total_cpu_hours=total/3600,
    primary_ratio=ratio,primary_interval=ci),indent=2))

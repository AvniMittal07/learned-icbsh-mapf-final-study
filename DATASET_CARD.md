# Dataset card — verified child-bound conflict preferences

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

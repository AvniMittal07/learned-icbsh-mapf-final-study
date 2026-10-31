# Publication provenance

This repository publishes the verified final submission snapshot of the learned
conflict prioritization MAPF project. It contains source, trained models,
datasets, saved experiment evidence, and the final research report.

The audited development snapshot was based on
[GabaSatvik/learned-icbsh-mapf](https://github.com/GabaSatvik/learned-icbsh-mapf/tree/phase3-cl-icbsh-final),
commit `8f26ac31c372a8c4e5a6426fed4e74da997b9676`. The original project context also
references [anay-28/learned-icbsh-mapf](https://github.com/anay-28/learned-icbsh-mapf).
This is a new repository containing a snapshot; it does not import the upstream
Git history or imply that all inherited implementation work is new.

`SOURCE_MANIFEST.json` records the frozen submission file hashes. Its
`remote_published: false` field records the state when that research package was
created, before this GitHub publication. It has deliberately been preserved with
the experiment evidence. This publication note is outside that frozen manifest.

The final test set is consumed. Model B did not establish a speed improvement
over residual selection. See `FINAL_RESEARCH_REPORT.md` for results and limits.

# Agent wiki index

Agent-first memory. Each entry states what question the page answers, so a search can stop at
this file when the answer is elsewhere.

## Start here

- `project.md` — invariants, architecture, dependency audit, migration state, verification status,
  and unresolved external validation. **Read first for anything about how the pipeline is built.**
- `hardening-backlog.md` — standing worklist with everything already closed and why. **Read before
  starting new work so a finished investigation is not repeated.**

## Verified axioms — checked empirically, do not re-derive

- `condition-alignment.md` — **do brain and model RDMs index the same conditions in the same
  order?** Yes. Verified through both sides plus the MPNet reference, and the once-implicit
  `np.unique` sort is now a named, tested contract. Also: which conditions the 8x100 sampling
  leaves unused, and why that is not a bias.
- `searchlight-coverage.md` — **what are the projection NaN warnings?** Not out-of-mask vertices:
  undefined searchlight centres, 0% of centres for subjects 1/4/7 but 11% for subject 8. Excluded
  correctly, and the NaN pattern is model-invariant so the paired contrast stays fair.

## Findings about the lens itself — model-side, no fMRI

- `lens-geometry.md` — **how much does J warp representational geometry, and do its directions
  matter?** The brain effect appears where the lens warps *least* (prediction inverted). All warp
  is anisotropy, none is rotation. The directions are data-aligned at 46.6 control-SDs beyond a
  spectrum-matched random map. Includes the `raw @ J^T` end-to-end extraction audit.

## Findings about the readout

- `why-pooling-won.md` — **why did pooling beat the endpoint readout 5-9x?** Because 73.7% of
  caption prompts end in the same token, a period. The endpoint RDM correlates 0.017-0.067 with an
  MPNet semantic reference; pooled reaches 0.41-0.58. Justifies the readout change without fMRI.
- `pooling-width.md` — **how many tokens are actually needed?** Four gives 6.7x the endpoint;
  saturates ~32 tokens at 97.5% of full. Punctuation is mildly *harmful*, so `no_punct` is a
  candidate improvement — untested brain-side.
- `whole-prompt-pooling.md` — the pooled readout's contract: scope, mask rule, statistical
  families, environment recipe, per-worktree venv trap, CPU-only TensorFlow, and the eight-subject
  result.

## Findings about the results

- `subject-consistency.md` — **is the J effect driven by one subject?** No: 7/8 signs, leave-one-out
  survives every drop. But per-subject variance tracks lens conditioning at Spearman +1.000, so the
  early-layer nulls are partly a precision problem, not only an effect-size one.
- `layer_performance_summary.md` — audited report semantics, derived-artifact exception, generator
  validation contract. **Historical: describes final-token results.**

## Superseded / historical

- `matched-readout-control.md` — the Lucas matched-readout control. Its README section was removed
  when pooling became the documented method; the code path and docs remain.
- `image_only_wikitext_pilot.md` — strict image-only transfer contract, implementation boundaries,
  run inputs, validation state. Independent of the readout change.

## Re-runnable checks (all in `scripts/`, all read committed artifacts)

| script | answers |
|---|---|
| `verify_mpnet_reference.py` | does the copied MPNet RDM match a fresh recomputation? |
| `audit_searchlight_coverage.py` | are undefined centres model-invariant, so the contrast is fair? |
| `analyze_subject_consistency.py` | is the effect distributed across subjects? |
| `analyze_lens_geometry.py` | how much does J warp the RDM, and how conditioned is it? |
| `analyze_lens_controls.py` | do the lens's directions matter, or only its spectrum? |
| `analyze_readout_semantics.py` | which readout carries caption semantics? |
| `analyze_pooling_variants.py` | how many tokens does the readout need? (needs GPU + model) |

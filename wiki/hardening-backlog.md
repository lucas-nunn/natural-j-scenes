# Hardening backlog

Standing worklist for the overnight hardening loop (started 2026-08-15). Ordered by value.
Each item records why it matters, so a later iteration does not re-derive it.

## Done

- [x] **Measure the metric-warp axiom.** Result inverts the naive prediction — see
      [[lens-geometry]]. `scripts/analyze_lens_geometry.py`, `docs/lens_geometry.json`.
- [x] **Condition ordering between brain and model RDMs.** Verified sound; the guarantee rested on
      an implicit `np.unique` sort, now extracted to `condition_column_index()` and pinned by
      `tests/test_condition_alignment.py`. See [[condition-alignment]].

## Open — assumptions and axioms

- [ ] **Rank-matched raw control.** The single highest-value follow-up. Project raw features onto
      the top-k right singular directions of `J_l` (k = that layer's effective rank) and re-run RSA.
      If it reproduces the J result, the effect is about conditioning, not about the lens direction.
      Would settle whether the layer profile means anything. See [[lens-geometry]].
- [ ] **Is the sampling really disjoint *and* exhaustive?** `sampling_is_disjoint` is checked, but
      8x100 = 800 of 835 conditions per subject. Confirm which 35 are dropped and whether the drop
      is systematic (e.g. always the same NSD ids across subjects).
- [ ] **MPNet reference is treated as a fixed anchor.** It matched to 4 decimals across two runs,
      which is strong. Confirm it is genuinely recomputed rather than copied from the MPNet tree.

## Open — correctness and simplification

- [ ] **`readout_mode` still defaults to `final_token`** while all documentation now says pooling is
      the method. Doc/code divergence. Changing the default is a behaviour change affecting the
      historical namespace — needs a decision, not a silent edit.
- [ ] **`uv.lock` is untracked and not ignored** in the main worktree. For a reproducibility-focused
      project an uncommitted lockfile is likely an oversight.
- [ ] **`_load_historical_final_token_scores` hardcodes `(8, 16, 23, 30)`** in its `wanted` list,
      duplicating the layer set that `resolve_source_layers` derives elsewhere. Single-source it.
- [ ] **Projection emits `Mean of empty slice` / `Degrees of freedom <= 0`** warnings. Almost
      certainly out-of-mask vertices, but unverified — confirm and silence deliberately, or fix.
- [ ] **The 5 model-extra tests skip in any venv without `[model]`.** They now pass when the extra is
      present. Consider a CI-visible marker so a green run cannot be mistaken for full coverage.

## Open — documentation

- [ ] **`docs/MATCHED_READOUT_*` and `docs/LAYER_PERFORMANCE_SUMMARY.md` still describe final-token
      results** that no longer appear in the README. Either mark them explicitly as superseded
      historical records, or remove them together with the code paths they document.
- [ ] **`docs/DESIGN.md` "Prompt" section still presents `visualize` as the chosen primary prompt**,
      but pooling is defined only for `plain`, so `visualize` is unreachable under the documented
      method. Contradiction to resolve.

## Open — extensions worth trying

- [ ] **Layer sweep.** Only 4 of 31 available lens layers are used. A cheap extraction over more
      layers would turn the 4-point layer profile into a real curve, which is what the depth claim
      needs.
- [ ] **Pooling variants.** Mean is one choice; last-k mean, or attention-mask-weighted variants,
      are one-line changes and would show whether the gain is about pooling per se or about
      averaging away endpoint noise.
- [ ] **Per-subject consistency.** The exact test only uses sign agreement. Report per-subject
      deltas so a single dominant subject cannot masquerade as a population effect.

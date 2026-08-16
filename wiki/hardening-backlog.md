# Hardening backlog

Standing worklist for the overnight hardening loop (started 2026-08-15). Ordered by value.
Each item records why it matters, so a later iteration does not re-derive it.

## Done

- [x] **Single-source the analysis layer set.** Was hardcoded in 8 places while extraction resolves
      layers at run time. Now `config.ANALYSIS_LAYERS` + `validate_analysis_layers`, enforced in
      `summarize` so a `--layers` run fails loudly instead of quietly shrinking the BH families.
- [x] **Commit `uv.lock`**; **resolve the DESIGN.md prompt contradiction**; **add layer-profile
      interpretation guardrails to the README**.
- [x] **Projection NaN warnings — NOT out-of-mask vertices.** They are undefined searchlight
      centres, 0% for subjects 1/4/7 but 11% for subject 8. Excluded correctly via nanmean, and the
      NaN pattern is model-invariant so the paired contrast is fair — now audited rather than
      assumed. See [[searchlight-coverage]].
- [x] **Per-subject consistency.** Effect is distributed (7/8 signs, LOO survives every drop), and
      per-subject variance tracks lens conditioning at Spearman +1.000 — early-layer nulls are partly
      a precision problem. See [[subject-consistency]].
- [x] **Do the lens directions matter, or only conditioning?** Both: rotation contributes nothing
      (all warp is anisotropy), but the directions are data-aligned — the real lens warps 46.6
      control-SDs more than a spectrum-matched random map. Still does not rescue the mechanism.
      See [[lens-geometry]]. Also audits `raw @ J^T` against stored features (~5e-07).
- [x] **Measure the metric-warp axiom.** Result inverts the naive prediction — see
      [[lens-geometry]]. `scripts/analyze_lens_geometry.py`, `docs/lens_geometry.json`.
- [x] **MPNet reference ordering.** Recomputed from raw embeddings; all 8 subjects reproduce the
      stored RDM exactly. Source table and index convention pinned. `scripts/verify_mpnet_reference.py`.
- [x] **Sampling coverage.** The 35 unused conditions per subject are the remainder of 8x100 of 835,
      differ per subject, and are uniformly positioned. No systematic exclusion.
- [x] **Condition ordering between brain and model RDMs.** Verified sound; the guarantee rested on
      an implicit `np.unique` sort, now extracted to `condition_column_index()` and pinned by
      `tests/test_condition_alignment.py`. See [[condition-alignment]].

## Open — assumptions and axioms

- [ ] **Brain-side rank-matched control.** The model-side question is now answered (directions are
      data-aligned), but the *brain* version is still open: feed a spectrum-matched random map's
      features through RSA and check it does not reproduce the layer-23/30 effect. Needs a
      searchlight run (~1 h CPU), so it is the main remaining compute-bound item.

## Open — correctness and simplification

*(cleared this round: hardcoded layer set, uv.lock, DESIGN.md prompt contradiction, README guardrails)*

- [ ] **`readout_mode` still defaults to `final_token`** while all documentation now says pooling is
      the method. Doc/code divergence. Changing the default is a behaviour change affecting the
      historical namespace — needs a decision, not a silent edit.
- [ ] **The 5 model-extra tests skip in any venv without `[model]`.** They now pass when the extra is
      present. Consider a CI-visible marker so a green run cannot be mistaken for full coverage.

- [ ] **Between-subject measurement support is unequal and unreported.** Subjects contribute means
      over 290k–406k centres, and subjects 6/8 lose 8–11% on top of that. The group mean weights
      subjects equally regardless. Not a bias in the paired contrast, but worth surfacing.

## Open — documentation

- [ ] **`docs/MATCHED_READOUT_*` and `docs/LAYER_PERFORMANCE_SUMMARY.md` still describe final-token
      results** that no longer appear in the README. Either mark them explicitly as superseded
      historical records, or remove them together with the code paths they document.

## Open — extensions worth trying

- [ ] **Layer sweep.** Only 4 of 31 available lens layers are used. A cheap extraction over more
      layers would turn the 4-point layer profile into a real curve, which is what the depth claim
      needs.
- [ ] **Pooling variants.** Mean is one choice; last-k mean, or attention-mask-weighted variants,
      are one-line changes and would show whether the gain is about pooling per se or about
      averaging away endpoint noise.

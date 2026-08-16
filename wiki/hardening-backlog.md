# Hardening backlog

Standing worklist for the overnight hardening loop (started 2026-08-15). Ordered by value.
Each item records why it matters, so a later iteration does not re-derive it.

## Done

- [x] **Whole-cortex localisation, with SNR control.** The absolute pattern tracks signal level
      (r=0.798); normalised, the J advantage is a near-uniform ~8% proportional gain everywhere,
      and frontal cortex is marginally *higher*, not deficient. See [[cortex-localisation]].
- [x] **Spatial localisation of the J advantage — workspace prediction fails.** No sensory->
      association gradient: the effect is 8/8 significant in *early visual* cortex and weakest in
      parietal. Limitation: `streams` is visual-only, so the frontoparietal test is still open.
      See [[stream-localisation]].
- [x] **Superseded docs marked HISTORICAL.** `LAYER_PERFORMANCE_SUMMARY.md` and both
      `MATCHED_READOUT_*` docs now carry a banner explaining the single-position readout and that
      the caption-only figures are close to a measurement of nothing.
- [x] **Between-subject coverage surfaced.** `summarize` now emits `subject_coverage`; support
      ranges 290,914–406,399 centres and 0.889–1.000 usable. Results unchanged.
- [x] **Layer 15 run against cortex — hypothesis refuted.** The semantic peak is not significant
      brain-side (p=0.28) and is roughly half the layers 23/30 effect. Semantic and brain layer
      profiles correlate at Spearman −0.500. Also verified end-to-end determinism to 9 decimal
      places. See [[layer15-experiment]].
- [x] **`readout_mode` default investigated — it was load-bearing.** `_load_historical_final_token_scores`
      resolved the comparator namespace via `group_name(profile)` with no explicit mode, so flipping
      the default would have silently repointed it into the pooled namespace. Now pinned to an
      explicit `config.FINAL_TOKEN`, with a test that survives a patched default. Flipping the
      default is now safe, and remains a deliberate decision rather than a silent edit.
- [x] **31-layer depth sweep.** Depth and effective rank are collinear at r=0.971 — this design
      *cannot* separate them, and the partials are unstable. The J advantage is non-monotone
      (negative at layers 0-6, peaks ~l15, negative again at l30). The MPNet proxy disagrees with the
      brain result in sign at l30, bounding where the proxy may be used. See [[depth-sweep]].
- [x] **Pooling width sweep.** Steep recovery (4 tokens = 6.7x the endpoint), saturates ~32 tokens
      (97.5% of full), and punctuation is mildly *harmful* — `no_punct` beats the full mean by
      +0.018. See [[pooling-width]].
- [x] **Explain WHY pooling won, without fMRI.** 73.7% of prompts end in the same token ('.'); the
      endpoint RDM correlates 0.017-0.067 with an MPNet semantic reference vs 0.41-0.58 for pooled,
      and the two readouts correlate only 0.04-0.15. See [[why-pooling-won]].
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

- [ ] **Decide whether to flip `DEFAULT_READOUT_MODE` to pooled.** Now *safe* to do (the historical
      namespace no longer follows it), but still a user-visible behaviour change: a bare
      `jlens-nsd extract` would switch readout. A decision for Lucas, not a silent edit.
- [ ] **The 5 model-extra tests skip in any venv without `[model]`.** They now pass when the extra is
      present. Consider a CI-visible marker so a green run cannot be mistaken for full coverage.


## Open — documentation


## Open — extensions worth trying




- [ ] **Test `no_punct` brain-side.** DELIBERATELY NOT RUN overnight: needs a third readout mode,
      which means touching ~10 `== ALL_TOKEN_MEAN` branches in a pipeline just verified
      deterministic to 9 d.p. Prior also dropped after layer 15 failed to transfer. Worth running
      rested — and it doubles as a test of where the semantic proxy's validity boundary sits, since
      this is a *readout* comparison (where the proxy worked) not a *layer* one (where it failed).
      Original note: Semantically it beats the production readout (+0.018), but
      that is one number with no significance test and no fMRI. Needs extraction + searchlight
      (~1 h) run through the predeclared machinery before it could be adopted. See [[pooling-width]].
- [ ] **Re-read historical caption-only conclusions.** The `plain` final-token numbers
      (0.003-0.006) are close to a measurement of nothing, not a weak finding. Any earlier claim
      resting on that condition needs revisiting. See [[why-pooling-won]].

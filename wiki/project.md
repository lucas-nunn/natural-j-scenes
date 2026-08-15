# Jacobian Lens × NSD agent memory

## Repository state

- Standalone root: `/home/chuddy/dev/research/jlens-nsd`.
- Active concise-montage worktree:
  `/home/chuddy/dev/research/jlens-nsd-four-subject-maps` on branch
  `concise-four-subject-maps`.
- Extracted from `neuroconnectionism/lucas_exploration/jlens_experiment` without
  changing the source checkout.
- Generated results, logs, arrays, maps, notebooks, model weights, and caches
  are intentionally absent and ignored.
- Package/import: `jacobian-lens-nsd` / `jlens_nsd`; CLI: `jlens-nsd` and
  `jlens-nsd-orchestrate`.

## Scientific invariants

- Subjects 1–8; sessions 1–10; sorted unique three-repeat conditions.
- NSD IDs remain 1-based except the single caption lookup subtraction.
- Established source-data counts: 835 conditions/subject, 6,148-ID union,
  eight mutually disjoint 100-image samples/subject. These facts come from the
  source project's prior data-backed run and require external data to recheck.
- Prompt text and order remain byte-for-byte in `prompts.py`: `visualize` then
  `plain`; no generation, chat template, or truncation.
- Feature order remains prompt → sorted selected layer → raw/J, then final.
- Hooked block `l` is checked against HF `hidden_states[l+1]` except the final
  block; final residual always comes from the block hook.
- Native float32 residual units, correlation-distance RDM, no normalization.
- Group manifest is the only mapping between lexical filenames, one-based
  model indices, projection surfaces, plots, and report features.

## Matched-readout extension

Lucas's new control does not supersede the historical prompt pair. It is an
isolated `matched_readout` prompt set and filesystem namespace. Detailed hard
gates, statistics, and live run provenance are cached in
`wiki/matched-readout-control.md` and documented for review in
`docs/MATCHED_READOUT_CONTROL.md`.

The matched-readout subject-1 validation and eight-subject run completed on
2026-08-15. Exact run provenance and results are in
`wiki/matched-readout-control.md`; human-facing results and compact artifacts
are in `docs/MATCHED_READOUT_RESULTS.md`.

## Dependency audit

- Audited upstream/merge-base commit:
  `a60e0eafb8d02841159e344adb732062734bc302`.
- Historical URL `KietzmannLab/nsd_visuo_semantics` is unavailable as a Git
  remote as of 2026-08-15; public continuation `adriendoerig/visuo_llm` has the
  same audited HEAD/history.
- Unchanged upstream APIs used: condition/mask/model-RDM/behavior helpers,
  BatchGen, RSASearchLight, TF RDM/sphere utilities, NSDmapdata.
- Local-only adapter requirements: streamed beta averages, streamed grouped
  correlations, explicit subject/session/sample selection, selective
  projection, named ROI contour plotting.
- Anthropic Jacobian Lens optional dependency pin:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

## Conflict with old project memory

The source wiki says implementation lives under
`lucas_exploration/jlens_experiment` and commands run from the old repository.
That location is now historical: the requested standalone repository replaces
it for future work. Scientific facts are preserved; paths/imports are migrated
to explicit CLI/env configuration. Existing results may remain in place and
be referenced without copying.

## Verification status

- Final integrated validation after all four feature merges: 40 tests pass,
  two extraction-helper tests skip because the optional model extra is absent,
  and eight montage subtests pass. This supersedes earlier 18/19/21-test
  counts. Run via `.venv/bin/python -m pytest` so the repository-root
  `scripts` namespace is on the import path.
- Ruff lint and formatting checks pass.
- Source distribution and wheel build successfully with `uv build`.
- A dry-run resolver check succeeds across all optional dependency groups.
- The wheel installs into a clean Python 3.12 environment; distribution import,
  console `--help`, and the model/data-free smoke command pass.
- `git diff --check` passes and only source/docs/tests/wiki are present.
- The main caption/matched-readout pipeline's generated GPU/model/NSD outputs
  remain external and ignored. The isolated image-only pilot was fully checked
  against those local inputs; its compact report is committed while generated
  features, maps, surfaces, and per-feature plots remain ignored.

## Four-branch integration (2026-08-15)

- Merge commits retained without rebasing: layer summary `25849d6`, concise
  four-subject maps `4c00271`, matched readout `afd235b`, and image-only pilot
  `0916455`. Feature tips `a4b5e31`, `f767863`, `36ee6b3`, and `7111ac6` are
  all ancestors of `main`.
- Repository-wide Ruff lint/format, `git diff --check`, checked-image opening,
  and the complete model-free suite pass after integration.
- README montages remain subjects 1–4 only; all population tables and
  statistics remain based on eight subjects. Peak values are explicitly
  searchlight-centre RSA correlations, never single-voxel correlations.

## README J-space readout figure

- `docs/assets/visualize_layer23_jspace_readouts.png` is generated by
  `scripts/make_jspace_readout_figure.py` from local artifacts; no image or
  vocabulary item is invented.
- Deterministic selection: filter the sorted 835-condition subject-1 set to
  local COCO metadata carrying CC BY 2.0 (146 conditions), then use indices
  `floor(n/6)`, `floor(n/2)`, and `floor(5n/6)`. Selected one-based NSD
  conditions are 10543, 34275, and 60417; COCO IDs are 23163, 104825, and
  207117.
- The exact stored feature is `visualize__l23__j` from the completed original
  all-subject chunks, matched by condition ID. Target and source manifests
  agree on model, captions, lens hash, adapter revision, prompt version,
  position, dimension, and residual normalization. Their prompt source hashes
  differ only because migration changed `typing.Sequence` to
  `collections.abc.Sequence`; prompt construction is unchanged.
- CPU unembedding mirrors `HFLensModel.unembed`: cast the stored float32
  transported vector to the tied head's bfloat16 dtype, apply Qwen3.5 final
  RMSNorm in float32 with epsilon `1e-6` and additive checkpoint weights
  `(1 + weight)`, cast back to bfloat16, then multiply by the tied token
  embedding/output-head matrix. There is no final logit softcap in this
  checkpoint.
- Display filtering is deterministic and formatting-only: special IDs, empty
  decodes, markup/entities, escaped whitespace, dot-prefixed file extensions,
  bare URL schemes, non-alphanumeric-only decodes, and NFKC-casefold
  duplicates. The figure preserves unfiltered vocabulary ranks and logits.
- The PNG is 1800×1080 RGB, embeds its audit metadata, and has SHA-256
  `02edd1d95138fc26044c6eb2b863c337fa3e13abbe9151c184e7e66bab6d2d35`.
  Stimulus/vector SHA-256 values are embedded in the PNG metadata and printed
  by the reproduction script.

## Caption-only layer-23 comparison

- The extraction manifest calls the caption-only prompt `plain`; the exact
  requested feature is therefore `plain__l23__j` (matched brain-map control
  `plain__l23__raw`), not `caption__l23__j`.
- `docs/assets/plain_layer23_jspace_readouts.png` uses the unchanged CC BY 2.0
  gate and deterministic condition rule from the visualize figure. It selects
  one-based NSD conditions 10543, 34275, and 60417 (COCO 23163, 104825, and
  207117). The PNG is 1800×1080 RGB with SHA-256
  `8fa9192694e1bcb0fb2d4ed83e6d5cd7359e8a0fe913702e174a1b60c7b38339`.
- Each PNG row records `source_array=plain__l23__j`, its source chunk, vector
  hash, and the distinct matched `visualize__l23__j` vector hash. Caption-only
  vector hashes are `b8d1a032...feeb2`, `11e27c40...ca42`, and
  `16adada4...0c9d`; the full values are embedded in the audit metadata.
- Actual filtered display readouts, preserving raw vocabulary rank, token ID,
  and logit, are: NSD 10543: What (rank 9, ID 3437, 13.25), Answer (10, 21134,
  12.9375), Correct (15, 38643, 12.5), Choose (19, 21513, 12.0), Why (23,
  8169, 11.6875); NSD 34275: Close (3, 12659, 13.1875), -close (15, 33318,
  10.125), Details (19, 11956, 9.9375), Description (20, 7419, 9.875), You
  (21, 1394, 9.8125); NSD 60417: Choose (8, 21513, 13.0625), Correct (9,
  38643, 13.0), What (11, 3437, 12.625), These (12, 4081, 12.25), Answer (13,
  21134, 12.1875).
- `docs/assets/plain_layer23_raw_then_j_subjects1-4.jpg` composes, without
  reanalysis, the existing raw then J maps for subjects 1–4. It is 2240×2064
  RGB with SHA-256
  `ad338c7d0b993fc9bbc8e0ec87228c38e524076ccac499042e9b5ad2c692b624`.
  EXIF audit metadata records all eight source filenames, dimensions, and
  hashes in subject-major/raw-then-J order; each source panel retains its
  original independent symmetric color scale.
- Vocabulary unembedding is interpretive only. Brain RDMs use the complete
  2,560-dimensional vectors; task-direction-like top tokens and
  representational geometry are different descriptive measurements.

## Concise README brain-map presentation

- README brain-map montages intentionally display subjects 1–4 only for
  concision. This is a presentation subset: every numerical/statistical
  analysis and table continues to use all eight subjects. There is no conflict
  with the eight-subject scientific invariant above.
- `scripts/make_layer23_brain_map_montage.py` defaults to the ordered subject
  tuple `(1, 2, 3, 4)` and accepts an explicit comma-separated `--subjects`
  list. It rejects duplicate, non-integer, empty, and out-of-range selections;
  canvas height is `HEADER_HEIGHT + row_count * ROW_STRIDE`.
- Both README montages use raw-left/J-right ordering and were regenerated from
  the authoritative qwen4b subject maps under the historical experiment result
  tree. No RSA, projection, thresholds, source plots, or color limits were
  recomputed.
- `docs/assets/visualize_layer23_raw_then_j_subjects1-4.jpg` is 2240×2064 RGB,
  contains eight audited panels, and has SHA-256
  `44f092c94f4a5699b1af574800e8d74a53e5158a3015919dc08b3e296fd894cf`.
- `docs/assets/plain_layer23_raw_then_j_subjects1-4.jpg` is 2240×2064 RGB,
  contains eight audited panels, and has SHA-256
  `ad338c7d0b993fc9bbc8e0ec87228c38e524076ccac499042e9b5ad2c692b624`.
- The obsolete `*_all_subjects.jpg` montages were removed. Tests cover subject
  parsing, dynamic dimensions, panel placement, exact source count/order,
  source hashes, and EXIF audit round-tripping for generated and checked-in
  assets.

## Subject-subset execution

- The CLI and durable orchestrator accept an explicit subject subset. It scopes
  condition preparation/union, extraction, grouped RDMs, searchlights,
  projection, plots, and summary artifacts while preserving the eight-subject
  default.
- A one-subject report is explicitly descriptive; its `n=1` confidence limits
  are undefined and must not be interpreted as population inference.

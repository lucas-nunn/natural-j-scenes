# Image-only WikiText transfer pilot memory

## Fixed contract

- Branch/worktree: `image-only-wikitext-pilot` in
  `/home/chuddy/dev/research/jlens-nsd-image-pilot`; never write to the main
  worktree, MPNet results, or source NSD trees.
- Subject 1, sessions 1–10, existing matched sampling row 0, exactly 100
  images. Subject conditions are sorted unique 73K IDs appearing exactly three
  times, matching upstream beta order.
- No semantic text or chat template. Literal processor input is only
  `<|vision_start|><|image_pad|><|vision_end|>`.
- Decoder block outputs 8, 16, 23, 30 have matched raw/J image-token means;
  block 31 is the un-normalized final decoder-block raw control.
- Released 2560×2560 float32 WikiText matrices are loaded unmodified. Label it
  exactly as an “image-only, decoder-residual, out-of-distribution transfer”
  after the Qwen vision tower, spatial merger, and 2560-D projection; it is not
  a vision-tower lens.
- Correlation-distance 100×100 RDMs (4,950 condensed entries), established
  radius-6 subject-1 brain searchlights, descriptive means only, no p-values.

## Token evidence established before implementation

Using Transformers 5.9.0 `Qwen3VLProcessor` on selected NSD condition 683:

- input length 171: IDs 248053, 169 × 248056, 248054;
- grid `[1, 26, 26]`, spatial merge 2, hence 169 decoder image positions;
- multimodal types 0, 169 × 1, 0;
- `(input_ids == image_token_id)` exactly equals `(mm_token_type_ids == 1)`.

The implementation rejects any extra attended token, non-contiguous image run,
boundary mismatch, modality-mask mismatch, or grid/count mismatch.

## Isolated implementation

- `src/jlens_nsd/image_only_pilot.py`: separate phase CLI and deterministic
  helpers; does not import or alter caption prompts/extraction.
- `scripts/run_image_only_wikitext_pilot.py`: experimental entry point.
- `tests/test_image_only_pilot.py`: model-free mask/pool/linearity tests.
- Generated artifacts live only under ignored
  `results/image_only_wikitext_pilot`.

## Validation/run status

2026-08-15 audit:

- Recovered a complete real 100-image `features.npz` and `features_complete`
  manifest, plus an RDM archive from an interrupted searchlight phase.
- Confirmed all authoritative default paths exist. The feature archive contains
  nine finite float32 `[100, 2560]` matrices; condition/sample identities agree;
  every image has grid `[1, 26, 26]` and 169 projected tokens; freshly computed
  correlation-distance RDMs exactly equal all nine saved RDM arrays.
- Hardened phase transitions and verification: status gates; size/SHA-256 checks
  for artifacts, authoritative inputs, model files, experiment source, and
  Transformers implementations; model composite and J-lens Git provenance;
  matched-sample reconstruction; feature/RDM equality; center and summary
  consistency; and all 100 source-pixel hashes.
- Full unit suite passes (24 passed, 2 skipped); Ruff lint and formatting pass.
- PID 240011 was no longer live when recovery began. Its backtrace showed a
  native abort in `libinfinipath` through UCX `libucs`; it had produced no map.
- A pure NumPy equivalence test matches direct SciPy correlation-distance RSA,
  but 365,127 mostly 895-voxel spheres make CPU execution impractical. The
  completed recovery therefore used the established streamed TensorFlow GPU
  path with UCX limited to `self,sm,cuda_copy,cuda_ipc` and PSM2/PSM3 limited
  to `self`. It completed without a native abort.
- Searchlight, direct single-sample fsaverage projection, individual surface
  plots, summary JSON/CSV, and compact report figure are complete. All 365,127
  centres are finite for all nine representations.
- Whole-searchlight raw/J means: layer 8, 0.026607/0.027302; layer 16,
  0.034328/0.032370; layer 23, 0.034379/0.032823; layer 30,
  0.025960/0.025510. Final block-31 raw control: 0.028943. Layers remain
  separate; these are descriptive searchlight-centre RSA correlations, not
  single-voxel correlations or population estimates.
- Compact figure `docs/assets/image_only_wikitext_pilot_scores.png` is
  1475×864 RGBA with SHA-256
  `c5036d83ba8bba9e9dbb61b7a56a8da8d0b2be27e996f3959973eea29cc3c744`.
- Image feature commit `7111ac6` is pushed; merge commit `0916455` integrates
  it after the other three feature merges. Handoff protocol is to push the
  validated `main` tip, then send exactly one Discord completion message.

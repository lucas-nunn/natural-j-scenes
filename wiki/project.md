# Jacobian Lens × NSD agent memory

## Repository state

- Standalone root: `/home/chuddy/dev/research/jlens-nsd`.
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

- 18 unit tests pass under Python 3.12 with the existing scientific environment.
- Ruff lint and formatting checks pass.
- Source distribution and wheel build successfully with `uv build`.
- A dry-run resolver check succeeds across all optional dependency groups.
- The wheel installs into a clean Python 3.12 environment; distribution import,
  console `--help`, and the model/data-free smoke command pass.
- `git diff --check` passes and only source/docs/tests/wiki are present.
- Full GPU/model/NSD validation remains external: the repository contains no
  weights, NSD beta data, prior MPNet artifacts, searchlight geometry, or
  pycortex/fsaverage assets.

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
- `docs/assets/plain_layer23_raw_then_j_all_subjects.jpg` composes, without
  reanalysis, the existing raw then J maps for subjects 1–8. It is 2240×4036
  RGB with SHA-256
  `45aae6f77287b485b0542fcb405f037920c5106be03fa3db013d33b120b52f16`.
  EXIF audit metadata records all 16 source filenames, dimensions, and hashes;
  each source panel retains its original independent symmetric color scale.
- Vocabulary unembedding is interpretive only. Brain RDMs use the complete
  2,560-dimensional vectors; task-direction-like top tokens and
  representational geometry are different descriptive measurements.

## Subject-subset execution

- The CLI and durable orchestrator accept an explicit subject subset. It scopes
  condition preparation/union, extraction, grouped RDMs, searchlights,
  projection, plots, and summary artifacts while preserving the eight-subject
  default.
- A one-subject report is explicitly descriptive; its `n=1` confidence limits
  are undefined and must not be interpreted as population inference.

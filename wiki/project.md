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

## Subject-subset execution

- The CLI and durable orchestrator accept an explicit subject subset. It scopes
  condition preparation/union, extraction, grouped RDMs, searchlights,
  projection, plots, and summary artifacts while preserving the eight-subject
  default.
- A one-subject report is explicitly descriptive; its `n=1` confidence limits
  are undefined and must not be interpreted as population inference.

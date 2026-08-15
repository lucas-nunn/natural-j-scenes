# Whole-prompt pooling agent memory

## Scope and immutable controls

- Branch/worktree only: `whole-prompt-pooling` at
  `/home/chuddy/dev/research/jlens-nsd-whole-prompt`.
- Historical prompt set and `plain` text/tokenizer path are reused, not copied.
- New axis: `readout_mode=all_token_mean`; default remains `final_token`.
- Run namespace: `qwen4b__plain_mean_pool`; feature namespace:
  `plain_mean_pool`; grouped namespace:
  `jlens_qwen4b__plain_mean_pool_group`.
- Layers: 8, 16, 23, 30 independently, plus pooled raw final block 31.
- Exact mask: attention-mask ones only. Existing tokenizer-added special tokens
  are included; right-padding zeros are excluded.
- J is applied to each valid token before pooling. Each batch checks the linear
  identity against J applied after pooling with atol=rtol=1e-5.

## Statistical contract

- Whole-searchlight mean: valid centres within sample, 8 samples within
  subject, then 8 subjects.
- Primary BH family `plain_mean_pool_j_vs_raw_4`: exactly four layer-matched
  J-minus-raw tests.
- Secondary BH family `plain_mean_pool_vs_final_token_9`: pooled-minus-
  historical for raw/J at 4 layers plus final raw. Never merge q-values.
- Historical source root, established by committed provenance:
  `/home/chuddy/dev/research/neuroconnectionism/lucas_exploration/jlens_experiment/results`.
- MPNet root:
  `/home/chuddy/dev/research/neuroconnectionism/lucas_exploration/results/mpnet_10_sessions`.

## External artifacts from manifests

- Captions:
  `/home/chuddy/dev/research/neuroconnectionism/lucas_exploration/results/saved_embeddings/nsd_allWords_per_image.pkl`, SHA-256
  `5fb429fba3addfd6c50a8951cae086e57b8361419b2538f56b73f76394c3125f`.
- Qwen3.5-4B: `/media/chuddy/Extreme SSD/models/Qwen3.5-4B`.
- Lens root: `/media/chuddy/Extreme SSD/models/jacobian-lens`; released file
  SHA-256 `1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e`.
- NSD: `/media/chuddy/Extreme SSD/data/NSD`.
- Jacobian Lens checkout:
  `/home/chuddy/dev/research/jacobian-lens`, revision
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

## Conflict audit

Existing wiki notes define historical `plain` as final-token. This control does
not supersede or conflict with that knowledge: it is an isolated readout mode.
Earlier statements that generated results were absent remain true until the new
ignored run roots are created; compact CSV/figure/docs are the declared commit
exception.

## Planned ignored roots

- Subject 1: `results/whole_prompt_subject1_20260815`.
- Full union: `results/whole_prompt_full_20260815`.

## Environment (built 2026-08-15) — no GPU env existed before this

No usable model environment existed anywhere on the box. Both `jlens-nsd/.venv` and the
main repo's environment carry only base + `dev`; `torch`, `transformers`, and `jlens` were
absent. Recipe that works on this machine:

```bash
uv venv --python 3.10 .venv
uv pip install -e ".[dev,model,nsd-upstream,nsd-runtime]"
```

Python 3.10 is required, not incidental: the `nsd-upstream` pin carries the marker
`python_version < '3.12'`, so `nsd_visuo_semantics` silently does not install on 3.12 and
`prepare` then dies with `ModuleNotFoundError: No module named 'nsd_visuo_semantics'`.

Resolved versions: torch `2.13.0+cu130`, transformers `5.15.0`, tensorflow `2.15.0`.

### TRAP: each worktree needs its OWN venv

`jlens-nsd/.venv` editable-installs from `/home/chuddy/dev/research/jlens-nsd/src`, i.e.
**main's** source tree. Running `jlens-nsd/.venv/bin/python -m pytest` from inside a feature
worktree imports main's code and **passes while testing the wrong thing** — no error, no
warning. Always create a venv inside the worktree and confirm before trusting a green run:

```bash
.venv/bin/python -c "import jlens_nsd; print(jlens_nsd.__file__)"   # must be the worktree path
```

### `~/dev/.jlens-venv` is a sandbox artifact — do not use it

Its `pyvenv.cfg` records `command = /usr/bin/python3 -m venv /workspace/.jlens-venv`, so it
was created **inside the OpenClaw container**. It holds CPU-only `torch 2.13.0+cpu` under
`lib/python3.11`, while its interpreter symlink resolves to the host's 3.12. Broken on the
host and useless for GPU work. See `wiki/system/openclaw-remote-control.md` in `~/dev`.

### TRAP: TensorFlow has no GPU here, so the searchlight is CPU-only

`tensorflow 2.15` needs CUDA 12.x + cuDNN 8.9. The only runtime CUDA in this venv is 13.0,
pulled in as torch's `nvidia-*` wheels. TF logs `Cannot dlopen some GPU libraries` then
`Skipping registering GPU devices...` and `tf.config.list_physical_devices('GPU')` is `[]`.

This affects **only** the searchlight stage. Extraction uses torch and does run on the GPU
(`sm_120` is in torch's `arch_list`; the RTX 5070 Ti is Blackwell).

Measured CPU cost, subject 1: **3m29s per sample** at ~173% CPU and 5.3 GB peak RSS, so
~28 min for eight samples. Extrapolated full eight-subject run ≈ **3.7 h**, but that assumes
per-subject `searchlight_indices` (2.4 GB) and `betas_average` (2.3 GB) are already
precomputed under the shared MPNet tree; only `subj01` was verified present.

`--max-samples N` is safe to probe with: each sample writes its own
`..._sample-<i>.npy` and the loop skips any that already exist, so a truncated run cannot
poison a later full run.

## Validation run — subject 1 (2026-08-15)

Root: `results/whole_prompt_subject1_20260815` (ignored, as planned).

| Stage | Outcome |
|---|---|
| pytest / ruff | 44 passed, 5 skipped (model extra), lint + format clean |
| `smoke` both modes | pass; `all_token_mean` correctly narrows to `plain` only |
| `prepare --subjects 1` | 835 conditions, sampling `(8,100)`, `sampling_is_disjoint: true` |
| `preflight` (cuda) | hook semantics exact: `max_abs_error = 0.0` at layers 8/16/23/30 |
| `extract` (cuda) | 835 conditions, 14 chunks, 9 features, 105 batches, **19 s** |
| `rdms` | 10 models under `jlens_qwen4b__plain_mean_pool_group` |
| `searchlight` sample 0 | 10×(81,104,83), zero NaN, 365,127 centres per model |

**The `prepare` output independently reconfirms a scientific invariant** that `wiki/project.md`
records as requiring external data to recheck: 835 conditions/subject and eight mutually
disjoint 100-image samples. Confirmed, not contradicted.

**The linearity proof holds on real data.** Worst tokenwise-J-then-pool vs pool-then-J
discrepancy across 105 batches / 55,040 valid tokens:

| layer | max_abs_error | tolerance |
|---|---|---|
| 8  | 9.54e-07 | 3.24e-05 |
| 16 | 3.34e-06 | 7.08e-05 |
| 23 | 5.72e-06 | 1.32e-04 |
| 30 | 9.54e-06 | 2.24e-04 |

Namespace isolation verified live: `run_name -> qwen4b__plain_mean_pool`, `group_name ->
jlens_qwen4b__plain_mean_pool_group`, features `plain_mean_pool__l{08,16,23,30}__{raw,j}`
plus `plain_mean_pool__final`. Historical `qwen4b` outputs were not touched. The guard
rejecting `all_token_mean` + `matched_readout` fires as designed.

Observed token facts for the `plain` prompt: `Qwen2Tokenizer`, right padding, pad id 248044,
52 valid tokens on condition 104, and **zero** tokenizer-added special tokens included — the
special-token audit machinery is exercised but finds none for this prompt set.

## Power ceiling of the exact sign-flip test — read before interpreting nulls

With 8 subjects the exact `2^8` sign-flip test has a **floor of p = 1/256 = 0.0039**.
After BH within each predeclared family the smallest attainable q is:

- `plain_mean_pool_j_vs_raw_4` (4 tests): **q = 0.0156** — comfortable headroom.
- `plain_mean_pool_vs_final_token_9` (9 tests): **q = 0.0352** — significance requires a
  *perfect* 8/8 sign agreement **and** the largest effect in the family.

So a non-significant secondary family is weak evidence, not evidence of absence. This is a
property of n=8, not a defect in the implementation. Reporting effect sizes and sign counts
alongside q-values is therefore necessary rather than optional.

## BUG FOUND AND FIXED: historical comparator guard rejected the real comparator

`_load_historical_final_token_scores` (new on this branch) read the comparator's subject list
as `manifest.get("subject_numbers", ())`. The **locked historical group manifest predates
subject-subset execution and has no such key** — it records `subjects` as a mapping keyed
`subj01`..`subj08`. The guard therefore compared `()` against `(1..8)` and always raised:

```
ValueError: historical comparator is not the locked eight-subject run
```

This blocked `summarize` for `all_token_mean` **unconditionally** — the entire
`plain_mean_pool_vs_final_token_9` family was unreachable, at any subject count. It was not
caught by the suite because no test exercised a legacy-schema manifest.

The historical root is immutable by contract, so the reader adapts, not the artifact.
`_manifest_subject_numbers()` now prefers `subject_numbers` and falls back to parsing
`subj(\d+)` keys from `subjects`, returning `()` for anything unrecognised so the guard still
fires on a genuinely wrong comparator. Four regression tests added in
`tests/test_summary_stats.py`.

Verified comparator identity (`historical_comparator` block, recorded per run):
`summary_sha256 d3baf8dd…`, `subject_scores_sha256 9aaa3507…`,
`group_manifest_sha256 1e3274bc…`.

## Subject-1 descriptive result (2026-08-15) — NOT inference

`summarize --subjects 1` emits both families with exactly 4 and 9 rows, never merged. **Every
`exact_p` is 1.0 and every CI is `nan`, which is correct, not broken:** the exact sign-flip
test over one subject enumerates `2^1` flips, so `|mean|>=|mean|` always holds and p is
identically 1. A one-subject run validates plumbing and can never test a hypothesis.

Mean searchlight-centre RSA correlation, subject 1:

| feature | pooled | historical final-token |
|---|---|---|
| `l08__raw` | 0.0264 | 0.0036 |
| `l08__j`   | 0.0257 | 0.0024 |
| `l16__raw` | 0.0301 | 0.0032 |
| `l16__j`   | 0.0308 | 0.0024 |
| `l23__raw` | 0.0311 | 0.0011 |
| `l23__j`   | 0.0334 | 0.0016 |
| `l30__raw` | 0.0310 | 0.0021 |
| `l30__j`   | 0.0322 | 0.0035 |
| `final`    | 0.0314 | 0.0038 |
| `mpnet_reference` | 0.0295 | 0.0295 |

### The mpnet_reference anchor is why the gap is believable

`mpnet_reference` is computed from the same MPNet embeddings in both pipelines and is
**identical to four decimals across the two runs (0.0295)**. That is an independent check that
betas, locked sampling, searchlight geometry, and the RSA metric are calibrated the same way,
so the ~10x pooled-vs-final-token gap **cannot be explained by a scaling or pipeline
difference**. Always cross-check this anchor before believing any cross-run comparison here.

Descriptively, pooling lifts caption features from barely-above-zero (0.001-0.004) to at or
slightly above the MPNet semantic reference, and J exceeds raw at layers 16/23/30 but not 08.
**Do not report any of that as a result.** n=1, no inference, and the primary
`plain_mean_pool_j_vs_raw_4` family needs all eight subjects.

### Timing measured on this box (CPU searchlight)

Subject 1: sample 0 alone 3m29s (173% CPU); the remaining 7 took 15m13s (283% CPU) for
~18.5 min per subject. Extrapolated eight-subject searchlight **≈ 2.5 h**, assuming each
subject's `searchlight_indices` and `betas_average` are already precomputed — only `subj01`
was verified. Extraction is GPU and negligible (19 s for 835 conditions).

## Remaining to do

- [ ] Eight-subject run (`prepare` through `summarize`, no `--subjects` filter). Only this
      can test either hypothesis family.
- [ ] `project` / `plot` stages never exercised on this branch.
- [ ] Confirm `searchlight_indices` + `betas_average` exist for subj02-08, or budget for
      recomputation.

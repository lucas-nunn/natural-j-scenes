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

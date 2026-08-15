# Attribution and dependency boundary

This repository contains the Jacobian Lens × NSD experiment authored in the
Lucas Nunn `neuroconnectionism` working tree. It does not redistribute NSD,
model weights, Jacobian-lens checkpoints, or generated experiment artifacts.

NSD condition, mask, model-RDM, sphere, beta, and fsaverage mapping primitives
come from Adrien Doerig and collaborators' `nsd_visuo_semantics` project. Its
historical URL is <https://github.com/KietzmannLab/nsd_visuo_semantics>; the
public repository is currently available as <https://github.com/adriendoerig/visuo_llm>
and is pinned at `a60e0eafb8d02841159e344adb732062734bc302`. Upstream is MIT
licensed (copyright Adrien Doerig, 2025).

Jacobian transport, Hugging Face model adaptation, and activation recording
come from Anthropic's `jacobian-lens`, Apache-2.0 licensed, pinned at
`581d398613e5602a5af361e1c34d3a92ea82ba8e` in the optional model extra.

The small compatibility implementation in `jlens_nsd.nsd_adapter` is adapted
from the local fork's memory-safety and selective-subject changes. The exact
delta and reasons for keeping it locally are recorded in
`docs/UPSTREAM_COMPATIBILITY.md`.

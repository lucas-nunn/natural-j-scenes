# Image-only WikiText Jacobian Lens transfer pilot

## Scope and predeclaration

This is exactly an **image-only, decoder-residual, out-of-distribution
transfer** of the released WikiText lens. It does not estimate a vision-tower
Jacobian Lens. Qwen first encodes pixels in its
vision tower, spatially merges and projects those features into the 2560-D
language-decoder residual stream, and only then are decoder block outputs
measured. Anthropic's released WikiText-average matrices are applied unchanged
to those decoder image-token states. The lens was fitted on text, not images;
any result is therefore descriptive evidence about transfer, not validation of
a vision-specific lens.

Before inspecting results, the experiment fixes subject 1, existing matched
sample row 0 (100 images), decoder blocks 8, 16, 23, and 30, plus block 31 as a
final raw control. Each selected block has matched raw and J-transformed
features from the same image-token positions. One image vector is the
deterministic float32 arithmetic mean across those positions. Model RDMs use
correlation distance. Whole-searchlight means are descriptive; there are no
inferential p-values for this one-subject, one-sample pilot.

## Minimal input and token boundary

No captions, descriptions, questions, instructions, generated text, or chat
template are used. The complete input string is only Qwen's mandatory control
sequence:

```text
<|vision_start|><|image_pad|><|vision_end|>
```

Qwen's processor expands the single placeholder according to
`prod(image_grid_thw) / spatial_merge_size²`. Pooling proceeds only when the
image-token-ID mask and Qwen's independent multimodal type-1 mask agree, the
image run is contiguous between exactly one boundary pair, and no other
attended token exists. The run manifest records the literal sequence, IDs,
grid, mask evidence, preprocessing, per-image counts, and file hashes.

## Analysis and limitations

The released Jacobian matrix at block *l* is multiplied with the mean raw
decoder residual from that same block. Numerical checks compare
`mean_patch(J h_patch)` with `J mean_patch(h_patch)` on a real selected image;
model-free tests exercise the same invariant. The 100 image vectors produce a
4,950-entry condensed correlation-distance RDM, which is compared with each
subject-1 brain searchlight RDM using the established pipeline correlation.

This pilot cannot support population inference, model selection, or a claim
that WikiText J-space generally improves visual representations. The final
decoder-block control has no fitted released J matrix and is not a matched J
comparison. Each numerical map is projected directly to `fsaverage`, without
averaging samples or subjects, and plotted on its own symmetric scale.

## Results

![Descriptive image-only pilot scores](assets/image_only_wikitext_pilot_scores.png)

Whole-searchlight means across 365,127 valid subject-1 searchlight centres:

| Decoder block | Raw | WikiText J | J − raw |
|---:|---:|---:|---:|
| 8 | 0.026607 | 0.027302 | +0.000695 |
| 16 | 0.034328 | 0.032370 | −0.001958 |
| 23 | 0.034379 | 0.032823 | −0.001556 |
| 30 | 0.025960 | 0.025510 | −0.000450 |
| 31 final residual control | 0.028943 | — | — |

Only layer 8 has a positive descriptive J-minus-raw difference. The other
three predeclared layers are negative. No population conclusion follows from
one subject and one sample, and layers are never averaged. These are
searchlight-centre RSA correlations, not single-voxel correlations.

The authoritative generated records are
`results/image_only_wikitext_pilot/manifest.json`, `summary.json`, and
`summary.csv`; projected surfaces and individual maps are under `surfaces/`
and `figures/` in the same ignored result directory.

## Searchlight failure and recovery

The first downstream process aborted in `libinfinipath` while UCX's
`libucs` error handler was building a backtrace. The recovery excluded the PSM
fabric (`PSM2_DEVICES=self`, `PSM3_DEVICES=self`) and limited UCX to local and
CUDA transports. TensorFlow then registered the GPU normally, and the existing
streamed float32 radius-6 searchlight completed. Extraction was not rerun;
`features.npz` and its deterministic sample identity were reused.

## References

- Gurnee et al., *The Jacobian Lens* (average layer-to-final residual
  Jacobians and lens interpretation):
  <https://transformer-circuits.pub/2026/workspace/index.html>
- Kriegeskorte, Mur, and Bandettini (2008), representational similarity
  analysis and RDM comparison: <https://doi.org/10.3389/neuro.06.004.2008>
- Allen et al. (2022), NSD repeated natural-scene measurements:
  <https://doi.org/10.1038/s41593-021-00962-x>

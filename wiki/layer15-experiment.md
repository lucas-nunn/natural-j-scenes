# Layer 15 against cortex — the semantic peak is a brain trough

Run 2026-08-16, root `results/layer_sweep_exploratory_20260816` (ignored), layers 8/15/16/23/30,
all eight subjects, ~60 min. `summarize` deliberately skipped; analysed by
`scripts/analyze_exploratory_layers.py`. Cached: `docs/exploratory_layers.json`.

**EXPLORATORY.** No family was predeclared over this layer set, so p-values are uncorrected and
these numbers are hypothesis-generating only.

## The hypothesis, and its refutation

The 31-layer model-side sweep ([[depth-sweep]]) found the J advantage peaking near **layer 15**
(+0.105 semantic) while the production layers 23 and 30 sat at +0.028 and −0.024. If semantic
alignment tracked cortical alignment, layer 15 should have been the strongest brain layer. It was
run for the first time to find out.

| layer | brain Δ (J−raw) | signs | p (uncorrected) | semantic Δ |
|---|---|---|---|---|
| 8  | −0.000476 | 4/8 | 0.6406 | +0.0654 |
| **15** | **+0.000787** | 6/8 | **0.2812** | **+0.1047** |
| 16 | +0.000356 | 6/8 | 0.6406 | +0.0899 |
| 23 | +0.001649 | 7/8 | 0.0156 | +0.0283 |
| 30 | +0.001459 | 7/8 | 0.0234 | −0.0241 |

**Layer 15 is not significant even uncorrected**, and its effect is roughly half that of layers 23
and 30. The semantic peak is, among the positive layers, the brain trough.

## The proxy is anti-predictive for the layer profile

Across the five shared layers:

```
Spearman(semantic advantage, brain advantage) = -0.500   (p = 0.391, n = 5)
Pearson                                       = -0.586
```

Negative. With n = 5 this is not significant, and the point is not that the relationship is
reliably inverse — it is that **there is no positive relationship to lean on**. Combined with the
sign disagreement at layer 30 noted in [[depth-sweep]], the conclusion from [[why-pooling-won]]
now has a hard boundary:

- **MPNet is valid for comparing readouts.** The endpoint-vs-pooled gap was 8-23x, both readouts
  were scored against the same reference, and the proxy predicted the brain ratio. That result
  stands.
- **MPNet is not valid for comparing representations across layers.** It is at best uninformative
  and at worst anti-predictive. Do not use it to choose layers or to argue about depth.

This is the second time tonight the proxy has been tested against cortex and failed on the layer
profile. It should now be treated as settled rather than re-examined.

## Bonus: end-to-end determinism, verified exactly

Layers 8, 16, 23 and 30 were re-extracted, re-RDM'd and re-searchlit from scratch in this run. All
four reproduce the original eight-subject result **to nine decimal places**:

```
l08  original -0.000475605   rerun -0.000475605
l16  original +0.000356224   rerun +0.000356224
l23  original +0.001649366   rerun +0.001649366
l30  original +0.001459367   rerun +0.001459367
```

Two things follow. The pipeline is **deterministic end to end** across extraction, grouped RDMs and
64 searchlights — a reproducibility property the project had never demonstrated. And **adding a
layer does not perturb the others**, confirming features are computed independently per layer with
no cross-contamination through the shared extraction pass.

## What is still not resolved

This does **not** address the depth/rank confound from [[depth-sweep]] (`r = 0.971`). Brain-side,
the effect still grows roughly with depth, and effective rank still grows with depth, so the two
remain inseparable. Layer 15 rules out the *semantic-peak* hypothesis specifically; it says nothing
about whether depth or conditioning drives the brain result.

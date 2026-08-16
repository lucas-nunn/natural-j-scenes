# All 31 lens layers: can depth be separated from conditioning?

Generator: `scripts/analyze_depth_sweep.py`. Cached: `docs/depth_sweep.json`.
Run 2026-08-16, 800 conditions, 32 samples of 100, semantic reference = MPNet.

## Answer: no, and now we can say why quantitatively

`corr(effective_rank, depth) = **0.971**` across the 31 fitted layers. Effective rank rises
monotonically from **182** at layer 0 to **2272** at layer 30. Depth and conditioning are very
nearly the same variable in this lens, exactly as predicted for an *average* Jacobian.

Partial correlations were computed and should **not** be trusted:

```
corr(advantage, depth)                 +0.585
corr(advantage, rank)                  +0.408
corr(rank, depth)                       0.971
partial(advantage, depth | rank)       +0.868
partial(advantage, rank | depth)       -0.829
```

Large partials of **opposite sign** under `r = 0.97` collinearity is the textbook signature of
unstable estimates, not of a clean dissociation. Reporting `+0.868` as "depth matters once rank is
controlled" would be a mistake. The correct statement is that **this design cannot separate them**,
and no amount of re-analysis of these 31 points will change that. Separating them needs a lens
fitted differently — more prompts, or per-layer rank matching.

## The J advantage is NOT monotone in depth

The four production layers (8, 16, 23, 30) sample a curve that turns out to have real structure:

| layers | J advantage | shape |
|---|---|---|
| 0–6 | **−0.224 to −0.109** | J actively *hurts*, badly |
| 7 | +0.024 | crossover |
| 8–18 | +0.050 to **+0.105** | broad peak, max at layer 15 |
| 19–23 | +0.020 to +0.039 | trough |
| 24–28 | +0.036 to +0.051 | second bump |
| 30 | **−0.024** | negative again |

The production sampling misses both the strongly negative early region and the peak near layer 15.
Anyone reading the 4-point profile as "the advantage grows with depth" is reading a curve that
does not do that.

## IMPORTANT: the semantic proxy does NOT track the brain layer profile

At layer 30 the J advantage is **negative semantically (−0.024)** while the eight-subject brain run
found J > raw at layer 30 (+0.0015, q = 0.047). They disagree in sign.

This bounds how far the MPNet proxy can be trusted, and the bound matters because the proxy did
excellent work in [[why-pooling-won]]:

- **Valid there:** the readout comparison. The endpoint-vs-pooled gap was 8-23x, both readouts were
  measured against the same reference, and the proxy predicted the brain ratio.
- **Not valid here:** adjudicating the J-vs-raw *layer profile*. The effects are ~20x smaller,
  MPNet encodes caption meaning rather than whatever cortex is responding to, and the two measures
  disagree in sign at the one layer where both are available.

Do not use semantic alignment as a stand-in for cortical alignment when comparing J against raw.
It earned trust on a large readout-level effect; it has not earned it on a small representational
one.

## What this changes

1. The depth confound is now **quantified** (`r = 0.971`) rather than asserted. The README guardrail
   understated it — this is not a confound to be controlled for, it is a near-identity.
2. A **denser brain-side layer sweep** would be worth more than any further model-side analysis, and
   is the natural next compute-bound job.
3. Layer 15 is the semantic peak and has never been run brain-side. If the workspace claim predicts
   anything about depth, that is the untested layer most likely to discriminate.

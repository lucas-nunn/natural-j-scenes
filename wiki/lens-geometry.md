# Lens geometry: how much does J actually warp the RDM?

Generator: `scripts/analyze_lens_geometry.py`. Cached result: `docs/lens_geometry.json`.
Uses stored pooled features + the released lens only; **no fMRI data touched**.

## The axiom being tested

The README argues that `J_l` is a fixed linear map, so `X_J = X_raw @ J_l.T` spans the same
subspace as `X_raw` when `J_l` is full rank, making the two feature spaces equivalent under OLS.
Any J-vs-raw difference under RSA must therefore come from the **metric change**: correlation
distance is not invariant under a general linear map, so `J_l` warps the RDM.

That was asserted, never measured. The natural prediction is **more warp -> more difference**.

## Result (2026-08-15, 6,148 conditions, 64 samples of 100)

| layer | RDM r (raw vs J) | effective rank | condition number | J−raw brain effect |
|---|---|---|---|---|
| 8  | **0.768** | 401 / 2560  | 1.9e6 | −0.0005 (ns) |
| 16 | 0.831 | 847 / 2560  | 9.1e5 | +0.0004 (ns) |
| 23 | 0.925 | 1877 / 2560 | 7.3e4 | **+0.0016** (q=0.047) |
| 30 | **0.957** | 2272 / 2560 | 7.6e3 | **+0.0015** (q=0.047) |

Effective rank is the participation ratio `(Σσ)²/Σσ²`; condition is `σ_max/σ_min`.

## THE PREDICTION IS INVERTED — this is the finding

The brain effect appears where the lens warps geometry **least**, and is absent where it warps
**most**. The ordering is perfectly monotone across all four layers and the spread is large
(RDM r 0.77 -> 0.96; effective rank 401 -> 2272; condition number spans 255x).

- Layer 8: `J_8` is severely rank-deficient — it uses ~400 of 2560 directions and has a condition
  number near 2 million. It is an aggressive, lossy compression. Brain alignment does **not**
  improve; the point estimate is negative.
- Layer 30: `J_30` is close to a mild, well-conditioned reweighting (89% effective rank). Brain
  alignment **does** improve, slightly.

So "J-space helps because it re-weights toward verbalizable directions" is **not supported by the
mechanism**: the layers where that re-weighting is strongest are exactly the layers where it does
not help. The late-layer gain is better described as a *gentle* reweighting that preserves nearly
all structure while slightly improving fit.

## Interpretation guardrails

- **n = 4 layers.** The monotone ordering is descriptive. It is a strong hint about mechanism, not
  an inferential claim, and no test is run over layers.
- **The low early-layer rank is expected and may be an artifact of the lens construction**, not a
  fact about the model. `J_l` is an *average* Jacobian over 1,000 WikiText prompts; the further
  `l` is from the final layer, the more nonlinear blocks are averaged through, so rank collapse
  with depth-distance is the null expectation. This confounds "early layer" with "low-rank lens".
- Consequently the layer profile of the J-vs-raw effect **cannot** currently distinguish "the
  workspace lives in late layers" from "the released lens is only usable in late layers".

## Follow-up that would separate those

Compare against a lens fitted with more prompts, or a rank-matched control: project raw features
onto the top-`k` right singular directions of `J_l` with `k` = that layer's effective rank, and
re-run. If the rank-matched raw control reproduces the J result, the effect is about conditioning
rather than about the lens direction.

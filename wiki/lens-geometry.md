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

---

# Follow-up: do the lens's *directions* matter, or only its conditioning?

Generator: `scripts/analyze_lens_controls.py`. Cached: `docs/lens_controls.json`. Run 2026-08-16.

Both controls are built from the layer's own SVD `J = U S V^T`, so nothing is imported from
outside the map itself:

- **`spectrum_matched`** — `Q1 S Q2^T` with fresh random orthogonal `Q1, Q2`. Identical singular
  values, random directions.
- **`orthogonal`** — `U V^T`, the lens's own rotation with the spectrum flattened to ones.

| layer | actual J | spectrum-matched | orthogonal (U Vᵀ) |
|---|---|---|---|
| 8  | 0.760 | 0.827 | 0.9999952 |
| 16 | 0.832 | 0.866 | 0.9999964 |
| 23 | 0.930 | 0.986 | 0.9999978 |
| 30 | 0.958 | 0.992 | 0.9999994 |

(RDM correlation against raw; lower = more warp.)

## 1. The rotation does nothing; all warp is anisotropy

`U V^T` leaves the RDM essentially untouched — `r = 0.999995` or better at every layer. So
correlation-distance geometry here is effectively invariant to the lens's rotation, and **100% of
the measured warp comes from the singular value spectrum**, i.e. from anisotropic rescaling. This
cleanly decomposes the effect and means "J rotates into a verbalizable basis" cannot by itself
change any RSA result.

## 2. The directions ARE data-aligned — conditioning alone does not explain the warp

At every layer the real lens warps **more** than a random matrix with the identical spectrum
(actual < matched). Robustness check at layer 23 over **8 independent random draws**:

```
actual                      0.9252
spectrum-matched  mean      0.9868   std 0.00132   range [0.9843, 0.9883]
gap                        +0.0616  = 46.6 control-SDs
```

Unambiguous. The lens's high-variance directions are aligned with the directions the *data*
actually occupies, so its rescaling lands where it has leverage; a random map with the same
spectrum spreads that rescaling across directions the data barely uses, and therefore does less.

**This is the strongest evidence so far that the released lens encodes something real** rather
than acting as an arbitrary ill-conditioned matrix.

## 3. But it does not rescue the mechanistic story

Combine with the layer profile above: the lens is demonstrably non-random, **and** the amount by
which it warps geometry still *anti*-correlates with brain benefit. Layer 8 has both the most warp
and the most data-aligned warp, and shows no brain effect at all. So "J helps because it
re-weights toward verbalizable directions" remains unsupported — the re-weighting is real and
data-aligned, it simply does not buy alignment where it is strongest.

## 4. End-to-end pipeline audit (bonus, and it passes)

The script re-derives `X_raw @ J^T` and compares against the stored J features. Relative error at
every layer is ~5e-07:

```
layer 8   max|diff| 1.05e-06   scale  2.25   rel 4.7e-07
layer 16  max|diff| 2.89e-06   scale  6.09   rel 4.7e-07
layer 23  max|diff| 6.59e-06   scale 12.21   rel 5.4e-07
layer 30  max|diff| 1.12e-05   scale 25.84   rel 4.3e-07
```

This audits extraction end-to-end from committed artifacts alone: correct matrix, correct layer,
correct orientation, float32 storage as declared. A transposed multiply or an off-by-one layer
index would fail loudly here.

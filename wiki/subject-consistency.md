# Per-subject robustness of the J-vs-raw effect

Generator: `scripts/analyze_subject_consistency.py`. Cached: `docs/subject_consistency.json`.
Run 2026-08-16 against the eight-subject pooled run.

## Why this was missing

The predeclared test is an exact sign-flip test, which uses only the **signs** of the eight paired
differences. That is conservative, but it means a reported group effect is formally compatible with
one subject carrying a huge delta while the other seven sit near zero. Nothing in the committed
summary artifacts reported whether the effect was distributed.

## Result — the effect is distributed, not driven by one subject

| layer | signs | mean Δ (×1e3) | std (×1e3) | \|mean\|/std | p | LOO p range |
|---|---|---|---|---|---|---|
| 8  | 4/8 | −0.476 | 2.557 | 0.186 | 0.6406 | [0.4062, 0.9219] |
| 16 | 6/8 | +0.356 | 2.151 | 0.166 | 0.6406 | [0.2500, 0.9531] |
| 23 | **7/8** | +1.649 | 1.235 | **1.335** | 0.0156 | **[0.0156, 0.0312]** |
| 30 | **7/8** | +1.459 | 1.131 | **1.290** | 0.0234 | **[0.0156, 0.0469]** |

Per-subject deltas (×1e3), layer 23: `+2.27 +3.83 +2.60 −0.14 +0.84 +0.78 +1.53 +1.49`
Layer 30: `+1.18 −0.06 +3.15 +1.68 +2.71 +0.02 +1.40 +1.60`

**Leave-one-out survives every drop.** Removing any single subject leaves the uncorrected exact p
at or below 0.0469 for both significant layers. The lone dissenting subject in each case is
essentially zero (−0.14e-3 at layer 23, −0.06e-3 at layer 30), not a genuine reversal.

LOO p-values are **uncorrected** and computed at n=7, where the exact floor is 2/128 = 0.0156.
They measure stability, not re-inference.

**Layer 23 is the more robust of the two.** Its LOO range tops out at 0.0312; layer 30 reaches
0.0469 in five of eight drops, so layer 30 sits closer to the edge.

## The early-layer nulls are a PRECISION problem, not only an effect-size problem

Across-subject variance tracks the lens's conditioning **perfectly** over these four layers
(Spearman rank correlation = **+1.000**, n = 4, descriptive):

| layer | delta std (×1e3) | condition number | effective rank |
|---|---|---|---|
| 8  | 2.557 | 1.9e6 | 401 |
| 16 | 2.151 | 9.1e5 | 847 |
| 23 | 1.235 | 7.3e4 | 1877 |
| 30 | 1.131 | 7.6e3 | 2272 |

The ill-conditioned early-layer lenses roughly **double** the per-subject noise, and the
effect-to-noise ratio is ~7x worse (0.17-0.19 versus 1.29-1.34).

This matters for interpretation. The layer profile should **not** be read as "there is no
workspace effect early". It is equally consistent with "the released lens is too ill-conditioned to
measure anything early". Together with [[lens-geometry]], the depth story now has two independent
confounds — rank collapse with distance from the final layer, and noise amplification — and neither
is separable with the current design.

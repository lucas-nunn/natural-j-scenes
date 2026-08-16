# Where does the J advantage live? Not where the workspace account predicts

Generator: `scripts/analyze_stream_localisation.py`. Cached: `docs/stream_localisation_l{23,30}.json`.
Run 2026-08-16 on the eight-subject pooled run.

**EXPLORATORY.** No family was predeclared over stream ROIs; p-values are uncorrected.

## Why this needed testing

The project's motivating idea is that J-space isolates a *verbalizable, broadcast-ready* subspace —
the LLM analogue of a global workspace. That is a claim about **which cortex** benefits, not only
about how much. A workspace account predicts a sensory -> association gradient: the advantage should
concentrate in higher-order cortex and should **not** be an early visual effect.

Everything reported so far is a whole-searchlight mean, which averages the entire sheet and is
silent about location. So the spatial half of the hypothesis had never been examined.

## Result — the advantage is everywhere, including early visual

Layer 23, paired J−raw within each NSD `streams` ROI, subjects as the independent unit:

| stream | mean Δ | signs | p (uncorr.) |
|---|---|---|---|
| 1 early | **+0.003560** | **8/8** | **0.0078** |
| 2 midventral | +0.005302 | 8/8 | 0.0078 |
| 3 midlateral | +0.003612 | 8/8 | 0.0078 |
| 4 midparietal | +0.002579 | 7/8 | 0.0391 |
| 5 ventral | +0.005186 | 8/8 | 0.0078 |
| 6 lateral | **+0.006568** | 8/8 | 0.0078 |
| 7 parietal | +0.000971 | 6/8 | 0.4453 |

**higher-order (5,6,7) minus early: +0.000682, 5/8 subjects, p = 0.617.**

Layer 30 gives the same picture: higher-order minus early = +0.001297, 5/8, p = 0.156.

**The predicted gradient is absent.** The J advantage is as reliable in *early visual cortex*
(8/8 subjects, p = 0.0078) as anywhere else, and the strongest stream at layer 23 is lateral, not
a canonical workspace region. Parietal — arguably the most workspace-like stream available here —
is the *weakest* and not significant.

This is the first direct test of the spatial half of the workspace claim, and the claim does not
survive it in the form stated.

## Two things worth noticing

**The effect is larger inside visual cortex than overall.** Per-stream deltas run +0.001 to +0.007
against a whole-searchlight mean of +0.0016. The `streams` ROI covers roughly 27k voxels of ~365k
searchlight centres, so the advantage is concentrated in visually-responsive cortex generally —
just not in any particular part of it.

**IMPORTANT LIMITATION: `streams` is a visual-hierarchy parcellation, not a whole-brain one.**
"Higher-order" here means higher-order *visual* streams (ventral, lateral, parietal). It contains
no prefrontal cortex and no default-mode network. A global-workspace account in the
Dehaene/Baars sense predicts **frontoparietal** involvement, which this ROI set cannot address.

So the precise finding is: **within visual cortex there is no sensory-to-association gradient.**
The stronger claim — that the advantage is not workspace-localised anywhere — is *not* established
and would need a whole-cortex parcellation.

## Follow-up

`HCP_MMP1.nii.gz` ships with NSD in the same `func1pt8mm` space and covers the whole cortical
sheet including frontal regions. Grouping its 180 parcels into networks would give the
frontoparietal test this analysis cannot provide. That is the natural next question and needs no
new runs — the searchlight volumes already on disk are sufficient.

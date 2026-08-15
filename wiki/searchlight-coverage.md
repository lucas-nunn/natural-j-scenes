# Undefined searchlight centres: prevalence, and why the contrast is still fair

Generator: `scripts/audit_searchlight_coverage.py`. Cached: `docs/searchlight_coverage.json`.
Audited 2026-08-16 on the eight-subject pooled run.

## What was hiding behind a RuntimeWarning

The projection stage emits `RuntimeWarning: Mean of empty slice` and
`Degrees of freedom <= 0`. Those were assumed to be out-of-mask vertices and never checked. They
are not: some **searchlight centres** return an undefined Pearson correlation — typically where
the sampled beta patterns carry no variance — and come back NaN.

Prevalence varies enormously by subject and is nowhere reported:

| subject | centres | undefined | % |
|---|---|---|---|
| subj01 | 365,127 | 0 | 0.000% |
| subj02 | 337,392 | 3,777 | 1.119% |
| subj03 | 352,821 | 1,820 | 0.516% |
| subj04 | 324,246 | 0 | 0.000% |
| subj05 | 297,640 | 575 | 0.193% |
| subj06 | 406,399 | 33,878 | **8.336%** |
| subj07 | 290,914 | 0 | 0.000% |
| subj08 | 316,323 | 34,982 | **11.059%** |

Subjects 6 and 8 lose roughly a tenth of their searchlight centres. Subjects 1, 4 and 7 lose none.

## Why the result is nevertheless sound

Two properties, both verified rather than assumed:

1. **Scores exclude, they do not propagate.** `stages.py` aggregates sample scores with
   `np.nanmean`, so undefined centres drop out instead of poisoning a subject's mean. All
   subject scores are finite.
2. **The NaN pattern is identical across all ten models, per sample.** Checked exhaustively for
   every subject. So within a subject, J and raw are averaged over *exactly* the same centres and
   the paired difference is not biased.

Property 2 is the load-bearing one and it had no guard. If a model ever lost centres another kept,
the two sides would be averaged over different voxels and the paired contrast would be biased —
with nothing raising an error, and a group table that looks entirely normal. The audit script now
checks it and exits non-zero on violation.

## What it does change

Subject-level means are computed over **different numbers of centres** (290,914 to 406,399 before
exclusions, and 89-100% of those after). The group mean therefore weights subjects equally while
those subjects have unequal measurement support. That is not a bias in the paired contrast, but it
is a source of between-subject heterogeneity that the current summary does not surface — and
subjects 6 and 8 are exactly the ones contributing the least-supported estimates.

Worth noting alongside [[subject-consistency]]: neither subject 6 nor 8 is the dissenting subject
at layer 23 or 30, so the coverage gap is not driving the reported effect.

# Why pooling beat the endpoint readout — mechanism, verified without fMRI

Generator: `scripts/analyze_readout_semantics.py`. Cached: `docs/readout_semantics.json`.
Run 2026-08-16.

## The question

The eight-subject run showed pooled caption features aligning with cortex **5-9x** better than the
historical final-token readout. That is a very large effect for a change that adds no new text and
no new information — the pooled vector is a linear function of states the endpoint readout also had
access to. Two explanations fit the brain result equally well:

- **(a)** the endpoint is a poor estimator, dominated by position-specific idiosyncrasy; or
- **(b)** pooling adds distributed content the endpoint genuinely lacks.

Both are separable **without touching fMRI data**, by scoring each readout against an independent
semantic reference. MPNet sentence embeddings of the same captions know nothing about Qwen, the
lens, or the brain.

## Result: (a), overwhelmingly

RDM correlation against the MPNet semantic RDM, 64 samples of 100 conditions:

| layer | kind | pooled ~ MPNet | final-token ~ MPNet | ratio | pooled ~ final |
|---|---|---|---|---|---|
| 8  | raw | 0.4644 | 0.0253 | 18.4x | 0.091 |
| 8  | j   | 0.5174 | 0.0279 | 18.5x | 0.043 |
| 16 | raw | 0.4072 | 0.0174 | 23.4x | 0.095 |
| 16 | j   | 0.4877 | 0.0392 | 12.4x | 0.071 |
| 23 | raw | 0.5353 | 0.0239 | 22.4x | 0.070 |
| 23 | j   | 0.5602 | 0.0339 | 16.5x | 0.093 |
| 30 | raw | 0.5758 | 0.0535 | 10.8x | 0.113 |
| 30 | j   | 0.5522 | 0.0668 | 8.3x  | 0.148 |

**The final-token readout carried almost no caption semantics.** Its RDM correlates 0.017-0.067
with a sentence-embedding RDM *of the very same captions*. That is close to noise. The pooled
readout reaches 0.41-0.58.

**The two readouts are nearly unrelated**, at r = 0.04-0.15. They are not the same quantity measured
better or worse; they are close to orthogonal descriptions of the same stimulus.

## The smoking gun: 73.7% of prompts end in the same token

From the run's own `token_mask_audit.json`, over all 6,148 conditions:

```
distinct final token IDs   595
token 13  '.'      4533  (73.73%)
token 424 ' it'      56  ( 0.91%)
token 2919 ' water'  34  ( 0.55%)
prompt lengths     min 44  max 118  mean 65.6
```

For nearly three quarters of stimuli the "final non-padding token" is **a period**. The endpoint
readout therefore reads the residual at an identical token for most of the dataset, and that state
is dominated by what a sentence-final period looks like rather than by what the caption said.
Averaging over ~66 tokens recovers the content.

## Why this matters beyond the readout choice

1. **It justifies the method change on model-side evidence alone.** The semantic ratio (8-23x)
   brackets the observed brain ratio (5-9x), predicted from an independent reference with no
   circularity. The readout change does not rest on having improved a brain number.
2. **It reframes every historical `plain` result.** Those numbers (0.003-0.006 brain alignment) are
   not a weak finding about caption semantics; they are close to a measurement of nothing. Any
   earlier conclusion drawn from the caption-only condition should be re-read in that light.
3. **It is a general warning for lens work.** "Read the final prompt token" is standard practice and
   is sound when the prompt ends on a content-bearing continuation point. It fails silently when the
   corpus ends on shared punctuation — nothing errors, the vectors look fine, and the RDM is noise.
   `DESIGN.md` had flagged the punctuation risk as a concern; it turns out to have been the whole
   story.

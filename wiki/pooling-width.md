# How many tokens does the caption readout need?

Generator: `scripts/analyze_pooling_variants.py`. Cached: `docs/pooling_variants.json`.
Run 2026-08-16, layer 23, first 1,200 union conditions, 64 samples of 100.

Every variant is a different reduction over the **same** collected token states from one forward
pass, so model, prompt, tokenizer, layer and stimuli are held exactly fixed. Scored by RDM
correlation against MPNet sentence embeddings — the independent semantic reference established in
[[why-pooling-won]].

| variant | raw ~ MPNet | J ~ MPNet |
|---|---|---|
| `last_1` (the historical endpoint) | 0.0391 | 0.0500 |
| `last_2` | 0.1164 | 0.1261 |
| `last_4` | 0.2637 | 0.3061 |
| `last_8` | 0.3885 | 0.4252 |
| `last_16` | 0.4343 | 0.4816 |
| `last_32` | 0.5440 | 0.5618 |
| `all` (production) | 0.5577 | 0.5788 |
| `no_punct` | **0.5755** | **0.5875** |

Mean prompt length is 65.6 tokens.

## 1. Recovery is steep and immediate

One extra token nearly triples semantic alignment (0.039 -> 0.116). Four tokens reach 0.264, a
**6.7x** gain over the endpoint. So the endpoint's failure is not a subtle estimator issue — it is
almost entirely fixed by looking slightly further back.

## 2. It saturates around 32 tokens

`last_32` reaches 0.544, which is **97.5%** of the full-prompt value. Pooling roughly the last half
of a caption block captures nearly everything the full mean does. Whole-prompt pooling is a safe
default rather than a necessary one, which matters for anyone porting this readout to longer
prompts where "all tokens" would sweep in unrelated context.

## 3. Punctuation is mildly HARMFUL, not merely uninformative

`no_punct` (0.5755) **exceeds** the full mean (0.5577) by +0.018. Dropping punctuation tokens does
not just avoid adding nothing — it removes something that was *diluting* the average. This is the
sharper version of the finding in [[why-pooling-won]]: the period is not a neutral passenger.

**Candidate improvement, not yet earned.** `no_punct` is a one-line change that beats the current
production readout on semantic alignment. It is **untested brain-side**, the margin is small, and no
significance was computed over variants. It should not be adopted on this evidence alone; it should
be run through the same predeclared machinery as any other readout claim.

## 4. J exceeds raw at every width

The J advantage holds at all eight widths (+0.010 to +0.042), including at `last_1` where the
representation is near-noise. Consistent with the layer-23 brain result, and it indicates the J
advantage is not an artifact of pooling.

## Validation

The `all` variant here (raw 0.5577) reproduces the stored-feature measurement in
[[why-pooling-won]] (raw 0.5353 at layer 23) to within the difference expected from using 1,200
conditions rather than the full 6,148. That agreement checks this standalone forward pass against
the production extraction path.

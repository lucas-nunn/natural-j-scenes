# Whole-prompt pooling control

## Scientific motivation

The historical `plain` condition reads one residual vector at the final
non-padding prompt token. For caption blocks this endpoint is often punctuation.
That state is causally informed by the preceding caption prefix, but its local
token identity and position can make it a fragile single-point summary. This is
a readout concern, not evidence that punctuation itself caused the historical
alignment result.

The scoped control asks whether distributing the readout over the unchanged
caption prompt improves raw- or J-space alignment. Decoder causal masking means
token position *t* can use its prefix through *t*, not future tokens; this
follows the decoder masking defined by
[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762). The control therefore
must not be described as bidirectional whole-caption contextualization.

## Exact representation

The readout is named **all-token mean-pooled causal decoder residuals** and is
selected by `--readout-mode all_token_mean`.

- Prompt: the byte-for-byte historical `plain` prompt produced by the existing
  `historical` prompt set. There is no instruction, suffix, chat template,
  generation, answer, caption edit, truncation, or new semantic text.
- Tokenization: the historical tokenizer call remains `add_special_tokens=True`,
  `truncation=False`, right padding, and an explicit attention mask.
- Mask: for each batch row, include exactly token positions where
  `attention_mask == 1`; exclude exactly padding positions where it is zero.
  Any tokenizer-added special token is included if and only if it is already in
  that historical encoding and has mask value one.
- Source blocks: 8, 16, 23, and 30 are kept separate. For each block, raw is the
  masked token mean. J-space is computed by applying that block's released
  WikiText matrix to every valid token vector and then taking the same masked
  mean. Layers are never averaged.
- Linearity proof: every extraction batch also computes J applied after raw
  pooling and requires the maximum absolute discrepancy from tokenwise-J then
  pooling to be no larger than `1e-5 + 1e-5 × max_abs(reference)`. The manifest
  records the worst observed error and tolerance per layer.
- Final control: block 31 residuals are mean-pooled with the identical mask and
  stored as raw only; no `J_final` is constructed.

The generated `token_mask_audit.json` records every condition's complete input
token IDs, attention mask, valid positions, tokenizer special-token mask,
included special-token positions/IDs, and a stable aggregate hash. This makes
token inclusion directly reviewable without committing large run products.

## Isolation and interpretation

Historical IDs such as `plain__l23__j` and their `qwen4b` filesystem paths are
unchanged. The control uses `qwen4b__plain_mean_pool`, grouped model
`jlens_qwen4b__plain_mean_pool_group`, and features such as
`plain_mean_pool__l23__j`.

Mean pooling removes dependence on one endpoint and uses evidence distributed
across the prompt. It does not make early token states aware of later caption
tokens, correct the released lens's WikiText-to-NSD domain shift, guarantee a
better sentence representation, or make tokens statistically independent. It
also weights tokenizer tokens equally, so words split into more subwords and
punctuation remain part of the declared estimand.

## Locked inference

Scores follow the established RSA pipeline: correlation-distance model RDMs
are compared with local fMRI-pattern RDMs, consistent with the RSA framework of
[Kriegeskorte, Mur, and Bandettini (2008)](https://doi.org/10.3389/neuro.06.004.2008)
and the eight-subject NSD resource of
[Allen et al. (2022)](https://doi.org/10.1038/s41593-021-00962-x).

Each sample score is the mean over valid searchlight centres; eight sample
scores are averaged within subject; the eight subjects are the independent
units. Two-sided exact tests enumerate all `2^8` sign flips of paired subject
differences, following the paired-label sign-flip principle used in
[permutation neuroimaging](https://doi.org/10.3389/fncom.2013.00171).

Benjamini–Hochberg correction
([Benjamini and Hochberg, 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x))
is applied separately to exactly these predeclared families:

1. `plain_mean_pool_j_vs_raw_4`: J minus raw at blocks 8, 16, 23, and 30.
2. `plain_mean_pool_vs_final_token_9`: pooled minus historical final-token for
   raw and J at each of four blocks, plus the pooled versus historical final
   raw control.

The two q-value families are never combined. MPNet remains an external semantic
reference. Any maxima are labelled descriptive peak searchlight-centre RSA
correlations and receive no peak-level inference.

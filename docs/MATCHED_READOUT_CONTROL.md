# Matched-readout prompt control

> **HISTORICAL — single-position readout.** Everything below was produced with the
> final-token readout, where the representation is read at one prompt position. That
> readout is no longer the project's method: 73.7% of caption prompts end in the same
> token (a period), and its RDM correlates 0.017–0.067 with a sentence-embedding
> reference of the same captions, against 0.41–0.58 for the pooled readout. These
> numbers are retained as a record of what was run, not as current results, and the
> caption-only figures in particular should be read as close to a measurement of
> nothing rather than as a weak effect. See `WHOLE_PROMPT_POOLING.md` and
> `../wiki/why-pooling-won.md`.

## Why this control is necessary

The historical prompt pair remains valid as an exploratory comparison, but it
does not isolate integration instructions. `plain` ends at punctuation supplied
by the final caption, whereas `visualize` ends at the fixed continuation cue
`Integrated visual scene:`. In a causal decoder, those are different token
positions with different immediate lexical contexts. Any representational
difference can therefore reflect the instruction, the readout cue, or both.

The new `matched_readout` prompt set is a paired intervention. For captions
represented below by `{caption block}`, its exact strings are:

```text
Source captions:
{caption block}

Scene representation:
```

and:

```text
Construct one coherent visual scene from the source captions. Reconcile their overlap into a single image: represent the entities, attributes, actions, spatial relations, and setting together. Infer only visually plausible structure. Form the scene rather than merely restating or listing caption words.

Source captions:
{caption block}

Scene representation:
```

Thus `integrate_readout == instruction_prefix + minimal_readout`. Caption text,
caption order, the full suffix beginning with `Source captions:`, and the final
readout token are identical. No chat template, generation, answer text,
caption modification, or truncation is allowed.

This is implemented as an immutable prompt-set registry, not a redefinition of
historical identifiers. The legacy default remains `historical` with output
name `qwen4b`; this control uses `qwen4b__matched_readout` and a correspondingly
isolated searchlight group. Model-free tests enforce the byte-level intervention.
The real-tokenizer preflight and extraction audit every selected condition,
record the UTF-8 suffix, suffix token IDs, final token ID, pairwise equality,
and stable hashes in `endpoint_audit.json`.

## Fixed analysis contract

- Same Qwen3.5-4B model, `qwen-n1000` lens, fitted layers 8/16/23/30,
  final block residual, final non-padding prompt position, native float32
  residuals, correlation-distance RDMs, and locked eight 100-image samples.
- Subjects are the independent units. Samples are averaged within subject
  before confidence intervals and exact two-sided sign-flip tests.
- BH family `matched_readout_j_vs_raw_8` contains exactly the four J-vs-raw
  contrasts for each of the two prompts.
- BH family `matched_readout_prompt_pair_9` contains exactly integrate-minus-
  minimal for raw and J at four layers plus the final residual. Its q-values
  are never pooled with the J-vs-raw family.
- Unembedding at layer 23 is descriptive only; RSA uses every component of the
  2,560-dimensional stored vector.

The paired design follows the general experimental principle that a control
should hold nuisance variables fixed. RSA and subject-level inference follow
Kriegeskorte, Mur, and Bandettini (2008), while false-discovery control follows
Benjamini and Hochberg (1995). The NSD repeated-measures structure is described
by Allen et al. (2022).

## References

- Kriegeskorte N, Mur M, Bandettini P. *Representational similarity analysis*.
  Frontiers in Systems Neuroscience (2008). https://doi.org/10.3389/neuro.06.004.2008
- Benjamini Y, Hochberg Y. *Controlling the false discovery rate*.
  Journal of the Royal Statistical Society B (1995).
  https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Allen EJ et al. *A massive 7T fMRI dataset to bridge cognitive neuroscience
  and artificial intelligence*. Nature Neuroscience (2022).
  https://doi.org/10.1038/s41593-021-00962-x

The completed execution record and results are in
[`MATCHED_READOUT_RESULTS.md`](MATCHED_READOUT_RESULTS.md).

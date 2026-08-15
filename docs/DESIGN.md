# Jacobian Lens × NSD: experiment design

## Fixed scientific contract

- Subjects 1–8, sessions 1–10, three-repeat conditions only.
- Reuse the existing eight disjoint 100-image samples per subject from
  `../results/mpnet_10_sessions`; never modify that result tree.
- Extract only the sorted union of required 1-based NSD condition IDs. The
  current data produce 835 conditions per subject and a 6,148-condition union.
- Compare transported and ordinary residuals from the same causal decoder.
  MPNet is retained only as a secondary reference.
- No held-out data, remote inference, or prompt-dependent generation.

## Choices considered

### Image-level position

1. Mean-pool every prompt token. This mixes instructions and lexical caption
   tokens, making a word-restatement explanation especially plausible.
2. Generate a scene description and pool its states. This introduces sampling,
   variable output lengths, and a second text-generation confound.
3. **Chosen: final non-padding prompt position.** A prompt ending in
   `Integrated visual scene:` makes that state the causal decoder's immediate
   scene-continuation state. This matches the J-lens source-position semantics
   and stays deterministic.

### Prompt

1. Caption text alone (**kept as the ablation**).
2. A chat-template instruction. Templates can change across Transformers/model
   revisions and may add hidden control tokens.
3. **Chosen primary:** deterministic plain text asking the model to reconcile
   the captions into one coherent scene, including entities, attributes,
   spatial relations, actions, and setting, and explicitly discouraging a word
   list. No answer is generated.

### Layer features

1. Every fitted layer: scientifically broad but expensive and multiplicity
   heavy for an overnight confirmatory run.
2. A single middle layer: cheap but risks missing the workspace transition.
3. **Chosen:** the fitted layers nearest 25%, 50%, 75%, and the penultimate
   block, resolved from the checkpoint at runtime. For each prompt/layer save
   raw block output `h_l` and transported `J_l h_l`; save final block output
   `h_final` once per prompt.

### Normalization and RDMs

Residuals are saved as finite float32 values in their native units. No
per-vector L2 normalization or feature standardization is applied. Subject
RDMs use SciPy correlation distance directly; it mean-centers each vector and
is invariant to positive scalar rescaling. Constant or non-finite rows are a
hard error rather than silently producing NaNs.

### Searchlight integration

1. Run the existing searchlight once per feature. Rejected because it repeats
   the expensive brain-RDM computation.
2. Change the project-wide pipeline API. Rejected because the current grouped
   model path already does the right computation.
3. **Chosen:** serialize every J/raw/final RDM plus the existing MPNet RDM into
   one lexically ordered model group per subject. The existing TensorFlow path
   then computes each brain RDM once and correlates it with the whole model
   matrix. A manifest fixes model-index-to-feature-name mapping for projection,
   plotting, and summaries.

The pinned upstream still materializes a multi-gigabyte full brain-RDM matrix
and lacks subject/session controls. `jlens_nsd.nsd_adapter` preserves its sphere
ordering and RDM math but correlates each TensorFlow batch immediately. The
audited interface delta is documented in `UPSTREAM_COMPATIBILITY.md`.

## Semantics and compatibility gates

Anthropic's `ActivationRecorder` hooks the output of decoder block `l`. In
Hugging Face `output_hidden_states`, that same tensor is `hidden_states[l+1]`
because element 0 is the embedding output. The final tuple element may be
post-final-normalization, so the equality is checked on every selected
non-final block; the final residual always comes from the reference recorder's
last-block hook. The extraction preflight validates this on the actual model
before writing chunks; it does not assume the index convention.

The lens checkpoint must agree with the loaded model on residual width, fitted
layer range, and source layer availability. The canonical pair is:

- `Qwen/Qwen3.5-4B`
- `neuronpedia/jacobian-lens`, revision `qwen-n1000`
- `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt`

The supported fallback pair is:

- `Qwen/Qwen3-1.7B`
- `neuronpedia/jacobian-lens`, revision `main`
- `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`

MPNet is not a valid J-lens target. It is a bidirectional sentence encoder, not
a causal decoder with a next-token unembedding and a future-token residual
mapping. There is also no matching pre-fitted MPNet checkpoint. Comparing Qwen
J-space to MPNet alone would confound representation with architecture/model;
the ordinary Qwen residuals are therefore the primary controls.

## Failure containment

- Deterministic sorted IDs, prompts, layers, model order, and seeds.
- One atomic NPZ per extraction chunk, with IDs inside every chunk.
- Manifest fingerprints cover configuration and the exact condition list.
- Resume validates completed chunk IDs, feature keys, shapes, and finiteness.
- GPU out-of-memory halves the microbatch down to one, clears the CUDA cache,
  and retries the same not-yet-published chunk.
- RDM and sampling outputs publish through same-directory temporary files.
- Searchlight stages run in subprocesses after extraction exits, preventing
  PyTorch and TensorFlow from retaining the GPU simultaneously.
- Existing MPNet artifacts are read-only inputs and are never overwritten.

## Scientific references

- Gurnee et al. define the Jacobian lens as an average layer-to-final residual
  Jacobian and distinguish it from ordinary logit-lens coordinates:
  https://transformer-circuits.pub/2026/workspace/index.html
- Kriegeskorte, Mur, and Bandettini introduce representational similarity
  analysis and its model/brain RDM comparison logic:
  https://doi.org/10.3389/neuro.06.004.2008
- Allen et al. describe NSD's repeated natural-scene measurements and stimulus
  organization: https://doi.org/10.1038/s41593-021-00962-x
- Lin et al. describe the COCO images and five-caption collection setup:
  https://doi.org/10.1007/978-3-319-10602-1_48

# Overnight runbook

## 1. Configure and validate

Export external paths once (examples are placeholders):

```bash
export JLENS_NSD_RESULTS=/path/to/jlens-results
export JLENS_NSD_NSD_DIR=/path/to/NSD
export JLENS_NSD_CAPTIONS=/path/to/nsd_allWords_per_image.pkl
export JLENS_NSD_MPNET_BASE=/path/to/mpnet_10_sessions
export JLENS_NSD_JLENS_CHECKOUT=/path/to/jacobian-lens  # optional
export JLENS_NSD_QWEN4B_MODEL=/path/to/Qwen3.5-4B      # optional
export JLENS_NSD_LENS_ROOT=/path/to/jacobian-lens-checkpoints  # optional
```

```bash
python -m pytest -q
ruff check .
ruff format --check .
jlens-nsd smoke
jlens-nsd prepare
jlens-nsd smoke --with-data
```

The established dataset contract is eight subjects, 10 sessions, 835 sorted
three-repeat conditions per subject, a 6,148-ID union, and eight disjoint
100-image samples per subject. Stop if data-backed smoke reports otherwise.

## 2. Resolve matched artifacts

The canonical matched pair is Qwen3.5-4B with the `qwen-n1000` lens; the
supported fallback is Qwen3-1.7B with its main-revision lens. Never mix model
and lens profiles.

```bash
jlens-nsd prefetch --profile qwen4b
jlens-nsd prefetch --profile qwen1.7b
```

Prefetch is the only stage that downloads by default. Extraction/preflight use
the local model directory or Hugging Face cache unless `--allow-download` is
explicitly passed.

## 3. GPU preflight

```bash
CUDA_VISIBLE_DEVICES=0 jlens-nsd preflight \
  --profile qwen4b --device cuda --max-length 256
```

Preflight validates residual width, fitted source layers, selected layers,
finite raw/J/final shapes, both prompt variants, and the recorder relationship
`block l == hidden_states[l+1]` for non-final blocks. Do not continue on a
failure; try the matched fallback profile only.

For Lucas's matched-readout control, select the scoped prompt set explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 jlens-nsd preflight \
  --profile qwen4b --prompt-set matched_readout \
  --device cuda --max-length 256
```

This additionally audits the common suffix tokens and identical final token ID
for every condition in the selected union. Outputs use the isolated namespace
`qwen4b__matched_readout`; historical `visualize/plain` outputs are unchanged.

## 4. Resume-safe orchestration

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
jlens-nsd-orchestrate \
  --prompt-set matched_readout \
  --profile qwen4b --fallback-profile qwen1.7b \
  --batch-size 8 --chunk-size 64 --max-length 256
```

To validate only a selected subject, pass the same explicit subset through the
orchestrator. Condition union, extraction, RDMs, searchlight, projection, plots,
and the descriptive report are then all scoped to that subset:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  jlens-nsd-orchestrate --subjects 1 \
  --profile qwen4b --fallback-profile qwen1.7b \
  --batch-size 8 --chunk-size 64 --max-length 256
```

```text
conditions → preflight → extraction → grouped RDMs
           → 8 streamed searchlights → projection → maps → report
```

Re-run the same command after interruption. Extraction validates atomic chunks;
searchlight and projection skip complete outputs. State and logs are below
`$JLENS_NSD_RESULTS`.

### Whole-prompt pooling control

The scoped control reuses only historical `plain` prompts and writes to
`qwen4b__plain_mean_pool`; historical and matched-readout outputs are untouched.
Its report requires the immutable historical full-run root for paired readout
comparisons:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
jlens-nsd-orchestrate \
  --readout-mode all_token_mean \
  --historical-results-root /path/to/historical/jlens-results \
  --profile qwen4b --fallback-profile qwen1.7b \
  --batch-size 8 --chunk-size 64 --max-length 256
```

The extraction manifest links `token_mask_audit.json` and records per-layer
tokenwise-J/pooling linearity errors. See
[`WHOLE_PROMPT_POOLING.md`](WHOLE_PROMPT_POOLING.md) for the exact estimand and
the separate 4-test and 9-test BH families.

## 5. Manual stages

```bash
jlens-nsd extract --profile qwen4b --device cuda \
  --batch-size 8 --chunk-size 64
jlens-nsd rdms --profile qwen4b
jlens-nsd searchlight --profile qwen4b --subject 1
jlens-nsd project --profile qwen4b --subjects 1,2,3,4,5,6,7,8
jlens-nsd plot --profile qwen4b --subjects 1,2,3,4,5,6,7,8
jlens-nsd summarize --profile qwen4b
```

Tiny extraction uses a distinct namespace and can never feed the full RDM
stage because the condition fingerprint differs:

```bash
jlens-nsd extract --profile qwen4b --device cuda --max-conditions 4 \
  --batch-size 2 --chunk-size 2 --output-name qwen4b_smoke
```

## Interpretation guardrails

- `J_l h_l` is the fitted Jacobian-transported residual, not a sparse or
  non-negative decomposition.
- The primary control is the raw residual from the same Qwen, prompt, layer,
  token position, and condition.
- MPNet is an external semantic reference, not a valid Jacobian-lens target.
- Report inference averages samples within subject, then uses subjects as the
  independent units for exact sign-flip tests and Benjamini–Hochberg control.
- These whole-searchlight summaries are exploratory and are not held-out model
  selection.

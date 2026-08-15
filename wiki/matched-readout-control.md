# Matched-readout control agent memory

## Request and design

- Authoritative request: Lucas matched-readout prompt-control experiment,
  implemented only on branch `matched-readout-control`.
- Chosen architecture: immutable prompt-set registry threaded through every
  model-dependent/downstream stage.
- Historical default: prompt set `historical`, prompt IDs `visualize/plain`,
  run name `qwen4b` (unchanged).
- New set: `matched_readout`, IDs `integrate_readout/minimal_readout`, run name
  `qwen4b__matched_readout`.
- Common fixed suffix is UTF-8 `\n\nScene representation:`. Integrated prompt
  is exactly the deterministic historical instruction plus two newlines plus
  the entire minimal prompt.
- Every real-tokenizer condition must pass the suffix-token and final-token-ID
  audit before extraction can publish chunks.

## Analysis contract

- Primary family `matched_readout_j_vs_raw_8`: two prompts × four layers.
- Paired family `matched_readout_prompt_pair_9`: integrate minus minimal for
  raw/J × four layers plus final. BH adjustment is within family only.
- Subjects are independent; eight samples are averaged within each subject.
- Unembedding uses deterministic licensed subject-1 conditions and the existing
  formatting-only filter, and remains interpretive.

## Status

- Subject-1 root: `results/matched_readout_subject1_20260815`; complete across
  835 conditions, 14 chunks, eight samples, projection, 19 plots, and report.
- Full root: `results/matched_readout_full_20260815`; complete across eight
  subjects, 6,148 conditions, 97 chunks, 64 samples, 304 required surfaces,
  152 plots, and 15 orchestrator stages.
- Full endpoint: suffix bytes
  `0a0a5363656e6520726570726573656e746174696f6e3a`, token IDs
  `[271,9723,12669,25]`, final ID 25, all 6,148 pairs pass, max length 181.
- Input hashes match the historical authoritative run: union fingerprint
  `f7b3ae...617e3`, captions `5fb429...125f`, lens `1f9a8f...534e`.
- Full statistical outputs: 19 scores and 17 comparisons; BH families contain
  exactly 8 J-vs-raw and 9 prompt-pair contrasts. Five J-vs-raw q values are
  below .05; no prompt-pair q value is below .05.
- Compact committed candidates: `docs/matched_readout_performance.csv`, endpoint
  JSON, layer summary PNG, and two full-vector unembedding PNGs. Large results
  remain ignored.

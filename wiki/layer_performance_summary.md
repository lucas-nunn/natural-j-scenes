# Layer performance summary task memory

## Scope and authoritative source

- Task date: 2026-08-15.
- Source is the completed `qwen4b` report with 8 subjects, 8 matched 100-image
  samples per subject, and 10 sessions. The subject-1 validation report is not
  used.
- Required source files: `feature_scores.csv`, `comparisons.csv`,
  `subject_scores.npy`, `sample_scores.npy`, and `summary.json`.
- Searchlights are not recomputed and decoder layers are never averaged.

## Summary semantics audited from upstream code

- Each sample/model score is `nanmean` across valid searchlight-center
  correlations.
- Each subject/model score is the arithmetic mean of that subject's eight
  sample scores.
- `mean_correlation` is the arithmetic mean across the eight subject scores.
- Feature CIs are two-sided 95% Student t intervals across subjects (df=7).
- Paired deltas are subject-wise J minus matched raw after within-subject
  aggregation. Their CIs use the same subject-level t interval.
- Exact p-values are two-sided exhaustive sign-flip tests of the absolute mean
  delta over all 2^8 assignments.
- BH q-values cover all 24 upstream comparisons: each of 8 J features versus
  matched raw, same-prompt final, and MPNet. The publication table selects the
  8 J-versus-raw comparisons without recalculating q-values.

## Derived artifact exception

Older wiki memory says generated results are absent from the standalone repo.
The current task explicitly requires small, auditable publication outputs under
`docs/`. This is a scoped exception for the compact CSV, PNG figure, and JSON
provenance record only. Large source reports, arrays, searchlights, and maps
remain external and ignored.

## Design and validation contract

- Figure: two prompt columns; upper raw/J trajectories plus a single final
  control; lower paired J-minus-raw deltas with subject-level CIs and stars only
  for source BH q < 0.05.
- Prompt display label is `Caption-only (plain)` while source IDs remain
  `plain`.
- MPNet is omitted from the figure to avoid clutter; it remains relevant only
  because its comparisons belong to the source BH family.
- Generator accepts a report directory argument and contains no workstation
  source path.
- NPY column order is validated against the complete lexical grouped-RDM model
  order. CSV and JSON rows must agree exactly; NPY-derived statistics must
  reproduce source values within floating-point roundoff; sample means must
  equal serialized subject scores bit-for-bit.

## Result pattern

- BH-significant J-minus-raw: `visualize` layer 23 (Δr 0.001261483580209547,
  p 0.0078125, q 0.01875) and layer 30 (Δr 0.0016450714013353058,
  p 0.0078125, q 0.01875).
- No `plain` layer and no earlier `visualize` layer is significant.
- Final controls: `visualize` 0.02293069928518198; `plain`
  0.00586490844657969.

## Completed verification

- Generator validation passed against all five authoritative report files.
- Two consecutive regenerations were byte-identical: CSV SHA-256
  `6ff448538c5f7aa352ef27b7fe276862b1cde6fe2f8b25663a5c4174adaaf2ff`;
  PNG SHA-256
  `e4823d1a16ac601c4e3f39def58baf7cc6ca54dc6281348293b39e8f6cc7489b`;
  metadata SHA-256
  `33d195cb40b8b532e4a0451fb98cc36fa89fdad1595da10539806f26265d3e10`.
- Full unit suite: 22 tests passed.
- Ruff lint, Ruff formatting check, `git diff --check`, CSV structure/content
  assertions, PNG metadata/dimensions, and visual inspection passed.
- Figure dimensions are 3000 × 2040 px at 300 dpi.

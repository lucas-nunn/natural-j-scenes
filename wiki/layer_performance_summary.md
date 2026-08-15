# Layer performance summary task memory

## Scope and authoritative source

- Task date: 2026-08-15.
- Source is the completed `qwen4b` report with 8 subjects, 8 matched 100-image
  samples per subject, and 10 sessions. The subject-1 validation report is not
  used.
- Required source files: `feature_scores.csv`, `comparisons.csv`,
  `subject_scores.npy`, `sample_scores.npy`, and `summary.json`.
- Peak extension authoritative root:
  `/home/chuddy/dev/research/neuroconnectionism/lucas_exploration/jlens_experiment/results`.
- Authoritative native centre arrays are the upstream per-subject files under
  `lucas_exploration/results/mpnet_10_sessions/precomputed` (the same files used
  by the searchlight stage). Searchlights are not recomputed, rendered/projected
  maps are never read, and decoder layers are never averaged.

## Peak extension and conflict resolution

- New task requirement conflicts with the earlier generator contract below
  saying it does not read searchlight volumes. The new authoritative contract
  supersedes that statement for descriptive peak computation only.
- Label exactly as peak SEARCHLIGHT-CENTRE RSA correlations, not single-voxel
  correlations. Each centre summarizes its local spherical searchlight.
- For each subject/feature, average its eight native sample correlation maps
  centrewise over only authoritative centre indices. A centre with any
  nonfinite sample is nonfinite in the subject mean and cannot win. Then take
  the maximum finite centre.
- Across subjects report mean subject peak, two-sided 95% subject t CI (df=7),
  and observed subject peak range. Never maximize 64 sample maps and never mix
  native coordinates between subjects.
- Peaks are descriptive/noise-sensitive. No peak delta, p-value, q-value, or
  significance claim is generated. Existing J-vs-raw inference remains based
  on paired subject whole-searchlight means.

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
- Generator requires explicit report and full result roots, permits an explicit
  centre-root override, and contains no authoritative-data workstation path.
- NPY column order is validated against the complete lexical grouped-RDM model
  order. CSV and JSON rows must agree exactly; NPY-derived statistics must
  reproduce source values within floating-point roundoff; sample means must
  equal serialized subject scores bit-for-bit.
- Group manifest feature/index/model-name order is validated against the same
  19-column contract. All 64 grouped native volumes must be float64 with 19
  models and stable within-subject spatial shapes; every model/sample must have
  a finite authoritative centre. Centre arrays must be nonempty, 1D, integer,
  unique, nonnegative, and in bounds. Native-volume means must reproduce all
  report sample scores.

## Result pattern

- BH-significant J-minus-raw: `visualize` layer 23 (Δr 0.001261483580209547,
  p 0.0078125, q 0.01875) and layer 30 (Δr 0.0016450714013353058,
  p 0.0078125, q 0.01875).
- No `plain` layer and no earlier `visualize` layer is significant.
- Final controls: `visualize` 0.02293069928518198; `plain`
  0.00586490844657969.
- Descriptive mean subject peaks (raw, J): visualize L8
  (0.061874174756667344, 0.056413084701489424), L16
  (0.03303575244353851, 0.029828819338945323), L23
  (0.19890948093961924, 0.20511762134265155), L30
  (0.1781312336679548, 0.19458385067991912); plain L8
  (0.03283017936428223, 0.03003315742898849), L16
  (0.029868879931200354, 0.033455957511250745), L23
  (0.03827422804170055, 0.04234938326771953), L30
  (0.0520069090925972, 0.05654084784328006).
- Descriptive final-control mean subject peaks: visualize
  0.19266082742251456; plain 0.05873851489741355.

## Completed verification

- Generator validation passed against all five authoritative report files, the
  grouped manifest, all 64 native grouped correlation volumes, and all eight
  authoritative centre arrays.
- Two consecutive regenerations were byte-identical: CSV SHA-256
  `9c835a095f305fdb43b48190547f451ebd25e225e6489f7effcba922e5f3a700`;
  PNG SHA-256
  `e4823d1a16ac601c4e3f39def58baf7cc6ca54dc6281348293b39e8f6cc7489b`;
  metadata SHA-256
  `92693420812c6ef4360801feda8842046799f63a88f50f238006a0751666bfa3`.
- Full unit suite: 27 tests passed, 2 skipped optional-dependency tests.
- Ruff lint, Ruff formatting check, `git diff --check`, CSV structure/content
  assertions, README-to-CSV checks, PNG metadata/dimensions, and preservation
  checks for unrelated assets passed.
- Figure dimensions are 3000 × 2040 px at 300 dpi.

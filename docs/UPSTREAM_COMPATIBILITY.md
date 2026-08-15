# Pinned upstream API audit

Audit base: `nsd_visuo_semantics` commit
`a60e0eafb8d02841159e344adb732062734bc302`, the merge base of the local fork
and current public `adriendoerig/visuo_llm` head inspected on 2026-08-15.

## Used unchanged

- `utils.nsd_get_data_light.get_subject_conditions`: unchanged signature used
  for sorted three-repeat condition alignment.
- `get_masks`, `get_model_rdms`, and `read_behavior`: upstream inputs retained.
- `utils.batch_gen.BatchGen`: condensed-RDM subset indexing retained.
- `searchlight_analyses.searchlight.RSASearchLight`: sphere construction retained.
- `utils.tf_utils.sort_spheres`, `chunking`, and `compute_rdm_batch`: upstream
  ordering and brain-RDM math retained.
- `nsdcode.nsd_mapdata.NSDmapdata.fit`: upstream projection primitive retained.

## Fork-only interfaces retained locally

| Local requirement | Upstream limitation | Local adapter |
|---|---|---|
| One subject, 10 sessions, optional sample cap | Searchlight loops all 8 subjects and fixes 40 sessions | `run_searchlight` |
| Grouped model correlations without a multi-GB brain-RDM matrix | Upstream materializes every brain RDM before correlation | `_tf_searchlight_corr` |
| Bounded-memory beta preparation | Upstream concatenates every session before averaging | `_compute_betas_average` |
| Explicit subject subset for projection | Upstream always projects all 8 subjects | `project_to_fsaverage` |
| Named ROI contours and current pycortex behavior | Upstream plotting API accepts neither | `plot_brain` |

The old fork also changed unrelated embedding/download utilities; none are
imported or copied here. The adapter deliberately refuses to generate missing
sample choices: locked MPNet samples must already have been copied by
`prepare`, preserving subject-condition alignment.

## Known packaging mismatch

The pinned upstream source declares TensorFlow 2.15 and many whole-project
dependencies. They are unnecessary for this experiment's lightweight tests
and unsuitable for modern Python environments. Therefore its exact VCS
dependency is an optional `nsd-upstream` extra, while the actual direct runtime
imports are separately declared in `nsd-runtime`. The README documents the
`--no-deps` source-install path; this does not alter upstream source.

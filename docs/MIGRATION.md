# Migration from `neuroconnectionism`

Nothing needs to be copied. Point the new package at existing inputs and, if
desired, the old J-lens output root.

| Old location | New configuration |
|---|---|
| `lucas_exploration/jlens_experiment/results/` | `JLENS_NSD_RESULTS` / `--results-dir` |
| `lucas_exploration/results/mpnet_10_sessions/` | `JLENS_NSD_MPNET_BASE` / `--mpnet-base` |
| `lucas_exploration/results/saved_embeddings/nsd_allWords_per_image.pkl` | `JLENS_NSD_CAPTIONS` / `--captions` |
| Former hard-coded NSD root | `JLENS_NSD_NSD_DIR` / `--nsd-dir` |
| Sibling `jacobian-lens/` checkout | `JLENS_NSD_JLENS_CHECKOUT` / `--jlens-checkout` |
| Former local Qwen/lens cache roots | model-specific env variables or `--model-path` / `--lens-root` |

To resume old J-lens chunks without moving them, set `JLENS_NSD_RESULTS` to the
absolute old `jlens_experiment/results` directory. Manifests contain historical
absolute provenance paths, but active resolution comes from current CLI/env
configuration. Resume fingerprints intentionally reject changed model, lens,
prompt, IDs, layers, chunk size, or token limits.

The old checkout remains untouched and should not be added to `PYTHONPATH`.
Imports change from `lucas_exploration.jlens_experiment.*` to `jlens_nsd.*`;
commands change from `python -m ...run` to `jlens-nsd`, and from
`python -m ...orchestrate` to `jlens-nsd-orchestrate`.

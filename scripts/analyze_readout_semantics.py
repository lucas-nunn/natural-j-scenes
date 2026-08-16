#!/usr/bin/env python3
"""Why does the pooled readout beat the single-endpoint readout?

The eight-subject run showed mean-pooled caption features aligning with cortex
5-9x better than the historical final-token readout. That is a large effect for
a change that adds no new text and no new information — the pooled vector is a
linear function of states the endpoint readout also had access to. Two
explanations fit the brain result equally well:

**(a) the endpoint is a poor estimator** — one position, often punctuation, is a
noisy summary of a caption block, so its RDM is dominated by position-specific
idiosyncrasy rather than content; or

**(b) pooling adds distributed content** the endpoint genuinely lacks.

These are separable without touching fMRI data, by scoring both readouts against
an independent *semantic* reference. MPNet sentence embeddings of the same
captions provide one: they encode caption meaning with no access to Qwen, the
lens, or the brain. If pooling mainly rescues semantic structure that the
endpoint discards, the pooled RDM will track the MPNet RDM far more closely —
and the size of that gap is a model-side prediction of the brain-side gap, with
no circularity.

Reports, per layer and representation, the correlation of each readout's RDM
with the MPNet RDM, plus the correlation between the two readouts themselves.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from analyze_lens_geometry import correlation_rdm, load_feature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-chunks", type=Path, required=True)
    parser.add_argument("--historical-chunks", type=Path, required=True)
    parser.add_argument("--mpnet-embeddings", type=Path, required=True)
    parser.add_argument("--layers", default="8,16,23,30")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.mpnet_embeddings.open("rb") as handle:
        mpnet = np.asarray(pickle.load(handle))

    rows = []
    for layer in (int(item) for item in args.layers.split(",")):
        for kind in ("raw", "j"):
            pooled_ids, pooled = load_feature(
                args.pooled_chunks, f"plain_mean_pool__l{layer:02d}__{kind}"
            )
            final_ids, final = load_feature(
                args.historical_chunks, f"plain__l{layer:02d}__{kind}"
            )
            if not np.array_equal(pooled_ids, final_ids):
                raise ValueError("the two runs cover different condition sets")

            # NSD IDs are 1-based; the embedding table is 0-based.
            semantic = mpnet[pooled_ids - 1]

            rng = np.random.default_rng(args.seed)
            pooled_vs_semantic, final_vs_semantic, pooled_vs_final = [], [], []
            for _ in range(args.n_samples):
                choice = rng.choice(
                    pooled.shape[0], size=args.sample_size, replace=False
                )
                a = correlation_rdm(pooled[choice])
                b = correlation_rdm(final[choice])
                c = correlation_rdm(semantic[choice])
                pooled_vs_semantic.append(float(np.corrcoef(a, c)[0, 1]))
                final_vs_semantic.append(float(np.corrcoef(b, c)[0, 1]))
                pooled_vs_final.append(float(np.corrcoef(a, b)[0, 1]))

            rows.append(
                {
                    "layer": layer,
                    "kind": kind,
                    "pooled_vs_semantic": float(np.mean(pooled_vs_semantic)),
                    "pooled_vs_semantic_std": float(np.std(pooled_vs_semantic, ddof=1)),
                    "final_token_vs_semantic": float(np.mean(final_vs_semantic)),
                    "final_token_vs_semantic_std": float(
                        np.std(final_vs_semantic, ddof=1)
                    ),
                    "ratio": float(
                        np.mean(pooled_vs_semantic) / np.mean(final_vs_semantic)
                    ),
                    "pooled_vs_final_token": float(np.mean(pooled_vs_final)),
                }
            )
            print(
                f"l{layer:02d} {kind:3}  pooled~MPNet={rows[-1]['pooled_vs_semantic']:.4f}  "
                f"final~MPNet={rows[-1]['final_token_vs_semantic']:.4f}  "
                f"ratio={rows[-1]['ratio']:.2f}x  "
                f"pooled~final={rows[-1]['pooled_vs_final_token']:.4f}",
                flush=True,
            )

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "sample_size": args.sample_size,
                    "n_samples": args.n_samples,
                    "seed": args.seed,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

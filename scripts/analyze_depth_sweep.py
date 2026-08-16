#!/usr/bin/env python3
"""Sweep every fitted lens layer to separate depth from conditioning.

The production analysis uses 4 of the 31 layers the lens was fitted for. Across
four points, "the J advantage grows with depth" and "the J advantage tracks how
well-conditioned the lens is" are indistinguishable — the two candidate
explanations are collinear, because an *average* Jacobian loses rank the further
its source layer sits from the final layer.

Sweeping all 31 layers turns that into a testable comparison. For each layer this
computes, from one forward pass:

- the lens's effective rank (participation ratio of its singular values);
- the semantic alignment of the pooled raw readout, and of its J-transported
  counterpart, scored by RDM correlation against MPNet sentence embeddings;
- their difference, the J advantage.

With 31 points the J advantage can be correlated against depth and against
effective rank, and against depth with rank partialled out. That cannot settle
the brain question — semantic alignment is not cortical alignment — but it can
say whether the depth story survives once conditioning is accounted for, which
four points cannot.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from analyze_lens_geometry import correlation_rdm


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Correlation of x and y with the linear effect of z removed from both."""

    def residual(target: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones_like(z), z])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return target - design @ coefficients

    return float(np.corrcoef(residual(x), residual(y))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--union-ids", type=Path, required=True)
    parser.add_argument("--mpnet-embeddings", type=Path, required=True)
    parser.add_argument("--n-conditions", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens_nsd.prompts import load_caption_table, prompts_for_condition

    union = np.load(args.union_ids, allow_pickle=False)[: args.n_conditions]
    table = load_caption_table(args.captions)
    prompts = [
        prompts_for_condition(table, int(cid), "historical")["plain"] for cid in union
    ]

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    tokenizer.padding_side = "right"
    model = (
        AutoModelForCausalLM.from_pretrained(str(args.model_path), dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )

    lens = torch.load(args.lens_path, map_location="cpu", weights_only=False)
    layers = sorted(int(layer) for layer in lens["J"])

    pooled: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            encoded = tokenizer(
                prompts[start : start + args.batch_size],
                add_special_tokens=True,
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            states = model(**encoded, output_hidden_states=True).hidden_states
            weights = encoded["attention_mask"].float().unsqueeze(-1)
            denominator = weights.sum(1)
            for layer in layers:
                mean = (states[layer + 1].float() * weights).sum(1) / denominator
                pooled[layer].append(mean.cpu().numpy().astype(np.float32))
    features = {layer: np.concatenate(blocks) for layer, blocks in pooled.items()}
    del model
    torch.cuda.empty_cache()

    with args.mpnet_embeddings.open("rb") as handle:
        semantic = np.asarray(pickle.load(handle))[union - 1]

    rng = np.random.default_rng(0)
    samples = [
        rng.choice(len(union), size=min(100, len(union)), replace=False)
        for _ in range(args.n_samples)
    ]
    references = [correlation_rdm(semantic[choice]) for choice in samples]

    rows = []
    for layer in layers:
        matrix = np.asarray(lens["J"][layer], dtype=np.float32).astype(np.float64)
        values = np.linalg.svd(matrix, compute_uv=False)
        values = values[values > 0]
        effective_rank = float(values.sum() ** 2 / np.square(values).sum())

        raw = features[layer].astype(np.float64)
        transported = raw @ matrix.T
        raw_scores, j_scores = [], []
        for choice, reference in zip(samples, references, strict=True):
            raw_scores.append(
                float(np.corrcoef(correlation_rdm(raw[choice]), reference)[0, 1])
            )
            j_scores.append(
                float(
                    np.corrcoef(correlation_rdm(transported[choice]), reference)[0, 1]
                )
            )
        rows.append(
            {
                "layer": layer,
                "effective_rank": effective_rank,
                "raw_vs_semantic": float(np.mean(raw_scores)),
                "j_vs_semantic": float(np.mean(j_scores)),
                "j_advantage": float(np.mean(j_scores) - np.mean(raw_scores)),
            }
        )
        print(
            f"l{layer:02d}  eff_rank={effective_rank:>7.1f}  "
            f"raw={rows[-1]['raw_vs_semantic']:.4f}  j={rows[-1]['j_vs_semantic']:.4f}  "
            f"adv={rows[-1]['j_advantage']:+.4f}",
            flush=True,
        )

    depth = np.array([row["layer"] for row in rows], dtype=float)
    rank = np.array([row["effective_rank"] for row in rows])
    advantage = np.array([row["j_advantage"] for row in rows])
    summary = {
        "n_layers": len(rows),
        "corr_advantage_depth": float(np.corrcoef(advantage, depth)[0, 1]),
        "corr_advantage_rank": float(np.corrcoef(advantage, rank)[0, 1]),
        "corr_rank_depth": float(np.corrcoef(rank, depth)[0, 1]),
        "partial_advantage_depth_given_rank": partial_correlation(
            advantage, depth, rank
        ),
        "partial_advantage_rank_given_depth": partial_correlation(
            advantage, rank, depth
        ),
    }
    print("\n" + json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        args.output.write_text(
            json.dumps(
                {"n_conditions": int(len(union)), "summary": summary, "rows": rows},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

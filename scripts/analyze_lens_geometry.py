#!/usr/bin/env python3
"""Measure how much the Jacobian lens warps representational geometry.

The project's design rests on an axiom: ``J_l`` is a fixed linear map, so
``X_J = X_raw @ J_l.T`` spans the same subspace as ``X_raw`` whenever ``J_l`` is
full rank. Under ordinary least squares the two feature spaces are therefore
exactly equivalent. Any measured J-vs-raw difference under RSA can only come
from the *metric change*: correlation distance is not invariant under a general
linear map, so ``J_l`` warps the RDM.

That argument is stated in the README but has never been measured. This script
quantifies the warp directly from stored features and the released lens, so the
size of the warp can be compared against the size of the observed brain effect.

Reported per layer:

``rdm_r``
    Pearson correlation between the raw and J correlation-distance RDMs, over
    samples of ``--sample-size`` conditions drawn to match the RSA sampling
    scale. ``1.0`` means the lens left representational geometry untouched, and
    no J-vs-raw brain difference is possible; lower values bound how much
    difference is available.
``effective_rank``
    Participation ratio of the singular values, ``(sum s)^2 / sum s^2``. How
    many directions the map meaningfully uses, out of ``d_model``.
``condition``
    ``s_max / s_min``. How anisotropically the map rescales directions.
``spectral_entropy``
    Shannon entropy of the normalised singular values, in nats.

Nothing here touches fMRI data; it is a property of the model features and the
lens alone.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_feature(chunk_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate one stored feature across chunks, ordered by condition id."""
    files = sorted(glob.glob(str(chunk_dir / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no chunks under {chunk_dir}")
    ids, blocks = [], []
    for path in files:
        with np.load(path) as handle:
            if name not in handle.files:
                raise KeyError(f"{name} missing from {path}")
            blocks.append(np.asarray(handle[name], dtype=np.float64))
            ids.append(np.asarray(handle["condition_ids"], dtype=np.int64))
    condition_ids = np.concatenate(ids)
    features = np.concatenate(blocks, axis=0)
    order = np.argsort(condition_ids)
    return condition_ids[order], features[order]


def correlation_rdm(matrix: np.ndarray) -> np.ndarray:
    """Upper-triangle correlation-distance RDM, matching the analysis metric."""
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("a feature vector is constant; correlation is undefined")
    unit = centered / norms
    similarity = unit @ unit.T
    index = np.triu_indices(matrix.shape[0], k=1)
    return 1.0 - similarity[index]


def spectrum_stats(matrix: np.ndarray) -> dict[str, float]:
    values = np.linalg.svd(matrix, compute_uv=False)
    values = values[values > 0]
    total = values.sum()
    proportions = values / total
    return {
        "effective_rank": float(total**2 / np.square(values).sum()),
        "condition": float(values.max() / values.min()),
        "spectral_entropy": float(-(proportions * np.log(proportions)).sum()),
        "n_singular_values": int(values.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument("--namespace", default="plain_mean_pool")
    parser.add_argument("--layers", default="8,16,23,30")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch

    lens = torch.load(args.lens_path, map_location="cpu", weights_only=False)
    layers = [int(item) for item in args.layers.split(",")]
    rng = np.random.default_rng(args.seed)

    rows = []
    for layer in layers:
        ids, raw = load_feature(args.chunk_dir, f"{args.namespace}__l{layer:02d}__raw")
        _, transported = load_feature(
            args.chunk_dir, f"{args.namespace}__l{layer:02d}__j"
        )
        if raw.shape != transported.shape:
            raise ValueError(f"shape mismatch at layer {layer}")

        correlations = []
        for _ in range(args.n_samples):
            choice = rng.choice(raw.shape[0], size=args.sample_size, replace=False)
            a = correlation_rdm(raw[choice])
            b = correlation_rdm(transported[choice])
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
        correlations = np.asarray(correlations)

        stats = spectrum_stats(
            np.asarray(lens["J"][layer], dtype=np.float32).astype(np.float64)
        )
        rows.append(
            {
                "layer": layer,
                "n_conditions": int(raw.shape[0]),
                "rdm_r_mean": float(correlations.mean()),
                "rdm_r_std": float(correlations.std(ddof=1)),
                "rdm_r_min": float(correlations.min()),
                **stats,
            }
        )

    report = {
        "namespace": args.namespace,
        "sample_size": args.sample_size,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "metric": "correlation",
        "rows": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

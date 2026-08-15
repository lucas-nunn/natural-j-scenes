#!/usr/bin/env python3
"""Ask whether the lens's *directions* matter, or only its conditioning.

`analyze_lens_geometry.py` established that ``J_l`` warps representational
geometry, and that the warp is *largest* at the layers where the brain effect is
absent. That leaves an obvious follow-up: is the warp a property of the
particular directions the lens learned, or merely of its singular value
spectrum — how anisotropically it rescales, regardless of what it rescales?

Two controls separate those, both built from the layer's own SVD
``J = U S V^T`` so nothing is imported from outside the map itself:

``spectrum_matched``
    ``Q1 S Q2^T`` with fresh random orthogonal ``Q1``, ``Q2``. Identical
    singular values, random directions. If this warps the RDM as much as the
    real lens, the warp is explained by conditioning alone and the learned
    directions add nothing measurable.
``orthogonal``
    ``U V^T`` — the lens's own rotation with its spectrum flattened to all ones.
    Isolates the opposite half: what the directions do once anisotropy is
    removed.

Also re-derives ``X_raw @ J^T`` and checks it against the stored J features.
That is an end-to-end audit of the extraction pipeline from committed artifacts
alone: if the stored features were produced by a different matrix, a different
layer, or a transposed multiply, this catches it.

No fMRI data is touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_lens_geometry import correlation_rdm, load_feature


def random_orthogonal(size: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.standard_normal((size, size))
    q, r = np.linalg.qr(matrix)
    return q * np.sign(np.diag(r))


def mean_rdm_correlation(
    reference: np.ndarray,
    other: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
    n_samples: int,
) -> tuple[float, float]:
    values = []
    for _ in range(n_samples):
        choice = rng.choice(reference.shape[0], size=sample_size, replace=False)
        values.append(
            float(
                np.corrcoef(
                    correlation_rdm(reference[choice]), correlation_rdm(other[choice])
                )[0, 1]
            )
        )
    array = np.asarray(values)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument("--namespace", default="plain_mean_pool")
    parser.add_argument("--layers", default="8,16,23,30")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch

    lens = torch.load(args.lens_path, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(args.seed)
    rows = []

    for layer in (int(item) for item in args.layers.split(",")):
        _, raw = load_feature(args.chunk_dir, f"{args.namespace}__l{layer:02d}__raw")
        _, stored_j = load_feature(args.chunk_dir, f"{args.namespace}__l{layer:02d}__j")
        matrix = np.asarray(lens["J"][layer], dtype=np.float32).astype(np.float64)

        # End-to-end audit: does the stored J feature equal raw @ J^T?
        rederived = raw @ matrix.T
        scale = float(np.max(np.abs(stored_j)))
        max_abs = float(np.max(np.abs(rederived - stored_j)))

        u, s, vt = np.linalg.svd(matrix)
        spectrum_matched = (
            random_orthogonal(matrix.shape[0], rng) * s
        ) @ random_orthogonal(matrix.shape[0], rng)
        orthogonal = u @ vt

        actual_mean, actual_std = mean_rdm_correlation(
            raw, rederived, rng, args.sample_size, args.n_samples
        )
        matched_mean, matched_std = mean_rdm_correlation(
            raw, raw @ spectrum_matched.T, rng, args.sample_size, args.n_samples
        )
        orth_mean, orth_std = mean_rdm_correlation(
            raw, raw @ orthogonal.T, rng, args.sample_size, args.n_samples
        )

        rows.append(
            {
                "layer": layer,
                "rederivation_max_abs_error": max_abs,
                "rederivation_feature_scale": scale,
                "rederivation_relative_error": max_abs / scale,
                "rdm_r_actual": actual_mean,
                "rdm_r_actual_std": actual_std,
                "rdm_r_spectrum_matched": matched_mean,
                "rdm_r_spectrum_matched_std": matched_std,
                "rdm_r_orthogonal": orth_mean,
                "rdm_r_orthogonal_std": orth_std,
            }
        )
        print(json.dumps(rows[-1], indent=2, sort_keys=True), flush=True)

    report = {
        "namespace": args.namespace,
        "sample_size": args.sample_size,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "rows": rows,
    }
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

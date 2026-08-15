#!/usr/bin/env python3
"""Per-subject robustness of the J-vs-raw contrast.

The predeclared inference is an exact sign-flip test, which uses only the
*signs* of the eight paired subject differences. That is deliberately
conservative, but it means the reported group effect is compatible with one
subject carrying an outsized delta while the rest hover near zero. The summary
artifacts report group means and q-values; nothing reports whether the effect is
distributed.

This script answers three questions the group table cannot:

1. **How many subjects agree in sign, and how large is each delta?**
2. **Does significance survive dropping any single subject?** Leave-one-out over
   all eight, recomputing the exact test at n=7 each time. Reported p-values are
   uncorrected: the point is stability of the effect, not re-inference.
3. **How noisy is each layer's estimate?** Reported as the across-subject
   standard deviation of the delta and the ratio ``|mean| / std``, so a layer
   that fails for lack of precision can be told apart from one that fails for
   lack of effect.

Reads only committed run artifacts: ``subject_scores.npy`` and the group
manifest that fixes the feature order.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def exact_sign_flip_p(differences: np.ndarray) -> float:
    """Two-sided exact test over all 2^n sign assignments."""
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-scores", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--namespace", default="plain_mean_pool")
    parser.add_argument("--layers", default="8,16,23,30")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scores = np.load(args.subject_scores, allow_pickle=False)
    manifest = json.loads(args.group_manifest.read_text(encoding="utf-8"))
    order = [item["feature"] for item in manifest["model_order"]]
    index = {feature: position for position, feature in enumerate(order)}
    if scores.shape[1] != len(order):
        raise ValueError(
            f"subject_scores has {scores.shape[1]} columns but the manifest "
            f"declares {len(order)} models"
        )

    rows = []
    for layer in (int(item) for item in args.layers.split(",")):
        j = scores[:, index[f"{args.namespace}__l{layer:02d}__j"]]
        raw = scores[:, index[f"{args.namespace}__l{layer:02d}__raw"]]
        delta = j - raw
        loo = [
            exact_sign_flip_p(np.delete(delta, position))
            for position in range(len(delta))
        ]
        rows.append(
            {
                "layer": layer,
                "n_subjects": int(delta.size),
                "per_subject_delta": [float(value) for value in delta],
                "n_positive": int((delta > 0).sum()),
                "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std(ddof=1)),
                "effect_to_noise": float(abs(delta.mean()) / delta.std(ddof=1)),
                "exact_p": exact_sign_flip_p(delta),
                "leave_one_out_p_min": float(min(loo)),
                "leave_one_out_p_max": float(max(loo)),
                "leave_one_out_p": [float(value) for value in loo],
            }
        )

    report = {"namespace": args.namespace, "rows": rows}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")

    print(
        f"{'layer':>5} {'signs':>6} {'mean x1e3':>10} {'std x1e3':>9} "
        f"{'|m|/sd':>7} {'p':>8} {'LOO p range':>18}"
    )
    for row in rows:
        print(
            f"{row['layer']:>5} {row['n_positive']:>4}/8 "
            f"{row['mean_delta'] * 1000:>+10.3f} {row['std_delta'] * 1000:>9.3f} "
            f"{row['effect_to_noise']:>7.3f} {row['exact_p']:>8.4f} "
            f"  [{row['leave_one_out_p_min']:.4f}, {row['leave_one_out_p_max']:.4f}]"
        )


if __name__ == "__main__":
    main()

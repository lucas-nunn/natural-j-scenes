#!/usr/bin/env python3
"""Score an exploratory layer set against cortex, outside the predeclared families.

``summarize`` deliberately refuses layer sets other than ``ANALYSIS_LAYERS``: the
BH families are predeclared over exactly four layers, and reporting a different
set through that path would misstate what was predeclared. Exploratory sweeps
therefore need their own entry point, and must be labelled as exploratory
wherever their numbers appear.

This computes, per layer, the paired J-minus-raw difference in whole-searchlight
mean RSA correlation with subjects as the independent unit — the same estimand
``summarize`` uses — and reports it with **uncorrected** exact sign-flip
p-values. No multiplicity correction is applied, because there is no predeclared
family to correct within. These numbers are hypothesis-generating. They cannot
be reported as confirmatory, and a layer that looks significant here earns a
predeclared run, not a claim.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
from pathlib import Path

import numpy as np


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--namespace", default="plain_mean_pool")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = (
        args.results_root
        / "searchlight"
        / "serialised_models_correlation"
        / args.group
        / "group_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = [item["feature"] for item in manifest["model_order"]]
    index = {feature: position for position, feature in enumerate(order)}
    layers = sorted(
        {
            int(match.group(1))
            for name in order
            if (match := re.search(r"__l(\d{2})__", name))
        }
    )

    subjects = [int(item) for item in args.subjects.split(",")]
    scores = np.full((len(subjects), len(order)), np.nan)
    for row, subject in enumerate(subjects):
        subj = f"subj{subject:02d}"
        directory = (
            args.results_root
            / "searchlight"
            / "searchlight_respectedsampling_correlation"
            / subj
            / args.group
            / "corr_vols_correlation"
        )
        files = sorted(glob.glob(str(directory / "*sample-*.npy")))
        if not files:
            raise FileNotFoundError(f"no searchlight samples in {directory}")
        per_sample = []
        for path in files:
            volumes = np.load(path, allow_pickle=False)
            flattened = volumes.reshape(volumes.shape[0], -1)
            centres = np.any(np.isfinite(flattened) & (flattened != 0), axis=0)
            per_sample.append(np.nanmean(flattened[:, centres], axis=1))
        scores[row] = np.mean(per_sample, axis=0)

    rows = []
    for layer in layers:
        j = scores[:, index[f"{args.namespace}__l{layer:02d}__j"]]
        raw = scores[:, index[f"{args.namespace}__l{layer:02d}__raw"]]
        delta = j - raw
        rows.append(
            {
                "layer": layer,
                "raw_mean": float(raw.mean()),
                "j_mean": float(j.mean()),
                "mean_delta": float(delta.mean()),
                "n_positive": int((delta > 0).sum()),
                "uncorrected_exact_p": exact_sign_flip_p(delta),
            }
        )
        print(
            f"l{layer:02d}  raw={rows[-1]['raw_mean']:.6f}  j={rows[-1]['j_mean']:.6f}  "
            f"delta={rows[-1]['mean_delta']:+.6f}  signs={rows[-1]['n_positive']}/"
            f"{len(subjects)}  p_uncorrected={rows[-1]['uncorrected_exact_p']:.4f}"
        )

    print(
        "\nEXPLORATORY: p-values are uncorrected and no family was predeclared. "
        "These are hypothesis-generating only."
    )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "status": "exploratory",
                    "multiplicity_correction": "none — no predeclared family",
                    "group": args.group,
                    "subjects": subjects,
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

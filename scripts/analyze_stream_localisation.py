#!/usr/bin/env python3
"""Where in cortex does the J-space advantage actually live?

The project's motivating idea is that J-space isolates a *verbalizable,
broadcast-ready* subspace — the language-model analogue of a global workspace.
That is a claim about **which cortex** should benefit, not only about how much:
a workspace account predicts the advantage concentrates in higher-order
association cortex, and specifically that it should not be an early visual
effect.

Nothing has tested that. The reported result is a whole-searchlight mean, which
averages the entire cortical sheet and is silent about location. This script
splits the same paired J-minus-raw difference by NSD's `streams` ROI, which
partitions cortex along an early -> mid -> higher-order gradient:

    1 early        2 midventral   3 midlateral   4 midparietal
    5 ventral      6 lateral      7 parietal

Per subject it averages the eight sample maps, takes the difference at valid
searchlight centres inside each stream, and then treats subjects as the
independent unit exactly as the main analysis does.

**EXPLORATORY.** No family was predeclared over stream ROIs, so p-values are
uncorrected and these numbers are hypothesis-generating. A stream that looks
significant here earns a predeclared test, not a claim.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np

STREAM_NAMES = {
    1: "early",
    2: "midventral",
    3: "midlateral",
    4: "midparietal",
    5: "ventral",
    6: "lateral",
    7: "parietal",
}
HIGHER_ORDER = (5, 6, 7)


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
    parser.add_argument("--nsd-dir", type=Path, required=True)
    parser.add_argument("--namespace", default="plain_mean_pool")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import nibabel as nib

    manifest = json.loads(
        (
            args.results_root
            / "searchlight"
            / "serialised_models_correlation"
            / args.group
            / "group_manifest.json"
        ).read_text(encoding="utf-8")
    )
    order = [item["feature"] for item in manifest["model_order"]]
    j_index = order.index(f"{args.namespace}__l{args.layer:02d}__j")
    raw_index = order.index(f"{args.namespace}__l{args.layer:02d}__raw")

    subjects = [int(item) for item in args.subjects.split(",")]
    per_subject: dict[int, dict[int, float]] = {}
    for subject in subjects:
        subj = f"subj{subject:02d}"
        streams = nib.load(
            args.nsd_dir
            / "nsddata"
            / "ppdata"
            / subj
            / "func1pt8mm"
            / "roi"
            / "streams.nii.gz"
        ).get_fdata()
        directory = (
            args.results_root
            / "searchlight"
            / "searchlight_respectedsampling_correlation"
            / subj
            / args.group
            / "corr_vols_correlation"
        )
        files = sorted(glob.glob(str(directory / "*sample-*.npy")))
        stack = np.stack([np.load(path, allow_pickle=False) for path in files])
        if stack.shape[2:] != streams.shape:
            raise ValueError(
                f"{subj}: searchlight {stack.shape[2:]} != streams {streams.shape}"
            )
        j_map = np.nanmean(stack[:, j_index], axis=0)
        raw_map = np.nanmean(stack[:, raw_index], axis=0)
        difference = j_map - raw_map
        valid = np.isfinite(difference) & (raw_map != 0)

        per_subject[subject] = {}
        for label in STREAM_NAMES:
            selection = valid & (streams == label)
            if selection.sum() == 0:
                continue
            per_subject[subject][label] = float(difference[selection].mean())

    rows = []
    for label, name in STREAM_NAMES.items():
        values = np.array(
            [per_subject[s][label] for s in subjects if label in per_subject[s]]
        )
        if values.size < len(subjects):
            continue
        rows.append(
            {
                "stream": label,
                "name": name,
                "mean_delta": float(values.mean()),
                "n_positive": int((values > 0).sum()),
                "uncorrected_exact_p": exact_sign_flip_p(values),
            }
        )
        print(
            f"{label} {name:12}  delta={rows[-1]['mean_delta']:+.6f}  "
            f"signs={rows[-1]['n_positive']}/{len(values)}  "
            f"p={rows[-1]['uncorrected_exact_p']:.4f}"
        )

    early = np.array([per_subject[s][1] for s in subjects])
    higher = np.array(
        [np.mean([per_subject[s][label] for label in HIGHER_ORDER]) for s in subjects]
    )
    contrast = higher - early
    summary = {
        "higher_order_minus_early_mean": float(contrast.mean()),
        "higher_order_minus_early_n_positive": int((contrast > 0).sum()),
        "higher_order_minus_early_p": exact_sign_flip_p(contrast),
    }
    print(
        f"\nhigher-order (ventral/lateral/parietal) minus early: "
        f"{summary['higher_order_minus_early_mean']:+.6f}  "
        f"signs={summary['higher_order_minus_early_n_positive']}/{len(subjects)}  "
        f"p={summary['higher_order_minus_early_p']:.4f}"
    )
    print("\nEXPLORATORY: uncorrected, no predeclared family over stream ROIs.")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "status": "exploratory",
                    "layer": args.layer,
                    "rows": rows,
                    "summary": summary,
                    "per_subject": {
                        str(s): {str(k): v for k, v in d.items()}
                        for s, d in per_subject.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

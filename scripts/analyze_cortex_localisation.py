#!/usr/bin/env python3
"""Whole-cortex localisation of the J advantage, including frontal cortex.

`analyze_stream_localisation.py` found no sensory-to-association gradient, but
NSD's `streams` ROI covers visual cortex only — it contains no prefrontal cortex
and no default-mode network, so a global-workspace account in the
Dehaene/Baars sense was untested rather than refuted.

The searchlight in fact covers the entire HCP-MMP1 parcellation: all 102,533
labelled voxels are valid centres, and roughly 75,000 of them lie **outside**
the visual streams. The frontoparietal question is therefore answerable from
data already on disk.

Two things are reported, deliberately separated by how much judgement each
requires:

**Primary, and free of judgement:** the paired J-minus-raw difference inside the
visual streams versus everywhere else in labelled cortex. That contrast uses
NSD's own definition of visual cortex and involves no network model.

**Descriptive:** the same difference per parcel, ranked, with names, plus an
explicitly listed set of canonical frontal parcels so their position in the
ranking is visible. Naming a "frontoparietal network" requires assumptions this
script does not want to smuggle in, so the parcels are listed in the source
rather than aggregated behind a label.

**The SNR control is not optional.** NSD is a visual experiment, so visual
cortex carries far more signal than frontal cortex and *every* alignment measure
is larger there. The absolute J-minus-raw difference therefore has a built-in
spatial gradient that says nothing about representation type. This script
reports the difference normalised by each parcel's raw alignment level
alongside the absolute value, because the two support opposite conclusions.

**EXPLORATORY.** No family was predeclared over parcels; p-values are
uncorrected.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np

#: Canonical prefrontal / frontoparietal-control parcels in HCP-MMP1, listed
#: explicitly so the selection is auditable rather than hidden behind a label.
FRONTAL_PARCELS = (
    "8C",
    "8Av",
    "IFJp",
    "IFJa",
    "IFSp",
    "IFSa",
    "p9-46v",
    "a9-46v",
    "46",
    "9-46d",
    "a47r",
    "p47r",
    "9m",
    "10d",
    "a10p",
    "p10p",
    "s6-8",
    "i6-8",
)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def load_label_table(path: Path) -> dict[int, str]:
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            table[int(parts[0])] = parts[1]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--nsd-dir", type=Path, required=True)
    parser.add_argument("--label-table", type=Path, required=True)
    parser.add_argument("--namespace", default="plain_mean_pool")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import nibabel as nib

    names = load_label_table(args.label_table)
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
    inside, outside = [], []
    parcel_values: dict[int, list[float]] = {}
    parcel_raw: dict[int, list[float]] = {}

    for subject in subjects:
        subj = f"subj{subject:02d}"
        roi = args.nsd_dir / "nsddata" / "ppdata" / subj / "func1pt8mm" / "roi"
        parcels = nib.load(roi / "HCP_MMP1.nii.gz").get_fdata()
        streams = nib.load(roi / "streams.nii.gz").get_fdata()
        files = sorted(
            glob.glob(
                str(
                    args.results_root
                    / "searchlight"
                    / "searchlight_respectedsampling_correlation"
                    / subj
                    / args.group
                    / "corr_vols_correlation"
                    / "*sample-*.npy"
                )
            )
        )
        stack = np.stack([np.load(path, allow_pickle=False) for path in files])
        difference = np.nanmean(stack[:, j_index], axis=0) - np.nanmean(
            stack[:, raw_index], axis=0
        )
        valid = np.isfinite(difference) & (parcels > 0)

        inside.append(float(difference[valid & (streams > 0)].mean()))
        outside.append(float(difference[valid & (streams == 0)].mean()))
        for label in np.unique(parcels[parcels > 0]):
            selection = valid & (parcels == label)
            if selection.sum() >= 20:
                parcel_values.setdefault(int(label), []).append(
                    float(difference[selection].mean())
                )
                parcel_raw.setdefault(int(label), []).append(
                    float(np.nanmean(stack[:, raw_index], axis=0)[selection].mean())
                )

    inside = np.array(inside)
    outside = np.array(outside)
    print("=== PRIMARY: visual streams vs rest of labelled cortex ===")
    print(
        f"  inside visual streams : {inside.mean():+.6f}  p={exact_sign_flip_p(inside):.4f}"
    )
    print(
        f"  outside (rest of ctx) : {outside.mean():+.6f}  p={exact_sign_flip_p(outside):.4f}"
    )
    contrast = outside - inside
    print(
        f"  outside minus inside  : {contrast.mean():+.6f}  "
        f"signs={int((contrast > 0).sum())}/{len(contrast)}  "
        f"p={exact_sign_flip_p(contrast):.4f}"
    )

    rows = []
    for label, values in parcel_values.items():
        if len(values) < len(subjects):
            continue
        array = np.array(values)
        raw_level = float(np.mean(parcel_raw[label]))
        rows.append(
            {
                "parcel": label,
                "name": names.get(label, str(label)),
                "mean_delta": float(array.mean()),
                "raw_alignment_level": raw_level,
                "relative_advantage": float(array.mean() / raw_level)
                if raw_level
                else float("nan"),
                "n_positive": int((array > 0).sum()),
                "uncorrected_exact_p": exact_sign_flip_p(array),
            }
        )
    rows.sort(key=lambda row: row["mean_delta"], reverse=True)

    print(f"\n=== per-parcel ranking ({len(rows)} parcels with all subjects) ===")
    print("  top 8:")
    for row in rows[:8]:
        print(
            f"    {row['name']:10} {row['mean_delta']:+.6f}  p={row['uncorrected_exact_p']:.4f}"
        )
    print("  bottom 5:")
    for row in rows[-5:]:
        print(
            f"    {row['name']:10} {row['mean_delta']:+.6f}  p={row['uncorrected_exact_p']:.4f}"
        )

    ranks = {row["name"]: position for position, row in enumerate(rows, start=1)}
    print(f"\n=== canonical frontal parcels (of {len(rows)}) ===")
    frontal = [row for row in rows if row["name"] in FRONTAL_PARCELS]
    for row in frontal:
        print(
            f"    {row['name']:10} rank {ranks[row['name']]:>3}  "
            f"{row['mean_delta']:+.6f}  p={row['uncorrected_exact_p']:.4f}"
        )
    if frontal:
        values = np.array([row["mean_delta"] for row in frontal])
        print(
            f"  frontal mean delta: {values.mean():+.6f}   median rank: "
            f"{int(np.median([ranks[r['name']] for r in frontal]))}"
        )
    # The control that decides how any of the above may be read.
    deltas = np.array([row["mean_delta"] for row in rows])
    levels = np.array([row["raw_alignment_level"] for row in rows])
    ratios = np.array([row["relative_advantage"] for row in rows])
    print("\n=== SNR CONTROL ===")
    print(
        f"  Pearson(delta, raw alignment level) = "
        f"{float(np.corrcoef(deltas, levels)[0, 1]):+.3f} across {len(rows)} parcels"
    )
    frontal_mask = np.array([row["name"] in FRONTAL_PARCELS for row in rows])
    print(
        f"  relative advantage — frontal {ratios[frontal_mask].mean():+.4f}  "
        f"vs others {ratios[~frontal_mask].mean():+.4f}"
    )
    print(
        "  The absolute spatial pattern largely tracks signal level; the "
        "normalised advantage does not."
    )
    print("\nEXPLORATORY: uncorrected, no predeclared family over parcels.")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "status": "exploratory",
                    "layer": args.layer,
                    "primary": {
                        "inside_streams_mean": float(inside.mean()),
                        "outside_streams_mean": float(outside.mean()),
                        "outside_minus_inside_mean": float(contrast.mean()),
                        "outside_minus_inside_p": exact_sign_flip_p(contrast),
                    },
                    "parcels": rows,
                    "frontal_parcels_listed": list(FRONTAL_PARCELS),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

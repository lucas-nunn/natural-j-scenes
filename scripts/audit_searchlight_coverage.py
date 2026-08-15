#!/usr/bin/env python3
"""Audit undefined searchlight centres and confirm the contrast stays fair.

Some searchlight spheres yield an undefined Pearson correlation — typically
where the sampled beta patterns carry no variance — and those centres come back
as NaN. The summary stage excludes them with ``np.nanmean``, so they never
propagate into a score, and the group table gives no sign they exist. Their
prevalence turns out to vary a lot by subject, from none to over a tenth of all
centres.

That is tolerable for the paired J-vs-raw contrast **only while the NaN pattern
is identical across models**. If one model ever lost centres the other kept, the
two sides would be averaged over different voxels and the paired difference
would be biased, with nothing raising an error. This audit checks that
invariant directly, per sample rather than only in aggregate, and reports
per-subject coverage so heterogeneity is visible rather than implicit.

Reads committed searchlight volumes only.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--searchlight-root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    violations = []
    for subject in (int(item) for item in args.subjects.split(",")):
        subj = f"subj{subject:02d}"
        directory = (
            args.searchlight_root
            / "searchlight_respectedsampling_correlation"
            / subj
            / args.group
            / "corr_vols_correlation"
        )
        files = sorted(glob.glob(str(directory / "*sample-*.npy")))
        if not files:
            raise FileNotFoundError(f"no searchlight samples in {directory}")
        stack = np.stack([np.load(path, allow_pickle=False) for path in files])

        reference = np.isnan(stack[:, 0])
        model_invariant = all(
            np.array_equal(reference, np.isnan(stack[:, model]))
            for model in range(stack.shape[1])
        )
        if not model_invariant:
            violations.append(subj)

        first = stack[0, 0]
        centres = int(((first != 0) | np.isnan(first)).sum())
        undefined = int(np.isnan(stack[:, 0]).all(axis=0).sum())
        rows.append(
            {
                "subject": subj,
                "n_samples": len(files),
                "n_models": int(stack.shape[1]),
                "n_centres": centres,
                "n_undefined_centres": undefined,
                "fraction_undefined": undefined / centres if centres else 0.0,
                "nan_mask_model_invariant": bool(model_invariant),
            }
        )
        print(
            f"{subj}  centres={centres:>7}  undefined={undefined:>6} "
            f"({100 * undefined / max(centres, 1):>6.3f}%)  "
            f"model-invariant={'yes' if model_invariant else 'NO'}"
        )

    if args.output:
        args.output.write_text(
            json.dumps({"group": args.group, "rows": rows}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    if violations:
        raise SystemExit(
            "NaN pattern differs between models for: "
            + ", ".join(violations)
            + " — the paired J-vs-raw contrast would be averaged over different "
            "voxels and is NOT valid for those subjects"
        )
    print(
        "\nNaN pattern is identical across all models for every subject: "
        "the paired contrast is averaged over the same centres."
    )


if __name__ == "__main__":
    main()

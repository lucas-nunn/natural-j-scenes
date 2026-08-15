#!/usr/bin/env python3
"""Verify the copied MPNet reference RDM against a fresh recomputation.

The MPNet reference is the only quantity shared between this project's runs and
the historical comparator, so it is the anchor used to argue that two runs sit
on the same footing. But it is **copied** from an external MPNet result tree
rather than recomputed, which means its condition ordering is inherited and
otherwise unverified. A permuted reference would silently weaken every
cross-run comparison that leans on it.

This script recomputes the correlation-distance RDM from the raw MPNet
sentence embeddings, indexing them by the subject's sorted condition IDs, and
compares it against the stored file. Agreement establishes that the reference
shares the sorted-ID ordering contract documented in
``nsd_adapter.condition_column_index``.

NSD condition IDs are 1-based; the embedding table is 0-based, so the lookup
subtracts one. That subtraction is the single index convention in the project,
and getting it wrong is loud rather than subtle: the off-by-one drops the
correlation to roughly zero instead of degrading it slightly.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

TOLERANCE = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--mpnet-base", type=Path, required=True)
    parser.add_argument("--conditions-dir", type=Path, required=True)
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--model-name", default="all-mpnet-base-v2")
    args = parser.parse_args()

    with args.embeddings.open("rb") as handle:
        embeddings = np.asarray(pickle.load(handle))
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2-D embedding table, got {embeddings.shape}")

    failures = 0
    for subject in (int(item) for item in args.subjects.split(",")):
        subj = f"subj{subject:02d}"
        ids = np.load(
            args.conditions_dir / f"{subj}_condition_ids.npy", allow_pickle=False
        )
        if not np.array_equal(ids, np.unique(ids)):
            raise ValueError(f"{subj} condition IDs are not sorted and unique")
        stored_path = (
            args.mpnet_base
            / "serialised_models_correlation"
            / args.model_name
            / f"{subj}_{args.model_name}_fullrdm.npy"
        )
        stored = np.load(stored_path, allow_pickle=False).astype(np.float64)

        recomputed = pdist(embeddings[ids - 1], metric="correlation")
        if recomputed.shape != stored.shape:
            raise ValueError(f"{subj}: length {recomputed.shape} != {stored.shape}")

        max_abs = float(np.max(np.abs(recomputed - stored)))
        correlation = float(np.corrcoef(recomputed, stored)[0, 1])
        ok = max_abs <= TOLERANCE
        failures += 0 if ok else 1
        print(
            f"{subj}  n={ids.size}  r={correlation:.10f}  "
            f"max|diff|={max_abs:.3e}  {'OK' if ok else 'MISMATCH'}"
        )

    if failures:
        raise SystemExit(f"{failures} subject(s) did not match the stored reference")
    print("\nAll subjects match: the MPNet reference uses sorted-condition-ID order.")


if __name__ == "__main__":
    main()

"""Materialize the exact sparse NSD scope and matched sample choices."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import (
    N_SAMPLES,
    N_SESSIONS,
    N_SUBJECTS,
    SAMPLE_SIZE,
    ExperimentPaths,
    validate_subjects,
)
from .io_utils import atomic_copy, atomic_json, atomic_npy, sha256_file, stable_hash


def _source_sampling_path(paths: ExperimentPaths, subject: int) -> Path:
    paths.require("mpnet_base")
    assert paths.mpnet_base is not None
    subj = f"subj{subject:02d}"
    return (
        paths.mpnet_base
        / "searchlight_respectedsampling_correlation"
        / subj
        / "saved_sampling"
        / f"{subj}_nsd-allsubstim_sampling.npy"
    )


def validate_sampling(choices: np.ndarray, n_conditions: int, subject: str) -> None:
    expected_shape = (N_SAMPLES, SAMPLE_SIZE)
    if choices.shape != expected_shape:
        raise ValueError(
            f"{subject} sampling shape {choices.shape}, expected {expected_shape}"
        )
    if np.any((choices < 0) | (choices >= n_conditions)):
        raise ValueError(f"{subject} sampling indices are out of range")
    if any(len(np.unique(row)) != SAMPLE_SIZE for row in choices):
        raise ValueError(f"{subject} has duplicates inside a sample")
    if len(np.unique(choices)) != N_SAMPLES * SAMPLE_SIZE:
        raise ValueError(f"{subject} samples are not mutually disjoint")


def prepare_conditions(
    paths: ExperimentPaths,
    subjects: Sequence[int] = tuple(range(1, N_SUBJECTS + 1)),
) -> dict:
    """Write auditable IDs and copies of the existing matched samples."""
    paths.require("nsd_dir", "mpnet_base")
    assert paths.nsd_dir is not None
    from nsd_visuo_semantics.utils.nsd_get_data_light import (
        get_subject_conditions,
    )

    subjects = validate_subjects(subjects)
    subject_records = []
    subject_ids: list[np.ndarray] = []
    sampling_dir = paths.conditions / "saved_sampling"
    paths.conditions.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        subj = f"subj{subject:02d}"
        _, _, conditions = get_subject_conditions(
            str(paths.nsd_dir), subj, N_SESSIONS, keep_only_3repeats=True
        )
        conditions = np.asarray(conditions, dtype=np.int64)
        if conditions.ndim != 1 or not np.array_equal(
            conditions, np.unique(conditions)
        ):
            raise ValueError(f"{subj} conditions are not sorted and unique")
        if np.any((conditions < 1) | (conditions > 73_000)):
            raise ValueError(f"{subj} contains invalid 1-based NSD IDs")

        source_sampling = _source_sampling_path(paths, subject)
        choices = np.load(source_sampling, allow_pickle=False)
        validate_sampling(choices, len(conditions), subj)

        condition_file = paths.conditions / f"{subj}_condition_ids.npy"
        sampling_file = sampling_dir / source_sampling.name
        sampled_id_file = paths.conditions / f"{subj}_sampling_condition_ids.npy"
        atomic_npy(condition_file, conditions)
        atomic_copy(source_sampling, sampling_file)
        atomic_npy(sampled_id_file, conditions[choices])

        subject_ids.append(conditions)
        subject_records.append(
            {
                "subject": subj,
                "n_conditions": len(conditions),
                "condition_ids_file": str(condition_file.resolve()),
                "condition_ids_sha256": sha256_file(condition_file),
                "sampling_source": str(source_sampling.resolve()),
                "sampling_source_sha256": sha256_file(source_sampling),
                "sampling_copy": str(sampling_file.resolve()),
                "sampling_copy_sha256": sha256_file(sampling_file),
                "sampling_shape": list(choices.shape),
                "sampling_is_disjoint": True,
            }
        )

    union_ids = np.unique(np.concatenate(subject_ids)).astype(np.int64)
    union_file = paths.conditions / "union_condition_ids.npy"
    atomic_npy(union_file, union_ids)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "nsd_id_base": 1,
        "caption_row_expression": "condition_id - 1",
        "n_subjects": len(subjects),
        "subject_numbers": list(subjects),
        "n_sessions": N_SESSIONS,
        "keep_only_3repeats": True,
        "n_union_conditions": len(union_ids),
        "union_min": int(union_ids.min()),
        "union_max": int(union_ids.max()),
        "union_file": str(union_file.resolve()),
        "union_sha256": sha256_file(union_file),
        "union_fingerprint": stable_hash(union_ids.tolist()),
        "subjects": subject_records,
    }
    atomic_json(paths.conditions / "manifest.json", manifest)
    return manifest


def load_union_ids(paths: ExperimentPaths) -> np.ndarray:
    ids = np.load(paths.conditions / "union_condition_ids.npy")
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or not np.array_equal(ids, np.unique(ids)):
        raise ValueError("union condition IDs are not sorted and unique")
    return ids

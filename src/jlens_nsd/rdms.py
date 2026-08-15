"""Build sparse subject RDMs and one grouped searchlight model batch."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from .conditions import load_union_ids
from .config import (
    DEFAULT_PROMPT_SET,
    N_SUBJECTS,
    ExperimentPaths,
    group_name,
    run_name,
    validate_subjects,
)
from .io_utils import atomic_copy, atomic_json, atomic_npy, sha256_file


def _embedding_manifest(
    paths: ExperimentPaths,
    profile: str,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
) -> tuple[Path, dict]:
    directory = paths.embeddings / run_name(profile, prompt_set_key)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"embedding manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"embedding extraction is incomplete: {manifest_path}")
    return directory, manifest


def _iter_chunks(directory: Path, manifest: dict) -> Iterator[tuple[Path, np.ndarray]]:
    for name in manifest["completed_chunks"]:
        path = directory / "chunks" / name
        if not path.exists():
            raise FileNotFoundError(f"manifest chunk missing: {path}")
        with np.load(path, allow_pickle=False) as chunk:
            yield path, np.asarray(chunk["condition_ids"], dtype=np.int64)


def _load_sparse_feature(
    directory: Path,
    manifest: dict,
    feature_name: str,
    union_ids: np.ndarray,
) -> np.ndarray:
    d_model = int(manifest["config"]["d_model"])
    matrix = np.empty((len(union_ids), d_model), dtype=np.float32)
    seen = np.zeros(len(union_ids), dtype=bool)
    for chunk_name in manifest["completed_chunks"]:
        path = directory / "chunks" / chunk_name
        with np.load(path, allow_pickle=False) as chunk:
            ids = np.asarray(chunk["condition_ids"], dtype=np.int64)
            positions = np.searchsorted(union_ids, ids)
            if np.any(positions >= len(union_ids)) or not np.array_equal(
                union_ids[positions], ids
            ):
                raise ValueError(f"chunk has IDs outside the union: {path}")
            if np.any(seen[positions]):
                raise ValueError(f"duplicate condition IDs across chunks: {path}")
            values = np.asarray(chunk[feature_name], dtype=np.float32)
            if values.shape != (len(ids), d_model):
                raise ValueError(f"{feature_name} in {path} has shape {values.shape}")
            matrix[positions] = values
            seen[positions] = True
    if not seen.all():
        raise ValueError(
            f"feature {feature_name} is missing {int((~seen).sum())} union rows"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"feature {feature_name} contains non-finite values")
    return matrix


def _validate_correlation_rows(matrix: np.ndarray, feature_name: str) -> None:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    bad = np.flatnonzero(~np.isfinite(norms) | (norms <= 1e-12))
    if len(bad):
        raise ValueError(
            f"correlation distance undefined for {len(bad)} constant/non-finite "
            f"rows in {feature_name}; first row index={int(bad[0])}"
        )


def _feature_model_name(group: str, feature_name: str) -> str:
    return f"{group}__{feature_name}"


def _subject_condition_ids(paths: ExperimentPaths, subject: int) -> np.ndarray:
    path = paths.conditions / f"subj{subject:02d}_condition_ids.npy"
    ids = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if ids.ndim != 1 or not np.array_equal(ids, np.unique(ids)):
        raise ValueError(f"invalid subject condition IDs: {path}")
    return ids


def _copy_sampling(paths: ExperimentPaths, subject: int) -> dict:
    subj = f"subj{subject:02d}"
    name = f"{subj}_nsd-allsubstim_sampling.npy"
    source = paths.conditions / "saved_sampling" / name
    destination = (
        paths.searchlight_base
        / "searchlight_respectedsampling_correlation"
        / subj
        / "saved_sampling"
        / name
    )
    if destination.exists() and sha256_file(destination) != sha256_file(source):
        raise RuntimeError(
            f"existing searchlight sampling differs from locked source: {destination}"
        )
    atomic_copy(source, destination)
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "sha256": sha256_file(destination),
    }


def prepare_grouped_rdms(
    paths: ExperimentPaths,
    profile: str,
    subjects: Sequence[int] = tuple(range(1, N_SUBJECTS + 1)),
    *,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
) -> dict:
    """Create selected-subject RDMs without ever allocating a 73K matrix."""
    paths.require("mpnet_base")
    assert paths.mpnet_base is not None
    subjects = validate_subjects(subjects)
    directory, embedding_manifest = _embedding_manifest(paths, profile, prompt_set_key)
    if embedding_manifest["config"].get("prompt_set", "historical") != prompt_set_key:
        raise RuntimeError("embedding manifest prompt set does not match RDM request")
    group = group_name(profile, prompt_set_key)
    union_ids = load_union_ids(paths)
    expected_hash = embedding_manifest["config"]["condition_ids_hash"]
    from .io_utils import stable_hash

    if expected_hash != stable_hash(union_ids.tolist()):
        raise RuntimeError(
            "embedding condition IDs do not match the full experiment union; "
            "smoke/partial extractions cannot feed the real RDM stage"
        )

    features = [item["name"] for item in embedding_manifest["features"]]
    output_dir = paths.searchlight_base / "serialised_models_correlation" / group
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_records = {f"subj{s:02d}": [] for s in subjects}

    subject_ids = {
        subject: _subject_condition_ids(paths, subject) for subject in subjects
    }
    subject_positions = {}
    for subject, ids in subject_ids.items():
        positions = np.searchsorted(union_ids, ids)
        if np.any(positions >= len(union_ids)) or not np.array_equal(
            union_ids[positions], ids
        ):
            raise ValueError(f"subj{subject:02d} IDs are missing from union")
        subject_positions[subject] = positions

    for feature_name in features:
        matrix = _load_sparse_feature(
            directory, embedding_manifest, feature_name, union_ids
        )
        _validate_correlation_rows(matrix, feature_name)
        model_name = _feature_model_name(group, feature_name)
        for subject in subjects:
            subj = f"subj{subject:02d}"
            subject_matrix = matrix[subject_positions[subject]]
            rdm = pdist(subject_matrix, metric="correlation").astype(np.float32)
            expected_length = len(subject_matrix) * (len(subject_matrix) - 1) // 2
            if rdm.shape != (expected_length,) or not np.isfinite(rdm).all():
                raise ValueError(f"invalid RDM for {subj}/{feature_name}")
            output = output_dir / f"{subj}_{model_name}_fullrdm.npy"
            atomic_npy(output, rdm)
            subject_records[subj].append(
                {
                    "feature": feature_name,
                    "model_name": model_name,
                    "path": str(output.resolve()),
                    "length": len(rdm),
                    "sha256": sha256_file(output),
                }
            )
        del matrix

    for subject in subjects:
        subj = f"subj{subject:02d}"
        source = (
            paths.mpnet_base
            / "serialised_models_correlation/all-mpnet-base-v2"
            / f"{subj}_all-mpnet-base-v2_fullrdm.npy"
        )
        model_name = f"{group}__mpnet_reference"
        output = output_dir / f"{subj}_{model_name}_fullrdm.npy"
        atomic_copy(source, output)
        rdm = np.load(output, allow_pickle=False, mmap_mode="r")
        expected = len(subject_ids[subject]) * (len(subject_ids[subject]) - 1) // 2
        if rdm.shape != (expected,) or not np.isfinite(rdm).all():
            raise ValueError(f"invalid MPNet reference RDM for {subj}")
        subject_records[subj].append(
            {
                "feature": "mpnet_reference",
                "model_name": model_name,
                "path": str(output.resolve()),
                "length": len(rdm),
                "sha256": sha256_file(output),
                "source": str(source.resolve()),
                "source_sha256": sha256_file(source),
            }
        )

    sampling = {
        f"subj{subject:02d}": _copy_sampling(paths, subject) for subject in subjects
    }
    # get_model_rdms sorts filenames. Record that exact order and a stable
    # 1-based index because projection files use model-1, model-2, ... .
    model_order = []
    first_subject = f"subj{subjects[0]:02d}"
    ordered = sorted(
        subject_records[first_subject], key=lambda record: Path(record["path"]).name
    )
    for index, record in enumerate(ordered, start=1):
        model_order.append(
            {
                "model_index": index,
                "feature": record["feature"],
                "model_name": record["model_name"],
            }
        )
    expected_names = [item["model_name"] for item in model_order]
    for subj, records in subject_records.items():
        names = [
            item["model_name"]
            for item in sorted(records, key=lambda item: Path(item["path"]).name)
        ]
        if names != expected_names:
            raise RuntimeError(f"grouped model order differs for {subj}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "group_name": group,
        "profile": profile,
        "prompt_set": prompt_set_key,
        "subject_numbers": list(subjects),
        "rdm_metric": "correlation",
        "normalization_before_rdm": "none",
        "embedding_manifest": str((directory / "manifest.json").resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_order": model_order,
        "subjects": subject_records,
        "matched_sampling": sampling,
    }
    atomic_json(output_dir / "group_manifest.json", manifest)
    return manifest

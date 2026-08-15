#!/usr/bin/env python3
"""Validate an eight-subject report and build the layer summary artifacts.

Example, from the repository root::

    python scripts/make_layer_performance_summary.py \
      --report-dir /path/to/results/reports/qwen4b \
      --result-root /path/to/results

The report directory must contain feature_scores.csv, comparisons.csv,
subject_scores.npy, sample_scores.npy, and summary.json from the completed
eight-subject run. The result root must contain the grouped model manifest and
all native-space searchlight correlation volumes. Searchlights are not
recomputed, and rendered or projected maps are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

LAYERS = (8, 16, 23, 30)
PROMPTS = ("visualize", "plain")
N_SUBJECTS = 8
N_SAMPLES = 8
N_SESSIONS = 10
ALPHA = 0.05
GROUP_NAME = "jlens_qwen4b_group"
REQUIRED_FILES = (
    "feature_scores.csv",
    "comparisons.csv",
    "subject_scores.npy",
    "sample_scores.npy",
    "summary.json",
)

# Summary columns follow the grouped-RDM manifest's lexical filename order.
# Keeping the complete expected order here makes NPY column interpretation
# explicit and ensures that a partial/single-subject report cannot slip through.
EXPECTED_MODEL_ORDER = (
    "mpnet_reference",
    "plain__final",
    "plain__l08__j",
    "plain__l08__raw",
    "plain__l16__j",
    "plain__l16__raw",
    "plain__l23__j",
    "plain__l23__raw",
    "plain__l30__j",
    "plain__l30__raw",
    "visualize__final",
    "visualize__l08__j",
    "visualize__l08__raw",
    "visualize__l16__j",
    "visualize__l16__raw",
    "visualize__l23__j",
    "visualize__l23__raw",
    "visualize__l30__j",
    "visualize__l30__raw",
)

TABLE_FIELDS = (
    "prompt",
    "layer",
    "row_type",
    "raw_feature",
    "j_feature",
    "control_feature",
    "raw_group_mean",
    "raw_ci_95",
    "raw_peak_searchlight_centre_mean",
    "raw_peak_searchlight_centre_ci_95",
    "raw_peak_searchlight_centre_range",
    "j_group_mean",
    "j_ci_95",
    "j_peak_searchlight_centre_mean",
    "j_peak_searchlight_centre_ci_95",
    "j_peak_searchlight_centre_range",
    "j_minus_raw_delta",
    "delta_ci_95",
    "exact_p_two_sided",
    "bh_q_24_comparisons",
    "bh_significant_q_lt_0_05",
    "final_control_group_mean",
    "final_control_ci_95",
    "final_control_peak_searchlight_centre_mean",
    "final_control_peak_searchlight_centre_ci_95",
    "final_control_peak_searchlight_centre_range",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        required=True,
        type=Path,
        help="completed qwen4b report directory containing the five source files",
    )
    parser.add_argument(
        "--result-root",
        required=True,
        type=Path,
        help=(
            "authoritative full result root containing searchlight/serialised_models_"
            "correlation and searchlight/searchlight_respectedsampling_correlation"
        ),
    )
    parser.add_argument(
        "--searchlight-centres-root",
        type=Path,
        help=(
            "optional directory containing subjXX authoritative searchlight-centre "
            "arrays; defaults to the upstream mpnet_10_sessions/precomputed sibling"
        ),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=root / "docs" / "layer_performance_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=root / "docs" / "assets" / "layer_performance_summary.png",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=root / "docs" / "layer_performance_summary.metadata.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    from scipy.stats import t

    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    half = float(t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - half, mean + half


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-15):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _float_row(row: dict[str, str], fields: Sequence[str]) -> dict[str, Any]:
    converted: dict[str, Any] = dict(row)
    for field in fields:
        converted[field] = float(row[field])
    return converted


def _validate_json_csv(
    summary_rows: list[dict[str, Any]],
    csv_rows: list[dict[str, Any]],
    label: str,
) -> None:
    if len(summary_rows) != len(csv_rows):
        raise ValueError(f"{label} JSON/CSV row count mismatch")
    for index, (json_row, csv_row) in enumerate(
        zip(summary_rows, csv_rows, strict=True)
    ):
        if set(json_row) != set(csv_row):
            raise ValueError(f"{label} JSON/CSV fields differ at row {index}")
        for key, json_value in json_row.items():
            csv_value = csv_row[key]
            if isinstance(json_value, float):
                if json_value != csv_value:
                    raise ValueError(
                        f"{label} JSON/CSV value differs at row {index}, field {key}"
                    )
            elif json_value != csv_value:
                raise ValueError(
                    f"{label} JSON/CSV value differs at row {index}, field {key}"
                )


def validate_report(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (report_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"report is missing required files: {missing}")

    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "schema_version": 1,
        "profile": "qwen4b",
        "group_name": "jlens_qwen4b_group",
        "independent_unit": "subject",
        "n_subjects": N_SUBJECTS,
        "n_sessions": N_SESSIONS,
        "n_samples_per_subject": N_SAMPLES,
    }
    metadata_mismatches = {
        key: (summary.get(key), expected)
        for key, expected in expected_metadata.items()
        if summary.get(key) != expected
    }
    if metadata_mismatches:
        raise ValueError(f"unexpected report metadata: {metadata_mismatches}")

    score_strings = read_csv(report_dir / "feature_scores.csv")
    comparison_strings = read_csv(report_dir / "comparisons.csv")
    score_rows = [
        _float_row(row, ("mean_correlation", "ci_low", "ci_high"))
        for row in score_strings
    ]
    comparison_rows = [
        _float_row(
            row,
            ("mean_delta", "ci_low", "ci_high", "exact_p", "fdr_q"),
        )
        for row in comparison_strings
    ]
    _validate_json_csv(summary["scores"], score_rows, "scores")
    _validate_json_csv(summary["comparisons"], comparison_rows, "comparisons")

    score_by_feature = {row["feature"]: row for row in score_rows}
    if len(score_by_feature) != len(score_rows):
        raise ValueError("duplicate feature in feature_scores.csv")
    if set(score_by_feature) != set(EXPECTED_MODEL_ORDER):
        missing_features = sorted(set(EXPECTED_MODEL_ORDER) - set(score_by_feature))
        extra_features = sorted(set(score_by_feature) - set(EXPECTED_MODEL_ORDER))
        raise ValueError(
            f"unexpected feature IDs; missing={missing_features}, extra={extra_features}"
        )

    expected_comparisons = {
        (f"{prompt}__l{layer:02d}__j", baseline)
        for prompt in PROMPTS
        for layer in LAYERS
        for baseline in (
            f"{prompt}__l{layer:02d}__raw",
            f"{prompt}__final",
            "mpnet_reference",
        )
    }
    comparison_by_key = {
        (row["j_feature"], row["baseline"]): row for row in comparison_rows
    }
    if len(comparison_by_key) != len(comparison_rows):
        raise ValueError("duplicate comparison in comparisons.csv")
    if set(comparison_by_key) != expected_comparisons:
        raise ValueError("comparison family is not the expected 24-test family")

    subject_scores = np.load(report_dir / "subject_scores.npy", allow_pickle=False)
    sample_scores = np.load(report_dir / "sample_scores.npy", allow_pickle=False)
    if subject_scores.shape != (N_SUBJECTS, len(EXPECTED_MODEL_ORDER)):
        raise ValueError(f"unexpected subject_scores shape: {subject_scores.shape}")
    if sample_scores.shape != (
        N_SUBJECTS,
        N_SAMPLES,
        len(EXPECTED_MODEL_ORDER),
    ):
        raise ValueError(f"unexpected sample_scores shape: {sample_scores.shape}")
    if subject_scores.dtype != np.float64 or sample_scores.dtype != np.float64:
        raise ValueError("score arrays must both have dtype float64")
    if not np.isfinite(subject_scores).all() or not np.isfinite(sample_scores).all():
        raise ValueError("score arrays contain non-finite values")
    recomputed_subject_scores = sample_scores.mean(axis=1)
    if not np.array_equal(recomputed_subject_scores, subject_scores):
        difference = float(np.max(np.abs(recomputed_subject_scores - subject_scores)))
        raise ValueError(
            f"subject_scores are not the exact sample mean (max difference={difference})"
        )

    column_by_feature = {
        feature: index for index, feature in enumerate(EXPECTED_MODEL_ORDER)
    }
    for feature, row in score_by_feature.items():
        mean, low, high = mean_ci(subject_scores[:, column_by_feature[feature]])
        assert_close(mean, row["mean_correlation"], f"{feature} mean")
        assert_close(low, row["ci_low"], f"{feature} CI low")
        assert_close(high, row["ci_high"], f"{feature} CI high")

    for key, row in comparison_by_key.items():
        j_feature, baseline = key
        differences = (
            subject_scores[:, column_by_feature[j_feature]]
            - subject_scores[:, column_by_feature[baseline]]
        )
        mean, low, high = mean_ci(differences)
        assert_close(mean, row["mean_delta"], f"{key} delta")
        assert_close(low, row["ci_low"], f"{key} CI low")
        assert_close(high, row["ci_high"], f"{key} CI high")
        assert_close(exact_sign_flip_p(differences), row["exact_p"], f"{key} p")

    recomputed_q = bh_adjust([row["exact_p"] for row in comparison_rows])
    for row, q_value in zip(comparison_rows, recomputed_q, strict=True):
        assert_close(q_value, row["fdr_q"], f"{row['j_feature']} q")

    artifact_names = {
        key: Path(value).name for key, value in summary["artifacts"].items()
    }
    for key, expected_name in {
        "subject_scores": "subject_scores.npy",
        "sample_scores": "sample_scores.npy",
        "feature_scores_csv": "feature_scores.csv",
        "comparisons_csv": "comparisons.csv",
    }.items():
        if artifact_names.get(key) != expected_name:
            raise ValueError(f"summary artifact {key} does not name {expected_name}")

    return {
        "summary": summary,
        "scores": score_by_feature,
        "comparisons": comparison_by_key,
        "subject_scores": subject_scores,
        "sample_scores": sample_scores,
        "source_hashes": {
            name: sha256_file(report_dir / name) for name in REQUIRED_FILES
        },
    }


def validate_model_order(manifest: dict[str, Any]) -> None:
    """Validate the manifest mapping used by grouped volume axis zero."""
    expected_metadata = {
        "schema_version": 1,
        "profile": "qwen4b",
        "group_name": GROUP_NAME,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_metadata.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"unexpected group manifest metadata: {mismatches}")

    model_order = manifest.get("model_order")
    if not isinstance(model_order, list):
        raise ValueError("group manifest model_order must be a list")
    features = tuple(item.get("feature") for item in model_order)
    indices = tuple(item.get("model_index") for item in model_order)
    model_names = tuple(item.get("model_name") for item in model_order)
    expected_names = tuple(
        f"{GROUP_NAME}__{feature}" for feature in EXPECTED_MODEL_ORDER
    )
    if features != EXPECTED_MODEL_ORDER:
        raise ValueError(
            f"group manifest model order mismatch: {features!r} != "
            f"{EXPECTED_MODEL_ORDER!r}"
        )
    if indices != tuple(range(1, len(EXPECTED_MODEL_ORDER) + 1)):
        raise ValueError("group manifest model indices are not consecutive and 1-based")
    if model_names != expected_names:
        raise ValueError("group manifest model names do not match feature order")


def _load_searchlight_centres(path: Path) -> np.ndarray:
    """Load and strictly validate one authoritative native-space centre array."""
    if not path.is_file():
        raise FileNotFoundError(f"searchlight-centre array not found: {path}")
    with path.open("rb") as handle:
        raw_centres = np.asarray(pickle.load(handle))
    if raw_centres.ndim != 1 or len(raw_centres) == 0:
        raise ValueError(f"searchlight centres must be a nonempty 1D array: {path}")
    if not np.issubdtype(raw_centres.dtype, np.integer):
        raise ValueError(f"searchlight centres must have integer dtype: {path}")
    centres = raw_centres.astype(np.int64, copy=False)
    if np.any(centres < 0):
        raise ValueError(f"searchlight centres contain negative indices: {path}")
    if len(np.unique(centres)) != len(centres):
        raise ValueError(f"searchlight centres contain duplicate indices: {path}")
    return centres


def validated_centre_values(
    volume: np.ndarray,
    centres: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select authoritative centres and return values plus per-model finite counts."""
    volume = np.asarray(volume)
    if volume.ndim != 4:
        raise ValueError(
            f"{label} must have shape (models, x, y, z), got {volume.shape}"
        )
    if volume.shape[0] != len(EXPECTED_MODEL_ORDER):
        raise ValueError(
            f"{label} has {volume.shape[0]} models, expected {len(EXPECTED_MODEL_ORDER)}"
        )
    if volume.dtype != np.float64:
        raise ValueError(f"{label} must have dtype float64, got {volume.dtype}")
    centres = np.asarray(centres)
    if centres.ndim != 1 or not np.issubdtype(centres.dtype, np.integer):
        raise ValueError(f"{label} centres must be a 1D integer array")
    spatial_size = int(np.prod(volume.shape[1:]))
    if len(centres) == 0 or np.any(centres < 0) or np.any(centres >= spatial_size):
        raise ValueError(f"{label} centres are empty or outside its spatial shape")
    values = np.asarray(volume.reshape(volume.shape[0], -1)[:, centres])
    finite_counts = np.isfinite(values).sum(axis=1)
    if np.any(finite_counts == 0):
        features = [
            EXPECTED_MODEL_ORDER[index] for index in np.flatnonzero(finite_counts == 0)
        ]
        raise ValueError(f"{label} has no finite centre values for {features}")
    return values, finite_counts


def aggregate_peak_searchlight_centres(
    sample_center_maps: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Average sample maps centrewise, then take each model's finite maximum.

    A centre is eligible only when all sample values are finite. The second
    return value is the number of finite samples at each winning centre.
    """
    if not sample_center_maps:
        raise ValueError("at least one sample centre map is required")
    expected_shape = np.asarray(sample_center_maps[0]).shape
    if len(expected_shape) != 2:
        raise ValueError("sample centre maps must have shape (models, centres)")
    sums = np.zeros(expected_shape, dtype=np.float64)
    counts = np.zeros(expected_shape, dtype=np.uint16)
    for sample_index, sample_map in enumerate(sample_center_maps):
        values = np.asarray(sample_map, dtype=np.float64)
        if values.shape != expected_shape:
            raise ValueError(
                f"sample centre map {sample_index} has shape {values.shape}, "
                f"expected {expected_shape}"
            )
        finite = np.isfinite(values)
        np.add(sums, values, out=sums, where=finite)
        counts += finite

    mean_maps = np.full(expected_shape, np.nan, dtype=np.float64)
    complete = counts == len(sample_center_maps)
    np.divide(sums, len(sample_center_maps), out=mean_maps, where=complete)
    finite_mean_counts = np.isfinite(mean_maps).sum(axis=1)
    if np.any(finite_mean_counts == 0):
        raise ValueError("a model has no finite subject-mean searchlight centres")
    safe_means = np.where(np.isfinite(mean_maps), mean_maps, -np.inf)
    peak_indices = np.argmax(safe_means, axis=1)
    model_indices = np.arange(expected_shape[0])
    peaks = mean_maps[model_indices, peak_indices]
    peak_sample_counts = counts[model_indices, peak_indices].astype(np.int64)
    return peaks, peak_sample_counts


def _default_searchlight_centres_root(result_root: Path) -> Path:
    return result_root.parent.parent / "results" / "mpnet_10_sessions" / "precomputed"


def load_peak_summaries(
    result_root: Path,
    report_data: dict[str, Any],
    searchlight_centres_root: Path | None = None,
) -> dict[str, Any]:
    """Validate native grouped volumes and compute subject-first peak summaries."""
    result_root = result_root.resolve()
    centres_root = (
        searchlight_centres_root.resolve()
        if searchlight_centres_root is not None
        else _default_searchlight_centres_root(result_root)
    )
    manifest_path = (
        result_root
        / "searchlight"
        / "serialised_models_correlation"
        / GROUP_NAME
        / "group_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"group manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_model_order(manifest)

    report_sample_scores = report_data["sample_scores"]
    subject_peaks = np.empty((N_SUBJECTS, len(EXPECTED_MODEL_ORDER)), dtype=np.float64)
    provenance_subjects: dict[str, Any] = {}
    for subject in range(1, N_SUBJECTS + 1):
        subj = f"subj{subject:02d}"
        centre_path = (
            centres_root / subj / f"{subj}-func1pt8mm-6rad-searchlight_centers.npy"
        )
        centres = _load_searchlight_centres(centre_path)
        volume_dir = (
            result_root
            / "searchlight"
            / "searchlight_respectedsampling_correlation"
            / subj
            / GROUP_NAME
            / "corr_vols_correlation"
        )
        expected_files = [
            volume_dir / f"{subj}_nsd-{GROUP_NAME}_func1pt8mm_sample-{sample}.npy"
            for sample in range(N_SAMPLES)
        ]
        actual_files = sorted(volume_dir.glob("*sample-*.npy"))
        if actual_files != expected_files:
            missing = sorted(
                str(path) for path in set(expected_files) - set(actual_files)
            )
            extra = sorted(
                str(path) for path in set(actual_files) - set(expected_files)
            )
            raise ValueError(
                f"expected exactly samples 0..{N_SAMPLES - 1} for {subj}; "
                f"missing={missing}, extra={extra}"
            )

        center_maps: list[np.ndarray] = []
        volume_records: list[dict[str, Any]] = []
        expected_shape: tuple[int, ...] | None = None
        for sample_index, volume_path in enumerate(expected_files):
            volume = np.load(volume_path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(value) for value in volume.shape)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    f"{volume_path} shape {shape} differs from {expected_shape}"
                )
            values, finite_counts = validated_centre_values(
                volume, centres, label=str(volume_path)
            )
            sample_means = np.divide(
                np.where(np.isfinite(values), values, 0.0).sum(axis=1),
                finite_counts,
            )
            for model_index, feature in enumerate(EXPECTED_MODEL_ORDER):
                assert_close(
                    sample_means[model_index],
                    report_sample_scores[
                        subject - 1,
                        sample_index,
                        model_index,
                    ],
                    f"{subj} sample {sample_index} {feature} searchlight mean",
                )
            center_maps.append(values)
            volume_records.append(
                {
                    "sample_index": sample_index,
                    "path": str(volume_path.resolve()),
                    "sha256": sha256_file(volume_path),
                    "shape_models_x_native_xyz": "x".join(map(str, shape)),
                    "finite_centre_counts_in_model_order_csv": ",".join(
                        map(str, finite_counts.tolist())
                    ),
                }
            )

        peaks, winning_counts = aggregate_peak_searchlight_centres(center_maps)
        subject_peaks[subject - 1] = peaks
        provenance_subjects[subj] = {
            "searchlight_centres": {
                "path": str(centre_path.resolve()),
                "sha256": sha256_file(centre_path),
                "count": len(centres),
            },
            "sample_volumes": volume_records,
            "winning_centre_finite_sample_counts_in_model_order": (
                winning_counts.tolist()
            ),
        }

    peaks_by_feature: dict[str, dict[str, Any]] = {}
    for model_index, feature in enumerate(EXPECTED_MODEL_ORDER):
        values = subject_peaks[:, model_index]
        mean, low, high = mean_ci(values)
        peaks_by_feature[feature] = {
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "range_low": float(values.min()),
            "range_high": float(values.max()),
            "subject_peaks": values.tolist(),
        }

    return {
        "peaks": peaks_by_feature,
        "subject_peaks": subject_peaks,
        "searchlight_provenance": {
            "result_root": str(result_root),
            "searchlight_centres_root": str(centres_root),
            "group_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "subjects": provenance_subjects,
        },
    }


def format_number(value: float) -> str:
    return repr(float(value))


def format_ci(row: dict[str, Any]) -> str:
    return f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}]"


def format_range(row: dict[str, Any]) -> str:
    return f"[{format_number(row['range_low'])}, {format_number(row['range_high'])}]"


def build_table_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    scores = data["scores"]
    comparisons = data["comparisons"]
    peaks = data["peaks"]
    rows: list[dict[str, str]] = []
    for prompt in PROMPTS:
        for layer in LAYERS:
            raw_feature = f"{prompt}__l{layer:02d}__raw"
            j_feature = f"{prompt}__l{layer:02d}__j"
            raw = scores[raw_feature]
            j_score = scores[j_feature]
            raw_peak = peaks[raw_feature]
            j_peak = peaks[j_feature]
            delta = comparisons[(j_feature, raw_feature)]
            rows.append(
                {
                    "prompt": prompt,
                    "layer": str(layer),
                    "row_type": "paired_raw_j",
                    "raw_feature": raw_feature,
                    "j_feature": j_feature,
                    "control_feature": "",
                    "raw_group_mean": format_number(raw["mean_correlation"]),
                    "raw_ci_95": format_ci(raw),
                    "raw_peak_searchlight_centre_mean": format_number(raw_peak["mean"]),
                    "raw_peak_searchlight_centre_ci_95": format_ci(raw_peak),
                    "raw_peak_searchlight_centre_range": format_range(raw_peak),
                    "j_group_mean": format_number(j_score["mean_correlation"]),
                    "j_ci_95": format_ci(j_score),
                    "j_peak_searchlight_centre_mean": format_number(j_peak["mean"]),
                    "j_peak_searchlight_centre_ci_95": format_ci(j_peak),
                    "j_peak_searchlight_centre_range": format_range(j_peak),
                    "j_minus_raw_delta": format_number(delta["mean_delta"]),
                    "delta_ci_95": format_ci(delta),
                    "exact_p_two_sided": format_number(delta["exact_p"]),
                    "bh_q_24_comparisons": format_number(delta["fdr_q"]),
                    "bh_significant_q_lt_0_05": str(delta["fdr_q"] < ALPHA).upper(),
                    "final_control_group_mean": "",
                    "final_control_ci_95": "",
                    "final_control_peak_searchlight_centre_mean": "",
                    "final_control_peak_searchlight_centre_ci_95": "",
                    "final_control_peak_searchlight_centre_range": "",
                }
            )
        control_feature = f"{prompt}__final"
        control = scores[control_feature]
        control_peak = peaks[control_feature]
        rows.append(
            {
                "prompt": prompt,
                "layer": "final",
                "row_type": "final_control",
                "raw_feature": "",
                "j_feature": "",
                "control_feature": control_feature,
                "raw_group_mean": "",
                "raw_ci_95": "",
                "raw_peak_searchlight_centre_mean": "",
                "raw_peak_searchlight_centre_ci_95": "",
                "raw_peak_searchlight_centre_range": "",
                "j_group_mean": "",
                "j_ci_95": "",
                "j_peak_searchlight_centre_mean": "",
                "j_peak_searchlight_centre_ci_95": "",
                "j_peak_searchlight_centre_range": "",
                "j_minus_raw_delta": "",
                "delta_ci_95": "",
                "exact_p_two_sided": "",
                "bh_q_24_comparisons": "",
                "bh_significant_q_lt_0_05": "",
                "final_control_group_mean": format_number(control["mean_correlation"]),
                "final_control_ci_95": format_ci(control),
                "final_control_peak_searchlight_centre_mean": format_number(
                    control_peak["mean"]
                ),
                "final_control_peak_searchlight_centre_ci_95": format_ci(control_peak),
                "final_control_peak_searchlight_centre_range": format_range(
                    control_peak
                ),
            }
        )
    return rows


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _asymmetric_errors(rows: list[dict[str, Any]], value_key: str) -> np.ndarray:
    values = np.asarray([row[value_key] for row in rows])
    low = np.asarray([row["ci_low"] for row in rows])
    high = np.asarray([row["ci_high"] for row in rows])
    return np.vstack((values - low, high - values))


def write_figure(path: Path, data: dict[str, Any]) -> tuple[int, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = data["scores"]
    comparisons = data["comparisons"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 6.8),
        dpi=300,
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": (2.8, 1.25), "hspace": 0.08, "wspace": 0.14},
    )
    raw_color = "#4C6572"
    j_color = "#C66A2B"
    delta_color = "#5B4A8A"
    control_color = "#20262E"
    x_layers = np.arange(len(LAYERS))

    for column, prompt in enumerate(PROMPTS):
        upper = axes[0, column]
        lower = axes[1, column]
        raw_rows = [scores[f"{prompt}__l{layer:02d}__raw"] for layer in LAYERS]
        j_rows = [scores[f"{prompt}__l{layer:02d}__j"] for layer in LAYERS]
        delta_rows = [
            comparisons[
                (
                    f"{prompt}__l{layer:02d}__j",
                    f"{prompt}__l{layer:02d}__raw",
                )
            ]
            for layer in LAYERS
        ]
        raw_means = np.asarray([row["mean_correlation"] for row in raw_rows])
        j_means = np.asarray([row["mean_correlation"] for row in j_rows])
        deltas = np.asarray([row["mean_delta"] for row in delta_rows])

        upper.errorbar(
            x_layers,
            raw_means,
            yerr=_asymmetric_errors(raw_rows, "mean_correlation"),
            color=raw_color,
            marker="o",
            markersize=4.6,
            linewidth=1.35,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            label="Raw residual",
            zorder=3,
        )
        upper.errorbar(
            x_layers,
            j_means,
            yerr=_asymmetric_errors(j_rows, "mean_correlation"),
            color=j_color,
            marker="s",
            markersize=4.3,
            linewidth=1.35,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            label="J-space",
            zorder=4,
        )
        control = scores[f"{prompt}__final"]
        upper.errorbar(
            [4],
            [control["mean_correlation"]],
            yerr=np.asarray(
                [
                    [control["mean_correlation"] - control["ci_low"]],
                    [control["ci_high"] - control["mean_correlation"]],
                ]
            ),
            color=control_color,
            marker="D",
            markersize=4.8,
            linestyle="none",
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            label="Final-layer control",
            zorder=5,
        )

        lower.axhline(0.0, color="#7B838B", linewidth=0.85, zorder=1)
        lower.errorbar(
            x_layers,
            deltas,
            yerr=_asymmetric_errors(delta_rows, "mean_delta"),
            color=delta_color,
            marker="o",
            markersize=4.5,
            linewidth=1.15,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            zorder=3,
        )
        for x_value, row in zip(x_layers, delta_rows, strict=True):
            if row["fdr_q"] < ALPHA:
                offset = max(0.00015, 0.08 * (row["ci_high"] - row["ci_low"]))
                lower.text(
                    x_value,
                    row["ci_high"] + offset,
                    "★",
                    color=delta_color,
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                    clip_on=False,
                )
        lower.text(
            4,
            0,
            "control\nonly",
            color="#7B838B",
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=0.95,
        )

        panel = "A" if column == 0 else "B"
        title = "Visualize" if prompt == "visualize" else "Caption-only (plain)"
        upper.set_title(f"{panel}   {title}", loc="left", fontweight="bold", pad=8)
        upper.grid(axis="y", color="#DDE2E6", linewidth=0.6, alpha=0.8)
        lower.grid(axis="y", color="#E3E7EA", linewidth=0.55, alpha=0.8)
        for axis in (upper, lower):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="out", length=3, width=0.8)
            axis.set_xlim(-0.35, 4.35)
        lower.set_xticks(np.arange(5), ["8", "16", "23", "30", "Final"])
        lower.set_xlabel("Decoder layer")

    axes[0, 0].set_ylabel("Group mean searchlight RSA correlation (r)")
    axes[1, 0].set_ylabel("J − raw Δr")
    axes[0, 0].set_ylim(-0.0015, 0.036)
    axes[1, 0].set_ylim(-0.0028, 0.0027)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.908),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.7,
    )
    fig.suptitle(
        "Raw versus J-space alignment with NSD searchlight geometry",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.947,
        "Points are means of 8 subject summaries; bars are 95% subject t CIs. "
        "★ BH q < 0.05 for paired J−raw tests (source 24-comparison family).",
        ha="left",
        va="top",
        fontsize=8.6,
        color="#3F4850",
    )
    fig.text(
        0.08,
        0.018,
        "Within each subject: searchlight centers averaged per 100-image sample, "
        "then 8 sample means averaged. Final is a single unpaired control.",
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#59636C",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.835, bottom=0.12)

    source_text = json.dumps(
        data["source_hashes"], sort_keys=True, separators=(",", ":")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        metadata={
            "Title": "Raw versus J-space NSD layer performance",
            "Author": "Deterministic repository generator",
            "Description": (
                "Completed eight-subject qwen4b run; subject-level 95% t CIs; "
                "exact paired sign-flip p-values; source BH q-values over 24 tests."
            ),
            "SourceFilesSHA256": source_text,
            "Software": "scripts/make_layer_performance_summary.py",
            "Date": None,
        },
    )
    width, height = (int(round(10.0 * 300)), int(round(6.8 * 300)))
    plt.close(fig)
    return width, height


def write_metadata(
    path: Path,
    data: dict[str, Any],
    table_output: Path,
    figure_output: Path,
    figure_dimensions: tuple[int, int],
) -> None:
    import matplotlib
    import scipy

    generator = Path(__file__).resolve()
    metadata = {
        "schema_version": 2,
        "generator": "scripts/make_layer_performance_summary.py",
        "generator_sha256": sha256_file(generator),
        "source_report_name": data["summary"]["profile"],
        "source_summary_created_at": data["summary"]["created_at"],
        "source_files_sha256": data["source_hashes"],
        "source_searchlight": data["searchlight_provenance"],
        "validated_contract": {
            "expected_feature_ids_in_npy_column_order": list(EXPECTED_MODEL_ORDER),
            "independent_unit": "subject",
            "n_subjects": N_SUBJECTS,
            "n_samples_per_subject": N_SAMPLES,
            "n_sessions": N_SESSIONS,
            "n_comparisons_in_bh_family": 24,
            "checks": [
                "JSON rows exactly equal source CSV rows",
                "subject_scores exactly equal sample_scores mean over samples",
                "CSV/JSON means, t CIs, deltas, exact p-values, and BH q-values reproduce from NPY arrays",
                "report score arrays are finite float64 with expected shapes",
                "grouped volume axis zero exactly matches the manifest model order",
                "all 8 subject x 8 sample native grouped volumes have expected shapes and a finite authoritative centre for every model",
                "whole-searchlight sample means recomputed from native volumes and authoritative centres reproduce sample_scores.npy",
            ],
        },
        "methods": {
            "mean_correlation": (
                "searchlight-center RSA correlations averaged within each sample, "
                "then 8 sample means averaged within subject, then 8 subject scores averaged"
            ),
            "confidence_interval": (
                "two-sided 95% Student t interval over 8 subject scores (df=7)"
            ),
            "delta": "paired within-subject J-space minus raw score, then group mean",
            "exact_p": (
                "two-sided exhaustive sign-flip test of the mean paired subject delta "
                "over all 2^8 sign assignments"
            ),
            "bh_q": (
                "Benjamini-Hochberg adjustment over the source report's 24 J-versus-"
                "raw/final/MPNet comparisons"
            ),
            "peak_label": (
                "peak SEARCHLIGHT-CENTRE RSA correlations, not single-voxel "
                "correlations"
            ),
            "peak_searchlight_centre": (
                "For each subject and feature, first average that subject's eight "
                "sample correlation volumes centrewise, restricted to the "
                "authoritative valid searchlight centres; a centre with any "
                "nonfinite sample value has a nonfinite subject mean and is "
                "excluded. Then take the maximum finite centre from that "
                "subject-mean map. Across the eight subjects, report mean subject "
                "peak, two-sided 95% subject t CI, and observed subject peak range. "
                "This avoids selecting a maximum over all 64 sample maps."
            ),
            "peak_inference_guardrail": (
                "Peaks are descriptive and noise-sensitive; no p or q significance "
                "is attached to peak differences. Existing J-versus-raw inference "
                "uses whole-searchlight means."
            ),
        },
        "derived_peak_subject_values_in_model_order": {
            f"subj{subject:02d}": data["subject_peaks"][subject - 1].tolist()
            for subject in range(1, N_SUBJECTS + 1)
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            "table": {
                "name": table_output.name,
                "sha256": sha256_file(table_output),
            },
            "figure": {
                "name": figure_output.name,
                "sha256": sha256_file(figure_output),
                "width_px": figure_dimensions[0],
                "height_px": figure_dimensions[1],
                "dpi": 300,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    data = validate_report(args.report_dir)
    peak_data = load_peak_summaries(
        args.result_root,
        data,
        args.searchlight_centres_root,
    )
    data.update(peak_data)
    rows = build_table_rows(data)
    write_table(args.table_output, rows)
    dimensions = write_figure(args.figure_output, data)
    write_metadata(
        args.metadata_output,
        data,
        args.table_output,
        args.figure_output,
        dimensions,
    )
    significant = [
        f"{row['prompt']}:L{row['layer']}"
        for row in rows
        if row["bh_significant_q_lt_0_05"] == "TRUE"
    ]
    print(
        "validated completed 8-subject report; "
        f"wrote {len(rows)} table rows and {dimensions[0]}x{dimensions[1]} figure; "
        f"BH-significant J-vs-raw: {', '.join(significant) or 'none'}"
    )


if __name__ == "__main__":
    main()

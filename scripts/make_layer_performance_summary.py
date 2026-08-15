#!/usr/bin/env python3
"""Validate an eight-subject report and build the layer summary artifacts.

Example, from the repository root::

    python scripts/make_layer_performance_summary.py \
      --report-dir /path/to/results/reports/qwen4b

The report directory must contain feature_scores.csv, comparisons.csv,
subject_scores.npy, sample_scores.npy, and summary.json from the completed
eight-subject run. The script does not read searchlight volumes or recompute
searchlights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
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
    "j_group_mean",
    "j_ci_95",
    "j_minus_raw_delta",
    "delta_ci_95",
    "exact_p_two_sided",
    "bh_q_24_comparisons",
    "bh_significant_q_lt_0_05",
    "final_control_group_mean",
    "final_control_ci_95",
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


def format_number(value: float) -> str:
    return repr(float(value))


def format_ci(row: dict[str, Any]) -> str:
    return f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}]"


def build_table_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    scores = data["scores"]
    comparisons = data["comparisons"]
    rows: list[dict[str, str]] = []
    for prompt in PROMPTS:
        for layer in LAYERS:
            raw_feature = f"{prompt}__l{layer:02d}__raw"
            j_feature = f"{prompt}__l{layer:02d}__j"
            raw = scores[raw_feature]
            j_score = scores[j_feature]
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
                    "j_group_mean": format_number(j_score["mean_correlation"]),
                    "j_ci_95": format_ci(j_score),
                    "j_minus_raw_delta": format_number(delta["mean_delta"]),
                    "delta_ci_95": format_ci(delta),
                    "exact_p_two_sided": format_number(delta["exact_p"]),
                    "bh_q_24_comparisons": format_number(delta["fdr_q"]),
                    "bh_significant_q_lt_0_05": str(delta["fdr_q"] < ALPHA).upper(),
                    "final_control_group_mean": "",
                    "final_control_ci_95": "",
                }
            )
        control_feature = f"{prompt}__final"
        control = scores[control_feature]
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
                "j_group_mean": "",
                "j_ci_95": "",
                "j_minus_raw_delta": "",
                "delta_ci_95": "",
                "exact_p_two_sided": "",
                "bh_q_24_comparisons": "",
                "bh_significant_q_lt_0_05": "",
                "final_control_group_mean": format_number(control["mean_correlation"]),
                "final_control_ci_95": format_ci(control),
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
        "schema_version": 1,
        "generator": "scripts/make_layer_performance_summary.py",
        "generator_sha256": sha256_file(generator),
        "source_report_name": data["summary"]["profile"],
        "source_summary_created_at": data["summary"]["created_at"],
        "source_files_sha256": data["source_hashes"],
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
                "all arrays are finite float64 with expected shapes",
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

"""Searchlight, projection, plotting, and quantitative summary stages."""

from __future__ import annotations

import csv
import html
import itertools
import json
import math
import pickle
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import (
    ALL_TOKEN_MEAN,
    DEFAULT_PROMPT_SET,
    DEFAULT_READOUT_MODE,
    N_SAMPLES,
    N_SESSIONS,
    N_SUBJECTS,
    ExperimentPaths,
    group_name,
    run_name,
    validate_subjects,
)
from .io_utils import atomic_json, atomic_npy


def _group_manifest(
    paths: ExperimentPaths,
    profile: str,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
    readout_mode: str = DEFAULT_READOUT_MODE,
) -> dict:
    path = (
        paths.searchlight_base
        / "serialised_models_correlation"
        / group_name(profile, prompt_set_key, readout_mode)
        / "group_manifest.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"group manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_searchlight_subject(
    paths: ExperimentPaths,
    profile: str,
    subject: int,
    *,
    allow_cpu: bool = False,
    max_samples: int | None = None,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
    readout_mode: str = DEFAULT_READOUT_MODE,
) -> None:
    """Correlate all model RDMs while computing each brain RDM once."""
    if not 1 <= subject <= N_SUBJECTS:
        raise ValueError(f"subject must be in 1..{N_SUBJECTS}")
    from .nsd_adapter import run_searchlight

    run_searchlight(
        paths,
        _group_manifest(paths, profile, prompt_set_key, readout_mode)["group_name"],
        subject,
        allow_cpu=allow_cpu,
        max_samples=max_samples,
    )


def project_subjects(
    paths: ExperimentPaths,
    profile: str,
    subjects: Sequence[int] = tuple(range(1, N_SUBJECTS + 1)),
    *,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
    readout_mode: str = DEFAULT_READOUT_MODE,
) -> None:
    from .nsd_adapter import project_to_fsaverage

    project_to_fsaverage(
        paths,
        _group_manifest(paths, profile, prompt_set_key, readout_mode)["group_name"],
        subjects,
    )


def _surface_path(
    paths: ExperimentPaths,
    group: str,
    subject: int,
    model_index: int,
    hemisphere: str,
) -> Path:
    subj = f"subj{subject:02d}"
    return (
        paths.searchlight_base
        / "searchlight_respectedsampling_correlation"
        / subj
        / group
        / f"{group}_correlation_fsaverage"
        / f"{hemisphere}.{subj}-model-{model_index}-surf.npy"
    )


def plot_individual_maps(
    paths: ExperimentPaths,
    profile: str,
    *,
    subjects: Sequence[int] = tuple(range(1, N_SUBJECTS + 1)),
    feature_names: Sequence[str] | None = None,
    roi_overlay: str | None = "streams",
    prompt_set_key: str = DEFAULT_PROMPT_SET,
    readout_mode: str = DEFAULT_READOUT_MODE,
) -> list[str]:
    """Plot projected surfaces using the manifest's model-index mapping."""
    manifest = _group_manifest(paths, profile, prompt_set_key, readout_mode)
    group = manifest["group_name"]
    selected = set(feature_names) if feature_names is not None else None
    output_dir = (
        paths.reports / "figures" / run_name(profile, prompt_set_key, readout_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    from .nsd_adapter import plot_brain

    outputs = []
    for model in manifest["model_order"]:
        feature = model["feature"]
        if selected is not None and feature not in selected:
            continue
        model_index = int(model["model_index"])
        for subject in subjects:
            surfaces = [
                np.load(
                    _surface_path(paths, group, subject, model_index, hemisphere),
                    allow_pickle=False,
                )
                for hemisphere in ("lh", "rh")
            ]
            values = np.concatenate(surfaces).astype(np.float32, copy=False)
            name = f"{feature}_subj{subject:02d}"
            output_path = output_dir / f"{name}.png"
            if output_path.exists() and output_path.stat().st_size > 0:
                outputs.append(str(output_path.resolve()))
                continue
            plot_brain(
                values,
                name,
                output_dir,
                roi_overlay=roi_overlay,
                nsd_dir=paths.nsd_dir,
            )
            outputs.append(str(output_path.resolve()))
    return outputs


def _searchlight_centers(paths: ExperimentPaths, subject: int) -> np.ndarray:
    subj = f"subj{subject:02d}"
    path = (
        paths.mpnet_precomputed
        / subj
        / f"{subj}-func1pt8mm-6rad-searchlight_centers.npy"
    )
    with path.open("rb") as handle:
        centers = np.asarray(pickle.load(handle), dtype=np.int64).ravel()
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError(f"invalid searchlight centers: {path}")
    return centers


def _sample_files(paths: ExperimentPaths, group: str, subject: int) -> list[Path]:
    subj = f"subj{subject:02d}"
    directory = (
        paths.searchlight_base
        / "searchlight_respectedsampling_correlation"
        / subj
        / group
        / "corr_vols_correlation"
    )
    files = sorted(directory.glob("*sample-*.npy"))
    if len(files) != N_SAMPLES:
        raise ValueError(
            f"expected {N_SAMPLES} searchlight samples in {directory}, "
            f"found {len(files)}"
        )
    return files


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    from scipy.stats import t

    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, math.nan, math.nan
    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    half = float(t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - half, mean + half


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def _comparison_rows(
    subject_scores: np.ndarray,
    feature_names: Sequence[str],
    prompt_set_key: str,
    readout_mode: str = DEFAULT_READOUT_MODE,
) -> list[dict]:
    """Build contrasts and adjust each predeclared BH family independently."""
    index_by_feature = {feature: index for index, feature in enumerate(feature_names)}
    specifications = []
    if readout_mode == ALL_TOKEN_MEAN:
        for layer in (8, 16, 23, 30):
            specifications.append(
                {
                    "comparison_type": "j_vs_raw",
                    "bh_family": "plain_mean_pool_j_vs_raw_4",
                    "feature": f"plain_mean_pool__l{layer:02d}__j",
                    "baseline": f"plain_mean_pool__l{layer:02d}__raw",
                    "representation_kind": "j_minus_raw",
                    "layer": layer,
                }
            )
    elif prompt_set_key == "matched_readout":
        for prompt in ("integrate_readout", "minimal_readout"):
            for layer in (8, 16, 23, 30):
                specifications.append(
                    {
                        "comparison_type": "j_vs_raw",
                        "bh_family": "matched_readout_j_vs_raw_8",
                        "feature": f"{prompt}__l{layer:02d}__j",
                        "baseline": f"{prompt}__l{layer:02d}__raw",
                        "representation_kind": "j_minus_raw",
                        "layer": layer,
                    }
                )
        for kind in ("raw", "j"):
            for layer in (8, 16, 23, 30):
                specifications.append(
                    {
                        "comparison_type": "integrate_vs_minimal",
                        "bh_family": "matched_readout_prompt_pair_9",
                        "feature": f"integrate_readout__l{layer:02d}__{kind}",
                        "baseline": f"minimal_readout__l{layer:02d}__{kind}",
                        "representation_kind": kind,
                        "layer": layer,
                    }
                )
        specifications.append(
            {
                "comparison_type": "integrate_vs_minimal",
                "bh_family": "matched_readout_prompt_pair_9",
                "feature": "integrate_readout__final",
                "baseline": "minimal_readout__final",
                "representation_kind": "final",
                "layer": None,
            }
        )
    else:
        for feature in feature_names:
            if not feature.endswith("__j"):
                continue
            prefix = feature[: -len("__j")]
            prompt = feature.split("__", 1)[0]
            for baseline in (
                f"{prefix}__raw",
                f"{prompt}__final",
                "mpnet_reference",
            ):
                specifications.append(
                    {
                        "comparison_type": "historical_j_vs_baseline",
                        "bh_family": "historical_j_vs_baselines",
                        "feature": feature,
                        "baseline": baseline,
                        "representation_kind": "j",
                        "layer": int(prefix.rsplit("l", 1)[1]),
                    }
                )

    rows = []
    for specification in specifications:
        feature = specification["feature"]
        baseline = specification["baseline"]
        if feature not in index_by_feature or baseline not in index_by_feature:
            raise ValueError(f"missing comparison feature: {feature} or {baseline}")
        differences = (
            subject_scores[:, index_by_feature[feature]]
            - subject_scores[:, index_by_feature[baseline]]
        )
        mean, low, high = _mean_ci(differences)
        rows.append(
            {
                **specification,
                "mean_delta": mean,
                "ci_low": low,
                "ci_high": high,
                "exact_p": _exact_sign_flip_p(differences),
                "fdr_q": math.nan,
            }
        )
    families = sorted({row["bh_family"] for row in rows})
    for family in families:
        indices = [
            index for index, row in enumerate(rows) if row["bh_family"] == family
        ]
        adjusted = _bh_adjust([rows[index]["exact_p"] for index in indices])
        for index, q_value in zip(indices, adjusted, strict=True):
            rows[index]["fdr_q"] = q_value
    return rows


def _whole_prompt_readout_rows(
    pooled_subject_scores: np.ndarray,
    pooled_feature_names: Sequence[str],
    historical_subject_scores: np.ndarray,
    historical_feature_names: Sequence[str],
) -> list[dict]:
    """Compare pooled and historical final-token readouts in one BH family."""
    pooled = {
        feature: pooled_subject_scores[:, index]
        for index, feature in enumerate(pooled_feature_names)
    }
    historical = {
        feature: historical_subject_scores[:, index]
        for index, feature in enumerate(historical_feature_names)
    }
    specifications = []
    for kind in ("raw", "j"):
        for layer in (8, 16, 23, 30):
            specifications.append((kind, layer))
    specifications.append(("final", None))
    rows = []
    for kind, layer in specifications:
        if kind == "final":
            feature = "plain_mean_pool__final"
            historical_feature = "plain__final"
        else:
            feature = f"plain_mean_pool__l{layer:02d}__{kind}"
            historical_feature = f"plain__l{layer:02d}__{kind}"
        if feature not in pooled or historical_feature not in historical:
            raise ValueError(
                f"missing pooled/historical readout pair: {feature}, "
                f"{historical_feature}"
            )
        differences = pooled[feature] - historical[historical_feature]
        mean, low, high = _mean_ci(differences)
        rows.append(
            {
                "comparison_type": "pooled_vs_historical_final_token",
                "bh_family": "plain_mean_pool_vs_final_token_9",
                "feature": feature,
                "baseline": f"historical_final_token__{historical_feature}",
                "representation_kind": kind,
                "layer": layer,
                "mean_delta": mean,
                "ci_low": low,
                "ci_high": high,
                "exact_p": _exact_sign_flip_p(differences),
                "fdr_q": math.nan,
            }
        )
    adjusted = _bh_adjust([row["exact_p"] for row in rows])
    for row, q_value in zip(rows, adjusted, strict=True):
        row["fdr_q"] = q_value
    return rows


def _manifest_subject_numbers(manifest: dict) -> tuple[int, ...]:
    """Read subject numbers from either group-manifest schema.

    Group manifests written before subject-subset execution record a
    ``subjects`` mapping keyed by ``subjNN`` and carry no ``subject_numbers``
    list. The locked historical comparator is one of those and is immutable,
    so the reader adapts instead of the artifact.
    """
    declared = manifest.get("subject_numbers")
    if declared is not None:
        return tuple(int(subject) for subject in declared)
    subjects = manifest.get("subjects")
    if not isinstance(subjects, dict):
        return ()
    numbers = []
    for key in subjects:
        match = re.fullmatch(r"subj(\d+)", str(key))
        if match is None:
            return ()
        numbers.append(int(match.group(1)))
    return tuple(sorted(numbers))


def _load_historical_final_token_scores(
    results_root: Path,
    profile: str,
    subjects: Sequence[int],
) -> tuple[np.ndarray, list[str], dict]:
    """Load and validate the immutable historical final-token comparator."""
    report_dir = results_root / "reports" / profile
    summary_path = report_dir / "summary.json"
    scores_path = report_dir / "subject_scores.npy"
    manifest_path = (
        results_root
        / "searchlight"
        / "serialised_models_correlation"
        / group_name(profile)
        / "group_manifest.json"
    )
    for path in (summary_path, scores_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"historical comparator artifact missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_subjects = _manifest_subject_numbers(manifest)
    if summary.get("profile") != profile or source_subjects != tuple(
        range(1, N_SUBJECTS + 1)
    ):
        raise ValueError("historical comparator is not the locked eight-subject run")
    all_names = [item["feature"] for item in manifest["model_order"]]
    all_scores = np.load(scores_path, allow_pickle=False)
    if (
        all_scores.shape != (N_SUBJECTS, len(all_names))
        or not np.isfinite(all_scores).all()
    ):
        raise ValueError("historical subject scores have invalid shape or values")
    wanted = []
    for kind in ("raw", "j"):
        for layer in (8, 16, 23, 30):
            wanted.append(f"plain__l{layer:02d}__{kind}")
    wanted.append("plain__final")
    missing = sorted(set(wanted) - set(all_names))
    if missing:
        raise ValueError(f"historical comparator lacks features: {missing}")
    columns = [all_names.index(feature) for feature in wanted]
    rows = [source_subjects.index(subject) for subject in subjects]
    selected = np.asarray(all_scores[np.ix_(rows, columns)], dtype=np.float64)
    from .io_utils import sha256_file

    provenance = {
        "results_root": str(results_root.resolve()),
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "subject_scores": str(scores_path.resolve()),
        "subject_scores_sha256": sha256_file(scores_path),
        "group_manifest": str(manifest_path.resolve()),
        "group_manifest_sha256": sha256_file(manifest_path),
        "subject_numbers": list(subjects),
        "feature_order": wanted,
    }
    return selected, wanted, provenance


def _performance_table_rows(
    score_rows: Sequence[dict], comparison_rows: Sequence[dict]
) -> list[dict]:
    """Join exact group scores to every inferential contrast for review."""
    scores = {row["feature"]: row for row in score_rows}
    output = []
    for comparison in comparison_rows:
        feature = scores[comparison["feature"]]
        baseline = scores[comparison["baseline"]]
        output.append(
            {
                "bh_family": comparison["bh_family"],
                "comparison_type": comparison["comparison_type"],
                "feature": comparison["feature"],
                "feature_group_mean": feature["mean_correlation"],
                "feature_ci_low": feature["ci_low"],
                "feature_ci_high": feature["ci_high"],
                "baseline": comparison["baseline"],
                "baseline_group_mean": baseline["mean_correlation"],
                "baseline_ci_low": baseline["ci_low"],
                "baseline_ci_high": baseline["ci_high"],
                "mean_delta": comparison["mean_delta"],
                "delta_ci_low": comparison["ci_low"],
                "delta_ci_high": comparison["ci_high"],
                "exact_p_two_sided": comparison["exact_p"],
                "bh_q_within_declared_family": comparison["fdr_q"],
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _html_report(summary: dict) -> str:
    def cells(values):
        return "".join(f"<td>{html.escape(str(value))}</td>" for value in values)

    score_rows = []
    maxima = max(abs(row["mean_correlation"]) for row in summary["scores"])
    maxima = maxima or 1.0
    for row in summary["scores"]:
        width = 100 * abs(row["mean_correlation"]) / maxima
        score_rows.append(
            "<tr>"
            + cells(
                [
                    row["feature"],
                    f"{row['mean_correlation']:.5f}",
                    f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}]",
                ]
            )
            + f'<td><span class="bar" style="width:{width:.1f}%"></span></td>'
            + "</tr>"
        )
    comparison_rows = []
    for row in summary["comparisons"]:
        comparison_rows.append(
            "<tr>"
            + cells(
                [
                    row["comparison_type"],
                    row["bh_family"],
                    row["feature"],
                    row["baseline"],
                    f"{row['mean_delta']:.5f}",
                    f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}]",
                    f"{row['exact_p']:.5f}",
                    f"{row['fdr_q']:.5f}",
                ]
            )
            + "</tr>"
        )
    if summary.get("readout_mode") == ALL_TOKEN_MEAN:
        comparison_heading = "Predeclared whole-prompt pooling comparisons"
        family_note = (
            "BH q-values are adjusted separately within exactly four pooled "
            "J-vs-raw tests and nine pooled-vs-historical-final-token tests."
        )
    elif summary.get("prompt_set") == "matched_readout":
        comparison_heading = "Predeclared matched-readout comparisons"
        family_note = (
            "BH q-values are adjusted separately within the eight J-vs-raw "
            "tests and the nine integrate-vs-minimal tests."
        )
    else:
        comparison_heading = "Matched J-space comparisons"
        family_note = "BH q-values use the historical comparison family."
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jacobian Lens × NSD summary</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17212b}}
h1,h2{{line-height:1.2}} .meta{{color:#586674}} table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{padding:.5rem .65rem;border-bottom:1px solid #dce3e8;text-align:left}} th{{position:sticky;top:0;background:#f5f8fa}}
.bar{{display:block;height:.8rem;min-width:2px;background:#356fc0;border-radius:3px}} code{{background:#edf2f5;padding:.12rem .25rem}}
</style></head><body>
<h1>Jacobian Lens × NSD</h1>
<p class="meta">Profile <code>{html.escape(summary["profile"])}</code>; {summary["n_subjects"]} subject(s) × {summary["n_sessions"]} sessions × {summary["n_samples_per_subject"]} matched 100-image samples. Generated {html.escape(summary["created_at"])}.</p>
<p>Values are searchlight-center correlations averaged within sample, then within subject. Confidence intervals and exact sign-flip tests use subjects as the independent unit (n={summary["n_subjects"]}). Runs with one subject are descriptive validations, not population inference. This is an exploratory summary, not a held-out model-selection analysis.</p>
<h2>Feature scores</h2>
<table><thead><tr><th>Feature</th><th>Mean r</th><th>95% subject CI</th><th>Relative magnitude</th></tr></thead><tbody>{"".join(score_rows)}</tbody></table>
<h2>{html.escape(comparison_heading)}</h2>
<p>{html.escape(family_note)}</p>
<table><thead><tr><th>Contrast</th><th>BH family</th><th>Feature</th><th>Baseline</th><th>Mean Δr</th><th>95% subject CI</th><th>Exact p</th><th>BH q</th></tr></thead><tbody>{"".join(comparison_rows)}</tbody></table>
</body></html>"""


def _plot_matched_layer_summary(summary: dict, path: Path) -> None:
    """Render the compact, review-oriented layer performance figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = {row["feature"]: row for row in summary["scores"]}
    layers = np.asarray([8, 16, 23, 30])
    colors = {"raw": "#7A8797", "j": "#2457C5"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, prompt in zip(
        axes, ("minimal_readout", "integrate_readout"), strict=True
    ):
        for kind in ("raw", "j"):
            rows = [scores[f"{prompt}__l{layer:02d}__{kind}"] for layer in layers]
            means = np.asarray([row["mean_correlation"] for row in rows])
            lower = means - np.asarray([row["ci_low"] for row in rows])
            upper = np.asarray([row["ci_high"] for row in rows]) - means
            axis.errorbar(
                layers,
                means,
                yerr=np.vstack([lower, upper]),
                marker="o",
                linewidth=2,
                capsize=4,
                color=colors[kind],
                label=kind.upper(),
            )
        axis.axhline(0, color="#C7CDD5", linewidth=1)
        axis.set_title(prompt.replace("_", " "))
        axis.set_xlabel("Qwen block")
        axis.set_xticks(layers)
        axis.grid(axis="y", color="#E4E8ED", linewidth=0.8)
    axes[0].set_ylabel("Mean searchlight correlation (subject-level 95% CI)")
    axes[1].legend(frameon=False)
    figure.suptitle("Matched-readout prompt control", fontsize=15, fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_whole_prompt_summary(summary: dict, path: Path) -> None:
    """Plot pooled and historical final-token trajectories for review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = {row["feature"]: row for row in summary["scores"]}
    layers = np.asarray([8, 16, 23, 30])
    colors = {"raw": "#7A8797", "j": "#2457C5"}
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for kind in ("raw", "j"):
        for prefix, label, line_style in (
            ("plain_mean_pool", "Pooled", "-"),
            ("historical_final_token__plain", "Historical final token", "--"),
        ):
            rows = [scores[f"{prefix}__l{layer:02d}__{kind}"] for layer in layers]
            means = np.asarray([row["mean_correlation"] for row in rows])
            lower = means - np.asarray([row["ci_low"] for row in rows])
            upper = np.asarray([row["ci_high"] for row in rows]) - means
            axis.errorbar(
                layers,
                means,
                yerr=np.vstack([lower, upper]),
                marker="o",
                linewidth=2,
                capsize=3,
                color=colors[kind],
                linestyle=line_style,
                label=f"{label} {kind.upper()}",
            )
    axis.axhline(0, color="#C7CDD5", linewidth=1)
    axis.set_xlabel("Qwen source block")
    axis.set_ylabel("Mean whole-searchlight RSA correlation\n(subject-level 95% CI)")
    axis.set_xticks(layers)
    axis.grid(axis="y", color="#E4E8ED", linewidth=0.8)
    axis.legend(frameon=False, ncol=2)
    figure.suptitle(
        "All-token mean-pooled causal decoder residuals",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(
    paths: ExperimentPaths,
    profile: str,
    subjects: Sequence[int] = tuple(range(1, N_SUBJECTS + 1)),
    *,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
    readout_mode: str = DEFAULT_READOUT_MODE,
    historical_results_root: Path | None = None,
) -> dict:
    """Aggregate samples within subjects and compare matched representations."""
    subjects = validate_subjects(subjects)
    manifest = _group_manifest(paths, profile, prompt_set_key, readout_mode)
    group = manifest["group_name"]
    model_order = manifest["model_order"]
    feature_names = [item["feature"] for item in model_order]
    n_models = len(feature_names)
    subject_scores = np.empty((len(subjects), n_models), dtype=np.float64)
    sample_scores = np.empty((len(subjects), N_SAMPLES, n_models), dtype=np.float64)

    for subject_index, subject in enumerate(subjects):
        centers = _searchlight_centers(paths, subject)
        for sample_index, path in enumerate(_sample_files(paths, group, subject)):
            volumes = np.load(path, allow_pickle=False)
            if volumes.shape[0] != n_models:
                raise ValueError(
                    f"{path} has {volumes.shape[0]} models, expected {n_models}"
                )
            flattened = volumes.reshape(n_models, -1)[:, centers]
            finite_counts = np.isfinite(flattened).sum(axis=1)
            if np.any(finite_counts == 0):
                raise ValueError(f"a model has no finite center values in {path}")
            sample_scores[subject_index, sample_index] = np.nanmean(flattened, axis=1)
        subject_scores[subject_index] = sample_scores[subject_index].mean(axis=0)

    score_rows = []
    for index, feature in enumerate(feature_names):
        mean, low, high = _mean_ci(subject_scores[:, index])
        score_rows.append(
            {
                "feature": feature,
                "mean_correlation": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    historical_subject_scores = None
    historical_provenance = None
    if readout_mode == ALL_TOKEN_MEAN:
        if historical_results_root is None:
            raise ValueError(
                "all_token_mean summary requires --historical-results-root for "
                "the predeclared secondary readout family"
            )
        (
            historical_subject_scores,
            historical_feature_names,
            historical_provenance,
        ) = _load_historical_final_token_scores(
            historical_results_root, profile, subjects
        )
        for index, feature in enumerate(historical_feature_names):
            mean, low, high = _mean_ci(historical_subject_scores[:, index])
            score_rows.append(
                {
                    "feature": f"historical_final_token__{feature}",
                    "mean_correlation": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    score_rows.sort(key=lambda row: row["mean_correlation"], reverse=True)

    comparisons = _comparison_rows(
        subject_scores, feature_names, prompt_set_key, readout_mode
    )
    if readout_mode == ALL_TOKEN_MEAN:
        assert historical_subject_scores is not None
        comparisons.extend(
            _whole_prompt_readout_rows(
                subject_scores,
                feature_names,
                historical_subject_scores,
                historical_feature_names,
            )
        )
    performance_rows = _performance_table_rows(score_rows, comparisons)

    output_dir = paths.reports / run_name(profile, prompt_set_key, readout_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npy(output_dir / "subject_scores.npy", subject_scores)
    atomic_npy(output_dir / "sample_scores.npy", sample_scores)
    if historical_subject_scores is not None:
        atomic_npy(
            output_dir / "historical_final_token_subject_scores.npy",
            historical_subject_scores,
        )
    _write_csv(output_dir / "feature_scores.csv", score_rows)
    _write_csv(output_dir / "comparisons.csv", comparisons)
    _write_csv(output_dir / "performance_table.csv", performance_rows)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "prompt_set": prompt_set_key,
        "group_name": group,
        "independent_unit": "subject",
        "n_subjects": len(subjects),
        "subject_numbers": list(subjects),
        "n_sessions": N_SESSIONS,
        "n_samples_per_subject": N_SAMPLES,
        "scores": score_rows,
        "comparisons": comparisons,
        "artifacts": {
            "subject_scores": str((output_dir / "subject_scores.npy").resolve()),
            "sample_scores": str((output_dir / "sample_scores.npy").resolve()),
            "feature_scores_csv": str((output_dir / "feature_scores.csv").resolve()),
            "comparisons_csv": str((output_dir / "comparisons.csv").resolve()),
            "performance_table_csv": str(
                (output_dir / "performance_table.csv").resolve()
            ),
        },
    }
    if readout_mode != DEFAULT_READOUT_MODE:
        summary["readout_mode"] = readout_mode
    if historical_provenance is not None:
        summary["historical_comparator"] = historical_provenance
        summary["artifacts"]["historical_final_token_subject_scores"] = str(
            (output_dir / "historical_final_token_subject_scores.npy").resolve()
        )
    atomic_json(output_dir / "summary.json", summary)
    report_path = output_dir / "report.html"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(_html_report(summary), encoding="utf-8")
    temporary.replace(report_path)
    summary["artifacts"]["html_report"] = str(report_path.resolve())
    if prompt_set_key == "matched_readout":
        figure_path = output_dir / "matched_readout_layer_summary.png"
        _plot_matched_layer_summary(summary, figure_path)
        summary["artifacts"]["layer_summary_figure"] = str(figure_path.resolve())
    if readout_mode == ALL_TOKEN_MEAN:
        figure_path = output_dir / "whole_prompt_pooling_summary.png"
        _plot_whole_prompt_summary(summary, figure_path)
        summary["artifacts"]["layer_summary_figure"] = str(figure_path.resolve())
    atomic_json(output_dir / "summary.json", summary)
    return summary

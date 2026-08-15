"""Minimal compatibility extensions around pinned ``nsd_visuo_semantics``.

Upstream supplies NSD condition, mask, RDM, searchlight-sphere, and mapping
logic.  This module keeps only the fork interfaces required by this experiment:
streamed beta averaging/correlation and subject-selective projection/plotting.
See ``docs/UPSTREAM_COMPATIBILITY.md`` for the audited API delta.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import N_SESSIONS, ExperimentPaths
from .io_utils import atomic_npy

VOXEL_SIZES = (
    (81, 104, 83),
    (82, 106, 84),
    (81, 106, 82),
    (85, 99, 80),
    (79, 97, 78),
    (85, 113, 83),
    (78, 95, 81),
    (80, 103, 78),
)


def bootstrap_cuda_library_path() -> None:
    """Restart once when pip-provided CUDA libraries need loader discovery."""
    marker = "_JLENS_NSD_CUDA_PATH_BOOTSTRAPPED"
    if os.environ.get(marker) == "1":
        return
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    root = Path(sys.prefix) / "lib" / py_version / "site-packages" / "nvidia"
    library_dirs = sorted(str(path) for path in root.glob("*/lib") if path.is_dir())
    current = [
        item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item
    ]
    missing = [item for item in library_dirs if item not in current]
    if not missing:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(missing + current)
    env[marker] = "1"
    original = getattr(sys, "orig_argv", [sys.executable, *sys.argv])
    os.execve(sys.executable, [sys.executable, *original[1:]], env)


def initialize_tensorflow(*, allow_cpu: bool = False):
    """Initialize TensorFlow with bounded GPU allocation."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and not allow_cpu:
        raise RuntimeError(
            "TensorFlow did not register a GPU; pass --allow-cpu only when intended"
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf


def _session_betas(folder: Path, session: int) -> np.ndarray:
    import nibabel as nib

    path = folder / f"betas_session{session:02d}.nii.gz"
    values = np.asarray(nib.load(path).dataobj, dtype=np.float32).squeeze()
    values /= 300.0
    return values


def condition_column_index(
    conditions_to_average: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map NSD condition IDs to averaged-beta columns in **sorted ID order**.

    This is the alignment contract between the brain and model sides, and it is
    load-bearing. Model RDMs are built from ``subj*_condition_ids.npy``, which
    ``prepare_conditions`` stores sorted and unique and ``_subject_condition_ids``
    re-checks on load. The averaged betas must use that same order, because the
    locked sampling file indexes both with one shared ``choices`` array.

    Ordering by first appearance instead would permute one side against the
    other. Nothing would raise: RSA would simply correlate mismatched conditions
    and report a weak, plausible-looking effect. Returning sorted order is
    therefore a contract, not an implementation detail.

    Returns the sorted unique IDs and a dense ``id -> column`` table whose
    entries are ``-1`` for IDs that are not present.
    """
    lookup = np.unique(np.asarray(conditions_to_average, dtype=np.int64))
    if lookup.size == 0:
        raise ValueError("no conditions to average")
    if lookup[0] < 1:
        raise ValueError("NSD condition IDs are 1-based")
    id_to_column = np.full(int(lookup.max()) + 1, -1, dtype=np.int64)
    id_to_column[lookup] = np.arange(len(lookup))
    return lookup, id_to_column


def _compute_betas_average(
    output: Path,
    nsd_dir: Path,
    subject: str,
    conditions_to_average: np.ndarray,
) -> np.memmap:
    """Stream sessions into a crash-safe averaged-beta memmap."""
    from nsd_visuo_semantics.utils.nsd_get_data_light import read_behavior

    folder = (
        nsd_dir
        / "nsddata_betas"
        / "ppdata"
        / subject
        / "func1pt8mm"
        / "betas_fithrf_GLMdenoise_RR"
    )
    sessions = [
        session
        for session in range(1, N_SESSIONS + 1)
        if (folder / f"betas_session{session:02d}.nii.gz").exists()
    ]
    if not sessions:
        raise FileNotFoundError(f"no session beta files found in {folder}")

    lookup, id_to_column = condition_column_index(conditions_to_average)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial.npy")
    counts_path = output.with_name(f".{output.name}.counts.npy")
    partial.unlink(missing_ok=True)
    counts_path.unlink(missing_ok=True)
    sums = counts = None

    try:
        for session in sessions:
            behavior = read_behavior(
                str(nsd_dir), subject=subject, session_index=session
            )
            presented = np.asarray(behavior["73KID"], dtype=np.int64)
            targets = np.full(presented.shape, -1, dtype=np.int64)
            in_range = presented < len(id_to_column)
            targets[in_range] = id_to_column[presented[in_range]]
            keep = np.flatnonzero(targets >= 0)
            if not len(keep):
                continue

            session_betas = _session_betas(folder, session)
            mean = session_betas.mean(axis=-1, keepdims=True)
            std = session_betas.std(axis=-1, keepdims=True)
            valid = np.isfinite(std[..., 0]) & (std[..., 0] != 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                session_betas = (session_betas - mean) / std
            session_betas[~valid] = 0

            if sums is None:
                shape = (*session_betas.shape[:-1], len(lookup))
                sums = np.lib.format.open_memmap(
                    partial, mode="w+", dtype=np.float32, shape=shape
                )
                counts = np.lib.format.open_memmap(
                    counts_path, mode="w+", dtype=np.uint8, shape=shape
                )
            assert counts is not None
            selected_targets = targets[keep]
            for target in np.unique(selected_targets):
                trials = keep[selected_targets == target]
                sums[..., target] += session_betas[..., trials].sum(axis=-1)
                counts[..., target] += valid.astype(np.uint8) * np.uint8(len(trials))
            sums.flush()
            counts.flush()

        if sums is None or counts is None:
            raise RuntimeError(f"no three-repeat trials found for {subject}")
        for start in range(sums.shape[0]):
            count_row = counts[start]
            with np.errstate(invalid="ignore", divide="ignore"):
                sums[start] /= count_row
            sums[start][count_row == 0] = np.nan
        sums.flush()
        del counts
        counts_path.unlink(missing_ok=True)
        os.replace(partial, output)
        return np.load(output, mmap_mode="r")
    finally:
        counts_path.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)


def _load_or_compute_betas(
    output: Path,
    nsd_dir: Path,
    subject: str,
    conditions_to_average: np.ndarray,
) -> np.ndarray:
    if output.exists():
        return np.load(output, mmap_mode="r")
    return _compute_betas_average(output, nsd_dir, subject, conditions_to_average)


def _tf_searchlight_corr(
    betas: np.ndarray,
    sphere_indices,
    sorted_indices,
    model_rdms: np.ndarray,
    batch_size: int = 50,
) -> np.ndarray:
    """Correlate each RDM batch before transfer to CPU."""
    import tensorflow as tf
    from nsd_visuo_semantics.utils.tf_utils import chunking, compute_rdm_batch

    n_conditions = betas.shape[-1]
    flat = tf.convert_to_tensor(betas.reshape((-1, n_conditions)), dtype=tf.float32)
    models = tf.convert_to_tensor(np.asarray(model_rdms), dtype=tf.float32)
    models -= tf.reduce_mean(models, axis=1, keepdims=True)
    models /= tf.sqrt(tf.einsum("ij,ij->i", models, models))[:, None]
    correlations = []
    for same_size in sorted_indices:
        for chunk in chunking(same_size, batch_size):
            indices = [sphere_indices[int(index)] for index in chunk]
            searchlights = tf.gather(flat, np.stack(indices))
            patterns = tf.transpose(searchlights, perm=[0, 2, 1])
            brain_rdms = compute_rdm_batch(patterns)
            brain_rdms -= tf.reduce_mean(brain_rdms, axis=1, keepdims=True)
            brain_rdms /= tf.sqrt(tf.einsum("ij,ij->i", brain_rdms, brain_rdms))[
                :, None
            ]
            correlations.append(np.asarray(tf.einsum("ik,jk->ij", brain_rdms, models)))
    return np.vstack(correlations)


def run_searchlight(
    paths: ExperimentPaths,
    group: str,
    subject: int,
    *,
    allow_cpu: bool = False,
    max_samples: int | None = None,
) -> None:
    """Run the fork-equivalent, memory-safe grouped searchlight."""
    paths.require("nsd_dir", "mpnet_base")
    assert paths.nsd_dir is not None
    from nsd_visuo_semantics.searchlight_analyses.searchlight import RSASearchLight
    from nsd_visuo_semantics.utils.batch_gen import BatchGen
    from nsd_visuo_semantics.utils.nsd_get_data_light import (
        get_masks,
        get_model_rdms,
        get_subject_conditions,
    )
    from nsd_visuo_semantics.utils.tf_utils import chunking, sort_spheres

    bootstrap_cuda_library_path()
    initialize_tensorflow(allow_cpu=allow_cpu)
    subj = f"subj{subject:02d}"
    model_dir = paths.searchlight_base / "serialised_models_correlation" / group
    model_rdms, _ = get_model_rdms(str(model_dir), subj, filt=group)
    mask = get_masks(str(paths.nsd_dir), subj, "func1pt8mm")

    precomputed = paths.mpnet_precomputed / subj
    precomputed.mkdir(parents=True, exist_ok=True)
    indices_path = precomputed / f"{subj}-func1pt8mm-6rad-searchlight_indices.npy"
    centers_path = precomputed / f"{subj}-func1pt8mm-6rad-searchlight_centers.npy"
    if indices_path.exists() and centers_path.exists():
        with indices_path.open("rb") as handle:
            sphere_indices = pickle.load(handle)
        with centers_path.open("rb") as handle:
            centers = pickle.load(handle)
    else:
        searchlight = RSASearchLight(mask, radius=6, thr=0.5, verbose=True)
        sphere_indices, centers = searchlight.allIndices, searchlight.centerIndices
        with indices_path.open("wb") as handle:
            pickle.dump(sphere_indices, handle)
        with centers_path.open("wb") as handle:
            pickle.dump(centers, handle)
    sorted_indices = sort_spheres(sphere_indices)
    rdm_order = np.hstack(
        [
            centers[chunk.astype(np.int32)]
            for same_size in sorted_indices
            for chunk in chunking(same_size, 50)
        ]
    ).astype(int)

    conditions, sampled_conditions, subject_conditions = get_subject_conditions(
        str(paths.nsd_dir), subj, N_SESSIONS, keep_only_3repeats=True
    )
    del conditions
    n_conditions = len(subject_conditions)
    selector = BatchGen(model_rdms, range(n_conditions))
    sampling_path = (
        paths.searchlight_base
        / "searchlight_respectedsampling_correlation"
        / subj
        / "saved_sampling"
        / f"{subj}_nsd-allsubstim_sampling.npy"
    )
    if not sampling_path.exists():
        raise FileNotFoundError(f"locked sampling file not found: {sampling_path}")
    samples = np.load(sampling_path, allow_pickle=False)
    if max_samples is not None:
        samples = samples[:max_samples]

    betas_path = (
        paths.mpnet_precomputed / "betas" / f"{subj}_betas_average_func1pt8mm.npy"
    )
    betas = _load_or_compute_betas(betas_path, paths.nsd_dir, subj, sampled_conditions)
    output_dir = (
        paths.searchlight_base
        / "searchlight_respectedsampling_correlation"
        / subj
        / group
        / "corr_vols_correlation"
    )
    for index, choices in enumerate(samples):
        output = output_dir / f"{subj}_nsd-{group}_func1pt8mm_sample-{index}.npy"
        if output.exists():
            continue
        sampled_betas = np.asarray(betas[..., choices], dtype=np.float32)
        sampled_rdms = np.asarray(selector.index_rdms(choices), dtype=np.float32)
        maps = _tf_searchlight_corr(
            sampled_betas, sphere_indices, sorted_indices, sampled_rdms
        )
        volumes = np.zeros((len(maps.T), *mask.shape), dtype=np.float64)
        for model_index, brain_map in enumerate(maps.T):
            flat = np.zeros(mask.size)
            flat[rdm_order] = brain_map
            volumes[model_index] = flat.reshape(mask.shape)
        atomic_npy(output, volumes)


def project_to_fsaverage(
    paths: ExperimentPaths, group: str, subjects: Sequence[int]
) -> None:
    """Upstream projection with an explicit subject subset."""
    paths.require("nsd_dir")
    assert paths.nsd_dir is not None
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    from nsdcode.nsd_mapdata import NSDmapdata

    nsd = NSDmapdata(str(paths.nsd_dir))
    fs_dir = Path(nsd.base_dir) / "nsddata" / "freesurfer" / "fsaverage"
    for subject in subjects:
        subj = f"subj{subject:02d}"
        input_dir = (
            paths.searchlight_base
            / "searchlight_respectedsampling_correlation"
            / subj
            / group
            / "corr_vols_correlation"
        )
        samples = sorted(input_dir.glob("*sample*.npy"))
        if not samples:
            raise FileNotFoundError(f"no searchlight samples found in {input_dir}")
        first = np.load(samples[0], allow_pickle=False, mmap_mode="r")
        output_dir = input_dir.parent / f"{group}_correlation_fsaverage"
        output_dir.mkdir(parents=True, exist_ok=True)
        for model_index in range(first.shape[0]):
            volumes = np.stack(
                [np.load(path, allow_pickle=False)[model_index] for path in samples]
            )
            # Some searchlight centres have an undefined correlation and come
            # back NaN; prevalence is strongly subject-dependent (none for
            # subjects 1/4/7, ~11% for subject 8). nanmean excludes them rather
            # than propagating, and the NaN pattern is identical across models,
            # so the paired J-vs-raw contrast stays averaged over the same
            # centres. A "Mean of empty slice" RuntimeWarning here is expected
            # for subjects that have all-NaN centres and is not a fault.
            # scripts/audit_searchlight_coverage.py checks the invariant.
            mean = np.nanmean(volumes, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                t_value = mean / (np.nanstd(volumes, axis=0) / np.sqrt(len(samples)))
            for suffix, source in (("", mean), ("-tvals", t_value)):
                destinations = {
                    hemi: output_dir
                    / f"{hemi}.{subj}-model-{model_index + 1}-surf{suffix}.npy"
                    for hemi in ("lh", "rh")
                }
                if all(path.exists() for path in destinations.values()):
                    continue
                for hemi in ("lh", "rh"):
                    depths = [
                        nsd.fit(
                            subject,
                            "func1pt8",
                            f"{hemi}.layerB{layer}",
                            source,
                            "cubic",
                            badval=0,
                        )
                        for layer in range(1, 4)
                    ]
                    native = np.nanmean(np.stack(depths), axis=0)
                    projected = nsd.fit(
                        subject,
                        f"{hemi}.white",
                        "fsaverage",
                        native,
                        interptype=None,
                        badval=0,
                        fsdir=str(fs_dir),
                    )
                    atomic_npy(destinations[hemi], projected)


def plot_brain(
    values: np.ndarray,
    name: str,
    output_dir: Path,
    *,
    nsd_dir: Path | None,
    roi_overlay: str | None,
) -> None:
    """Render the current project's individual fsaverage map contract."""
    import cortex
    import matplotlib.pyplot as plt

    configured = Path(cortex.db.filestore)
    if not (configured / "fsaverage" / "surfaces" / "wm_lh.gii").exists():
        fallback = Path(sys.prefix) / "share" / "pycortex" / "db"
        if not (fallback / "fsaverage" / "surfaces" / "wm_lh.gii").exists():
            raise FileNotFoundError(
                "pycortex fsaverage surfaces are unavailable in its configured "
                "filestore and the active environment"
            )
        cortex.db.filestore = str(fallback)
        cortex.db._subjects = None
        cortex.options.config.set("basic", "filestore", str(fallback))

    bound = np.nanmax(np.abs(values))
    vertex = cortex.dataset.Vertex(
        values, "fsaverage", cmap="RdBu_r", vmin=-bound, vmax=bound
    )
    figure = cortex.quickflat.make_figure(
        vertex, height=480, with_colorbar=True, with_rois=False
    )
    if roi_overlay is not None:
        if nsd_dir is None:
            raise ValueError("NSD path is required for a named ROI overlay")
        from cortex.quickflat.utils import make_flatmap_image
        from nsd_visuo_semantics.utils.nsd_get_data_light import get_rois

        roi_dir = nsd_dir / "nsddata" / "freesurfer" / "fsaverage" / "label"
        roi_values, _ = get_rois(roi_overlay, str(roi_dir))
        ax = figure.axes[0]
        for label in np.unique(roi_values[np.isfinite(roi_values)]):
            if label == 0:
                continue
            mask = cortex.dataset.Vertex(
                (roi_values == label).astype(np.float32),
                "fsaverage",
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            image, extent = make_flatmap_image(mask, height=480)
            ax.contour(
                np.nan_to_num(image),
                levels=[0.5],
                colors=["black"],
                linewidths=0.8,
                origin="upper",
                extent=extent,
            )
    figure.suptitle(f"{name} - max abs val: {bound:.2f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

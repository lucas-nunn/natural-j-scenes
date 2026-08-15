"""Strict image-only WikiText Jacobian Lens transfer pilot.

This module is intentionally separate from the caption extraction pipeline.  It
uses only Qwen's vision boundary/placeholder tokens, pools only decoder states
at projected image-token positions, and applies the released WikiText average
Jacobians without fitting or modification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist

from .io_utils import atomic_json, atomic_npy, atomic_npz, sha256_file, stable_hash

PILOT_LAYERS = (8, 16, 23, 30)
FINAL_BLOCK = 31
SUBJECT = 1
SAMPLE_ROW = 0
SAMPLE_SIZE = 100
N_SESSIONS = 10
MINIMAL_IMAGE_SEQUENCE = "<|vision_start|><|image_pad|><|vision_end|>"
FEATURE_ORDER = tuple(
    name for layer in PILOT_LAYERS for name in (f"l{layer:02d}_raw", f"l{layer:02d}_j")
) + ("final_raw",)
PILOT_LABEL = "image-only, decoder-residual, out-of-distribution transfer"


def strict_image_token_mask(
    input_ids: np.ndarray,
    mm_token_type_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    vision_start_token_id: int,
    image_token_id: int,
    vision_end_token_id: int,
    expected_image_tokens: int,
) -> np.ndarray:
    """Return the projected-image-token mask or fail on any extra token.

    The only accepted attended sequence is one vision-start token, one
    contiguous run of image placeholders, and one vision-end token.  The
    multimodal type mask must independently identify exactly that placeholder
    run as image modality 1.
    """
    ids = np.asarray(input_ids)
    types = np.asarray(mm_token_type_ids)
    attention = np.asarray(attention_mask)
    if ids.ndim != 1 or types.shape != ids.shape or attention.shape != ids.shape:
        raise ValueError("token IDs, modality types, and attention must be 1-D peers")
    if expected_image_tokens <= 0:
        raise ValueError("expected image-token count must be positive")
    active = attention.astype(bool)
    if int(active.sum()) != expected_image_tokens + 2:
        raise ValueError("minimal image sequence has an unexpected attended length")
    active_ids = ids[active]
    expected_ids = np.asarray(
        [vision_start_token_id]
        + [image_token_id] * expected_image_tokens
        + [vision_end_token_id],
        dtype=ids.dtype,
    )
    if not np.array_equal(active_ids, expected_ids):
        raise ValueError(
            "attended sequence is not exactly vision_start + image tokens + vision_end"
        )
    active_types = types[active]
    expected_types = np.asarray(
        [0] + [1] * expected_image_tokens + [0], dtype=types.dtype
    )
    if not np.array_equal(active_types, expected_types):
        raise ValueError("multimodal type IDs do not isolate the image-token run")
    id_mask = (ids == image_token_id) & active
    type_mask = (types == 1) & active
    if not np.array_equal(id_mask, type_mask):
        raise ValueError("image-token ID mask disagrees with modality-type mask")
    if int(id_mask.sum()) != expected_image_tokens:
        raise ValueError("projected image-token count disagrees with image grid")
    return id_mask


def mean_pool_image_tokens(hidden: np.ndarray, image_mask: np.ndarray) -> np.ndarray:
    """Mean-pool exactly the masked decoder residuals in float32."""
    values = np.asarray(hidden)
    mask = np.asarray(image_mask, dtype=bool)
    if values.ndim != 2 or mask.ndim != 1 or values.shape[0] != mask.shape[0]:
        raise ValueError("hidden must be [tokens, width] with a peer token mask")
    if not mask.any():
        raise ValueError("cannot pool an empty image-token mask")
    patches = np.asarray(values[mask], dtype=np.float32)
    if not np.isfinite(patches).all():
        raise ValueError("image-token residuals contain non-finite values")
    return patches.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)


def apply_j_and_check_linearity(
    image_token_residuals: np.ndarray,
    jacobian: np.ndarray,
    *,
    rtol: float = 5e-4,
    atol: float = 5e-4,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply J to a pooled vector and verify patchwise/poolwise equivalence."""
    patches = np.asarray(image_token_residuals, dtype=np.float32)
    matrix = np.asarray(jacobian, dtype=np.float32)
    if patches.ndim != 2 or matrix.shape != (patches.shape[1], patches.shape[1]):
        raise ValueError("J must be square and match the residual width")
    if not np.isfinite(patches).all() or not np.isfinite(matrix).all():
        raise ValueError("linearity inputs must be finite")
    raw_mean = patches.mean(axis=0, dtype=np.float32)
    projected_after_pool = raw_mean @ matrix.T
    projected_before_pool = (patches @ matrix.T).mean(axis=0, dtype=np.float32)
    difference = np.abs(projected_after_pool - projected_before_pool)
    max_abs = float(difference.max(initial=0.0))
    scale = float(np.abs(projected_after_pool).max(initial=0.0))
    tolerance = float(atol + rtol * scale)
    if not np.allclose(
        projected_after_pool,
        projected_before_pool,
        rtol=rtol,
        atol=atol,
    ):
        raise RuntimeError(
            f"J/mean linearity check failed: max_abs={max_abs:.6g}, "
            f"tolerance_bound={tolerance:.6g}"
        )
    return (
        raw_mean.astype(np.float32),
        projected_after_pool.astype(np.float32),
        {
            "max_abs_error": max_abs,
            "max_abs_reference": scale,
            "rtol": float(rtol),
            "atol": float(atol),
            "tolerance_bound": tolerance,
        },
    )


def numpy_searchlight_corr(
    betas: np.ndarray,
    sphere_indices: list[np.ndarray],
    model_rdms: np.ndarray,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute the established Pearson RSA on CPU without TensorFlow/UCX.

    Results are returned in the original sphere/center order. Spheres are
    grouped by voxel count only to permit batched matrix operations.
    """
    values = np.asarray(betas, dtype=np.float32)
    models = np.asarray(model_rdms, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != SAMPLE_SIZE:
        raise ValueError("betas must be a 4-D volume ending in 100 conditions")
    if models.shape != (len(FEATURE_ORDER), 4950):
        raise ValueError("model RDMs have an unexpected shape")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    models = models - models.mean(axis=1, keepdims=True, dtype=np.float32)
    model_norms = np.sqrt(np.einsum("ij,ij->i", models, models))
    if not np.isfinite(models).all() or np.any(model_norms == 0):
        raise ValueError("model RDMs cannot be centered and normalized")
    models = models / model_norms[:, None]

    flat = values.reshape((-1, SAMPLE_SIZE))
    correlations = np.empty((len(sphere_indices), len(FEATURE_ORDER)), np.float32)
    sizes = np.asarray([len(indices) for indices in sphere_indices])
    triangle = np.triu_indices(SAMPLE_SIZE, k=1)
    for size in np.unique(sizes):
        same_size = np.flatnonzero(sizes == size)
        for start in range(0, len(same_size), batch_size):
            sphere_ids = same_size[start : start + batch_size]
            indices = np.stack([sphere_indices[int(index)] for index in sphere_ids])
            patterns = np.transpose(flat[indices], (0, 2, 1))
            if not np.isfinite(patterns).all():
                raise ValueError("searchlight contains non-finite beta values")
            patterns -= patterns.mean(axis=2, keepdims=True, dtype=np.float32)
            pattern_norms = np.sqrt(np.einsum("bij,bij->bi", patterns, patterns))
            if np.any(pattern_norms == 0) or not np.isfinite(pattern_norms).all():
                raise ValueError("searchlight contains a constant condition pattern")
            patterns /= pattern_norms[:, :, None]
            square_rdms = 1.0 - np.einsum("bik,bjk->bij", patterns, patterns)
            brain_rdms = square_rdms[:, triangle[0], triangle[1]]
            brain_rdms -= brain_rdms.mean(axis=1, keepdims=True, dtype=np.float32)
            brain_norms = np.sqrt(np.einsum("ij,ij->i", brain_rdms, brain_rdms))
            if np.any(brain_norms == 0) or not np.isfinite(brain_norms).all():
                raise ValueError("brain RDM cannot be centered and normalized")
            brain_rdms /= brain_norms[:, None]
            correlations[sphere_ids] = brain_rdms @ models.T
    if not np.isfinite(correlations).all():
        raise RuntimeError("CPU searchlight produced non-finite correlations")
    return correlations


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_file_record(name: str, record: dict[str, Any]) -> Path:
    """Validate one manifest file record and return its resolved path."""
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise RuntimeError(f"recorded file is missing: {name}: {path}")
    size = path.stat().st_size
    if size != record["size_bytes"]:
        raise RuntimeError(
            f"file size mismatch: {name}: {size} != {record['size_bytes']}"
        )
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise RuntimeError(f"file hash mismatch: {name}: {path}")
    return path


def _tree_records(root: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".git" in path.parts or ".cache" in path.parts:
            continue
        records[str(path.relative_to(root))] = _file_record(path)
    return records


def _git_record(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).splitlines()
    return {"path": str(root.resolve()), "commit": commit, "dirty_paths": status}


def _load_subject_condition_ids(responses_tsv: Path) -> np.ndarray:
    presented = []
    with responses_tsv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if 1 <= int(row["SESSION"]) <= N_SESSIONS:
                presented.append(int(row["73KID"]))
    counts = Counter(presented)
    ids = np.asarray(sorted(key for key, count in counts.items() if count == 3))
    if ids.shape != (835,) or ids.dtype.kind not in "iu":
        raise ValueError(
            f"unexpected subject-1 three-repeat condition set: {ids.shape}"
        )
    return ids.astype(np.int64)


def _selected_sample(
    responses_tsv: Path, sampling_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_ids = _load_subject_condition_ids(responses_tsv)
    samples = np.load(sampling_path, allow_pickle=False)
    if samples.ndim != 2 or samples.shape[1] != SAMPLE_SIZE:
        raise ValueError(f"matched sampling has unexpected shape {samples.shape}")
    choices = np.asarray(samples[SAMPLE_ROW], dtype=np.int64)
    if len(np.unique(choices)) != SAMPLE_SIZE:
        raise ValueError("selected matched-sampling row is not 100 unique indices")
    if choices.min() < 0 or choices.max() >= len(subject_ids):
        raise ValueError("matched-sampling row indexes outside subject conditions")
    return subject_ids, choices, subject_ids[choices]


def _stimulus_metadata(stim_info_csv: Path, condition_ids: np.ndarray) -> list[dict]:
    wanted = set(int(item) for item in condition_ids)
    by_condition: dict[int, dict] = {}
    with stim_info_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition_id = int(row["nsdId"]) + 1
            if condition_id in wanted:
                by_condition[condition_id] = {
                    "condition_id_73k": condition_id,
                    "nsd_index_zero_based": int(row["nsdId"]),
                    "coco_id": int(row["cocoId"]),
                    "coco_split": row["cocoSplit"],
                }
    if set(by_condition) != wanted:
        raise ValueError("stimulus metadata is missing selected conditions")
    return [by_condition[int(item)] for item in condition_ids]


def _validate_feature_matrix(values: np.ndarray, name: str) -> None:
    if values.shape != (SAMPLE_SIZE, 2560) or values.dtype != np.float32:
        raise ValueError(
            f"{name} has invalid shape/dtype: {values.shape}/{values.dtype}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} has a constant row; correlation distance undefined")


def _update_manifest(output_root: Path, **sections: Any) -> dict:
    path = output_root / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update(sections)
    manifest["updated_at"] = _now()
    atomic_json(path, manifest)
    return manifest


def _processor_config(processor) -> dict[str, Any]:
    config = processor.image_processor.to_dict()
    keep = {
        "do_convert_rgb",
        "do_normalize",
        "do_rescale",
        "do_resize",
        "image_mean",
        "image_std",
        "merge_size",
        "patch_size",
        "resample",
        "rescale_factor",
        "size",
        "temporal_patch_size",
    }
    return {key: config[key] for key in sorted(keep & set(config))}


def extract(args: argparse.Namespace) -> None:
    """Extract raw/J image-token means from the predeclared 100 images."""
    import h5py
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    args.output_root.mkdir(parents=True, exist_ok=True)
    subject_ids, sample_indices, condition_ids = _selected_sample(
        args.responses_tsv, args.sampling
    )
    image_metadata = _stimulus_metadata(args.stim_info_csv, condition_ids)

    if str(args.jlens_checkout) not in sys.path:
        sys.path.insert(0, str(args.jlens_checkout))
    import jlens

    lens = jlens.JacobianLens.load(str(args.lens))
    if lens.d_model != 2560 or not set(PILOT_LAYERS).issubset(lens.source_layers):
        raise ValueError(
            f"released lens mismatch: d_model={lens.d_model}, layers={lens.source_layers}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    minimal = (
        processor.vision_start_token
        + processor.image_token
        + processor.vision_end_token
    )
    if minimal != MINIMAL_IMAGE_SEQUENCE:
        raise RuntimeError(f"processor minimal sequence changed: {minimal!r}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    wrapped = jlens.from_hf(model, processor.tokenizer, force_bos=False)
    expected_layers = model.model.language_model.layers
    if wrapped.layers is not expected_layers or len(wrapped.layers) != FINAL_BLOCK + 1:
        raise RuntimeError(
            "Anthropic adapter did not resolve Qwen language decoder blocks"
        )
    lens_matrices = {
        layer: lens.jacobians[layer].to(device=device, dtype=torch.float32)
        for layer in PILOT_LAYERS
    }

    features = {
        name: np.empty((SAMPLE_SIZE, 2560), dtype=np.float32) for name in FEATURE_ORDER
    }
    token_counts = np.empty(SAMPLE_SIZE, dtype=np.int64)
    grids = np.empty((SAMPLE_SIZE, 3), dtype=np.int64)
    image_hashes = []
    token_evidence = None
    hook_checks = None
    linearity_checks = {}
    record_at = (*PILOT_LAYERS, FINAL_BLOCK)

    with h5py.File(args.stimuli, "r") as handle:
        dataset = handle["imgBrick"]
        if dataset.shape != (73000, 425, 425, 3) or dataset.dtype != np.uint8:
            raise ValueError(
                f"unexpected NSD stimulus dataset: {dataset.shape}/{dataset.dtype}"
            )
        for row, condition_id in enumerate(condition_ids):
            pixels = np.asarray(dataset[int(condition_id) - 1])
            image_hashes.append(hashlib.sha256(pixels.tobytes(order="C")).hexdigest())
            encoded = processor(
                images=Image.fromarray(pixels),
                text=minimal,
                return_tensors="pt",
                return_mm_token_type_ids=True,
            )
            grid = np.asarray(encoded["image_grid_thw"][0], dtype=np.int64)
            expected_count = int(
                np.prod(grid) // processor.image_processor.merge_size**2
            )
            image_mask = strict_image_token_mask(
                encoded["input_ids"][0].numpy(),
                encoded["mm_token_type_ids"][0].numpy(),
                encoded["attention_mask"][0].numpy(),
                vision_start_token_id=processor.vision_start_token_id,
                image_token_id=processor.image_token_id,
                vision_end_token_id=processor.vision_end_token_id,
                expected_image_tokens=expected_count,
            )
            token_counts[row] = expected_count
            grids[row] = grid
            if token_evidence is None:
                ids = encoded["input_ids"][0].numpy()
                types = encoded["mm_token_type_ids"][0].numpy()
                token_evidence = {
                    "condition_id_73k": int(condition_id),
                    "literal_sequence": minimal,
                    "sequence_length": int(len(ids)),
                    "input_ids_rle": [
                        [int(ids[0]), 1],
                        [int(processor.image_token_id), expected_count],
                        [int(ids[-1]), 1],
                    ],
                    "tokens_rle": [
                        [processor.vision_start_token, 1],
                        [processor.image_token, expected_count],
                        [processor.vision_end_token, 1],
                    ],
                    "mm_token_type_ids_rle": [[0, 1], [1, expected_count], [0, 1]],
                    "attention_all_ones": bool(
                        np.all(encoded["attention_mask"][0].numpy() == 1)
                    ),
                    "image_id_mask_equals_mm_type_1": bool(
                        np.array_equal(ids == processor.image_token_id, types == 1)
                    ),
                    "image_grid_thw": grid.tolist(),
                    "spatial_merge_size": int(processor.image_processor.merge_size),
                    "expected_and_observed_image_tokens": expected_count,
                }

            model_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key
                in {
                    "input_ids",
                    "attention_mask",
                    "mm_token_type_ids",
                    "pixel_values",
                    "image_grid_thw",
                }
            }
            validate_hooks = row == 0
            with (
                torch.inference_mode(),
                jlens.ActivationRecorder(wrapped.layers, at=record_at) as recorder,
            ):
                outputs = model.model(
                    **model_inputs,
                    use_cache=False,
                    output_hidden_states=validate_hooks,
                    return_dict=True,
                )
            torch_mask = torch.as_tensor(image_mask, device=device)
            if validate_hooks:
                hidden_states = outputs.hidden_states
                if hidden_states is None or len(hidden_states) != FINAL_BLOCK + 2:
                    raise RuntimeError(
                        "Qwen did not expose expected block-indexed hidden states"
                    )
                hook_checks = {}
                for layer in PILOT_LAYERS:
                    hooked = recorder.activations[layer][0, torch_mask].float()
                    reference = hidden_states[layer + 1][0, torch_mask].float()
                    max_abs = float((hooked - reference).abs().max().cpu())
                    if max_abs != 0.0:
                        raise RuntimeError(
                            f"hook/HF hidden-state mismatch at block {layer}: {max_abs}"
                        )
                    hook_checks[str(layer)] = {
                        "block_index_zero_based": layer,
                        "hf_hidden_states_index": layer + 1,
                        "max_abs_error": max_abs,
                    }
            for layer in PILOT_LAYERS:
                patches = recorder.activations[layer][0, torch_mask].float()
                raw = patches.mean(dim=0)
                projected = raw @ lens_matrices[layer].T
                if row == 0:
                    patchwise = (patches @ lens_matrices[layer].T).mean(dim=0)
                    difference = (projected - patchwise).abs()
                    max_abs = float(difference.max().cpu())
                    scale = float(projected.abs().max().cpu())
                    rtol = atol = 5e-4
                    tolerance = atol + rtol * scale
                    if not torch.allclose(projected, patchwise, rtol=rtol, atol=atol):
                        raise RuntimeError(
                            f"actual J/mean linearity failed at layer {layer}: "
                            f"{max_abs} > {tolerance}"
                        )
                    linearity_checks[str(layer)] = {
                        "condition_id_73k": int(condition_id),
                        "formula_left": "mean_patch(J @ h_patch)",
                        "formula_right": "J @ mean_patch(h_patch)",
                        "max_abs_error": max_abs,
                        "max_abs_reference": scale,
                        "rtol": rtol,
                        "atol": atol,
                        "tolerance_bound": tolerance,
                    }
                features[f"l{layer:02d}_raw"][row] = raw.cpu().numpy()
                features[f"l{layer:02d}_j"][row] = projected.cpu().numpy()
            final_patches = recorder.activations[FINAL_BLOCK][0, torch_mask].float()
            features["final_raw"][row] = final_patches.mean(dim=0).cpu().numpy()
            if (row + 1) % 10 == 0:
                print(f"extracted {row + 1}/{SAMPLE_SIZE}", flush=True)
            del encoded, model_inputs, outputs, recorder

    for name, values in features.items():
        _validate_feature_matrix(values, name)
    feature_path = args.output_root / "features.npz"
    atomic_npz(
        feature_path,
        condition_ids=condition_ids,
        sample_indices=sample_indices,
        image_token_counts=token_counts,
        image_grid_thw=grids,
        **features,
    )

    model_files = _tree_records(args.model)
    model_source_hash = stable_hash(
        {name: record["sha256"] for name, record in model_files.items()}
    )
    transformer_root = Path(__import__("transformers").__file__).parent
    implementation_files = [
        transformer_root / "models/qwen3_5/modeling_qwen3_5.py",
        transformer_root / "models/qwen3_vl/processing_qwen3_vl.py",
        transformer_root / "models/qwen2_vl/image_processing_qwen2_vl.py",
    ]
    repository_root = Path(__file__).resolve().parents[2]
    experiment_files = [
        Path(__file__).resolve(),
        repository_root / "scripts/run_image_only_wikitext_pilot.py",
    ]
    input_files = {
        "lens_checkpoint": _file_record(args.lens),
        "stimuli_hdf5": _file_record(args.stimuli),
        "stimulus_metadata_csv": _file_record(args.stim_info_csv),
        "subject_responses_tsv": _file_record(args.responses_tsv),
        "matched_sampling": _file_record(args.sampling),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "image_only_wikitext_pilot_subj01_sample00",
        "created_at": _now(),
        "status": "features_complete",
        "scientific_scope": {
            "description": "OOD transfer of released WikiText average J matrices to Qwen decoder image-token residuals",
            "image_only": True,
            "vision_tower_lens": False,
            "semantic_text_present": False,
            "chat_template_used": False,
            "generation_used": False,
            "lens_fitted_or_modified": False,
            "subject": SUBJECT,
            "sessions": N_SESSIONS,
            "sample_row_zero_based": SAMPLE_ROW,
            "sample_size": SAMPLE_SIZE,
            "predeclared_layers": list(PILOT_LAYERS),
            "final_control_block": FINAL_BLOCK,
            "rdm_metric": "correlation distance",
            "inference": "descriptive only; no inferential p-values",
        },
        "minimal_image_input": token_evidence,
        "layer_semantics": {
            "hook": "Anthropic jlens.ActivationRecorder",
            "module": "Qwen3_5ForConditionalGeneration.model.language_model.layers",
            "indexing": "zero-based decoder block output; HF hidden_states index = block + 1",
            "selected_hook_checks": hook_checks,
            "final_control": "un-normalized output of decoder block 31, before final RMSNorm",
        },
        "pooling": {
            "rule": "deterministic arithmetic mean over only projected image-token decoder residuals",
            "raw_and_j_same_token_mask": True,
            "accumulation_dtype": "float32",
            "linearity_checks": linearity_checks,
        },
        "representations": {
            "feature_order": list(FEATURE_ORDER),
            "shape_each": [SAMPLE_SIZE, 2560],
            "dtype": "float32",
            "normalization": "none",
        },
        "condition_ids": condition_ids.tolist(),
        "sample_indices_into_sorted_subject_conditions": sample_indices.tolist(),
        "subject_condition_count": int(len(subject_ids)),
        "images": [
            {
                **meta,
                "pixel_sha256": digest,
                "image_token_count": int(count),
                "image_grid_thw": grid.tolist(),
            }
            for meta, digest, count, grid in zip(
                image_metadata, image_hashes, token_counts, grids, strict=True
            )
        ],
        "image_preprocessing": _processor_config(processor),
        "software": {
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "experiment_source": {
                path.name: _file_record(path) for path in experiment_files
            },
            "experiment_repository": _git_record(repository_root),
            "jlens_source": _git_record(args.jlens_checkout),
            "transformers_implementation_files": {
                path.name: _file_record(path) for path in implementation_files
            },
        },
        "model": {
            "path": str(args.model.resolve()),
            "composite_sha256": model_source_hash,
            "files": model_files,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(device),
            "d_model": 2560,
            "decoder_blocks": FINAL_BLOCK + 1,
        },
        "lens": {
            "path": str(args.lens.resolve()),
            "sha256": input_files["lens_checkpoint"]["sha256"],
            "source_layers": lens.source_layers,
            "matrix_dtype": "float32",
            "dataset": "Salesforce WikiText",
            "released_checkpoint_unmodified": True,
        },
        "input_files": input_files,
        "artifacts": {"features": _file_record(feature_path)},
        "visualization": {
            "generated": False,
            "reason": "numerical pilot retained; existing surface projection aggregates repeated samples and is not a clean single-row visualization path",
        },
    }
    _update_manifest(args.output_root, **manifest)
    print(f"wrote {feature_path}")


def rdm_searchlight(args: argparse.Namespace) -> None:
    """Build RDMs and run the subject-1 searchlight on the CPU-safe path."""
    import nibabel as nib

    manifest_path = args.output_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run extract before rdm-searchlight")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "features_complete":
        raise RuntimeError(
            "rdm-searchlight requires a features_complete manifest; "
            f"found {manifest.get('status')!r}"
        )
    _validate_file_record("features", manifest["artifacts"]["features"])
    feature_path = args.output_root / "features.npz"
    with np.load(feature_path, allow_pickle=False) as payload:
        if not np.array_equal(payload["condition_ids"], manifest["condition_ids"]):
            raise ValueError("feature condition IDs disagree with manifest")
        features = {
            name: np.asarray(payload[name], dtype=np.float32) for name in FEATURE_ORDER
        }
    rdms = {}
    for name, values in features.items():
        _validate_feature_matrix(values, name)
        rdm = pdist(values, metric="correlation").astype(np.float32)
        if rdm.shape != (4950,) or not np.isfinite(rdm).all():
            raise ValueError(f"invalid 100-image correlation-distance RDM: {name}")
        rdms[name] = rdm
    rdm_path = args.output_root / "rdms.npz"
    atomic_npz(rdm_path, **rdms)

    mask = np.asarray(nib.load(args.brain_mask).dataobj)
    with args.searchlight_indices.open("rb") as handle:
        sphere_indices = pickle.load(handle)
    with args.searchlight_centers.open("rb") as handle:
        centers = np.asarray(pickle.load(handle), dtype=np.int64)
    if len(sphere_indices) != len(centers) or len(np.unique(centers)) != len(centers):
        raise RuntimeError("searchlight center ordering is not one-to-one")
    if centers.min() < 0 or centers.max() >= mask.size:
        raise ValueError("searchlight centers fall outside the brain-mask volume")

    betas = np.load(args.betas, mmap_mode="r", allow_pickle=False)
    sample_indices = np.asarray(
        manifest["sample_indices_into_sorted_subject_conditions"], dtype=np.int64
    )
    if betas.shape != (81, 104, 83, 835) or betas.dtype != np.float32:
        raise ValueError(
            f"unexpected subject-1 beta cache: {betas.shape}/{betas.dtype}"
        )
    selected_betas = np.asarray(betas[..., sample_indices], dtype=np.float32)
    model_rdms = np.stack([rdms[name] for name in FEATURE_ORDER])
    if args.searchlight_backend == "tensorflow":
        # Exclude the host's broken Infinipath/PSM transport before importing
        # TensorFlow. The searchlight is single-process and needs no UCX fabric.
        os.environ.setdefault("UCX_TLS", "self,sm,cuda_copy,cuda_ipc")
        os.environ.setdefault("PSM2_DEVICES", "self")
        os.environ.setdefault("PSM3_DEVICES", "self")
        from .nsd_adapter import (
            _tf_searchlight_corr,
            bootstrap_cuda_library_path,
            initialize_tensorflow,
        )

        bootstrap_cuda_library_path()
        initialize_tensorflow(allow_cpu=False)
        sizes = np.asarray([len(indices) for indices in sphere_indices])
        sorted_indices = [np.flatnonzero(sizes == size) for size in np.unique(sizes)]
        maps = _tf_searchlight_corr(
            selected_betas,
            sphere_indices,
            sorted_indices,
            model_rdms,
            batch_size=args.searchlight_batch_size,
        )
        rdm_order = np.hstack([centers[group] for group in sorted_indices]).astype(
            np.int64
        )
        backend = (
            "TensorFlow float32 GPU with UCX limited to self/shared-memory/CUDA; "
            "PSM2/PSM3 limited to self"
        )
    else:
        maps = numpy_searchlight_corr(
            selected_betas,
            sphere_indices,
            model_rdms,
            batch_size=args.searchlight_batch_size,
        )
        rdm_order = centers
        backend = "NumPy float32 CPU; no TensorFlow, UCX, or PSM import"
    if maps.shape != (len(centers), len(FEATURE_ORDER)):
        raise ValueError(f"unexpected searchlight correlation shape {maps.shape}")
    volumes = np.zeros((len(FEATURE_ORDER), *mask.shape), dtype=np.float32)
    for feature_index, brain_map in enumerate(maps.T):
        flat = np.zeros(mask.size, dtype=np.float32)
        flat[rdm_order] = brain_map.astype(np.float32)
        volumes[feature_index] = flat.reshape(mask.shape)
    maps_path = args.output_root / "searchlight_maps.npy"
    atomic_npy(maps_path, volumes)

    artifacts = dict(manifest["artifacts"])
    artifacts.update(
        {
            "rdms": _file_record(rdm_path),
            "searchlight_maps": _file_record(maps_path),
        }
    )
    input_files = dict(manifest["input_files"])
    input_files.update(
        {
            "subject1_beta_cache": _file_record(args.betas),
            "subject1_brain_mask": _file_record(args.brain_mask),
            "subject1_searchlight_indices": _file_record(args.searchlight_indices),
            "subject1_searchlight_centers": _file_record(args.searchlight_centers),
        }
    )
    _update_manifest(
        args.output_root,
        status="searchlight_complete",
        input_files=input_files,
        artifacts=artifacts,
        rdm_searchlight={
            "feature_order": list(FEATURE_ORDER),
            "rdm_shape_each": [4950],
            "rdm_dtype": "float32",
            "rdm_metric": "correlation distance",
            "brain_model_similarity": "Pearson correlation of condensed RDMs",
            "searchlight_radius_voxels": 6,
            "searchlight_threshold": 0.5,
            "n_searchlight_centers": int(len(centers)),
            "map_shape": list(volumes.shape),
            "map_dtype": "float32",
            "matched_sample_row_zero_based": SAMPLE_ROW,
            "backend": backend,
            "recovery_reason": "prior native abort in libinfinipath/libucs",
        },
    )
    print(f"wrote {maps_path}")


def project(args: argparse.Namespace) -> None:
    """Project each subject-1 pilot map directly to fsaverage."""
    manifest = json.loads((args.output_root / "manifest.json").read_text())
    if manifest.get("status") != "searchlight_complete":
        raise RuntimeError(
            "project requires a searchlight_complete manifest; "
            f"found {manifest.get('status')!r}"
        )
    _validate_file_record("searchlight_maps", manifest["artifacts"]["searchlight_maps"])
    volumes = np.load(args.output_root / "searchlight_maps.npy", allow_pickle=False)
    if volumes.shape != tuple(manifest["rdm_searchlight"]["map_shape"]):
        raise ValueError("searchlight map shape disagrees with manifest")

    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    from nsdcode.nsd_mapdata import NSDmapdata

    nsd = NSDmapdata(str(args.nsd_dir))
    fs_dir = Path(nsd.base_dir) / "nsddata" / "freesurfer" / "fsaverage"
    surface_dir = args.output_root / "surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    artifacts = dict(manifest["artifacts"])
    for feature_index, feature_name in enumerate(FEATURE_ORDER):
        for hemisphere in ("lh", "rh"):
            depths = [
                nsd.fit(
                    SUBJECT,
                    "func1pt8",
                    f"{hemisphere}.layerB{depth}",
                    volumes[feature_index],
                    "cubic",
                    badval=0,
                )
                for depth in range(1, 4)
            ]
            native = np.nanmean(np.stack(depths), axis=0)
            surface = np.asarray(
                nsd.fit(
                    SUBJECT,
                    f"{hemisphere}.white",
                    "fsaverage",
                    native,
                    interptype=None,
                    badval=0,
                    fsdir=str(fs_dir),
                ),
                dtype=np.float32,
            )
            if surface.ndim != 1 or not np.isfinite(surface).all():
                raise ValueError(
                    f"invalid projected surface: {feature_name}/{hemisphere}"
                )
            path = surface_dir / f"{hemisphere}.subj01-{feature_name}.npy"
            atomic_npy(path, surface)
            artifacts[f"surface_{hemisphere}_{feature_name}"] = _file_record(path)
    _update_manifest(
        args.output_root,
        status="projection_complete",
        artifacts=artifacts,
        projection={
            "subject": SUBJECT,
            "target": "fsaverage",
            "depths": ["layerB1", "layerB2", "layerB3"],
            "sample_aggregation": "none; direct projection of sample row 0",
            "feature_order": list(FEATURE_ORDER),
        },
    )
    print(f"wrote {surface_dir}")


def plot(args: argparse.Namespace) -> None:
    """Render auditable per-feature subject-1 surface plots."""
    manifest = json.loads((args.output_root / "manifest.json").read_text())
    if manifest.get("status") != "projection_complete":
        raise RuntimeError(
            "plot requires a projection_complete manifest; "
            f"found {manifest.get('status')!r}"
        )
    from .nsd_adapter import plot_brain

    figure_dir = args.output_root / "figures"
    artifacts = dict(manifest["artifacts"])
    for feature_name in FEATURE_ORDER:
        hemispheres = []
        for hemisphere in ("lh", "rh"):
            record = manifest["artifacts"][f"surface_{hemisphere}_{feature_name}"]
            path = _validate_file_record(f"surface_{hemisphere}_{feature_name}", record)
            hemispheres.append(np.load(path, allow_pickle=False))
        values = np.concatenate(hemispheres).astype(np.float32, copy=False)
        name = f"image-only_subj01_sample00_{feature_name}"
        plot_brain(values, name, figure_dir, nsd_dir=args.nsd_dir, roi_overlay=None)
        path = figure_dir / f"{name}.png"
        artifacts[f"plot_{feature_name}"] = _file_record(path)
    _update_manifest(
        args.output_root,
        status="plots_complete",
        artifacts=artifacts,
        plots={
            "subject": SUBJECT,
            "sample_row_zero_based": SAMPLE_ROW,
            "feature_order": list(FEATURE_ORDER),
            "scaling": "independent symmetric scale per feature",
        },
    )
    print(f"wrote {figure_dir}")


def _plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    """Write the compact descriptive score figure used by the report."""
    import matplotlib.pyplot as plt

    layers = [str(row["layer"]) for row in rows]
    raw = [row["raw_mean_brain_rdm_correlation"] for row in rows]
    j_values = [row["j_mean_brain_rdm_correlation"] for row in rows[:-1]]
    x = np.arange(len(layers))
    figure, axis = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    width = 0.34
    axis.bar(x[:-1] - width / 2, raw[:-1], width, label="Raw residual", color="#4477AA")
    axis.bar(x[:-1] + width / 2, j_values, width, label="WikiText J", color="#EE6677")
    axis.bar(
        x[-1], raw[-1], width * 1.45, label="Final residual control", color="#999999"
    )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_xticks(x, layers)
    axis.set_xlabel("Decoder block (31 is a separate raw control)")
    axis.set_ylabel("Mean searchlight-centre RSA correlation")
    axis.set_title(PILOT_LABEL)
    axis.legend(frameon=False, ncols=3, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.01,
        0.01,
        "Subject 1 · deterministic 100-image sample 0 · descriptive only",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def summarize(args: argparse.Namespace) -> None:
    """Write descriptive means and within-layer J-minus-raw deltas."""
    manifest_path = args.output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") not in {"plots_complete", "complete"}:
        raise RuntimeError(
            "summarize requires a plots_complete manifest; "
            f"found {manifest.get('status')!r}"
        )
    for name in ("searchlight_maps",):
        _validate_file_record(name, manifest["artifacts"][name])
    volumes = np.load(args.output_root / "searchlight_maps.npy", allow_pickle=False)
    with Path(manifest["input_files"]["subject1_searchlight_centers"]["path"]).open(
        "rb"
    ) as handle:
        centers = np.asarray(pickle.load(handle), dtype=np.int64)
    if volumes.shape[0] != len(FEATURE_ORDER) or volumes.dtype != np.float32:
        raise ValueError("searchlight map feature axis/dtype is invalid")
    means = {}
    finite_counts = {}
    for index, name in enumerate(FEATURE_ORDER):
        values = volumes[index].reshape(-1)[centers]
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError(f"{name} has no finite searchlight-center values")
        means[name] = float(values[finite].mean(dtype=np.float64))
        finite_counts[name] = int(finite.sum())

    rows = []
    for layer in PILOT_LAYERS:
        raw_name = f"l{layer:02d}_raw"
        j_name = f"l{layer:02d}_j"
        rows.append(
            {
                "layer": layer,
                "condition": "matched_raw_and_j",
                "raw_mean_brain_rdm_correlation": means[raw_name],
                "j_mean_brain_rdm_correlation": means[j_name],
                "j_minus_raw": means[j_name] - means[raw_name],
                "finite_searchlight_centers": min(
                    finite_counts[raw_name], finite_counts[j_name]
                ),
            }
        )
    rows.append(
        {
            "layer": FINAL_BLOCK,
            "condition": "final_decoder_block_raw_control",
            "raw_mean_brain_rdm_correlation": means["final_raw"],
            "j_mean_brain_rdm_correlation": None,
            "j_minus_raw": None,
            "finite_searchlight_centers": finite_counts["final_raw"],
        }
    )
    summary = {
        "experiment_id": manifest["experiment_id"],
        "label": PILOT_LABEL,
        "subject": SUBJECT,
        "sample_row_zero_based": SAMPLE_ROW,
        "n_images": SAMPLE_SIZE,
        "n_searchlight_centers": manifest["rdm_searchlight"]["n_searchlight_centers"],
        "description": (
            f"{PILOT_LABEL}; single-subject, single-sample descriptive pilot; "
            "no inferential p-values"
        ),
        "rows": rows,
    }
    json_path = args.output_root / "summary.json"
    csv_path = args.output_root / "summary.csv"
    atomic_json(json_path, summary)
    temporary = csv_path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    _plot_summary(rows, args.report_figure)
    artifacts = dict(manifest["artifacts"])
    artifacts.update(
        {
            "summary_json": _file_record(json_path),
            "summary_csv": _file_record(csv_path),
            "report_figure": _file_record(args.report_figure),
        }
    )
    scientific_scope = dict(manifest["scientific_scope"])
    scientific_scope["label"] = PILOT_LABEL
    _update_manifest(
        args.output_root,
        status="complete",
        completed_at=_now(),
        artifacts=artifacts,
        scientific_scope=scientific_scope,
        summary=summary,
    )
    print(json.dumps(summary, indent=2))


def verify(args: argparse.Namespace) -> None:
    """Re-hash artifacts and revalidate every generated numerical array."""
    manifest = json.loads((args.output_root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"pilot manifest is not complete: {manifest.get('status')}")
    scope = manifest.get("scientific_scope", {})
    required_scope = {
        "image_only": True,
        "semantic_text_present": False,
        "chat_template_used": False,
        "generation_used": False,
        "lens_fitted_or_modified": False,
    }
    if any(scope.get(key) is not value for key, value in required_scope.items()):
        raise RuntimeError("manifest violates the strict image-only scientific scope")
    if (
        scope.get("label") != PILOT_LABEL
        or manifest["summary"].get("label") != PILOT_LABEL
    ):
        raise RuntimeError("manifest does not use the required scientific label")
    minimal = manifest.get("minimal_image_input", {})
    if (
        minimal.get("literal_sequence") != MINIMAL_IMAGE_SEQUENCE
        or not minimal.get("attention_all_ones")
        or not minimal.get("image_id_mask_equals_mm_type_1")
    ):
        raise RuntimeError("manifest minimal-image token evidence is invalid")

    for name, record in manifest["artifacts"].items():
        _validate_file_record(f"artifact:{name}", record)
    for name, record in manifest["input_files"].items():
        _validate_file_record(f"input:{name}", record)
    for name, record in manifest["model"]["files"].items():
        _validate_file_record(f"model:{name}", record)
    model_hash = stable_hash(
        {name: record["sha256"] for name, record in manifest["model"]["files"].items()}
    )
    if model_hash != manifest["model"]["composite_sha256"]:
        raise RuntimeError("model composite hash disagrees with model file records")
    for name, record in manifest["software"].get("experiment_source", {}).items():
        _validate_file_record(f"experiment_source:{name}", record)
    for name, record in manifest["software"][
        "transformers_implementation_files"
    ].items():
        _validate_file_record(f"transformers_source:{name}", record)
    jlens_record = manifest["software"]["jlens_source"]
    current_jlens = _git_record(Path(jlens_record["path"]))
    if current_jlens != jlens_record:
        raise RuntimeError("Jacobian Lens source checkout changed since extraction")

    expected_condition_ids = np.asarray(manifest["condition_ids"], dtype=np.int64)
    expected_sample_indices = np.asarray(
        manifest["sample_indices_into_sorted_subject_conditions"], dtype=np.int64
    )
    subject_ids, sample_indices, condition_ids = _selected_sample(
        Path(manifest["input_files"]["subject_responses_tsv"]["path"]),
        Path(manifest["input_files"]["matched_sampling"]["path"]),
    )
    if len(subject_ids) != manifest["subject_condition_count"]:
        raise ValueError("subject condition count disagrees with source inputs")
    if not np.array_equal(sample_indices, expected_sample_indices):
        raise ValueError("manifest sample indices disagree with matched sampling")
    if not np.array_equal(condition_ids, expected_condition_ids):
        raise ValueError("manifest condition IDs disagree with source inputs")
    with np.load(args.output_root / "features.npz", allow_pickle=False) as payload:
        if set(payload.files) != set(FEATURE_ORDER) | {
            "condition_ids",
            "sample_indices",
            "image_token_counts",
            "image_grid_thw",
        }:
            raise ValueError("unexpected feature archive keys")
        if not np.array_equal(payload["condition_ids"], expected_condition_ids):
            raise ValueError("feature condition IDs disagree with manifest")
        if not np.array_equal(payload["sample_indices"], expected_sample_indices):
            raise ValueError("feature sample indices disagree with manifest")
        token_counts = payload["image_token_counts"]
        grids = payload["image_grid_thw"]
        if token_counts.shape != (SAMPLE_SIZE,) or token_counts.dtype != np.int64:
            raise ValueError("invalid image-token count vector")
        if grids.shape != (SAMPLE_SIZE, 3) or grids.dtype != np.int64:
            raise ValueError("invalid image-grid array")
        merge_size = int(manifest["image_preprocessing"]["merge_size"])
        expected_counts = np.prod(grids, axis=1) // merge_size**2
        if not np.array_equal(token_counts, expected_counts):
            raise ValueError("image token counts disagree with image grids")
        manifest_counts = np.asarray(
            [image["image_token_count"] for image in manifest["images"]],
            dtype=np.int64,
        )
        if not np.array_equal(token_counts, manifest_counts):
            raise ValueError("feature image-token counts disagree with manifest")
        for name in FEATURE_ORDER:
            _validate_feature_matrix(payload[name], name)
        feature_rdms = {
            name: pdist(payload[name], metric="correlation").astype(np.float32)
            for name in FEATURE_ORDER
        }
    with np.load(args.output_root / "rdms.npz", allow_pickle=False) as payload:
        if set(payload.files) != set(FEATURE_ORDER):
            raise ValueError("unexpected RDM archive keys")
        for name in FEATURE_ORDER:
            if payload[name].shape != (4950,) or payload[name].dtype != np.float32:
                raise ValueError(f"invalid RDM array: {name}")
            if not np.isfinite(payload[name]).all():
                raise ValueError(f"non-finite RDM array: {name}")
            if not np.array_equal(payload[name], feature_rdms[name]):
                raise ValueError(f"RDM does not exactly match features: {name}")
    maps = np.load(args.output_root / "searchlight_maps.npy", allow_pickle=False)
    if maps.shape != tuple(manifest["rdm_searchlight"]["map_shape"]):
        raise ValueError("searchlight map shape mismatch")
    if maps.dtype != np.float32 or not np.isfinite(maps).all():
        raise ValueError("searchlight maps are not finite float32")
    with Path(manifest["input_files"]["subject1_searchlight_centers"]["path"]).open(
        "rb"
    ) as handle:
        centers = np.asarray(pickle.load(handle), dtype=np.int64)
    if (
        centers.ndim != 1
        or len(centers) != manifest["rdm_searchlight"]["n_searchlight_centers"]
        or len(np.unique(centers)) != len(centers)
        or centers.min() < 0
        or centers.max() >= maps[0].size
    ):
        raise ValueError("invalid searchlight-center index vector")
    if "libinfinipath/libucs" not in manifest["rdm_searchlight"].get(
        "recovery_reason", ""
    ):
        raise ValueError("searchlight recovery provenance is missing")
    for name in FEATURE_ORDER:
        for hemisphere in ("lh", "rh"):
            record = manifest["artifacts"][f"surface_{hemisphere}_{name}"]
            surface = np.load(_validate_file_record(name, record), allow_pickle=False)
            if surface.shape != (163842,) or surface.dtype != np.float32:
                raise ValueError(f"invalid fsaverage surface: {hemisphere}/{name}")
            if not np.isfinite(surface).all():
                raise ValueError(f"non-finite fsaverage surface: {hemisphere}/{name}")

    summary_path = args.output_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary != manifest.get("summary"):
        raise ValueError("summary.json disagrees with manifest summary")
    with (args.output_root / "summary.csv").open(encoding="utf-8", newline="") as h:
        csv_rows = list(csv.DictReader(h))
    if len(csv_rows) != len(summary["rows"]):
        raise ValueError("summary.csv row count disagrees with summary.json")
    for json_row, csv_row in zip(summary["rows"], csv_rows, strict=True):
        parsed_csv_row = {
            "layer": int(csv_row["layer"]),
            "condition": csv_row["condition"],
            "raw_mean_brain_rdm_correlation": float(
                csv_row["raw_mean_brain_rdm_correlation"]
            ),
            "j_mean_brain_rdm_correlation": (
                float(csv_row["j_mean_brain_rdm_correlation"])
                if csv_row["j_mean_brain_rdm_correlation"]
                else None
            ),
            "j_minus_raw": (
                float(csv_row["j_minus_raw"]) if csv_row["j_minus_raw"] else None
            ),
            "finite_searchlight_centers": int(csv_row["finite_searchlight_centers"]),
        }
        if parsed_csv_row != json_row:
            raise ValueError(
                f"summary.csv disagrees with summary.json at layer {json_row['layer']}"
            )
    recalculated_means = {
        name: float(maps[index].reshape(-1)[centers].mean(dtype=np.float64))
        for index, name in enumerate(FEATURE_ORDER)
    }
    for row in summary["rows"]:
        layer = int(row["layer"])
        raw_name = "final_raw" if layer == FINAL_BLOCK else f"l{layer:02d}_raw"
        if row["raw_mean_brain_rdm_correlation"] != recalculated_means[raw_name]:
            raise ValueError(f"summary raw mean mismatch at layer {layer}")
        if layer != FINAL_BLOCK:
            j_name = f"l{layer:02d}_j"
            j_mean = recalculated_means[j_name]
            if row["j_mean_brain_rdm_correlation"] != j_mean:
                raise ValueError(f"summary J mean mismatch at layer {layer}")
            if row["j_minus_raw"] != j_mean - recalculated_means[raw_name]:
                raise ValueError(f"summary delta mismatch at layer {layer}")

    import h5py

    with h5py.File(Path(manifest["input_files"]["stimuli_hdf5"]["path"]), "r") as h:
        dataset = h["imgBrick"]
        for image in manifest["images"]:
            pixels = np.asarray(dataset[int(image["condition_id_73k"]) - 1])
            digest = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
            if digest != image["pixel_sha256"]:
                raise ValueError(
                    "stimulus pixel hash mismatch for condition "
                    f"{image['condition_id_73k']}"
                )

    n_hashed = (
        len(manifest["artifacts"])
        + len(manifest["input_files"])
        + len(manifest["model"]["files"])
        + len(manifest["software"].get("experiment_source", {}))
        + len(manifest["software"]["transformers_implementation_files"])
    )
    print(
        f"verified {n_hashed} hashed files, {len(manifest['images'])} pixel hashes, "
        "manifest scope, summaries, and all arrays"
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path("results/image_only_wikitext_pilot")
    nsd = Path("/media/chuddy/Extreme SSD/data/NSD")
    mpnet = Path(
        "/home/chuddy/dev/research/neuroconnectionism/lucas_exploration/results/mpnet_10_sessions"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "extract",
            "rdm-searchlight",
            "project",
            "plot",
            "summarize",
            "verify",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--nsd-dir",
        type=Path,
        default=nsd,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/media/chuddy/Extreme SSD/models/Qwen3.5-4B"),
    )
    parser.add_argument(
        "--lens",
        type=Path,
        default=Path(
            "/media/chuddy/Extreme SSD/models/jacobian-lens/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
        ),
    )
    parser.add_argument(
        "--jlens-checkout",
        type=Path,
        default=Path("/home/chuddy/dev/research/jacobian-lens"),
    )
    parser.add_argument(
        "--stimuli",
        type=Path,
        default=nsd / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
    )
    parser.add_argument(
        "--stim-info-csv",
        type=Path,
        default=nsd / "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
    )
    parser.add_argument(
        "--responses-tsv",
        type=Path,
        default=nsd / "nsddata/ppdata/subj01/behav/responses.tsv",
    )
    parser.add_argument(
        "--brain-mask",
        type=Path,
        default=nsd / "nsddata/ppdata/subj01/func1pt8mm/brainmask.nii.gz",
    )
    parser.add_argument(
        "--sampling",
        type=Path,
        default=mpnet
        / "searchlight_respectedsampling_correlation/subj01/saved_sampling/subj01_nsd-allsubstim_sampling.npy",
    )
    parser.add_argument(
        "--betas",
        type=Path,
        default=mpnet / "precomputed/betas/subj01_betas_average_func1pt8mm.npy",
    )
    parser.add_argument(
        "--searchlight-indices",
        type=Path,
        default=Path(
            "/home/chuddy/dev/research/neuroconnectionism/lucas_exploration/results/searchlight/subj01/subj01-func1pt8mm-6rad-searchlight_indices.npy"
        ),
    )
    parser.add_argument(
        "--searchlight-centers",
        type=Path,
        default=Path(
            "/home/chuddy/dev/research/neuroconnectionism/lucas_exploration/results/searchlight/subj01/subj01-func1pt8mm-6rad-searchlight_centers.npy"
        ),
    )
    parser.add_argument(
        "--searchlight-backend",
        choices=("tensorflow", "numpy"),
        default="tensorflow",
    )
    parser.add_argument("--searchlight-batch-size", type=int, default=250)
    parser.add_argument(
        "--report-figure",
        type=Path,
        default=Path("docs/assets/image_only_wikitext_pilot_scores.png"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for name in (
        "output_root",
        "nsd_dir",
        "model",
        "lens",
        "jlens_checkout",
        "stimuli",
        "stim_info_csv",
        "responses_tsv",
        "brain_mask",
        "sampling",
        "betas",
        "searchlight_indices",
        "searchlight_centers",
        "report_figure",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    globals()[args.phase.replace("-", "_")](args)


if __name__ == "__main__":
    main()

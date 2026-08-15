"""Batched, resumable residual and J-transport extraction."""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .conditions import load_union_ids
from .config import DEFAULT_PROMPT_SET, ExperimentPaths, ModelSpec, run_name
from .io_utils import atomic_json, atomic_npz, sha256_file, stable_hash
from .prompts import (
    MATCHED_READOUT_SUFFIX,
    captions_for_condition,
    load_caption_table,
    matched_prompt_contract,
    prompt_set,
    prompts_for_condition,
)

SCHEMA_VERSION = 1


def _load_jlens(reference_repo: Path | None):
    """Load jlens from an explicit checkout, or from the installed extra."""
    if reference_repo is not None:
        if not (reference_repo / "jlens/__init__.py").exists():
            raise FileNotFoundError(
                f"Jacobian Lens checkout not found: {reference_repo}"
            )
        reference = str(reference_repo.resolve())
        if reference not in sys.path:
            sys.path.insert(0, reference)
    try:
        import jlens
    except ImportError as error:
        raise RuntimeError(
            "Jacobian Lens is unavailable; install the 'model' extra or pass "
            "--jlens-checkout"
        ) from error

    module_path = Path(jlens.__file__).resolve()
    if (
        reference_repo is not None
        and reference_repo.resolve() not in module_path.parents
    ):
        raise RuntimeError(
            f"imported jlens from {module_path}, expected {reference_repo}"
        )
    return jlens


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_lens_path(
    spec: ModelSpec,
    *,
    lens_path: Path | None,
    allow_download: bool,
) -> Path:
    if lens_path is not None:
        if not lens_path.is_file():
            raise FileNotFoundError(f"lens checkpoint not found: {lens_path}")
        return lens_path.resolve()
    if spec.local_lens_root is not None:
        local_lens = spec.local_lens_root / spec.lens_filename
        if local_lens.is_file():
            return local_lens.resolve()
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=spec.lens_repo,
                filename=spec.lens_filename,
                revision=spec.lens_revision,
                local_files_only=not allow_download,
            )
        ).resolve()
    except Exception as error:
        mode = "download enabled" if allow_download else "local cache only"
        raise RuntimeError(
            f"could not resolve matched lens ({mode}): {spec.lens_repo}@"
            f"{spec.lens_revision}/{spec.lens_filename}"
        ) from error


def prefetch_artifacts(spec: ModelSpec) -> dict[str, str]:
    """Explicitly download weights; extraction itself defaults to offline."""
    from huggingface_hub import snapshot_download

    model_path = (
        str(spec.local_model_path.resolve())
        if spec.local_model_path is not None and spec.local_model_path.is_dir()
        else snapshot_download(repo_id=spec.model_name)
    )
    lens_path = resolve_lens_path(
        spec,
        lens_path=None,
        allow_download=True,
    )
    return {"model_path": model_path, "lens_path": str(lens_path)}


def resolve_source_layers(
    source_layers: Sequence[int],
    n_layers: int,
    explicit_layers: Sequence[int] | None = None,
) -> list[int]:
    fitted = sorted(set(int(layer) for layer in source_layers))
    if not fitted:
        raise ValueError("lens contains no source layers")
    if explicit_layers is not None:
        selected = [int(layer) for layer in explicit_layers]
        unknown = sorted(set(selected) - set(fitted))
        if unknown:
            raise ValueError(
                f"requested layers {unknown} are not fitted; available={fitted}"
            )
        if len(selected) != len(set(selected)):
            raise ValueError("requested layers contain duplicates")
        return selected

    targets = [
        round((n_layers - 1) * 0.25),
        round((n_layers - 1) * 0.50),
        round((n_layers - 1) * 0.75),
        n_layers - 2,
    ]
    selected: list[int] = []
    for target in targets:
        candidates = sorted(fitted, key=lambda layer: (abs(layer - target), layer))
        choice = next((layer for layer in candidates if layer not in selected), None)
        if choice is not None:
            selected.append(choice)
    if len(selected) < min(4, len(fitted)):
        for layer in reversed(fitted):
            if layer not in selected:
                selected.append(layer)
            if len(selected) == min(4, len(fitted)):
                break
    return sorted(selected)


def feature_registry(
    layers: Sequence[int],
    prompt_kinds: Sequence[str] = ("visualize", "plain"),
) -> list[dict[str, Any]]:
    features = []
    for prompt_kind in prompt_kinds:
        for layer in layers:
            features.extend(
                [
                    {
                        "name": f"{prompt_kind}__l{layer:02d}__raw",
                        "prompt": prompt_kind,
                        "kind": "raw",
                        "layer": layer,
                    },
                    {
                        "name": f"{prompt_kind}__l{layer:02d}__j",
                        "prompt": prompt_kind,
                        "kind": "j",
                        "layer": layer,
                    },
                ]
            )
        features.append(
            {
                "name": f"{prompt_kind}__final",
                "prompt": prompt_kind,
                "kind": "final",
                "layer": None,
            }
        )
    return features


def _token_ids(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        return_attention_mask=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer returned multiple rows for one prompt")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def audit_matched_prompt_endpoints(
    tokenizer,
    prompt_rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Prove the declared suffix and actual final token match for every pair."""
    if not prompt_rows:
        raise ValueError("endpoint audit requires at least one prompt row")
    suffix_bytes = MATCHED_READOUT_SUFFIX.encode("utf-8")
    suffix_ids = _token_ids(tokenizer, MATCHED_READOUT_SUFFIX, add_special_tokens=False)
    if not suffix_ids:
        raise ValueError("matched readout suffix tokenized to an empty sequence")
    endpoint_ids = []
    pair_token_hashes = []
    max_tokens = 0
    for prompts in prompt_rows:
        matched_prompt_contract(prompts)
        minimal_ids = _token_ids(
            tokenizer, prompts["minimal_readout"], add_special_tokens=True
        )
        integrate_ids = _token_ids(
            tokenizer, prompts["integrate_readout"], add_special_tokens=True
        )
        if minimal_ids[-len(suffix_ids) :] != suffix_ids:
            raise ValueError("minimal prompt token suffix differs from declaration")
        if integrate_ids[-len(suffix_ids) :] != suffix_ids:
            raise ValueError("integrated prompt token suffix differs from declaration")
        if minimal_ids[-1] != integrate_ids[-1]:
            raise ValueError("matched prompts have different final readout token IDs")
        endpoint_ids.append(minimal_ids[-1])
        pair_token_hashes.append(
            stable_hash(
                {
                    "minimal_suffix": minimal_ids[-len(suffix_ids) :],
                    "integrate_suffix": integrate_ids[-len(suffix_ids) :],
                }
            )
        )
        max_tokens = max(max_tokens, len(minimal_ids), len(integrate_ids))
    return {
        "n_conditions": len(prompt_rows),
        "common_suffix": MATCHED_READOUT_SUFFIX,
        "common_suffix_utf8_hex": suffix_bytes.hex(),
        "common_suffix_nbytes": len(suffix_bytes),
        "common_suffix_token_ids": suffix_ids,
        "final_readout_token_id": suffix_ids[-1],
        "all_pair_final_token_ids_match": True,
        "all_prompts_end_with_declared_suffix_tokens": True,
        "endpoint_ids_hash": stable_hash(endpoint_ids),
        "paired_suffix_token_hashes_hash": stable_hash(pair_token_hashes),
        "max_prompt_tokens": max_tokens,
        "tokenizer_class": type(tokenizer).__name__,
    }


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def _load_model_and_lens(
    paths: ExperimentPaths,
    spec: ModelSpec,
    *,
    device: str,
    allow_download: bool,
    lens_path: Path | None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    jlens = _load_jlens(paths.jlens_checkout)
    resolved_lens = resolve_lens_path(
        spec, lens_path=lens_path, allow_download=allow_download
    )
    lens = jlens.JacobianLens.load(str(resolved_lens))

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a GPU. "
            "Do not start the overnight run until this preflight passes."
        )
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    model_source: str | Path = spec.model_name
    if spec.local_model_path is not None and spec.local_model_path.is_dir():
        # Downloads for this workstation deliberately live on the external SSD
        # rather than in Hugging Face's root-filesystem cache.
        model_source = spec.local_model_path.resolve()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=not allow_download,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=not allow_download,
        )
    except Exception as error:
        mode = "download enabled" if allow_download else "local cache only"
        raise RuntimeError(
            f"could not load {spec.model_name} ({mode}); run the explicit "
            "prefetch command first"
        ) from error

    model.to(target)
    model.eval()
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    wrapped = jlens.from_hf(model, tokenizer, force_bos=False)
    if wrapped.d_model != lens.d_model:
        raise ValueError(
            f"model d_model={wrapped.d_model}, lens d_model={lens.d_model}"
        )
    invalid = [
        layer for layer in lens.source_layers if not 0 <= layer < wrapped.n_layers
    ]
    if invalid:
        raise ValueError(
            f"lens layers {invalid} outside model range 0..{wrapped.n_layers - 1}"
        )
    return model, tokenizer, wrapped, lens, resolved_lens, str(model_source)


def _gather_last(hidden, attention_mask):
    import torch

    positions = attention_mask.sum(dim=1, dtype=torch.long) - 1
    if torch.any(positions < 0):
        raise ValueError("encountered an empty tokenized prompt")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, positions]


def _validate_hook_semantics(recorder, outputs, layers, attention_mask) -> dict:
    """Check HF's embedding-offset convention against reference hooks."""
    import torch

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("model did not return hidden_states for validation")
    checks = {}
    for layer in layers:
        # HF tuple element zero is the embedding output. For non-final blocks,
        # element layer+1 is the block output/input to the next block. The last
        # tuple element may be post-final-norm, so it is intentionally excluded.
        hooked = _gather_last(recorder.activations[layer], attention_mask).float()
        hf_state = _gather_last(hidden_states[layer + 1], attention_mask).float()
        max_abs = float(torch.max(torch.abs(hooked - hf_state)).cpu())
        reference = float(torch.max(torch.abs(hooked)).cpu())
        tolerance = 2e-4 + 2e-4 * reference
        if max_abs > tolerance:
            raise RuntimeError(
                f"hook/HF hidden-state mismatch at block {layer}: "
                f"max_abs={max_abs:.6g}, tolerance={tolerance:.6g}"
            )
        checks[str(layer)] = {
            "hf_hidden_states_index": layer + 1,
            "max_abs_error": max_abs,
            "tolerance": tolerance,
        }
    return checks


def _forward_prompt_batch(
    prompts: Sequence[str],
    *,
    tokenizer,
    wrapped,
    lens_matrices,
    layers: Sequence[int],
    max_length: int,
    validate_semantics: bool,
):
    import torch
    from jlens.hooks import ActivationRecorder

    encoded = tokenizer(
        list(prompts),
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    sequence_length = int(encoded.input_ids.shape[1])
    if sequence_length > max_length:
        raise ValueError(
            f"prompt batch needs {sequence_length} tokens, exceeding the "
            f"configured guard {max_length}; raise --max-length rather than "
            "truncating captions"
        )
    encoded = {key: value.to(wrapped.input_device) for key, value in encoded.items()}
    final_layer = wrapped.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    context = torch.no_grad()
    with context, ActivationRecorder(wrapped.layers, at=record_at) as recorder:
        # Call the bare decoder exactly as the reference adapter does. Calling
        # *ForCausalLM would allocate vocabulary logits that this experiment
        # never uses (hundreds of MB for a padded batch).
        outputs = wrapped._text_module(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            output_hidden_states=validate_semantics,
            return_dict=True,
        )
    attention_mask = encoded["attention_mask"]
    semantic_checks = None
    if validate_semantics:
        semantic_checks = _validate_hook_semantics(
            recorder,
            outputs,
            [layer for layer in layers if layer < final_layer],
            attention_mask,
        )

    result = {}
    for layer in layers:
        raw = _gather_last(recorder.activations[layer], attention_mask).float()
        transported = raw @ lens_matrices[layer].T
        result[("raw", layer)] = raw.cpu().numpy().astype(np.float32)
        result[("j", layer)] = transported.cpu().numpy().astype(np.float32)
    final = _gather_last(recorder.activations[final_layer], attention_mask).float()
    result[("final", None)] = final.cpu().numpy().astype(np.float32)
    return result, sequence_length, semantic_checks


def _chunk_is_valid(
    path: Path,
    expected_ids: np.ndarray,
    features: Sequence[dict[str, Any]],
    d_model: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as chunk:
            if set(chunk.files) != {"condition_ids"} | {
                item["name"] for item in features
            }:
                return False
            if not np.array_equal(chunk["condition_ids"], expected_ids):
                return False
            for item in features:
                array = chunk[item["name"]]
                if array.shape != (len(expected_ids), d_model):
                    return False
                if array.dtype != np.float32 or not np.isfinite(array).all():
                    return False
    except (OSError, ValueError, EOFError):
        return False
    return True


def _manifest_config(
    paths: ExperimentPaths,
    spec: ModelSpec,
    union_ids: np.ndarray,
    *,
    layers: Sequence[int],
    d_model: int,
    n_layers: int,
    chunk_size: int,
    max_length: int,
    seed: int,
    lens_path: Path,
    model_source: str,
    prompt_set_key: str,
) -> dict:
    selected_prompt_set = prompt_set(prompt_set_key)
    config = {
        "schema_version": SCHEMA_VERSION,
        "model_profile": spec.key,
        "model_name": spec.model_name,
        "model_source": model_source,
        "lens_repo": spec.lens_repo,
        "lens_revision": spec.lens_revision,
        "lens_filename": spec.lens_filename,
        "lens_path": str(lens_path),
        "lens_sha256": sha256_file(lens_path),
        "jlens_source": (
            str(paths.jlens_checkout.resolve())
            if paths.jlens_checkout is not None
            else "installed-package"
        ),
        "jlens_git_revision": (
            _git_revision(paths.jlens_checkout)
            if paths.jlens_checkout is not None
            else None
        ),
        "prompt_set": selected_prompt_set.key,
        "prompt_version": selected_prompt_set.version,
        "prompt_source_sha256": sha256_file(
            Path(sys.modules[prompts_for_condition.__module__].__file__)
        ),
        "prompt_kinds": list(selected_prompt_set.kinds),
        "caption_file": str(paths.captions.resolve()),
        "caption_sha256": sha256_file(paths.captions),
        "nsd_id_base": 1,
        "n_conditions": len(union_ids),
        "condition_ids_hash": stable_hash(union_ids.tolist()),
        "layers": list(layers),
        "n_model_layers": n_layers,
        "d_model": d_model,
        "position": "last_nonpadding_prompt_token",
        "normalization": "none; native residual units cast to float32",
        "chunk_size": chunk_size,
        "max_length_guard_no_truncation": max_length,
        "seed": seed,
    }
    config["fingerprint"] = stable_hash(config)
    return config


def extract_embeddings(
    paths: ExperimentPaths,
    spec: ModelSpec,
    *,
    device: str = "cuda",
    allow_download: bool = False,
    lens_path: Path | None = None,
    explicit_layers: Sequence[int] | None = None,
    batch_size: int = 8,
    chunk_size: int = 64,
    max_length: int = 256,
    seed: int = 0,
    max_conditions: int | None = None,
    output_name: str | None = None,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
) -> dict:
    """Extract all configured features, resuming at atomic chunk boundaries."""
    import torch

    paths.require("captions")
    assert paths.captions is not None
    if batch_size < 1 or chunk_size < 1:
        raise ValueError("batch_size and chunk_size must be positive")
    _configure_determinism(seed)
    union_ids = load_union_ids(paths)
    if max_conditions is not None:
        union_ids = union_ids[:max_conditions]
    if len(union_ids) == 0:
        raise ValueError("no condition IDs selected")

    _model, tokenizer, wrapped, lens, resolved_lens, model_source = (
        _load_model_and_lens(
            paths,
            spec,
            device=device,
            allow_download=allow_download,
            lens_path=lens_path,
        )
    )
    layers = resolve_source_layers(
        lens.source_layers, wrapped.n_layers, explicit_layers
    )
    selected_prompt_set = prompt_set(prompt_set_key)
    features = feature_registry(layers, selected_prompt_set.kinds)
    output_dir = paths.embeddings / (output_name or run_name(spec.key, prompt_set_key))
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    config = _manifest_config(
        paths,
        spec,
        union_ids,
        layers=layers,
        d_model=wrapped.d_model,
        n_layers=wrapped.n_layers,
        chunk_size=chunk_size,
        max_length=max_length,
        seed=seed,
        lens_path=resolved_lens,
        model_source=model_source,
        prompt_set_key=prompt_set_key,
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config", {}).get("fingerprint") != config["fingerprint"]:
            raise RuntimeError(
                f"resume configuration differs from {manifest_path}; use a "
                "different --output-name to preserve the existing run"
            )

    target_device = wrapped.input_device
    lens_matrices = {}
    for layer in layers:
        matrix = lens.jacobians[layer]
        if matrix.shape != (wrapped.d_model, wrapped.d_model):
            raise ValueError(
                f"lens layer {layer} shape {tuple(matrix.shape)} does not match "
                f"({wrapped.d_model}, {wrapped.d_model})"
            )
        if not torch.isfinite(matrix).all():
            raise ValueError(f"lens layer {layer} contains non-finite values")
        lens_matrices[layer] = matrix.to(target_device, dtype=torch.float32)

    caption_table = load_caption_table(paths.captions)
    preview_positions = np.linspace(
        0, len(union_ids) - 1, num=min(8, len(union_ids)), dtype=int
    )
    preview = []
    for position in preview_positions:
        condition_id = int(union_ids[position])
        preview.append(
            {
                "condition_id_1based": condition_id,
                "captions": captions_for_condition(caption_table, condition_id),
                "prompts": prompts_for_condition(
                    caption_table, condition_id, prompt_set_key
                ),
            }
        )
    preview_path = output_dir / "prompt_preview.json"
    atomic_json(
        preview_path,
        {
            "prompt_set": selected_prompt_set.key,
            "prompt_version": selected_prompt_set.version,
            "nsd_id_base": 1,
            "examples": preview,
        },
    )
    endpoint_audit = None
    if selected_prompt_set.matched_readout:
        all_prompt_rows = [
            prompts_for_condition(caption_table, int(condition_id), prompt_set_key)
            for condition_id in union_ids
        ]
        endpoint_audit = audit_matched_prompt_endpoints(tokenizer, all_prompt_rows)
        endpoint_audit_path = output_dir / "endpoint_audit.json"
        atomic_json(endpoint_audit_path, endpoint_audit)
    else:
        endpoint_audit_path = None
    semantic_checks = None
    max_observed_tokens = 0
    effective_batch_size = batch_size
    completed_chunks = []
    started_at = datetime.now(timezone.utc).isoformat()

    for start in range(0, len(union_ids), chunk_size):
        stop = min(start + chunk_size, len(union_ids))
        chunk_ids = union_ids[start:stop]
        chunk_path = chunks_dir / f"chunk_{start:06d}_{stop:06d}.npz"
        if _chunk_is_valid(chunk_path, chunk_ids, features, wrapped.d_model):
            completed_chunks.append(chunk_path.name)
            continue

        prompt_rows = [
            prompts_for_condition(caption_table, int(condition_id), prompt_set_key)
            for condition_id in chunk_ids
        ]
        arrays: dict[str, list[np.ndarray]] = {
            feature["name"]: [] for feature in features
        }
        for prompt_kind in selected_prompt_set.kinds:
            offset = 0
            while offset < len(prompt_rows):
                current = min(effective_batch_size, len(prompt_rows) - offset)
                prompts = [
                    row[prompt_kind] for row in prompt_rows[offset : offset + current]
                ]
                try:
                    batch_result, n_tokens, checks = _forward_prompt_batch(
                        prompts,
                        tokenizer=tokenizer,
                        wrapped=wrapped,
                        lens_matrices=lens_matrices,
                        layers=layers,
                        max_length=max_length,
                        validate_semantics=semantic_checks is None,
                    )
                except torch.OutOfMemoryError:
                    if effective_batch_size == 1:
                        raise RuntimeError(
                            "GPU OOM at batch size 1; use the supported 1.7B "
                            "fallback profile or a shorter max-length guard"
                        ) from None
                    effective_batch_size = max(1, effective_batch_size // 2)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                max_observed_tokens = max(max_observed_tokens, n_tokens)
                if checks is not None:
                    semantic_checks = checks
                for layer in layers:
                    arrays[f"{prompt_kind}__l{layer:02d}__raw"].append(
                        batch_result[("raw", layer)]
                    )
                    arrays[f"{prompt_kind}__l{layer:02d}__j"].append(
                        batch_result[("j", layer)]
                    )
                arrays[f"{prompt_kind}__final"].append(batch_result[("final", None)])
                offset += current

        published = {"condition_ids": chunk_ids}
        for feature in features:
            value = np.concatenate(arrays[feature["name"]], axis=0)
            if value.shape != (len(chunk_ids), wrapped.d_model):
                raise RuntimeError(
                    f"bad feature shape for {feature['name']}: {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite extraction values in {feature['name']}")
            published[feature["name"]] = value
        atomic_npz(chunk_path, **published)
        if not _chunk_is_valid(chunk_path, chunk_ids, features, wrapped.d_model):
            raise RuntimeError(f"published chunk failed validation: {chunk_path}")
        completed_chunks.append(chunk_path.name)
        atomic_json(
            manifest_path,
            {
                "config": config,
                "features": features,
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "complete": False,
                "prompt_preview": str(preview_path.resolve()),
                "endpoint_audit": (
                    str(endpoint_audit_path.resolve())
                    if endpoint_audit_path is not None
                    else None
                ),
                "completed_chunks": completed_chunks,
                "effective_batch_size": effective_batch_size,
                "max_observed_tokens": max_observed_tokens,
                "hook_semantics": semantic_checks,
            },
        )

    manifest = {
        "config": config,
        "features": features,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "prompt_preview": str(preview_path.resolve()),
        "endpoint_audit": (
            str(endpoint_audit_path.resolve())
            if endpoint_audit_path is not None
            else None
        ),
        "completed_chunks": completed_chunks,
        "effective_batch_size": effective_batch_size,
        "max_observed_tokens": max_observed_tokens,
        "hook_semantics": semantic_checks,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def preflight(
    paths: ExperimentPaths,
    spec: ModelSpec,
    *,
    device: str = "cuda",
    allow_download: bool = False,
    lens_path: Path | None = None,
    max_length: int = 256,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
) -> dict:
    """Load the matched pair and run two prompts without publishing chunks."""
    import torch

    paths.require("captions")
    assert paths.captions is not None
    _configure_determinism(0)
    _model, tokenizer, wrapped, lens, resolved_lens, model_source = (
        _load_model_and_lens(
            paths,
            spec,
            device=device,
            allow_download=allow_download,
            lens_path=lens_path,
        )
    )
    layers = resolve_source_layers(lens.source_layers, wrapped.n_layers)
    lens_matrices = {
        layer: lens.jacobians[layer].to(wrapped.input_device, dtype=torch.float32)
        for layer in layers
    }
    table = load_caption_table(paths.captions)
    union_ids = load_union_ids(paths)
    condition_id = int(union_ids[0])
    selected_prompt_set = prompt_set(prompt_set_key)
    prompt_row = prompts_for_condition(table, condition_id, prompt_set_key)
    endpoint_audit = None
    if selected_prompt_set.matched_readout:
        endpoint_audit = audit_matched_prompt_endpoints(
            tokenizer,
            [
                prompts_for_condition(table, int(item), prompt_set_key)
                for item in union_ids
            ],
        )
        if endpoint_audit["max_prompt_tokens"] > max_length:
            raise ValueError(
                "matched prompt endpoint audit found a prompt longer than the "
                "configured no-truncation guard"
            )
    results = {}
    semantics = None
    for kind in selected_prompt_set.kinds:
        batch, n_tokens, checks = _forward_prompt_batch(
            [prompt_row[kind]],
            tokenizer=tokenizer,
            wrapped=wrapped,
            lens_matrices=lens_matrices,
            layers=layers,
            max_length=max_length,
            validate_semantics=semantics is None,
        )
        semantics = checks or semantics
        results[kind] = {
            "n_tokens": n_tokens,
            "shapes": {
                f"{feature_kind}:{layer}": list(value.shape)
                for (feature_kind, layer), value in batch.items()
            },
            "finite": all(np.isfinite(value).all() for value in batch.values()),
        }
    return {
        "model": spec.model_name,
        "model_source": model_source,
        "lens": str(resolved_lens),
        "device": device,
        "n_layers": wrapped.n_layers,
        "d_model": wrapped.d_model,
        "selected_layers": layers,
        "condition_id": condition_id,
        "prompt_set": selected_prompt_set.key,
        "prompt_version": selected_prompt_set.version,
        "endpoint_audit": endpoint_audit,
        "hook_semantics": semantics,
        "prompts": results,
    }

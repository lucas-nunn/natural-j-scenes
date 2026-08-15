#!/usr/bin/env python3
"""Build the README's audited NSD/J-space vocabulary-readout figure.

Example (from the repository root):

    PYTHONPATH=src /path/to/python scripts/make_jspace_readout_figure.py \
      --result-root /path/to/subject1_result \
      --embedding-root /path/to/completed/qwen4b/embeddings \
      --nsd-hdf5 /path/to/nsd_stimuli.hdf5 \
      --stim-info /path/to/nsd_stim_info_merged.csv \
      --coco-annotations-dir /path/to/nsd/annotations \
      --captions /path/to/nsd_allWords_per_image.pkl \
      --model-dir /path/to/Qwen3.5-4B \
      --feature visualize__l23__j \
      --output docs/assets/visualize_layer23_jspace_readouts.png

The rows are selected without looking at their content. The subject-1 set is
restricted to images whose local COCO metadata says CC BY 2.0, then indices
floor(n / 6), floor(n / 2), and floor(5n / 6) sample the centers of three
equal spans in the sorted eligible set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from safetensors import safe_open
from transformers import AutoTokenizer

from jlens_nsd.prompts import captions_for_condition, load_caption_table

DEFAULT_FEATURE = "visualize__l23__j"
FEATURES = {
    "visualize__l23__j": {
        "prompt_kind": "visualize",
        "column_title": "J-space top words",
        "footer_label": "visualize prompt",
    },
    "plain__l23__j": {
        "prompt_kind": "plain",
        "column_title": "Caption-only top words",
        "footer_label": "caption-only (plain) prompt",
        "counterpart": "visualize__l23__j",
    },
    "integrate_readout__l23__j": {
        "prompt_kind": "integrate_readout",
        "column_title": "Integrated J-space top words",
        "footer_label": "matched readout + integration instruction",
        "counterpart": "minimal_readout__l23__j",
    },
    "minimal_readout__l23__j": {
        "prompt_kind": "minimal_readout",
        "column_title": "Minimal J-space top words",
        "footer_label": "matched readout without integration instruction",
        "counterpart": "integrate_readout__l23__j",
    },
}
FEATURES["visualize__l23__j"]["counterpart"] = "plain__l23__j"
MODEL_NAME = "Qwen/Qwen3.5-4B"
HEAD_KEY = "model.language_model.embed_tokens.weight"
NORM_KEY = "model.language_model.norm.weight"
TOP_K = 5
MARKUP_ONLY = re.compile(
    r"(?:</?[A-Za-z][^>]*>?|&(?:[A-Za-z][A-Za-z0-9]+|#[0-9]+|#x[0-9A-Fa-f]+);)$"
)
ESCAPED_WHITESPACE_ONLY = re.compile(r"(?:\\[nrtfv])+$")
FILE_EXTENSION_ONLY = re.compile(r"\.[A-Za-z0-9]{1,8}$")
URL_SCHEME_ONLY = re.compile(r"(?:https?|ftp)(?:://)?$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--nsd-hdf5", type=Path, required=True)
    parser.add_argument("--stim-info", type=Path, required=True)
    parser.add_argument("--coco-annotations-dir", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default="qwen4b",
        help="Embedding namespace below result-root (default: %(default)s)",
    )
    parser.add_argument(
        "--feature",
        choices=tuple(FEATURES),
        default=DEFAULT_FEATURE,
        help="Exact manifest feature to unembed (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifests(
    target_manifest: dict[str, Any],
    source_manifest: dict[str, Any],
    captions: Path,
    feature: str,
) -> None:
    if not target_manifest.get("complete") or not source_manifest.get("complete"):
        raise ValueError("both extraction manifests must be complete")
    target = target_manifest["config"]
    source = source_manifest["config"]
    shared_fields = (
        "caption_sha256",
        "d_model",
        "jlens_git_revision",
        "lens_sha256",
        "model_name",
        "normalization",
        "nsd_id_base",
        "position",
        "prompt_kinds",
        "prompt_version",
        "prompt_set",
    )
    mismatches = {
        field: (target.get(field), source.get(field))
        for field in shared_fields
        if target.get(field) != source.get(field)
    }
    if mismatches:
        raise ValueError(f"target/source extraction mismatch: {mismatches}")
    if target["model_name"] != MODEL_NAME or target["d_model"] != 2560:
        raise ValueError(f"unexpected model configuration: {target}")
    if target["caption_sha256"] != sha256_file(captions):
        raise ValueError("caption file hash does not match extraction manifest")
    if feature not in {entry["name"] for entry in target_manifest["features"]}:
        raise ValueError(f"{feature} absent from target manifest")
    if feature not in {entry["name"] for entry in source_manifest["features"]}:
        raise ValueError(f"{feature} absent from source manifest")
    prompt_kind = FEATURES[feature]["prompt_kind"]
    if (
        prompt_kind not in target["prompt_kinds"]
        or prompt_kind not in source["prompt_kinds"]
    ):
        raise ValueError(f"prompt kind {prompt_kind} absent from extraction config")


def select_conditions(
    condition_ids: np.ndarray, stim_info: Path, annotations_dir: Path
) -> tuple[list[int], dict[int, dict[str, Any]], int]:
    if condition_ids.ndim != 1 or len(condition_ids) < 3:
        raise ValueError("condition IDs must be a one-dimensional array")
    if not np.all(condition_ids[1:] > condition_ids[:-1]):
        raise ValueError("subject condition IDs must be strictly increasing")

    wanted_nsd_ids = {int(condition_id) - 1 for condition_id in condition_ids}
    rows_by_nsd_id: dict[int, dict[str, str]] = {}
    with stim_info.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            nsd_id = int(row["nsdId"])
            if nsd_id in wanted_nsd_ids:
                rows_by_nsd_id[nsd_id] = row
    if set(rows_by_nsd_id) != wanted_nsd_ids:
        raise ValueError("stimulus metadata does not cover the subject condition set")

    wanted_coco_ids: dict[str, set[int]] = {}
    for row in rows_by_nsd_id.values():
        wanted_coco_ids.setdefault(row["cocoSplit"], set()).add(int(row["cocoId"]))
    coco_images: dict[tuple[str, int], dict[str, Any]] = {}
    coco_licenses: dict[tuple[str, int], dict[str, Any]] = {}
    for split, wanted_ids in wanted_coco_ids.items():
        annotations = load_json(annotations_dir / f"captions_{split}.json")
        for license_entry in annotations["licenses"]:
            coco_licenses[(split, int(license_entry["id"]))] = license_entry
        for image in annotations["images"]:
            coco_id = int(image["id"])
            if coco_id in wanted_ids:
                coco_images[(split, coco_id)] = image

    records: dict[int, dict[str, Any]] = {}
    eligible: list[int] = []
    for condition_id_raw in condition_ids:
        condition_id = int(condition_id_raw)
        row = rows_by_nsd_id[condition_id - 1]
        split = row["cocoSplit"]
        coco_id = int(row["cocoId"])
        image = coco_images[(split, coco_id)]
        license_entry = coco_licenses[(split, int(image["license"]))]
        record = {
            "nsd_metadata_id_0based": int(row["nsdId"]),
            "coco_id": coco_id,
            "coco_split": split,
            "flickr_url": image["flickr_url"],
            "license_name": license_entry["name"],
            "license_url": license_entry["url"].replace("http://", "https://"),
        }
        records[condition_id] = record
        if record["license_url"].rstrip("/") == (
            "https://creativecommons.org/licenses/by/2.0"
        ):
            eligible.append(condition_id)
    if len(eligible) < 6:
        raise ValueError("fewer than six CC BY 2.0 subject conditions")
    indices = (len(eligible) // 6, len(eligible) // 2, 5 * len(eligible) // 6)
    selected = [eligible[index] for index in indices]
    return selected, records, len(eligible)


def load_vectors(
    embedding_root: Path,
    source_manifest: dict[str, Any],
    selected: list[int],
    feature: str,
) -> tuple[np.ndarray, dict[int, str], dict[int, str]]:
    wanted = set(selected)
    found: dict[int, np.ndarray] = {}
    chunk_for: dict[int, str] = {}
    counterpart_hash_for: dict[int, str] = {}
    counterpart_feature = FEATURES[feature]["counterpart"]
    chunks_dir = embedding_root / "chunks"
    for chunk_name in source_manifest["completed_chunks"]:
        path = chunks_dir / chunk_name
        with np.load(path, allow_pickle=False) as chunk:
            ids = chunk["condition_ids"]
            for row in np.flatnonzero(np.isin(ids, selected)):
                condition_id = int(ids[row])
                vector = chunk[feature][row]
                if condition_id in found:
                    raise ValueError(f"duplicate vector for condition {condition_id}")
                if vector.shape != (2560,) or vector.dtype != np.float32:
                    raise ValueError(
                        f"invalid {feature} vector for condition {condition_id}"
                    )
                if not np.isfinite(vector).all():
                    raise ValueError(f"non-finite vector for condition {condition_id}")
                counterpart = chunk[counterpart_feature][row]
                if (
                    counterpart.shape != vector.shape
                    or counterpart.dtype != vector.dtype
                ):
                    raise ValueError(
                        f"invalid counterpart vector for condition {condition_id}"
                    )
                if np.array_equal(vector, counterpart):
                    raise ValueError(
                        f"{feature} aliases {counterpart_feature} for condition "
                        f"{condition_id}"
                    )
                found[condition_id] = vector.copy()
                chunk_for[condition_id] = chunk_name
                counterpart_hash_for[condition_id] = sha256_array(counterpart)
        if set(found) == wanted:
            break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"conditions absent from completed chunks: {sorted(missing)}")
    return (
        np.stack([found[condition_id] for condition_id in selected]),
        chunk_for,
        counterpart_hash_for,
    )


def tensor_location(index: dict[str, Any], key: str, model_dir: Path) -> Path:
    try:
        return model_dir / index["weight_map"][key]
    except KeyError as error:
        raise ValueError(f"model index does not contain {key}") from error


def exact_unembed(
    vectors: np.ndarray, model_dir: Path
) -> tuple[np.ndarray, AutoTokenizer, dict[str, Any]]:
    config = load_json(model_dir / "config.json")
    text_config = config["text_config"]
    index = load_json(model_dir / "model.safetensors.index.json")
    if text_config["hidden_size"] != vectors.shape[1]:
        raise ValueError("model hidden size does not match stored vectors")
    if not text_config.get("tie_word_embeddings", config.get("tie_word_embeddings")):
        raise ValueError("expected the Qwen output head to use tied embeddings")
    if "lm_head.weight" in index["weight_map"]:
        raise ValueError("checkpoint unexpectedly stores a separate LM head")

    norm_path = tensor_location(index, NORM_KEY, model_dir)
    with safe_open(str(norm_path), framework="pt", device="cpu") as tensors:
        norm_weight = tensors.get_tensor(NORM_KEY).float()
    if tuple(norm_weight.shape) != (vectors.shape[1],):
        raise ValueError("unexpected final RMSNorm shape")

    # Match HFLensModel.unembed exactly. It first casts residuals to the tied
    # head dtype. Qwen3.5RMSNorm then normalizes in fp32, applies (1 + weight),
    # and casts back before the bias-free tied output projection.
    residual = torch.from_numpy(vectors).to(torch.bfloat16)
    eps = float(text_config["rms_norm_eps"])
    normalized = residual.float()
    normalized = normalized * torch.rsqrt(
        normalized.pow(2).mean(-1, keepdim=True) + eps
    )
    normalized = (normalized * (1.0 + norm_weight)).to(torch.bfloat16)

    vocab_size = int(text_config["vocab_size"])
    logits = np.empty((len(vectors), vocab_size), dtype=np.float32)
    head_path = tensor_location(index, HEAD_KEY, model_dir)
    chunk_size = 16_384
    with safe_open(str(head_path), framework="pt", device="cpu") as tensors:
        head = tensors.get_slice(HEAD_KEY)
        if tuple(head.get_shape()) != (vocab_size, vectors.shape[1]):
            raise ValueError("unexpected tied output-head shape")
        if str(head.get_dtype()) != "BF16":
            raise ValueError(f"unexpected output-head dtype: {head.get_dtype()}")
        for start in range(0, vocab_size, chunk_size):
            stop = min(start + chunk_size, vocab_size)
            weight = head[start:stop]
            chunk_logits = F.linear(normalized, weight).float()
            softcap = text_config.get("final_logit_softcapping")
            if softcap is not None:
                chunk_logits = float(softcap) * torch.tanh(
                    chunk_logits / float(softcap)
                )
            logits[:, start:stop] = chunk_logits.numpy()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    audit = {
        "adapter": "Qwen3.5 RMSNorm(additive weight) + tied embedding head",
        "epsilon": eps,
        "head_dtype": "bfloat16",
        "head_key": HEAD_KEY,
        "norm_key": NORM_KEY,
        "softcap": text_config.get("final_logit_softcapping"),
        "vocab_size": vocab_size,
    }
    return logits, tokenizer, audit


def token_display(tokenizer: AutoTokenizer, token_id: int) -> str:
    decoded = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return " ".join(decoded.split())


def top_words(
    logits: np.ndarray, tokenizer: AutoTokenizer
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    special_ids = set(tokenizer.all_special_ids)
    token_ids = np.arange(logits.shape[1])
    rows: list[list[dict[str, Any]]] = []
    rejected_counts = {
        "special": 0,
        "empty": 0,
        "formatting_control": 0,
        "non_alphanumeric": 0,
        "duplicate": 0,
    }
    for scores in logits:
        # Logits are bfloat16-valued. Token ID ascending is the deterministic
        # tie-break, and ranks below refer to this complete unfiltered ordering.
        order = np.lexsort((token_ids, -scores))
        retained: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_rank, token_id_raw in enumerate(order, start=1):
            token_id = int(token_id_raw)
            if token_id in special_ids:
                rejected_counts["special"] += 1
                continue
            display = token_display(tokenizer, token_id)
            if not display:
                rejected_counts["empty"] += 1
                continue
            if (
                MARKUP_ONLY.fullmatch(display)
                or ESCAPED_WHITESPACE_ONLY.fullmatch(display)
                or FILE_EXTENSION_ONLY.fullmatch(display)
                or URL_SCHEME_ONLY.fullmatch(display)
            ):
                rejected_counts["formatting_control"] += 1
                continue
            if not any(unicodedata.category(char)[0] in {"L", "N"} for char in display):
                rejected_counts["non_alphanumeric"] += 1
                continue
            duplicate_key = unicodedata.normalize("NFKC", display).casefold()
            if duplicate_key in seen:
                rejected_counts["duplicate"] += 1
                continue
            seen.add(duplicate_key)
            retained.append(
                {
                    "token": display,
                    "token_id": token_id,
                    "raw_rank": raw_rank,
                    "logit": float(scores[token_id]),
                }
            )
            if len(retained) == TOP_K:
                break
        if len(retained) != TOP_K:
            raise ValueError("could not retain enough vocabulary tokens")
        rows.append(retained)
    filter_audit = {
        "rule": (
            "skip special IDs, empty decodes, full-string markup/entities or "
            "escaped-whitespace controls, dot-prefixed file extensions, or bare URL "
            "schemes; skip decodes without any Unicode letter/number and "
            "NFKC-casefold duplicates; preserve raw rank"
        ),
        "rejected_before_top_k": rejected_counts,
    }
    return rows, filter_audit


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    root = Path("/usr/share/fonts/truetype/dejavu")
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"required figure font missing: {path}")
    return ImageFont.truetype(str(path), size=size)


def fit_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    selected_font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=selected_font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_figure(
    images: list[Image.Image],
    captions: list[list[str]],
    decoded: list[list[dict[str, Any]]],
    conditions: list[int],
    output: Path,
    metadata: dict[str, Any],
    feature: str,
) -> None:
    width, height = 1800, 1080
    margin = 52
    header_h = 92
    row_h = 286
    image_w, captions_w, words_w = 340, 780, 500
    gap = 38
    x_image = margin
    x_captions = x_image + image_w + gap
    x_words = x_captions + captions_w + gap
    palette = {
        "ink": "#172033",
        "muted": "#657083",
        "line": "#DDE3EA",
        "panel": "#F5F7FA",
        "accent": "#2457C5",
        "accent_soft": "#E9F0FF",
        "white": "#FFFFFF",
    }
    canvas = Image.new("RGB", (width, height), palette["white"])
    draw = ImageDraw.Draw(canvas)
    title_font = font("DejaVuSans-Bold.ttf", 32)
    arrow_font = font("DejaVuSans.ttf", 34)
    condition_font = font("DejaVuSans-Bold.ttf", 22)
    caption_font = font("DejaVuSans.ttf", 23)
    caption_italic = font("DejaVuSans.ttf", 21)
    mono_font = font("DejaVuSansMono.ttf", 25)
    token_font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=25
    )
    footer_font = font("DejaVuSans.ttf", 20)

    headers = (
        ("NSD stimulus", x_image, image_w),
        ("Human captions", x_captions, captions_w),
        (FEATURES[feature]["column_title"], x_words, words_w),
    )
    for text, x, column_width in headers:
        text_width = draw.textlength(text, font=title_font)
        draw.text(
            (x + (column_width - text_width) / 2, 32),
            text,
            font=title_font,
            fill=palette["ink"],
        )
    for arrow_x in (x_captions - gap / 2, x_words - gap / 2):
        draw.text((arrow_x - 12, 31), "→", font=arrow_font, fill=palette["accent"])

    for row, (stimulus, row_captions, row_words, condition_id) in enumerate(
        zip(images, captions, decoded, conditions, strict=True)
    ):
        y = header_h + row * row_h
        draw.rounded_rectangle(
            (margin, y, width - margin, y + row_h - 14),
            radius=20,
            fill=palette["panel"],
        )
        image_size = 236
        image_x = x_image + (image_w - image_size) // 2
        image_y = y + 25
        draw.rounded_rectangle(
            (
                image_x - 4,
                image_y - 4,
                image_x + image_size + 4,
                image_y + image_size + 4,
            ),
            radius=16,
            fill=palette["white"],
            outline=palette["line"],
            width=2,
        )
        resized = stimulus.resize((image_size, image_size), Image.Resampling.LANCZOS)
        mask = Image.new("L", (image_size, image_size), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, image_size - 1, image_size - 1), radius=12, fill=255
        )
        canvas.paste(resized, (image_x, image_y), mask)
        badge = f"NSD {condition_id}"
        badge_w = math.ceil(draw.textlength(badge, font=condition_font)) + 26
        draw.rounded_rectangle(
            (image_x + 12, image_y + 12, image_x + 12 + badge_w, image_y + 48),
            radius=12,
            fill=palette["accent"],
        )
        draw.text(
            (image_x + 25, image_y + 16),
            badge,
            font=condition_font,
            fill=palette["white"],
        )

        caption_x = x_captions + 24
        caption_y = y + 28
        caption_line_h = 31
        for caption in row_captions[:2]:
            lines = fit_lines(draw, f"• {caption}", caption_font, captions_w - 50)
            for line in lines:
                draw.text(
                    (caption_x, caption_y), line, font=caption_font, fill=palette["ink"]
                )
                caption_y += caption_line_h
            caption_y += 11
        remaining = len(row_captions) - 2
        if remaining:
            suffix = "caption" if remaining == 1 else "captions"
            draw.text(
                (caption_x, caption_y),
                f"… +{remaining} additional human {suffix}",
                font=caption_italic,
                fill=palette["muted"],
            )

        words_x = x_words + 22
        words_y = y + 24
        for item in row_words:
            rank = item["raw_rank"]
            token = item["token"]
            score = item["logit"]
            draw.rounded_rectangle(
                (words_x, words_y, x_words + words_w - 22, words_y + 42),
                radius=10,
                fill=palette["accent_soft"],
            )
            draw.text(
                (words_x + 14, words_y + 6),
                f"#{rank}",
                font=mono_font,
                fill=palette["ink"],
            )
            draw.text(
                (words_x + 92, words_y + 5),
                token[:16],
                font=token_font,
                fill=palette["ink"],
            )
            score_label = f"{score:.2f}"
            score_width = draw.textlength(score_label, font=mono_font)
            draw.text(
                (x_words + words_w - 36 - score_width, words_y + 6),
                score_label,
                font=mono_font,
                fill=palette["ink"],
            )
            words_y += 48

    footer_y = header_h + 3 * row_h + 4
    footer = (
        f"{FEATURES[feature]['footer_label']}  •  layer 23  •  "
        "Qwen3.5-4B vocabulary rank + logit  •  deterministic artifact filtering"
    )
    footer_w = draw.textlength(footer, font=footer_font)
    draw.text(
        ((width - footer_w) / 2, footer_y),
        footer,
        font=footer_font,
        fill=palette["muted"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    description = "NSD examples with audited layer-23 J-space readouts"
    if feature != DEFAULT_FEATURE:
        description = f"{description} ({feature})"
    png_info.add_text(
        "Description",
        description,
    )
    png_info.add_text(
        "Audit", json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    canvas.save(output, format="PNG", optimize=True, pnginfo=png_info)


def main() -> None:
    args = parse_args()
    condition_path = args.result_root / "conditions" / "subj01_condition_ids.npy"
    target_manifest_path = (
        args.result_root / "embeddings" / args.run_name / "manifest.json"
    )
    source_manifest_path = args.embedding_root / "manifest.json"
    target_manifest = load_json(target_manifest_path)
    source_manifest = load_json(source_manifest_path)
    validate_manifests(target_manifest, source_manifest, args.captions, args.feature)

    condition_ids = np.load(condition_path, allow_pickle=False)
    selected, stimulus_metadata, n_license_eligible = select_conditions(
        condition_ids, args.stim_info, args.coco_annotations_dir
    )
    vectors, chunk_for, counterpart_hash_for = load_vectors(
        args.embedding_root, source_manifest, selected, args.feature
    )
    logits, tokenizer, unembed_audit = exact_unembed(vectors, args.model_dir)
    decoded, filter_audit = top_words(logits, tokenizer)

    caption_table = load_caption_table(args.captions)
    row_captions = [captions_for_condition(caption_table, cid) for cid in selected]
    with h5py.File(args.nsd_hdf5, "r") as handle:
        stimuli = handle["imgBrick"]
        if stimuli.shape != (73_000, 425, 425, 3) or stimuli.dtype != np.uint8:
            raise ValueError(f"unexpected NSD stimulus dataset: {stimuli.shape}")
        raw_images = [stimuli[cid - 1] for cid in selected]
    images = [Image.fromarray(array, mode="RGB") for array in raw_images]

    rows = []
    for cid, image_array, captions, vector, words in zip(
        selected, raw_images, row_captions, vectors, decoded, strict=True
    ):
        row = {
            "condition_id_1based": cid,
            "caption_count": len(captions),
            "image_sha256": sha256_array(image_array),
            "vector_sha256": sha256_array(vector),
            "source_chunk": chunk_for[cid],
            "top_words": words,
            **stimulus_metadata[cid],
        }
        if args.feature != DEFAULT_FEATURE:
            row.update(
                {
                    "source_array": args.feature,
                    "counterpart_vector_sha256": counterpart_hash_for[cid],
                }
            )
        rows.append(row)
    metadata = {
        "feature": args.feature,
        "selection": (
            "sorted subject-1 CC BY 2.0 subset indices "
            "[floor(n/6), floor(n/2), floor(5n/6)]"
        ),
        "n_license_eligible": n_license_eligible,
        "rows": rows,
        "unembedding": unembed_audit,
        "filter": filter_audit,
    }
    if args.feature != DEFAULT_FEATURE:
        metadata.update(
            {
                "prompt_kind": FEATURES[args.feature]["prompt_kind"],
                "counterpart_feature": FEATURES[args.feature]["counterpart"],
                "provenance": {
                    "captions_sha256": sha256_file(args.captions),
                    "source_manifest_sha256": sha256_file(source_manifest_path),
                    "target_manifest_sha256": sha256_file(target_manifest_path),
                    "source_condition_ids_hash": source_manifest["config"][
                        "condition_ids_hash"
                    ],
                    "target_condition_ids_hash": target_manifest["config"][
                        "condition_ids_hash"
                    ],
                    "source_prompt_sha256": source_manifest["config"][
                        "prompt_source_sha256"
                    ],
                    "target_prompt_sha256": target_manifest["config"][
                        "prompt_source_sha256"
                    ],
                },
            }
        )
    draw_figure(
        images,
        row_captions,
        decoded,
        selected,
        args.output,
        metadata,
        args.feature,
    )

    with Image.open(args.output) as rendered:
        if rendered.size != (1800, 1080) or rendered.mode != "RGB":
            raise ValueError("rendered figure failed size/mode validation")
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"figure_sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()

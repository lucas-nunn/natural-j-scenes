#!/usr/bin/env python3
"""How many tokens does the caption readout actually need?

Pooling beat the single-endpoint readout by 8-23x on semantic alignment, and the
reason is that 73.7% of caption prompts end in the same token, a period. That
raises the practical question the brain run cannot answer cheaply: is the whole
prompt required, or does the endpoint simply need a little company?

This sweeps pooling width in one forward pass — every variant is a different
reduction over the same collected token states, so the comparison holds model,
prompt, tokenizer, layer and stimuli exactly fixed. Variants:

``last_k``
    mean over the final ``k`` valid tokens. ``last_1`` reproduces the historical
    endpoint readout; ``all`` reproduces the production pooled readout, which
    doubles as a correctness check against the stored features.
``no_punct``
    mean over valid tokens excluding a small punctuation set. Distinguishes
    "punctuation is uninformative" from "punctuation is actively harmful": if
    dropping it beats the full mean, the period is diluting the average, not
    merely failing to contribute.

Each variant is scored by RDM correlation against MPNet sentence embeddings of
the same captions — an independent semantic reference with no access to Qwen,
the lens, or the brain.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from analyze_lens_geometry import correlation_rdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--union-ids", type=Path, required=True)
    parser.add_argument("--mpnet-embeddings", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens_nsd.prompts import load_caption_table, prompts_for_condition

    union = np.load(args.union_ids, allow_pickle=False)
    if args.max_conditions:
        union = union[: args.max_conditions]
    table = load_caption_table(args.captions)
    prompts = [
        prompts_for_condition(table, int(cid), "historical")["plain"] for cid in union
    ]

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    tokenizer.padding_side = "right"
    model = (
        AutoModelForCausalLM.from_pretrained(str(args.model_path), dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )

    # Punctuation-only tokens, resolved from the tokenizer rather than assumed.
    punctuation = {
        token_id
        for token_id in range(tokenizer.vocab_size)
        if (text := tokenizer.decode([token_id]).strip())
        and all(character in ".,;:!?'\"-()" for character in text)
    }

    widths = [1, 2, 4, 8, 16, 32]
    names = [f"last_{k}" for k in widths] + ["all", "no_punct"]
    sums = {name: [] for name in names}

    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            hidden = (
                model(**encoded, output_hidden_states=True)
                .hidden_states[args.layer + 1]
                .float()
            )
            mask = encoded["attention_mask"]
            ids = encoded["input_ids"]
            lengths = mask.sum(dim=1)

            for name in names:
                if name == "all":
                    weights = mask.float()
                elif name == "no_punct":
                    is_punct = torch.zeros_like(mask, dtype=torch.bool)
                    for token_id in punctuation:
                        is_punct |= ids == token_id
                    weights = (mask.bool() & ~is_punct).float()
                    empty = weights.sum(dim=1) == 0
                    weights[empty] = mask.float()[empty]
                else:
                    k = int(name.split("_")[1])
                    positions = torch.arange(mask.shape[1], device=mask.device)
                    lower = (lengths - k).clamp(min=0).unsqueeze(1)
                    weights = (mask.bool() & (positions.unsqueeze(0) >= lower)).float()
                pooled = (hidden * weights.unsqueeze(-1)).sum(1) / weights.sum(
                    1, keepdim=True
                )
                sums[name].append(pooled.cpu().numpy().astype(np.float32))

    features = {name: np.concatenate(blocks) for name, blocks in sums.items()}

    lens = torch.load(args.lens_path, map_location="cpu", weights_only=False)
    matrix = np.asarray(lens["J"][args.layer], dtype=np.float32).astype(np.float64)

    with args.mpnet_embeddings.open("rb") as handle:
        mpnet = np.asarray(pickle.load(handle))
    semantic = mpnet[union - 1]

    rng = np.random.default_rng(0)
    samples = [
        rng.choice(len(union), size=min(100, len(union)), replace=False)
        for _ in range(64)
    ]

    rows = []
    for name in names:
        raw = features[name].astype(np.float64)
        transported = raw @ matrix.T
        scores = {"raw": [], "j": []}
        for choice in samples:
            reference = correlation_rdm(semantic[choice])
            scores["raw"].append(
                float(np.corrcoef(correlation_rdm(raw[choice]), reference)[0, 1])
            )
            scores["j"].append(
                float(
                    np.corrcoef(correlation_rdm(transported[choice]), reference)[0, 1]
                )
            )
        rows.append(
            {
                "variant": name,
                "raw_vs_semantic": float(np.mean(scores["raw"])),
                "j_vs_semantic": float(np.mean(scores["j"])),
            }
        )
        print(
            f"{name:10}  raw~MPNet={rows[-1]['raw_vs_semantic']:.4f}  "
            f"j~MPNet={rows[-1]['j_vs_semantic']:.4f}",
            flush=True,
        )

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "layer": args.layer,
                    "n_conditions": int(len(union)),
                    "n_punctuation_tokens": len(punctuation),
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

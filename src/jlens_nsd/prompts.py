"""Deterministic reconstruction and prompting of tokenized COCO captions."""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from pathlib import Path

TERMINATORS = {".", "!", "?"}
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}"}
NO_SPACE_AFTER = {"(", "[", "{"}


def detokenize(tokens: Sequence[str]) -> str:
    """Reverse the relevant NLTK word-tokenization rules without data files."""
    output = ""
    previous = ""
    for token in (str(item) for item in tokens):
        attach = (
            not output
            or token in NO_SPACE_BEFORE
            or token.startswith("'")
            or token == "n't"
            or previous in NO_SPACE_AFTER
        )
        output += token if attach else f" {token}"
        previous = token
    output = re.sub(r"\s+", " ", output).strip()
    return output


def split_caption_tokens(tokens: Sequence[str]) -> list[str]:
    """Recover sentence/caption boundaries retained as punctuation tokens."""
    sentences: list[str] = []
    current: list[str] = []
    for token in tokens:
        current.append(str(token))
        if token in TERMINATORS:
            sentence = detokenize(current)
            if sentence:
                sentences.append(sentence)
            current = []
    if current:
        sentence = detokenize(current)
        if sentence:
            sentences.append(sentence)
    if not sentences:
        raise ValueError("caption token row is empty")
    return sentences


def load_caption_table(path: Path) -> list[list[str]]:
    with path.open("rb") as handle:
        table = pickle.load(handle)
    if not isinstance(table, list) or len(table) != 73_000:
        raise ValueError(
            f"expected a 73,000-row caption list, found {type(table)} "
            f"with length {len(table) if hasattr(table, '__len__') else 'NA'}"
        )
    return table


def captions_for_condition(
    table: Sequence[Sequence[str]], condition_id: int
) -> list[str]:
    """Look up a 1-based NSD condition ID at the one explicit -1 boundary."""
    if not 1 <= condition_id <= len(table):
        raise IndexError(f"NSD condition ID out of range: {condition_id}")
    return split_caption_tokens(table[condition_id - 1])


def build_prompt(captions: Sequence[str], kind: str) -> str:
    clean = [re.sub(r"\s+", " ", caption).strip() for caption in captions]
    if not clean or any(not caption for caption in clean):
        raise ValueError("captions must contain non-empty strings")
    caption_block = "\n".join(f"- {caption}" for caption in clean)
    if kind == "plain":
        return caption_block
    if kind != "visualize":
        raise ValueError(f"unknown prompt kind: {kind}")
    return (
        "Construct one coherent visual scene from the source captions. "
        "Reconcile their overlap into a single image: represent the entities, "
        "attributes, actions, spatial relations, and setting together. Infer "
        "only visually plausible structure. Form the scene rather than merely "
        "restating or listing caption words.\n\n"
        f"Source captions:\n{caption_block}\n\n"
        "Integrated visual scene:"
    )


def prompts_for_condition(
    table: Sequence[Sequence[str]], condition_id: int
) -> dict[str, str]:
    captions = captions_for_condition(table, condition_id)
    return {
        "visualize": build_prompt(captions, "visualize"),
        "plain": build_prompt(captions, "plain"),
    }

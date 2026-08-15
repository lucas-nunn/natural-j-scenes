#!/usr/bin/env python3
"""Compose existing layer-23 subject maps into a matched raw-then-J montage.

This script does not recompute RSA, projection, thresholds, or color scales. It
only resizes and places the already-rendered per-subject maps. Each source panel
retains its independently symmetric color limit and title from the analysis.
The ordered subject selection defaults to 1,2,3,4 and can be set explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

AVAILABLE_SUBJECTS = tuple(range(1, 9))
DEFAULT_SUBJECTS = (1, 2, 3, 4)
KINDS = ("raw", "j")
# Source panels must all share one size so placement is exact. The specific
# size is a property of the render, not a constant: readout modes and pycortex
# versions produce different canvases. The invariant is agreement, so it is
# derived from the first panel and enforced across the rest.
HEADER_HEIGHT = 92
ROW_STRIDE = 493
PANEL_SIZE = (1120, 487)


def parse_subjects(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, ordered subject selection."""
    try:
        subjects = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "subjects must be comma-separated integers"
        ) from exc
    if not subjects or any(subject not in AVAILABLE_SUBJECTS for subject in subjects):
        raise argparse.ArgumentTypeError("subjects must be selected from 1..8")
    if len(subjects) != len(set(subjects)):
        raise argparse.ArgumentTypeError("subjects must be unique")
    return subjects


def output_size(subjects: Sequence[int]) -> tuple[int, int]:
    """Return montage dimensions for the requested subject rows."""
    if not subjects:
        raise ValueError("at least one subject is required")
    return (len(KINDS) * PANEL_SIZE[0], HEADER_HEIGHT + len(subjects) * ROW_STRIDE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-root",
        type=Path,
        required=True,
        help="Directory containing the existing qwen4b per-subject PNG maps",
    )
    parser.add_argument(
        "--prompt-kind",
        choices=("plain", "visualize", "plain_mean_pool"),
        default="plain",
        help="Exact manifest prompt kind to compose (default: %(default)s)",
    )
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=DEFAULT_SUBJECTS,
        metavar="ID,ID,...",
        help="Ordered subject IDs to compose (default: 1,2,3,4)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(root: Path, prompt_kind: str, kind: str, subject: int) -> Path:
    return root / f"{prompt_kind}__l23__{kind}_subj{subject:02d}.png"


def centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    selected_font: ImageFont.FreeTypeFont,
) -> None:
    width = draw.textlength(text, font=selected_font)
    draw.text((center_x - width / 2, y), text, font=selected_font, fill="#111111")


def compose(args: argparse.Namespace) -> dict[str, Any]:
    subjects = tuple(args.subjects)
    dimensions = output_size(subjects)
    sources: dict[tuple[str, int], Image.Image] = {}
    source_audit: list[dict[str, Any]] = []
    source_size: tuple[int, int] | None = None
    for subject in subjects:
        for kind in KINDS:
            path = source_path(args.figure_root, args.prompt_kind, kind, subject)
            if not path.is_file():
                raise FileNotFoundError(f"missing source map: {path}")
            with Image.open(path) as opened:
                opened.load()
                if source_size is None:
                    source_size = opened.size
                elif opened.size != source_size:
                    raise ValueError(
                        f"map dimensions differ for {path}: {opened.size} "
                        f"!= {source_size}"
                    )
                image = opened.convert("RGB")
            sources[(kind, subject)] = image
            source_audit.append(
                {
                    "feature": f"{args.prompt_kind}__l23__{kind}",
                    "subject": subject,
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "dimensions": list(source_size),
                }
            )

    canvas = Image.new("RGB", dimensions, "white")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=38
    )
    centered_text(draw, "Raw LLM (hₗ)", PANEL_SIZE[0] // 2, 25, heading)
    centered_text(
        draw,
        "J-space (Jₗhₗ)",
        PANEL_SIZE[0] + PANEL_SIZE[0] // 2,
        25,
        heading,
    )

    for row, subject in enumerate(subjects):
        y = HEADER_HEIGHT + row * ROW_STRIDE
        for column, kind in enumerate(KINDS):
            resized = sources[(kind, subject)].resize(
                PANEL_SIZE, Image.Resampling.LANCZOS
            )
            canvas.paste(resized, (column * PANEL_SIZE[0], y))

    audit = {
        "prompt_kind": args.prompt_kind,
        "features": [
            f"{args.prompt_kind}__l23__raw",
            f"{args.prompt_kind}__l23__j",
        ],
        "subjects": list(subjects),
        "column_order": ["raw", "j"],
        "source_dimensions": list(source_size),
        "output_dimensions": list(dimensions),
        "operation": "composition only; no RSA, projection, or map recomputation",
        "scaling": (
            "source plots retained unchanged except LANCZOS resize; each panel keeps "
            "its original independent symmetric color scale printed in its title"
        ),
        "sources": source_audit,
    }
    exif = Image.Exif()
    exif[270] = json.dumps(audit, sort_keys=True, separators=(",", ":"))
    exif[305] = "make_layer23_brain_map_montage.py"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        args.output,
        format="JPEG",
        quality=92,
        optimize=True,
        progressive=True,
        exif=exif,
    )

    with Image.open(args.output) as rendered:
        rendered.load()
        if rendered.size != dimensions or rendered.mode != "RGB":
            raise ValueError("rendered montage failed size/mode validation")
        stored_audit = json.loads(rendered.getexif()[270])
        if stored_audit != audit:
            raise ValueError("rendered montage audit metadata failed round-trip")
    return audit


def main() -> None:
    args = parse_args()
    audit = compose(args)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"figure_sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()

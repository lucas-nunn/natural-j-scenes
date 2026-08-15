"""Command-line entry point for isolated Jacobian Lens × NSD stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from .conditions import load_union_ids, prepare_conditions
from .config import (
    DEFAULT_PROMPT_SET,
    MODEL_SPECS,
    ExperimentPaths,
    model_spec,
    run_name,
    validate_subjects,
)
from .extract import (
    extract_embeddings,
    prefetch_artifacts,
    preflight,
)
from .io_utils import atomic_json
from .prompts import PROMPT_SETS, load_caption_table, prompts_for_condition
from .rdms import prepare_grouped_rdms
from .stages import (
    plot_individual_maps,
    project_subjects,
    run_searchlight_subject,
    summarize,
)


def _paths(args) -> ExperimentPaths:
    return ExperimentPaths.from_values(
        results=args.results_dir,
        nsd_dir=args.nsd_dir,
        captions=args.captions,
        mpnet_base=args.mpnet_base,
        jlens_checkout=args.jlens_checkout,
    )


def _spec(args):
    return model_spec(
        args.profile,
        local_model_path=getattr(args, "model_path", None),
        local_lens_root=getattr(args, "lens_root", None),
    )


def _layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("--layers cannot be empty")
    return layers


def _subjects(value: str) -> list[int]:
    subjects = [int(item.strip()) for item in value.split(",") if item.strip()]
    try:
        return list(validate_subjects(subjects))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def smoke(
    paths: ExperimentPaths,
    *,
    with_data: bool = False,
    subjects: list[int] | None = None,
    prompt_set_key: str = DEFAULT_PROMPT_SET,
) -> dict:
    """Fast model/GPU-free validation, optionally including configured data."""
    table = [["A", "dog", "runs", ".", "The", "dog", "crosses", "grass", "."]]
    prompts = [prompts_for_condition(table, 1, prompt_set_key)]
    rng = np.random.default_rng(0)
    toy = rng.normal(size=(5, 16)).astype(np.float32)
    rdm = pdist(toy, metric="correlation").astype(np.float32)
    if rdm.shape != (10,) or not np.isfinite(rdm).all():
        raise RuntimeError("toy condensed RDM smoke failed")

    result = {
        "ok": True,
        "prompt_lengths_chars": {
            kind: [len(row[kind]) for row in prompts]
            for kind in PROMPT_SETS[prompt_set_key].kinds
        },
        "prompt_set": prompt_set_key,
        "toy_rdm_length": len(rdm),
        "results_dir": str(paths.results),
    }
    if with_data:
        selected_subjects = validate_subjects(subjects or range(1, 9))
        paths.require("captions")
        assert paths.captions is not None
        union_ids = load_union_ids(paths)
        captions = load_caption_table(paths.captions)
        prompts_for_condition(captions, int(union_ids[0]), prompt_set_key)
        for subject in selected_subjects:
            ids = np.load(
                paths.conditions / f"subj{subject:02d}_condition_ids.npy",
                allow_pickle=False,
            )
            sampled = np.load(
                paths.conditions / f"subj{subject:02d}_sampling_condition_ids.npy",
                allow_pickle=False,
            )
            if ids.shape != (835,) or sampled.shape != (8, 100):
                raise RuntimeError(f"subj{subject:02d} alignment smoke failed")
        result["data_contract"] = {
            "n_union_ids": len(union_ids),
            "subjects": list(selected_subjects),
            "ok": True,
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--nsd-dir", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--mpnet-base", type=Path)
    parser.add_argument("--jlens-checkout", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="lock condition IDs and sample choices"
    )
    prepare.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    smoke_parser = subparsers.add_parser(
        "smoke", help="run the fast model-free smoke test"
    )
    smoke_parser.add_argument("--with-data", action="store_true")
    smoke_parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    smoke_parser.add_argument(
        "--prompt-set", choices=sorted(PROMPT_SETS), default=DEFAULT_PROMPT_SET
    )

    for command in ("prefetch", "preflight", "extract", "rdms"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--profile", choices=sorted(MODEL_SPECS), default="qwen4b"
        )
        if command != "prefetch":
            subparser.add_argument(
                "--prompt-set",
                choices=sorted(PROMPT_SETS),
                default=DEFAULT_PROMPT_SET,
            )
        if command in {"prefetch", "preflight", "extract"}:
            subparser.add_argument("--model-path", type=Path)
            subparser.add_argument("--lens-root", type=Path)
        if command in {"preflight", "extract"}:
            subparser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            subparser.add_argument("--allow-download", action="store_true")
            subparser.add_argument("--lens-path", type=Path)
            subparser.add_argument("--max-length", type=int, default=256)
        if command == "extract":
            subparser.add_argument("--layers")
            subparser.add_argument("--batch-size", type=int, default=8)
            subparser.add_argument("--chunk-size", type=int, default=64)
            subparser.add_argument("--seed", type=int, default=0)
            subparser.add_argument("--max-conditions", type=int)
            subparser.add_argument("--output-name")
        if command == "rdms":
            subparser.add_argument("--subjects", default="1,2,3,4,5,6,7,8")

    searchlight = subparsers.add_parser("searchlight")
    searchlight.add_argument("--profile", choices=sorted(MODEL_SPECS), default="qwen4b")
    searchlight.add_argument(
        "--prompt-set", choices=sorted(PROMPT_SETS), default=DEFAULT_PROMPT_SET
    )
    searchlight.add_argument("--subject", type=int, required=True)
    searchlight.add_argument("--allow-cpu", action="store_true")
    searchlight.add_argument("--max-samples", type=int)

    project = subparsers.add_parser("project")
    project.add_argument("--profile", choices=sorted(MODEL_SPECS), default="qwen4b")
    project.add_argument(
        "--prompt-set", choices=sorted(PROMPT_SETS), default=DEFAULT_PROMPT_SET
    )
    project.add_argument("--subjects", default="1,2,3,4,5,6,7,8")

    plot = subparsers.add_parser("plot")
    plot.add_argument("--profile", choices=sorted(MODEL_SPECS), default="qwen4b")
    plot.add_argument(
        "--prompt-set", choices=sorted(PROMPT_SETS), default=DEFAULT_PROMPT_SET
    )
    plot.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    plot.add_argument(
        "--features",
        help="optional comma-separated feature names; default plots all",
    )
    plot.add_argument("--roi-overlay", default="streams")

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--profile", choices=sorted(MODEL_SPECS), default="qwen4b")
    summary.add_argument(
        "--prompt-set", choices=sorted(PROMPT_SETS), default=DEFAULT_PROMPT_SET
    )
    summary.add_argument("--subjects", default="1,2,3,4,5,6,7,8")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = _paths(args)
    if args.command == "prepare":
        _json(prepare_conditions(paths, _subjects(args.subjects)))
    elif args.command == "smoke":
        _json(
            smoke(
                paths,
                with_data=args.with_data,
                subjects=_subjects(args.subjects),
                prompt_set_key=args.prompt_set,
            )
        )
    elif args.command == "prefetch":
        _json(prefetch_artifacts(_spec(args)))
    elif args.command == "preflight":
        result = preflight(
            paths,
            _spec(args),
            device=args.device,
            allow_download=args.allow_download,
            lens_path=args.lens_path,
            max_length=args.max_length,
            prompt_set_key=args.prompt_set,
        )
        destination = (
            paths.results
            / "preflight"
            / f"{run_name(args.profile, args.prompt_set)}.json"
        )
        atomic_json(destination, result)
        _json(result)
    elif args.command == "extract":
        _json(
            extract_embeddings(
                paths,
                _spec(args),
                device=args.device,
                allow_download=args.allow_download,
                lens_path=args.lens_path,
                explicit_layers=_layers(args.layers),
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                max_length=args.max_length,
                seed=args.seed,
                max_conditions=args.max_conditions,
                output_name=args.output_name,
                prompt_set_key=args.prompt_set,
            )
        )
    elif args.command == "rdms":
        _json(
            prepare_grouped_rdms(
                paths,
                args.profile,
                _subjects(args.subjects),
                prompt_set_key=args.prompt_set,
            )
        )
    elif args.command == "searchlight":
        run_searchlight_subject(
            paths,
            args.profile,
            args.subject,
            allow_cpu=args.allow_cpu,
            max_samples=args.max_samples,
            prompt_set_key=args.prompt_set,
        )
    elif args.command == "project":
        project_subjects(
            paths,
            args.profile,
            _subjects(args.subjects),
            prompt_set_key=args.prompt_set,
        )
    elif args.command == "plot":
        features = args.features.split(",") if args.features else None
        _json(
            plot_individual_maps(
                paths,
                args.profile,
                subjects=_subjects(args.subjects),
                feature_names=features,
                roi_overlay=(
                    None if args.roi_overlay.lower() == "none" else args.roi_overlay
                ),
                prompt_set_key=args.prompt_set,
            )
        )
    elif args.command == "summarize":
        _json(
            summarize(
                paths,
                args.profile,
                _subjects(args.subjects),
                prompt_set_key=args.prompt_set,
            )
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

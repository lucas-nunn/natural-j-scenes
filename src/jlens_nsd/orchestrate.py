"""Durable overnight orchestration with canonical-to-fallback preflight."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_PROMPT_SET, MODEL_SPECS, ExperimentPaths, validate_subjects
from .io_utils import atomic_json
from .prompts import PROMPT_SETS

MODULE = "jlens_nsd.cli"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Orchestrator:
    def __init__(self, paths: ExperimentPaths, args) -> None:
        self.paths = paths
        self.args = args
        self.state_path = paths.results / "orchestrator_state.json"
        self.state = {
            "schema_version": 1,
            "started_at": _now(),
            "status": "running",
            "selected_profile": None,
            "subject_numbers": list(args.subjects),
            "prompt_set": args.prompt_set,
            "stages": [],
        }
        paths.logs.mkdir(parents=True, exist_ok=True)
        atomic_json(self.state_path, self.state)

    def run(self, name: str, arguments: list[str], *, check: bool = True) -> bool:
        command = [
            sys.executable,
            "-m",
            MODULE,
            "--results-dir",
            str(self.paths.results),
        ]
        for option, value in (
            ("--nsd-dir", self.paths.nsd_dir),
            ("--captions", self.paths.captions),
            ("--mpnet-base", self.paths.mpnet_base),
            ("--jlens-checkout", self.paths.jlens_checkout),
        ):
            if value is not None:
                command.extend([option, str(value)])
        command.extend(arguments)
        log_path = self.paths.logs / f"{name}.log"
        record = {
            "name": name,
            "command": command,
            "log": str(log_path.resolve()),
            "started_at": _now(),
            "status": "running",
        }
        self.state["stages"].append(record)
        atomic_json(self.state_path, self.state)
        print(f"[{record['started_at']}] {name}: {' '.join(command)}", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=False,
            )
        record["completed_at"] = _now()
        record["returncode"] = result.returncode
        record["status"] = "complete" if result.returncode == 0 else "failed"
        atomic_json(self.state_path, self.state)
        if check and result.returncode != 0:
            raise RuntimeError(f"stage {name} failed; see {log_path}")
        return result.returncode == 0

    def finish(self, status: str, error: str | None = None) -> None:
        self.state["status"] = status
        self.state["completed_at"] = _now()
        if error is not None:
            self.state["error"] = error
        atomic_json(self.state_path, self.state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(MODEL_SPECS), default="qwen4b")
    parser.add_argument(
        "--prompt-set", choices=sorted(PROMPT_SETS), default=DEFAULT_PROMPT_SET
    )
    parser.add_argument(
        "--fallback-profile", choices=sorted(MODEL_SPECS), default="qwen1.7b"
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--nsd-dir", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--mpnet-base", type=Path)
    parser.add_argument("--jlens-checkout", type=Path)
    parser.add_argument("--lens-root", type=Path)
    parser.add_argument("--qwen4b-model", type=Path)
    parser.add_argument("--qwen1-7b-model", type=Path)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--allow-cpu-searchlight", action="store_true")
    parser.add_argument(
        "--subjects",
        default="1,2,3,4,5,6,7,8",
        help="comma-separated subject numbers; default: all",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.subjects = validate_subjects(
            int(item.strip()) for item in args.subjects.split(",") if item.strip()
        )
    except ValueError as error:
        raise SystemExit(f"invalid --subjects: {error}") from error
    paths = ExperimentPaths.from_values(
        results=args.results_dir,
        nsd_dir=args.nsd_dir,
        captions=args.captions,
        mpnet_base=args.mpnet_base,
        jlens_checkout=args.jlens_checkout,
    )
    runner = Orchestrator(paths, args)
    try:
        subjects_csv = ",".join(str(subject) for subject in args.subjects)
        prompt_args = ["--prompt-set", args.prompt_set]
        runner.run("prepare", ["prepare", "--subjects", subjects_csv])
        candidates = []
        for profile in (args.profile, args.fallback_profile):
            if profile not in candidates:
                candidates.append(profile)
        selected = None
        for profile in candidates:
            artifact_args = (
                ["--lens-root", str(args.lens_root)] if args.lens_root else []
            )
            model_path = (
                args.qwen4b_model if profile == "qwen4b" else args.qwen1_7b_model
            )
            if model_path:
                artifact_args.extend(["--model-path", str(model_path)])
            if args.prefetch:
                if not runner.run(
                    f"prefetch_{profile}",
                    ["prefetch", "--profile", profile, *artifact_args],
                    check=False,
                ):
                    continue
            preflight_args = [
                "preflight",
                "--profile",
                profile,
                "--device",
                "cuda",
                "--max-length",
                str(args.max_length),
                *prompt_args,
                *artifact_args,
            ]
            if runner.run(f"preflight_{profile}", preflight_args, check=False):
                selected = profile
                break
        if selected is None:
            raise RuntimeError(
                "canonical and fallback preflights both failed; inspect logs "
                "before starting extraction"
            )
        runner.state["selected_profile"] = selected
        atomic_json(runner.state_path, runner.state)
        selected_model_path = (
            args.qwen4b_model if selected == "qwen4b" else args.qwen1_7b_model
        )

        runner.run(
            f"extract_{selected}",
            [
                "extract",
                "--profile",
                selected,
                "--device",
                "cuda",
                "--batch-size",
                str(args.batch_size),
                "--chunk-size",
                str(args.chunk_size),
                "--max-length",
                str(args.max_length),
                *prompt_args,
                *(["--lens-root", str(args.lens_root)] if args.lens_root else []),
                *(
                    ["--model-path", str(selected_model_path)]
                    if selected_model_path
                    else []
                ),
            ],
        )
        runner.run(
            f"rdms_{selected}",
            [
                "rdms",
                "--profile",
                selected,
                *prompt_args,
                "--subjects",
                subjects_csv,
            ],
        )
        for subject in args.subjects:
            searchlight_args = [
                "searchlight",
                "--profile",
                selected,
                "--subject",
                str(subject),
                *prompt_args,
            ]
            if args.allow_cpu_searchlight:
                searchlight_args.append("--allow-cpu")
            runner.run(f"searchlight_subj{subject:02d}", searchlight_args)
        runner.run(
            f"project_{selected}",
            [
                "project",
                "--profile",
                selected,
                *prompt_args,
                "--subjects",
                subjects_csv,
            ],
        )
        if not args.skip_plots:
            runner.run(
                f"plot_{selected}",
                [
                    "plot",
                    "--profile",
                    selected,
                    *prompt_args,
                    "--subjects",
                    subjects_csv,
                ],
            )
        runner.run(
            f"summarize_{selected}",
            [
                "summarize",
                "--profile",
                selected,
                *prompt_args,
                "--subjects",
                subjects_csv,
            ],
        )
        runner.finish("complete")
        print(f"overnight pipeline complete for {selected}", flush=True)
    except Exception as error:
        runner.finish("failed", str(error))
        raise


if __name__ == "__main__":
    main()

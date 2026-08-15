"""Scientific constants and explicit path resolution for the experiment."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from .prompts import PROMPT_SETS


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _resolved(value: Path | None) -> Path | None:
    return value.expanduser().resolve() if value is not None else None


@dataclass(frozen=True)
class ModelSpec:
    """A causal decoder and its exactly matched fitted Jacobian lens."""

    key: str
    model_name: str
    lens_repo: str
    lens_revision: str
    lens_filename: str
    local_model_path: Path | None = None
    local_lens_root: Path | None = None


MODEL_SPECS = {
    "qwen4b": ModelSpec(
        key="qwen4b",
        model_name="Qwen/Qwen3.5-4B",
        lens_repo="neuronpedia/jacobian-lens",
        lens_revision="qwen-n1000",
        lens_filename=(
            "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
        ),
    ),
    "qwen1.7b": ModelSpec(
        key="qwen1.7b",
        model_name="Qwen/Qwen3-1.7B",
        lens_repo="neuronpedia/jacobian-lens",
        lens_revision="main",
        lens_filename=(
            "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
        ),
    ),
}


def model_spec(
    key: str,
    *,
    local_model_path: Path | None = None,
    local_lens_root: Path | None = None,
) -> ModelSpec:
    """Resolve optional CLI overrides without mutating fixed model metadata."""
    spec = MODEL_SPECS[key]
    model_env = _env_path(f"JLENS_NSD_{key.upper().replace('.', '_')}_MODEL")
    lens_env = _env_path("JLENS_NSD_LENS_ROOT")
    return replace(
        spec,
        local_model_path=local_model_path or model_env,
        local_lens_root=local_lens_root or lens_env,
    )


@dataclass(frozen=True)
class ExperimentPaths:
    """External inputs and the output root; no workstation paths are implicit."""

    results: Path
    nsd_dir: Path | None = None
    captions: Path | None = None
    mpnet_base: Path | None = None
    jlens_checkout: Path | None = None

    @classmethod
    def from_values(
        cls,
        *,
        results: Path | None = None,
        nsd_dir: Path | None = None,
        captions: Path | None = None,
        mpnet_base: Path | None = None,
        jlens_checkout: Path | None = None,
    ) -> ExperimentPaths:
        return cls(
            results=(results or _env_path("JLENS_NSD_RESULTS") or Path("results"))
            .expanduser()
            .resolve(),
            nsd_dir=_resolved(nsd_dir or _env_path("JLENS_NSD_NSD_DIR")),
            captions=_resolved(captions or _env_path("JLENS_NSD_CAPTIONS")),
            mpnet_base=_resolved(mpnet_base or _env_path("JLENS_NSD_MPNET_BASE")),
            jlens_checkout=_resolved(
                jlens_checkout or _env_path("JLENS_NSD_JLENS_CHECKOUT")
            ),
        )

    def require(self, *names: str) -> None:
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            raise ValueError(f"missing required path configuration: {flags}")

    @property
    def conditions(self) -> Path:
        return self.results / "conditions"

    @property
    def embeddings(self) -> Path:
        return self.results / "embeddings"

    @property
    def searchlight_base(self) -> Path:
        return self.results / "searchlight"

    @property
    def reports(self) -> Path:
        return self.results / "reports"

    @property
    def logs(self) -> Path:
        return self.results / "logs"

    @property
    def mpnet_precomputed(self) -> Path:
        self.require("mpnet_base")
        assert self.mpnet_base is not None
        return self.mpnet_base / "precomputed"


N_SUBJECTS = 8
N_SESSIONS = 10
N_SAMPLES = 8
SAMPLE_SIZE = 100
DEFAULT_PROMPT_SET = "historical"
PROMPT_KINDS = PROMPT_SETS[DEFAULT_PROMPT_SET].kinds


def validate_subjects(subjects: Iterable[int]) -> tuple[int, ...]:
    """Return a stable, unique subject subset within the NSD contract."""
    selected = tuple(subjects)
    if not selected:
        raise ValueError("at least one subject is required")
    if len(set(selected)) != len(selected):
        raise ValueError("subjects must be unique")
    if any(subject < 1 or subject > N_SUBJECTS for subject in selected):
        raise ValueError(f"subjects must be in 1..{N_SUBJECTS}")
    return selected


def run_name(profile: str, prompt_set_key: str = DEFAULT_PROMPT_SET) -> str:
    """Namespace non-historical experiments without renaming legacy outputs."""
    if prompt_set_key not in PROMPT_SETS:
        raise ValueError(f"unknown prompt set: {prompt_set_key}")
    if prompt_set_key == DEFAULT_PROMPT_SET:
        return profile
    return f"{profile}__{prompt_set_key}"


def group_name(profile: str, prompt_set_key: str = DEFAULT_PROMPT_SET) -> str:
    """Return a filesystem-safe grouped model name for one experiment run."""
    return f"jlens_{run_name(profile, prompt_set_key).replace('.', '_')}_group"

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from jlens_nsd.cli import build_parser, smoke
from jlens_nsd.config import (
    ExperimentPaths,
    group_name,
    model_spec,
    run_name,
    validate_readout_mode,
    validate_subjects,
)


class ConfigAndSmokeTests(unittest.TestCase):
    def test_default_smoke_needs_no_model_or_data(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            paths = ExperimentPaths.from_values(results=Path("synthetic-results"))
        result = smoke(paths)
        self.assertTrue(result["ok"])
        self.assertEqual(result["toy_rdm_length"], 10)
        self.assertFalse(paths.results.exists())

    def test_paths_and_model_locations_are_explicit(self) -> None:
        environment = {
            "JLENS_NSD_NSD_DIR": "/example/nsd",
            "JLENS_NSD_CAPTIONS": "/example/captions.pkl",
            "JLENS_NSD_MPNET_BASE": "/example/mpnet",
            "JLENS_NSD_QWEN4B_MODEL": "/example/qwen",
            "JLENS_NSD_LENS_ROOT": "/example/lenses",
        }
        with patch.dict(os.environ, environment, clear=True):
            paths = ExperimentPaths.from_values(results=Path("out"))
            spec = model_spec("qwen4b")
        self.assertEqual(paths.nsd_dir, Path("/example/nsd"))
        self.assertEqual(paths.captions, Path("/example/captions.pkl"))
        self.assertEqual(spec.local_model_path, Path("/example/qwen"))
        self.assertEqual(spec.local_lens_root, Path("/example/lenses"))

    def test_missing_external_path_fails_with_cli_name(self) -> None:
        paths = ExperimentPaths.from_values(results=Path("out"))
        with self.assertRaisesRegex(ValueError, "--nsd-dir"):
            paths.require("nsd_dir")

    def test_subject_subset_is_validated_and_stable(self) -> None:
        self.assertEqual(validate_subjects([1, 3]), (1, 3))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_subjects([1, 1])
        with self.assertRaisesRegex(ValueError, "1..8"):
            validate_subjects([9])

    def test_matched_prompt_set_has_an_isolated_namespace(self) -> None:
        self.assertEqual(run_name("qwen4b"), "qwen4b")
        self.assertEqual(
            run_name("qwen4b", "matched_readout"), "qwen4b__matched_readout"
        )

    def test_all_token_mean_has_isolated_namespace_and_plain_only_smoke(self) -> None:
        self.assertEqual(
            run_name("qwen4b", readout_mode="all_token_mean"),
            "qwen4b__plain_mean_pool",
        )
        self.assertEqual(
            group_name("qwen4b", readout_mode="all_token_mean"),
            "jlens_qwen4b__plain_mean_pool_group",
        )
        with patch.dict(os.environ, {}, clear=True):
            paths = ExperimentPaths.from_values(results=Path("synthetic-results"))
        result = smoke(paths, readout_mode="all_token_mean")
        self.assertEqual(set(result["prompt_lengths_chars"]), {"plain"})

    def test_readout_cli_and_config_are_isolated_from_historical_defaults(self) -> None:
        parser = build_parser()
        historical = parser.parse_args(["extract"])
        pooled = parser.parse_args(["extract", "--readout-mode", "all_token_mean"])
        self.assertEqual(historical.readout_mode, "final_token")
        self.assertEqual(pooled.readout_mode, "all_token_mean")
        self.assertEqual(run_name("qwen4b"), "qwen4b")
        with self.assertRaisesRegex(ValueError, "historical plain"):
            validate_readout_mode("all_token_mean", "matched_readout")

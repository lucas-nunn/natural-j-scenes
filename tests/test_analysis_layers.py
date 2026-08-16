"""Pin the predeclared analysis layer set against a run's actual layers.

Extraction layers are resolved at run time and accept a ``--layers`` override,
but the BH families are predeclared over a fixed set. If a run used different
layers, the reporting stage would look for feature names that do not exist and
build its families from whatever happened to match — reporting fewer tests than
the family names claim, without raising.
"""

from __future__ import annotations

import unittest

from jlens_nsd.config import ANALYSIS_LAYERS, validate_analysis_layers
from jlens_nsd.stages import feature_layers


class FeatureLayerParsingTests(unittest.TestCase):
    def test_parses_layers_and_skips_unlayered_features(self) -> None:
        names = [
            "mpnet_reference",
            "plain_mean_pool__final",
            "plain_mean_pool__l08__raw",
            "plain_mean_pool__l08__j",
            "plain_mean_pool__l23__j",
        ]
        self.assertEqual(feature_layers(names), (8, 23))

    def test_deduplicates_and_sorts(self) -> None:
        names = ["a__l30__j", "a__l08__raw", "a__l30__raw", "a__l16__j"]
        self.assertEqual(feature_layers(names), (8, 16, 30))

    def test_empty_when_nothing_carries_a_layer(self) -> None:
        self.assertEqual(feature_layers(["mpnet_reference", "x__final"]), ())


class AnalysisLayerValidationTests(unittest.TestCase):
    def test_accepts_the_predeclared_set(self) -> None:
        self.assertEqual(validate_analysis_layers(ANALYSIS_LAYERS), ANALYSIS_LAYERS)

    def test_rejects_a_different_set(self) -> None:
        with self.assertRaises(ValueError):
            validate_analysis_layers((4, 12, 20, 28))

    def test_rejects_a_subset_even_though_names_would_partly_match(self) -> None:
        with self.assertRaises(ValueError):
            validate_analysis_layers((8, 16, 23))

    def test_rejects_reordered_layers(self) -> None:
        with self.assertRaises(ValueError):
            validate_analysis_layers((30, 23, 16, 8))

    def test_real_feature_names_round_trip(self) -> None:
        names = [f"plain_mean_pool__l{layer:02d}__j" for layer in ANALYSIS_LAYERS]
        names += [f"plain_mean_pool__l{layer:02d}__raw" for layer in ANALYSIS_LAYERS]
        names += ["mpnet_reference", "plain_mean_pool__final"]
        self.assertEqual(
            validate_analysis_layers(feature_layers(names)), ANALYSIS_LAYERS
        )


if __name__ == "__main__":
    unittest.main()

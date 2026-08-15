from __future__ import annotations

import unittest

import numpy as np

from scripts.make_layer_performance_summary import (
    EXPECTED_MODEL_ORDER,
    GROUP_NAME,
    aggregate_peak_searchlight_centres,
    bh_adjust,
    exact_sign_flip_p,
    validate_model_order,
    validated_centre_values,
)


class LayerPerformanceSummaryTests(unittest.TestCase):
    def test_expected_model_order_is_complete_lexical_order(self) -> None:
        self.assertEqual(len(EXPECTED_MODEL_ORDER), 19)
        self.assertEqual(EXPECTED_MODEL_ORDER, tuple(sorted(EXPECTED_MODEL_ORDER)))
        self.assertEqual(EXPECTED_MODEL_ORDER[0], "mpnet_reference")
        self.assertEqual(EXPECTED_MODEL_ORDER[-1], "visualize__l30__raw")

    def test_exact_sign_flip_uses_all_subject_assignments(self) -> None:
        self.assertEqual(exact_sign_flip_p(np.ones(8)), 2 / 256)

    def test_bh_adjustment_matches_reference_example(self) -> None:
        np.testing.assert_allclose(
            bh_adjust([0.01, 0.04, 0.03]),
            [0.03, 0.04, 0.04],
            rtol=0,
            atol=0,
        )

    def test_peak_averages_maps_before_taking_maximum(self) -> None:
        peaks, counts = aggregate_peak_searchlight_centres(
            [
                np.array([[10.0, 0.0]]),
                np.array([[0.0, 10.0]]),
            ]
        )
        np.testing.assert_array_equal(peaks, [5.0])
        np.testing.assert_array_equal(counts, [2])

    def test_peak_uses_only_authoritative_searchlight_centres(self) -> None:
        volume = np.zeros((len(EXPECTED_MODEL_ORDER), 1, 1, 4), dtype=np.float64)
        volume[:, 0, 0, 0] = 99.0
        volume[:, 0, 0, 1] = 1.0
        volume[:, 0, 0, 3] = 3.0
        selected, finite_counts = validated_centre_values(
            volume,
            np.array([1, 3], dtype=np.int64),
            label="synthetic",
        )
        peaks, _ = aggregate_peak_searchlight_centres([selected, selected])
        np.testing.assert_array_equal(peaks, np.full(len(EXPECTED_MODEL_ORDER), 3.0))
        np.testing.assert_array_equal(
            finite_counts, np.full(len(EXPECTED_MODEL_ORDER), 2)
        )

    def test_peak_excludes_centres_with_nonfinite_sample_values(self) -> None:
        peaks, counts = aggregate_peak_searchlight_centres(
            [
                np.array([[np.nan, 2.0, np.inf]]),
                np.array([[4.0, 6.0, 8.0]]),
            ]
        )
        np.testing.assert_array_equal(peaks, [4.0])
        np.testing.assert_array_equal(counts, [2])

    def test_volume_shape_and_dtype_are_validated(self) -> None:
        centres = np.array([0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "shape"):
            validated_centre_values(
                np.zeros((len(EXPECTED_MODEL_ORDER), 2, 2)),
                centres,
                label="synthetic",
            )
        with self.assertRaisesRegex(ValueError, "models"):
            validated_centre_values(
                np.zeros((len(EXPECTED_MODEL_ORDER) - 1, 1, 1, 1)),
                centres,
                label="synthetic",
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            validated_centre_values(
                np.zeros((len(EXPECTED_MODEL_ORDER), 1, 1, 1), dtype=np.float32),
                centres,
                label="synthetic",
            )

    def test_manifest_model_order_is_validated(self) -> None:
        manifest = {
            "schema_version": 1,
            "profile": "qwen4b",
            "group_name": GROUP_NAME,
            "model_order": [
                {
                    "model_index": index,
                    "feature": feature,
                    "model_name": f"{GROUP_NAME}__{feature}",
                }
                for index, feature in enumerate(EXPECTED_MODEL_ORDER, start=1)
            ],
        }
        validate_model_order(manifest)
        manifest["model_order"][2], manifest["model_order"][3] = (
            manifest["model_order"][3],
            manifest["model_order"][2],
        )
        with self.assertRaisesRegex(ValueError, "model order mismatch"):
            validate_model_order(manifest)

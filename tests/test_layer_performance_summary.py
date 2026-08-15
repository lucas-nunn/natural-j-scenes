from __future__ import annotations

import unittest

import numpy as np

from scripts.make_layer_performance_summary import (
    EXPECTED_MODEL_ORDER,
    bh_adjust,
    exact_sign_flip_p,
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

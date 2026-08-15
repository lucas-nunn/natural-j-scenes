from __future__ import annotations

import unittest

import numpy as np

from jlens_nsd.stages import (
    _bh_adjust,
    _exact_sign_flip_p,
    _mean_ci,
)


class SummaryStatisticsTests(unittest.TestCase):
    def test_exact_sign_flip_uses_subjects_as_units(self) -> None:
        differences = np.ones(8)
        self.assertAlmostEqual(_exact_sign_flip_p(differences), 2 / 256)

    def test_confidence_interval_and_bh_adjustment_are_finite(self) -> None:
        mean, low, high = _mean_ci(np.arange(1, 9, dtype=float))
        self.assertLess(low, mean)
        self.assertLess(mean, high)
        adjusted = _bh_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])

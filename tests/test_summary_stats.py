from __future__ import annotations

import unittest

import numpy as np

from jlens_nsd.stages import (
    _bh_adjust,
    _comparison_rows,
    _exact_sign_flip_p,
    _mean_ci,
    _performance_table_rows,
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

    def test_matched_contrasts_have_two_separate_predeclared_bh_families(self) -> None:
        features = ["mpnet_reference"]
        for prompt in ("integrate_readout", "minimal_readout"):
            for layer in (8, 16, 23, 30):
                features.extend(
                    [f"{prompt}__l{layer:02d}__raw", f"{prompt}__l{layer:02d}__j"]
                )
            features.append(f"{prompt}__final")
        scores = np.arange(8 * len(features), dtype=float).reshape(8, len(features))
        rows = _comparison_rows(scores, features, "matched_readout")
        families = {}
        for row in rows:
            families.setdefault(row["bh_family"], []).append(row)
        self.assertEqual(
            {family: len(items) for family, items in families.items()},
            {
                "matched_readout_j_vs_raw_8": 8,
                "matched_readout_prompt_pair_9": 9,
            },
        )
        self.assertTrue(all(np.isfinite(row["fdr_q"]) for row in rows))

        score_rows = [
            {
                "feature": feature,
                "mean_correlation": float(index),
                "ci_low": float(index) - 0.1,
                "ci_high": float(index) + 0.1,
            }
            for index, feature in enumerate(features)
        ]
        table = _performance_table_rows(score_rows, rows)
        self.assertEqual(len(table), 17)
        self.assertEqual(
            {row["bh_family"] for row in table},
            {"matched_readout_j_vs_raw_8", "matched_readout_prompt_pair_9"},
        )

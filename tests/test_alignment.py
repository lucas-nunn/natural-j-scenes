from __future__ import annotations

import unittest

import numpy as np

from jlens_nsd.conditions import validate_sampling


class AlignmentTests(unittest.TestCase):
    def test_eight_nonoverlapping_samples_validate(self) -> None:
        choices = np.arange(800, dtype=np.int64).reshape(8, 100)
        validate_sampling(choices, 835, "subj01")

    def test_sampling_overlap_is_rejected(self) -> None:
        choices = np.arange(800, dtype=np.int64).reshape(8, 100)
        choices[1, 0] = choices[0, 0]
        with self.assertRaisesRegex(ValueError, "not mutually disjoint"):
            validate_sampling(choices, 835, "subj01")

    def test_sampling_out_of_range_is_rejected(self) -> None:
        choices = np.arange(800, dtype=np.int64).reshape(8, 100)
        choices[-1, -1] = 835
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_sampling(choices, 835, "subj01")

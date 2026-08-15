"""Pin the brain/model condition-alignment contract.

A permutation between the averaged-beta column order and the model RDM row
order does not raise anywhere. It silently correlates mismatched conditions and
yields a weak but plausible effect, which is indistinguishable from a real
small result. These tests exist so that failure mode cannot be introduced
quietly.
"""

from __future__ import annotations

import unittest

import numpy as np

from jlens_nsd.nsd_adapter import condition_column_index


class ConditionColumnIndexTests(unittest.TestCase):
    def test_columns_follow_sorted_id_order_not_first_appearance(self) -> None:
        # Deliberately unsorted, with repeats, like the real trial-level list.
        conditions = np.array([430, 104, 430, 1066, 104, 430, 120], dtype=np.int64)
        lookup, id_to_column = condition_column_index(conditions)

        self.assertTrue(np.array_equal(lookup, np.array([104, 120, 430, 1066])))
        # First appearance would put 430 in column 0; sorted order puts it at 2.
        self.assertEqual(id_to_column[430], 2)
        self.assertEqual(id_to_column[104], 0)
        self.assertEqual(id_to_column[120], 1)
        self.assertEqual(id_to_column[1066], 3)

    def test_absent_ids_map_to_negative_one(self) -> None:
        lookup, id_to_column = condition_column_index(np.array([5, 9], dtype=np.int64))
        self.assertEqual(id_to_column[7], -1)
        self.assertEqual(len(lookup), 2)

    def test_lookup_matches_numpy_unique_of_input(self) -> None:
        rng = np.random.default_rng(0)
        conditions = rng.integers(1, 5000, size=4000)
        lookup, _ = condition_column_index(conditions)
        self.assertTrue(np.array_equal(lookup, np.unique(conditions)))

    def test_column_assignment_is_a_bijection_onto_range(self) -> None:
        conditions = np.array([70, 3, 3, 900, 12], dtype=np.int64)
        lookup, id_to_column = condition_column_index(conditions)
        columns = id_to_column[lookup]
        self.assertTrue(np.array_equal(np.sort(columns), np.arange(len(lookup))))

    def test_rejects_empty_and_non_one_based_ids(self) -> None:
        with self.assertRaises(ValueError):
            condition_column_index(np.array([], dtype=np.int64))
        with self.assertRaises(ValueError):
            condition_column_index(np.array([0, 4], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()

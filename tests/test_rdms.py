from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from jlens_nsd.io_utils import atomic_npz
from jlens_nsd.rdms import (
    _load_sparse_feature,
    _validate_correlation_rows,
)


class RDMTests(unittest.TestCase):
    def test_sparse_chunks_reassemble_by_explicit_ids(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            chunks = root / "chunks"
            chunks.mkdir()
            atomic_npz(
                chunks / "second.npz",
                condition_ids=np.asarray([30, 40]),
                feature=np.asarray([[3, 4], [4, 5]], dtype=np.float32),
            )
            atomic_npz(
                chunks / "first.npz",
                condition_ids=np.asarray([10, 20]),
                feature=np.asarray([[1, 2], [2, 3]], dtype=np.float32),
            )
            manifest = {
                "completed_chunks": ["second.npz", "first.npz"],
                "config": {"d_model": 2},
            }
            matrix = _load_sparse_feature(
                root, manifest, "feature", np.asarray([10, 20, 30, 40])
            )
            np.testing.assert_array_equal(matrix[:, 0], [1, 2, 3, 4])

    def test_condensed_rdm_length_and_finiteness(self) -> None:
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(100, 12)).astype(np.float32)
        _validate_correlation_rows(matrix, "test")
        rdm = pdist(matrix, metric="correlation")
        self.assertEqual(rdm.shape, (4_950,))
        self.assertTrue(np.isfinite(rdm).all())

    def test_constant_rows_fail_before_scipy_nan(self) -> None:
        matrix = np.asarray([[1, 1, 1], [1, 2, 3]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "correlation distance undefined"):
            _validate_correlation_rows(matrix, "constant")

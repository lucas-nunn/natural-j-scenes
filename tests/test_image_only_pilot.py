from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.distance import pdist

from jlens_nsd.image_only_pilot import (
    _file_record,
    _validate_file_record,
    apply_j_and_check_linearity,
    mean_pool_image_tokens,
    numpy_searchlight_corr,
    strict_image_token_mask,
)


class ImageOnlyPilotHelperTests(unittest.TestCase):
    def test_numpy_searchlight_matches_direct_correlation_rsa(self) -> None:
        rng = np.random.default_rng(20260815)
        betas = rng.normal(size=(2, 2, 2, 100)).astype(np.float32)
        spheres = [np.asarray([0, 1, 2]), np.asarray([3, 4, 5, 6])]
        models = rng.normal(size=(9, 4950)).astype(np.float32)
        actual = numpy_searchlight_corr(betas, spheres, models, batch_size=1)
        expected = np.empty_like(actual)
        flat = betas.reshape(-1, 100)
        for sphere_index, indices in enumerate(spheres):
            brain_rdm = pdist(flat[indices].T, metric="correlation")
            for model_index, model_rdm in enumerate(models):
                expected[sphere_index, model_index] = np.corrcoef(brain_rdm, model_rdm)[
                    0, 1
                ]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_strict_mask_accepts_only_minimal_boundary_sequence(self) -> None:
        ids = np.asarray([10, 20, 20, 20, 30])
        types = np.asarray([0, 1, 1, 1, 0])
        attention = np.ones(5, dtype=np.int64)
        mask = strict_image_token_mask(
            ids,
            types,
            attention,
            vision_start_token_id=10,
            image_token_id=20,
            vision_end_token_id=30,
            expected_image_tokens=3,
        )
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_strict_mask_rejects_text_control_or_modality_disagreement(self) -> None:
        kwargs = {
            "vision_start_token_id": 10,
            "image_token_id": 20,
            "vision_end_token_id": 30,
            "expected_image_tokens": 3,
        }
        with self.assertRaisesRegex(ValueError, "not exactly"):
            strict_image_token_mask(
                np.asarray([10, 20, 99, 20, 30]),
                np.asarray([0, 1, 0, 1, 0]),
                np.ones(5),
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "multimodal"):
            strict_image_token_mask(
                np.asarray([10, 20, 20, 20, 30]),
                np.asarray([0, 1, 0, 1, 0]),
                np.ones(5),
                **kwargs,
            )

    def test_pooling_excludes_boundaries_and_is_float32(self) -> None:
        hidden = np.arange(15, dtype=np.float64).reshape(5, 3)
        mask = np.asarray([False, True, True, True, False])
        pooled = mean_pool_image_tokens(hidden, mask)
        np.testing.assert_array_equal(pooled, hidden[1:4].mean(axis=0))
        self.assertEqual(pooled.dtype, np.float32)

    def test_j_commutes_with_mean_within_float32_tolerance(self) -> None:
        rng = np.random.default_rng(20260815)
        patches = rng.normal(size=(169, 32)).astype(np.float32)
        jacobian = rng.normal(size=(32, 32)).astype(np.float32)
        raw, projected, check = apply_j_and_check_linearity(patches, jacobian)
        self.assertEqual(raw.shape, (32,))
        self.assertEqual(projected.dtype, np.float32)
        self.assertLessEqual(check["max_abs_error"], check["tolerance_bound"])

    def test_pooling_rejects_empty_or_nonfinite_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            mean_pool_image_tokens(np.ones((2, 3)), np.zeros(2, dtype=bool))
        values = np.ones((2, 3))
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mean_pool_image_tokens(values, np.asarray([True, False]))

    def test_file_record_validation_checks_size_and_hash(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"validated payload")
            record = _file_record(path)
            self.assertEqual(_validate_file_record("artifact", record), path.resolve())
            path.write_bytes(b"tampered payload!")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                _validate_file_record("artifact", record)

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:  # lightweight core test environment
    torch = None

from jlens_nsd.extract import (
    _chunk_is_valid,
    _gather_last,
    _validate_hook_semantics,
    feature_registry,
    resolve_source_layers,
)
from jlens_nsd.io_utils import atomic_npz


class ExtractionHelperTests(unittest.TestCase):
    def test_relative_layer_selection_uses_fitted_layers(self) -> None:
        self.assertEqual(resolve_source_layers(range(12), 12), [3, 6, 8, 10])
        self.assertEqual(resolve_source_layers([0, 4, 8, 11], 12), [0, 4, 8, 11])
        self.assertEqual(resolve_source_layers(range(12), 12, [2, 7]), [2, 7])
        with self.assertRaisesRegex(ValueError, "not fitted"):
            resolve_source_layers(range(4), 4, [9])
        features = feature_registry([1, 2, 3, 4])
        self.assertEqual(len(features), 18)
        self.assertEqual(
            [item["name"] for item in features[:5]],
            [
                "visualize__l01__raw",
                "visualize__l01__j",
                "visualize__l02__raw",
                "visualize__l02__j",
                "visualize__l03__raw",
            ],
        )
        self.assertEqual(features[8]["name"], "visualize__final")
        self.assertEqual(features[-1]["name"], "plain__final")

    @unittest.skipUnless(torch is not None, "requires the optional model extra")
    def test_last_nonpadding_gather(self) -> None:
        hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        gathered = _gather_last(hidden, mask)
        torch.testing.assert_close(gathered[0], hidden[0, 1])
        torch.testing.assert_close(gathered[1], hidden[1, 2])

    @unittest.skipUnless(torch is not None, "requires the optional model extra")
    def test_hook_matches_hf_embedding_offset_semantics(self) -> None:
        mask = torch.tensor([[1, 1, 1]])
        embedding = torch.randn(1, 3, 4)
        block0 = torch.randn(1, 3, 4)
        block1 = torch.randn(1, 3, 4)
        recorder = SimpleNamespace(activations={0: block0, 1: block1})
        outputs = SimpleNamespace(hidden_states=(embedding, block0, block1))
        checks = _validate_hook_semantics(recorder, outputs, [0, 1], mask)
        self.assertEqual(checks["0"]["hf_hidden_states_index"], 1)
        self.assertEqual(checks["1"]["max_abs_error"], 0.0)

    def test_atomic_chunk_validation_supports_resume(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as name:
            layers = [1]
            features = feature_registry(layers)
            ids = np.asarray([17, 42], dtype=np.int64)
            values = {
                item["name"]: np.ones((2, 4), dtype=np.float32) for item in features
            }
            chunk = Path(name) / "chunk.npz"
            atomic_npz(chunk, condition_ids=ids, **values)
            self.assertTrue(_chunk_is_valid(chunk, ids, features, 4))
            self.assertFalse(_chunk_is_valid(chunk, ids[::-1], features, 4))

            values[features[0]["name"]][0, 0] = np.nan
            atomic_npz(chunk, condition_ids=ids, **values)
            self.assertFalse(_chunk_is_valid(chunk, ids, features, 4))

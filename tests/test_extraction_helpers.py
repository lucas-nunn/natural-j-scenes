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
    _masked_mean,
    _readout,
    _transport_tokens_then_mean,
    _validate_hook_semantics,
    audit_all_token_mask,
    audit_matched_prompt_endpoints,
    feature_registry,
    resolve_source_layers,
)
from jlens_nsd.io_utils import atomic_npz
from jlens_nsd.prompts import prompts_for_condition


class ExtractionHelperTests(unittest.TestCase):
    def test_all_token_audit_includes_only_historical_encoded_special_tokens(
        self,
    ) -> None:
        class AuditTokenizer:
            padding_side = "right"
            pad_token_id = 0

            def __call__(
                self,
                text,
                *,
                add_special_tokens,
                truncation,
                return_attention_mask,
                return_special_tokens_mask,
            ):
                self.last_text = text
                self.last_options = (add_special_tokens, truncation)
                return {
                    "input_ids": [101, 7, 8],
                    "attention_mask": [1, 1, 1],
                    "special_tokens_mask": [1, 0, 0],
                }

        tokenizer = AuditTokenizer()
        audit = audit_all_token_mask(tokenizer, [42], ["unchanged plain prompt."])
        self.assertEqual(tokenizer.last_text, "unchanged plain prompt.")
        self.assertEqual(tokenizer.last_options, (True, False))
        self.assertEqual(audit["n_conditions"], 1)
        self.assertEqual(audit["records"][0]["valid_positions_0based"], [0, 1, 2])
        self.assertEqual(audit["records"][0]["included_special_token_ids"], [101])

    def test_model_free_endpoint_audit_covers_every_condition(self) -> None:
        class CharacterTokenizer:
            def __call__(
                self,
                text,
                *,
                add_special_tokens,
                truncation,
                return_attention_mask,
            ):
                del truncation, return_attention_mask
                ids = [ord(character) for character in text]
                return {"input_ids": ([1] + ids) if add_special_tokens else ids}

        tables = [
            [["A", "dog", "runs", "."]],
            [["Two", "birds", "fly", "!", "A", "tree", "."]],
        ]
        rows = [prompts_for_condition(table, 1, "matched_readout") for table in tables]
        audit = audit_matched_prompt_endpoints(CharacterTokenizer(), rows)
        self.assertEqual(audit["n_conditions"], 2)
        self.assertTrue(audit["all_pair_final_token_ids_match"])
        self.assertTrue(audit["all_prompts_end_with_declared_suffix_tokens"])
        self.assertEqual(audit["final_readout_token_id"], ord(":"))

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
        matched = feature_registry(
            [8, 16, 23, 30], ("integrate_readout", "minimal_readout")
        )
        self.assertEqual(matched[0]["name"], "integrate_readout__l08__raw")
        self.assertEqual(matched[-1]["name"], "minimal_readout__final")
        pooled = feature_registry(
            [8, 16, 23, 30],
            ("plain",),
            feature_namespace="plain_mean_pool",
        )
        self.assertEqual(len(pooled), 9)
        self.assertEqual(pooled[0]["name"], "plain_mean_pool__l08__raw")
        self.assertEqual(pooled[-1]["name"], "plain_mean_pool__final")
        self.assertTrue(all(item["prompt"] == "plain" for item in pooled))

    @unittest.skipUnless(torch is not None, "requires the optional model extra")
    def test_last_nonpadding_gather(self) -> None:
        hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        gathered = _gather_last(hidden, mask)
        torch.testing.assert_close(gathered[0], hidden[0, 1])
        torch.testing.assert_close(gathered[1], hidden[1, 2])

    @unittest.skipUnless(torch is not None, "requires the optional model extra")
    def test_attention_mask_mean_pooling_and_padding_invariance(self) -> None:
        hidden = torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0], [999.0, -999.0]],
                [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        pooled = _masked_mean(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0], [4.0, 6.0]]))
        hidden[0, 2] = torch.tensor([-1e9, 1e9])
        torch.testing.assert_close(_masked_mean(hidden, mask), pooled)

    @unittest.skipUnless(torch is not None, "requires the optional model extra")
    def test_tokenwise_j_application_and_pool_linearity(self) -> None:
        hidden = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [500.0, 600.0]]], dtype=torch.float32
        )
        mask = torch.tensor([[1, 1, 0]])
        matrix = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], dtype=torch.float32)
        raw, transported, check = _transport_tokens_then_mean(hidden, mask, matrix)
        expected_tokens = hidden[0, :2] @ matrix.T
        torch.testing.assert_close(raw, torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(transported, expected_tokens.mean(dim=0)[None])
        torch.testing.assert_close(transported, raw @ matrix.T)
        self.assertLessEqual(check["max_abs_error"], check["tolerance"])
        self.assertEqual(check["n_valid_tokens"], 2)

    @unittest.skipUnless(torch is not None, "requires the optional model extra")
    def test_historical_readout_regression_is_still_last_nonpadding(self) -> None:
        hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        torch.testing.assert_close(
            _readout(hidden, mask, "final_token"), _gather_last(hidden, mask)
        )

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

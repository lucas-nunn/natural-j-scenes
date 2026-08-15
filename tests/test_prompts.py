from __future__ import annotations

import unittest

from jlens_nsd.prompts import (
    build_prompt,
    captions_for_condition,
    detokenize,
    split_caption_tokens,
)


class PromptTests(unittest.TestCase):
    def test_detokenize_and_caption_boundaries(self) -> None:
        tokens = [
            "A",
            "cat",
            "is",
            "n't",
            "sleeping",
            ".",
            "It",
            "'s",
            "awake",
            "!",
        ]
        self.assertEqual(detokenize(tokens), "A cat isn't sleeping. It's awake!")
        self.assertEqual(
            split_caption_tokens(tokens),
            ["A cat isn't sleeping.", "It's awake!"],
        )

    def test_condition_lookup_is_explicitly_one_based(self) -> None:
        table = [["first", "."], ["second", "."]]
        self.assertEqual(captions_for_condition(table, 1), ["first."])
        self.assertEqual(captions_for_condition(table, 2), ["second."])
        with self.assertRaises(IndexError):
            captions_for_condition(table, 0)

    def test_visualization_and_plain_ablation_are_deterministic(self) -> None:
        captions = ["A dog runs.", "The dog crosses grass."]
        visual = build_prompt(captions, "visualize")
        plain = build_prompt(captions, "plain")
        self.assertEqual(visual, build_prompt(captions, "visualize"))
        self.assertEqual(
            visual,
            "Construct one coherent visual scene from the source captions. "
            "Reconcile their overlap into a single image: represent the entities, "
            "attributes, actions, spatial relations, and setting together. Infer "
            "only visually plausible structure. Form the scene rather than merely "
            "restating or listing caption words.\n\nSource captions:\n- A dog runs.\n"
            "- The dog crosses grass.\n\nIntegrated visual scene:",
        )
        self.assertTrue(visual.endswith("Integrated visual scene:"))
        self.assertIn("spatial relations", visual)
        self.assertIn("merely restating", visual)
        self.assertEqual(plain, "- A dog runs.\n- The dog crosses grass.")
        self.assertNotIn("Construct", plain)

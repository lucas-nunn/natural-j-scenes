from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "make_layer23_brain_map_montage.py"
)
SPEC = importlib.util.spec_from_file_location(
    "make_layer23_brain_map_montage", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load montage generator: {SCRIPT_PATH}")
montage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(montage)


class BrainMapMontageTests(unittest.TestCase):
    def test_subject_parser_is_explicit_ordered_and_defaults_to_one_through_four(
        self,
    ) -> None:
        args = montage.parse_args(
            ["--figure-root", "/maps", "--output", "/tmp/montage.jpg"]
        )
        self.assertEqual(args.subjects, (1, 2, 3, 4))
        self.assertEqual(montage.parse_subjects("4,2,1"), (4, 2, 1))

        for invalid in ("", "1,,2", "one", "0", "9", "1,1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    montage.parse_subjects(invalid)

    def test_output_dimensions_follow_selected_subject_count(self) -> None:
        self.assertEqual(montage.output_size((1,)), (2240, 585))
        self.assertEqual(montage.output_size((1, 2, 3, 4)), (2240, 2064))
        self.assertEqual(montage.output_size(tuple(range(1, 9))), (2240, 4036))
        with self.assertRaisesRegex(ValueError, "at least one"):
            montage.output_size(())

    def test_compose_preserves_source_order_hashes_and_exif_audit(self) -> None:
        subjects = (1, 2, 3, 4)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            figure_root = root / "sources"
            figure_root.mkdir()
            expected_sources = []
            expected_values = {}
            for row, subject in enumerate(subjects):
                for column, kind in enumerate(montage.KINDS):
                    value = 20 + row * 40 + column * 15
                    expected_values[(row, column)] = value
                    path = montage.source_path(figure_root, "plain", kind, subject)
                    Image.new("RGB", (16, 8), (value, value, value)).save(path)
                    expected_sources.append(
                        {
                            "feature": f"plain__l23__{kind}",
                            "subject": subject,
                            "filename": path.name,
                            "sha256": montage.sha256_file(path),
                            "dimensions": [16, 8],
                        }
                    )

            output = root / "montage.jpg"
            args = argparse.Namespace(
                figure_root=figure_root,
                prompt_kind="plain",
                subjects=subjects,
                output=output,
            )
            with (
                patch.object(montage, "SOURCE_SIZE", (16, 8)),
                patch.object(montage, "PANEL_SIZE", (10, 6)),
                patch.object(montage, "HEADER_HEIGHT", 4),
                patch.object(montage, "ROW_STRIDE", 8),
                patch.object(montage, "centered_text"),
            ):
                audit = montage.compose(args)

            self.assertEqual(audit["subjects"], [1, 2, 3, 4])
            self.assertEqual(audit["column_order"], ["raw", "j"])
            self.assertEqual(audit["sources"], expected_sources)
            self.assertEqual(len(audit["sources"]), 8)
            self.assertEqual(audit["output_dimensions"], [20, 36])
            self.assertIn("no RSA", audit["operation"])

            with Image.open(output) as rendered:
                rendered.load()
                self.assertEqual(rendered.size, (20, 36))
                self.assertEqual(rendered.mode, "RGB")
                self.assertEqual(json.loads(rendered.getexif()[270]), audit)
                self.assertEqual(
                    rendered.getexif()[305], "make_layer23_brain_map_montage.py"
                )
                for (row, column), expected_value in expected_values.items():
                    pixel = rendered.getpixel((column * 10 + 5, 4 + row * 8 + 3))
                    self.assertTrue(
                        all(abs(channel - expected_value) <= 3 for channel in pixel),
                        (row, column, pixel, expected_value),
                    )

    def test_checked_in_montages_have_eight_ordered_source_panels(self) -> None:
        asset_root = SCRIPT_PATH.parents[1] / "docs" / "assets"
        for prompt_kind in ("visualize", "plain"):
            with self.subTest(prompt_kind=prompt_kind):
                path = asset_root / f"{prompt_kind}_layer23_raw_then_j_subjects1-4.jpg"
                with Image.open(path) as rendered:
                    rendered.load()
                    audit = json.loads(rendered.getexif()[270])
                    self.assertEqual(rendered.size, (2240, 2064))
                    self.assertEqual(rendered.mode, "RGB")
                    self.assertEqual(
                        rendered.getexif()[305],
                        "make_layer23_brain_map_montage.py",
                    )

                expected_order = [
                    (
                        f"{prompt_kind}__l23__{kind}",
                        subject,
                        f"{prompt_kind}__l23__{kind}_subj{subject:02d}.png",
                    )
                    for subject in (1, 2, 3, 4)
                    for kind in ("raw", "j")
                ]
                actual_order = [
                    (source["feature"], source["subject"], source["filename"])
                    for source in audit["sources"]
                ]
                self.assertEqual(actual_order, expected_order)
                self.assertEqual(len(audit["sources"]), 8)
                self.assertEqual(audit["subjects"], [1, 2, 3, 4])
                self.assertEqual(audit["column_order"], ["raw", "j"])
                self.assertEqual(audit["source_dimensions"], [3315, 1440])
                self.assertEqual(audit["output_dimensions"], [2240, 2064])
                for source in audit["sources"]:
                    self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()

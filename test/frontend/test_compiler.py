"""Test the project importer path against the actual HEIR frontend."""

import os
from pathlib import Path
import unittest

from heir.heir_cli.heir_cli import CLIError
from snabf_frontend import compile_reference, import_mlir


class FrontendTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.model = root / "test/fixtures/linear_silu_linear.mlir"
        heir_bin = Path(os.environ.get("HEIR_BIN_DIR", root / "extern/heir-release/heir"))
        self.options = {
            "profile": self.model.with_name("reference_profile.json"),
            "min_slots": 4096,
            "heir_opt": heir_bin / "heir-opt",
            "heir_translate": heir_bin / "heir-translate",
        }

    def test_custom_importer_is_lowered(self):
        def custom_importer(path):
            self.assertEqual(path, self.model)
            return import_mlir(path).replace("@linear_silu_linear", "@extended_entry")

        result = compile_reference(self.model, importer=custom_importer, **self.options)
        self.assertIn("@extended_entry", result.input_mlir)
        self.assertIn("@extended_entry", result.lowered_mlir)
        self.assertIn("func Extended_entry(", result.go_source)

    def test_heir_rejects_invalid_import(self):
        with self.assertRaises(CLIError):
            compile_reference(self.model, importer=lambda _: "this is not MLIR", **self.options)

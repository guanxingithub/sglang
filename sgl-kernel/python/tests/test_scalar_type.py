# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.scalar_type (no package init required)."""

import unittest

from support import PackageExportMixin, load_sgl_source_module


class TestScalarTypePackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.scalar_type"
    exported_names = ("ScalarType", "scalar_types", "NanRepr")


class TestScalarType(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = load_sgl_source_module("sgl_scalar_type_ut", "scalar_type.py")

    def test_scalar_types_have_ids(self):
        scalar_types = self.st.scalar_types
        for name in (
            "int4",
            "uint4",
            "int8",
            "uint8",
            "float8_e4m3fn",
            "float8_e5m2",
            "bfloat16",
            "float16",
        ):
            with self.subTest(name=name):
                t = getattr(scalar_types, name)
                self.assertIsInstance(t.id, int)

    def test_from_id_roundtrip(self):
        scalar_types = self.st.scalar_types
        t = scalar_types.float16
        self.assertIs(self.st.ScalarType.from_id(t.id), t)

    def test_from_id_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.st.ScalarType.from_id(-1)

    def test_int_properties(self):
        t = self.st.scalar_types.int8
        self.assertTrue(t.is_integer())
        self.assertFalse(t.is_floating_point())
        self.assertTrue(t.is_signed())
        self.assertEqual(t.size_bits, 8)

    def test_float_ieee_properties(self):
        t = self.st.scalar_types.float8_e5m2
        self.assertTrue(t.is_floating_point())
        self.assertTrue(t.is_ieee_754())
        self.assertTrue(t.has_nans())

    def test_str_and_repr(self):
        t = self.st.scalar_types.uint4b8
        self.assertIn("uint4", str(t))
        self.assertTrue(repr(t).startswith("ScalarType."))

    def test_len_raises_typeerror(self):
        t = self.st.scalar_types.int8
        with self.assertRaises(TypeError):
            len(t)  # noqa: B018 — required for PyTorch opcheck compatibility


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.utils."""

import unittest

from support import PackageExportMixin, load_sgl_source_module, torch_mod


class TestSglUtilsPackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.utils"
    exported_names = ("is_arch_support_pdl",)


class TestSglUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        cls.u = load_sgl_source_module("sgl_utils_ut", "utils.py")

    def test_to_tensor_scalar_tuple_tensor(self):
        t = torch_mod.tensor([1], dtype=torch_mod.int32)
        a, b = self.u._to_tensor_scalar_tuple(t)
        self.assertIs(a, t)
        self.assertEqual(b, 0)

    def test_to_tensor_scalar_tuple_int(self):
        a, b = self.u._to_tensor_scalar_tuple(3)
        self.assertIsNone(a)
        self.assertEqual(b, 3)

    def test_get_cache_buf_cpu(self):
        buf = self.u._get_cache_buf("ut_buf", 128, torch_mod.device("cpu"))
        self.assertEqual(buf.shape, (128,))
        self.assertEqual(buf.dtype, torch_mod.uint8)
        again = self.u._get_cache_buf("ut_buf", 128, torch_mod.device("cpu"))
        self.assertIs(again, buf)

    @unittest.skipUnless(
        torch_mod is not None and torch_mod.cuda.is_available(),
        "CUDA not available",
    )
    def test_is_arch_support_pdl_returns_bool(self):
        self.assertIsInstance(self.u.is_arch_support_pdl(), bool)


if __name__ == "__main__":
    unittest.main()

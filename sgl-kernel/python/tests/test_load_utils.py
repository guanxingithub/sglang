# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.load_utils helpers."""

import unittest
from unittest import mock

from support import PackageExportMixin, load_sgl_source_module, torch_mod


class TestLoadUtilsPackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.load_utils"
    exported_names = (
        "_get_compute_capability",
        "_filter_compiled_extensions",
        "_load_architecture_specific_ops",
        "_find_cuda_home",
        "_preload_cuda_library",
    )


class TestLoadUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        cls.lu = load_sgl_source_module("sgl_load_utils_ut", "load_utils.py")

    def test_filter_compiled_extensions_prioritizes_so(self):
        files = [
            "/tmp/common_ops.py",
            "/tmp/common_ops.cpython-312-x86_64-linux-gnu.so",
            "/tmp/common_ops.abi3.so",
        ]
        ordered = self.lu._filter_compiled_extensions(files)
        self.assertTrue(str(ordered[0]).endswith(".so") or ".so" in str(ordered[0]))

    def test_get_compute_capability_none_without_cuda(self):
        import torch

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIsNone(self.lu._get_compute_capability())

    def test_find_cuda_home_returns_str(self):
        home = self.lu._find_cuda_home()
        self.assertIsInstance(home, str)
        self.assertTrue(len(home) > 0)

    def test_get_compute_capability_with_mock_gpu(self):
        import torch

        mock_props = mock.Mock()
        mock_props.major = 9
        mock_props.minor = 0
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.object(torch.cuda, "current_device", return_value=0):
                with mock.patch.object(
                    torch.cuda, "get_device_properties", return_value=mock_props
                ):
                    self.assertEqual(self.lu._get_compute_capability(), 90)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin, load_sgl_source_module, torch_mod


class TestKvcacheioPure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")

    def test_is_hip_returns_bool(self):
        kv = load_sgl_source_module("sgl_kvcacheio_ut", "kvcacheio.py")
        self.assertIsInstance(kv.is_hip(), bool)


class TestKvcacheioSmoke(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.kvcacheio"
    exported_names = (
        "is_hip",
        "transfer_kv_per_layer",
        "transfer_kv_per_layer_pf_lf",
        "transfer_kv_per_layer_ph_lf",
        "transfer_kv_all_layer",
        "transfer_kv_all_layer_lf_pf",
        "transfer_kv_all_layer_lf_ph",
        "transfer_kv_direct",
        "transfer_kv_per_layer_direct_pf_lf",
        "transfer_kv_all_layer_direct_lf_pf",
        "transfer_kv_per_layer_mla",
        "transfer_kv_per_layer_mla_pf_lf",
        "transfer_kv_all_layer_mla",
        "transfer_kv_all_layer_mla_lf_pf",
    )


if __name__ == "__main__":
    unittest.main()

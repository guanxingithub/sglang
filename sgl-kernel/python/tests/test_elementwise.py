# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin, import_sgl_submodule


class TestElementwise(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.elementwise"
    exported_names = (
        "rmsnorm",
        "fused_add_rmsnorm",
        "gemma_rmsnorm",
        "gemma_fused_add_rmsnorm",
        "silu_and_mul",
        "gelu_tanh_and_mul",
        "gelu_and_mul",
        "rotary_embedding",
        "copy_to_gpu_no_ce",
        "concat_mla_k",
        "concat_mla_absorb_q",
    )

    def test_gelu_quick_on_hip(self):
        import torch

        if torch.version.hip is None:
            self.skipTest("ROCm / HIP build only")
        mod = import_sgl_submodule(self.module_path)
        self.assertTrue(hasattr(mod, "gelu_quick"), "gelu_quick missing on HIP")


if __name__ == "__main__":
    unittest.main()

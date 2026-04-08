# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestQuantizationPackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.quantization"
    exported_names = (
        "ggml_dequantize",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_mul_mat_a8",
        "ggml_mul_mat_vec_a8",
        "ggml_moe_get_block_size",
    )


class TestQuantizationGguf(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.quantization.gguf"
    exported_names = (
        "ggml_dequantize",
        "ggml_mul_mat_vec_a8",
        "ggml_mul_mat_a8",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
    )


if __name__ == "__main__":
    unittest.main()

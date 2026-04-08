# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestGemm(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.gemm"
    exported_names = (
        "awq_dequantize",
        "int8_scaled_mm",
        "fp8_blockwise_scaled_mm",
        "fp8_scaled_mm",
        "bmm_fp8",
        "dsv3_fused_a_gemm",
        "sgl_per_token_group_quant_8bit",
        "sgl_per_token_group_quant_fp8",
        "sgl_per_token_group_quant_int8",
        "sgl_per_token_quant_fp8",
        "qserve_w4a8_per_chn_gemm",
        "qserve_w4a8_per_group_gemm",
        "dsv3_router_gemm",
        "shuffle_rows",
        "gptq_gemm",
        "gptq_shuffle",
    )


if __name__ == "__main__":
    unittest.main()

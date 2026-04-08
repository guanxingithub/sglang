# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestMoe(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.moe"
    exported_names = (
        "moe_align_block_size",
        "topk_softmax",
        "topk_sigmoid",
        "moe_sum_reduce",
        "moe_sum",
        "moe_fused_gate",
        "kimi_k2_moe_fused_gate",
        "fp8_blockwise_scaled_grouped_mm",
        "prepare_moe_input",
        "apply_shuffle_mul_sum",
        "fused_qk_norm_rope",
    )


if __name__ == "__main__":
    unittest.main()

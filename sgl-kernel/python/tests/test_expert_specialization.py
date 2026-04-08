# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestExpertSpecialization(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.expert_specialization"
    exported_names = (
        "es_fp8_blockwise_scaled_grouped_mm",
        "es_sm100_mxfp8_blockscaled_grouped_mm",
        "es_sm100_mxfp8_blockscaled_grouped_quant",
    )


if __name__ == "__main__":
    unittest.main()

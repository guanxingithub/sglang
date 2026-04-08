# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestCutlassMoe(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.cutlass_moe"
    exported_names = ("get_cutlass_w4a8_moe_mm_data", "cutlass_w4a8_moe_mm")


if __name__ == "__main__":
    unittest.main()

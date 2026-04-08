# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestFlashAttn(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.flash_attn"
    exported_names = (
        "is_fa3_supported",
        "maybe_contiguous",
        "flash_attn_with_kvcache",
        "flash_attn_varlen_func",
        "get_scheduler_metadata",
    )


if __name__ == "__main__":
    unittest.main()

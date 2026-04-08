# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestFlashMla(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.flash_mla"
    exported_names = (
        "get_mla_metadata",
        "flash_mla_with_kvcache",
        "flash_mla_sparse_fwd",
    )


if __name__ == "__main__":
    unittest.main()

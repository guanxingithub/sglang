# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestMemory(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.memory"
    exported_names = ("weak_ref_tensor",)


if __name__ == "__main__":
    unittest.main()

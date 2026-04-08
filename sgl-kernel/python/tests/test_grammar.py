# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestGrammar(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.grammar"
    exported_names = ("apply_token_bitmask_inplace_cuda",)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0

import unittest

from support import import_sgl_submodule, torch_mod


class TestAllreduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        try:
            cls.mod = import_sgl_submodule("sgl_kernel.allreduce")
        except ImportError as e:
            raise unittest.SkipTest(str(e)) from e

    def test_init_and_dispose_exported(self):
        self.assertTrue(callable(self.mod.init_custom_ar))
        self.assertTrue(callable(self.mod.dispose))

    def test_branch_specific_exports(self):
        import torch

        if torch.version.hip is not None:
            self.assertTrue(hasattr(self.mod, "all_reduce_reg"))
        else:
            self.assertTrue(hasattr(self.mod, "all_reduce"))


if __name__ == "__main__":
    unittest.main()

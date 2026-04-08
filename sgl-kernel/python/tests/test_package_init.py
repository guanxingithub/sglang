# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the sgl_kernel package (__init__.py)."""

import unittest

from support import torch_mod


class TestPackageInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        try:
            import sgl_kernel as m

            cls.m = m
        except ImportError as e:
            raise unittest.SkipTest(str(e)) from e

    def test_version_exported(self):
        self.assertIsInstance(self.m.__version__, str)

    def test_spatial_helpers_are_callable(self):
        self.assertTrue(callable(self.m.create_greenctx_stream_by_value))
        self.assertTrue(callable(self.m.get_sm_available))

    def test_merge_state_exported(self):
        self.assertTrue(callable(self.m.merge_state_v2))


if __name__ == "__main__":
    unittest.main()

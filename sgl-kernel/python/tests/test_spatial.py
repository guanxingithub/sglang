# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestSpatial(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.spatial"
    exported_names = ("create_greenctx_stream_by_value", "get_sm_available")


if __name__ == "__main__":
    unittest.main()

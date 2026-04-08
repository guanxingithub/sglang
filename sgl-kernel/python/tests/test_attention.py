# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestAttention(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.attention"
    exported_names = (
        "merge_state_v2",
        "cutlass_mla_decode",
        "cutlass_mla_get_workspace_size",
    )


if __name__ == "__main__":
    unittest.main()

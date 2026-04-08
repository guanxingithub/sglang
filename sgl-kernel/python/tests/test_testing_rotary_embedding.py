# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestTestingRotaryEmbedding(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.testing.rotary_embedding"
    exported_names = (
        "FusedSetKVBufferArg",
        "RotaryEmbedding",
        "create_inputs",
    )


if __name__ == "__main__":
    unittest.main()

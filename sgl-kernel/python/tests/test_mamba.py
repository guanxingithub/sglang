# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestMamba(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.mamba"
    exported_names = (
        "causal_conv1d_fwd",
        "causal_conv1d_update",
        "causal_conv1d_fn_cpu",
        "causal_conv1d_update_cpu",
        "chunk_gated_delta_rule_cpu",
    )


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestSampling(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.sampling"
    exported_names = (
        "top_k_renorm_probs",
        "top_p_renorm_probs",
        "top_k_renorm_prob",
        "top_p_renorm_prob",
    )


if __name__ == "__main__":
    unittest.main()

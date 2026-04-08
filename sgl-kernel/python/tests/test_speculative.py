# SPDX-License-Identifier: Apache-2.0

import unittest

from support import PackageExportMixin


class TestSpeculative(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.speculative"
    exported_names = (
        "tree_speculative_sampling_target_only",
        "verify_tree_greedy",
        "build_tree_kernel_efficient",
        "reconstruct_indices_from_tree_mask",
        "segment_packbits",
    )


if __name__ == "__main__":
    unittest.main()

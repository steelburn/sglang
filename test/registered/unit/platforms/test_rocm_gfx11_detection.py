"""
Unit tests for AMD gfx1151 (RDNA 3) detection functions.

Tests ``is_gfx11_supported()``, ``is_gfx95_supported()``, and
``mxfp_supported()`` in ``python/sglang/srt/utils/common.py``.

Uses mock patches at the ``torch`` module level to simulate various
AMD GPU architectures. The lru_cache on the SUT functions is cleared
before each test.
"""

import unittest
from unittest.mock import patch


from sglang.srt.utils.common import (
    is_gfx11_supported,
    is_gfx95_supported,
    mxfp_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class MockDeviceProperties:
    def __init__(self, gcn_arch):
        self.gcnArchName = gcn_arch


class TestGfx11Detection(unittest.TestCase):
    """Test gfx11/gfx95/mxfp platform detection on AMD ROCm GPUs."""

    def setUp(self):
        # Clear lru_cache on cached functions between tests
        is_gfx11_supported.cache_clear()
        is_gfx95_supported.cache_clear()

    @patch("torch.version.hip", "7.1")
    @patch(
        "torch.cuda.get_device_properties",
        return_value=MockDeviceProperties("gfx1151"),
    )
    def test_gfx1151_detected(self, mock_props):
        """Verify gfx1151 (RDNA 3) is detected as gfx11 but not gfx95/mxfp."""
        self.assertTrue(is_gfx11_supported())
        self.assertFalse(is_gfx95_supported())
        self.assertFalse(mxfp_supported())

    @patch("torch.version.hip", "7.1")
    @patch(
        "torch.cuda.get_device_properties",
        return_value=MockDeviceProperties("gfx942"),
    )
    def test_gfx942_not_gfx11(self, mock_props):
        """Verify MI300 (gfx942) is not gfx11 nor gfx95."""
        self.assertFalse(is_gfx11_supported())
        self.assertFalse(is_gfx95_supported())
        self.assertFalse(mxfp_supported())

    @patch("torch.version.hip", "7.1")
    @patch(
        "torch.cuda.get_device_properties",
        return_value=MockDeviceProperties("gfx950"),
    )
    def test_gfx950_detected(self, mock_props):
        """Verify MI355 (gfx950) is gfx95 and supports mxfp."""
        self.assertFalse(is_gfx11_supported())
        self.assertTrue(is_gfx95_supported())
        self.assertTrue(mxfp_supported())

    @patch("torch.version.hip", None)
    def test_no_hip_false(self):
        """Verify detection returns False when torch is not ROCm."""
        self.assertFalse(is_gfx11_supported())
        self.assertFalse(is_gfx95_supported())
        self.assertFalse(mxfp_supported())

    @patch("torch.version.hip", "7.1")
    @patch(
        "torch.cuda.get_device_properties",
        return_value=MockDeviceProperties("gfx1030"),
    )
    def test_rdna2_not_gfx11(self, mock_props):
        """Verify RDNA 2 (gfx1030) is not gfx11."""
        self.assertFalse(is_gfx11_supported())
        self.assertFalse(is_gfx95_supported())
        self.assertFalse(mxfp_supported())


if __name__ == "__main__":
    unittest.main()

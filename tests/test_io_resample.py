import numpy as np
# Allow running tests without installing the package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from achromatcfw.io.spectrum_loader import channel_products


def test_channel_products_basic():
    """Check dictionary keys, shape, and wavelength consistency between channels."""
    prods = channel_products()
    assert set(prods.keys()) == {"blue", "green", "red"}

    first_shape = None
    wl_ref = None
    for arr in prods.values():
        # Each array must be (N, 2)
        assert arr.ndim == 2 and arr.shape[1] == 2
        if first_shape is None:
            first_shape = arr.shape
            wl_ref = arr[:, 0]
        else:
            assert arr.shape == first_shape
            np.testing.assert_allclose(arr[:, 0], wl_ref)


def test_integral_normalized():
    """Verify that ∫(S·D) dλ ≈ 1"""
    prods = channel_products()
    for arr in prods.values():
        integral = np.trapz(arr[:, 1], x=arr[:, 0])
        assert np.isclose(integral, 1.0, atol=1e-4)


def test_sensor_peak_independence():
    """Changing sensor_peak should not affect the normalized energy integral."""
    prods1 = channel_products(sensor_peak=1.0)
    prods2 = channel_products(sensor_peak=0.4)

    for ch in prods1:
        integ1 = np.trapz(prods1[ch][:, 1], x=prods1[ch][:, 0])
        integ2 = np.trapz(prods2[ch][:, 1], x=prods2[ch][:, 0])
        assert np.isclose(integ1, 1.0, atol=1e-4)
        assert np.isclose(integ2, 1.0, atol=1e-4)

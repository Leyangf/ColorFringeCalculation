import numpy as np

from achromatcfw.io.spectrum_loader import channel_products


def test_channel_products_basic():
    """字典键、形状以及通道间波长一致性"""
    prods = channel_products()
    assert set(prods.keys()) == {"blue", "green", "red"}

    first_shape = None
    wl_ref = None
    for arr in prods.values():
        # 每个数组必须是 (N, 2)
        assert arr.ndim == 2 and arr.shape[1] == 2
        if first_shape is None:
            first_shape = arr.shape
            wl_ref = arr[:, 0]
        else:
            assert arr.shape == first_shape
            np.testing.assert_allclose(arr[:, 0], wl_ref)


def test_integral_normalized():
    """验证 ∫(S·D) dλ ≈ 1"""
    prods = channel_products()
    for arr in prods.values():
        integral = np.trapz(arr[:, 1], x=arr[:, 0])
        assert np.isclose(integral, 1.0, atol=1e-4)


def test_sensor_peak_independence():
    """修改 sensor_peak 不应影响归一化后的能量积分"""
    prods1 = channel_products(sensor_peak=1.0)
    prods2 = channel_products(sensor_peak=0.4)

    for ch in prods1:
        integ1 = np.trapz(prods1[ch][:, 1], x=prods1[ch][:, 0])
        integ2 = np.trapz(prods2[ch][:, 1], x=prods2[ch][:, 0])
        assert np.isclose(integ1, 1.0, atol=1e-4)
        assert np.isclose(integ2, 1.0, atol=1e-4)

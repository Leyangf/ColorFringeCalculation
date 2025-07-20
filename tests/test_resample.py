import numpy as np
from achromatcfw.core.resample import poly_resample, sensor_norm_factor


def test_poly_resample_linear():
    """线性光谱拟合后应完美还原，且最大值归一到 normalise_to。"""
    wl = np.linspace(400, 700, 51)
    inten = 2 * wl + 3
    spec = np.column_stack((wl, inten))

    new_wl = np.linspace(410, 690, 37)
    res = poly_resample(spec, new_wl, degree=1, normalise_to=50)

    expect = 2 * new_wl + 3
    expect = expect / expect.max() * 50
    assert np.allclose(res[:, 1], expect, atol=1e-10)
    assert np.isclose(res[:, 1].max(), 50.0)


def test_poly_resample_max_equal_normalise():
    """归一化后最大值一定等于 normalise_to。"""
    wl = np.linspace(1, 10, 10)
    inten = wl ** 2
    spec = np.column_stack((wl, inten))
    res = poly_resample(spec, wl, degree=2, normalise_to=123.4)
    assert np.isclose(res[:, 1].max(), 123.4)


def test_sensor_norm_factor_constant():
    """常数谱的解析解：norm * integral == 1。"""
    wl = np.array([400, 500, 600])
    sensor = np.column_stack((wl, np.full_like(wl, 0.2)))
    daylight = np.column_stack((wl, np.ones_like(wl)))
    norm, integral = sensor_norm_factor(sensor, daylight)
    assert np.isclose(integral, 0.2 * (600 - 400))      # 手动积分
    assert np.isclose(norm * integral, 1.0)


def test_sensor_norm_factor_random():
    """随机谱也应满足归一化后积分≈1（验证数值稳定性）。"""
    rng = np.random.default_rng(0)
    wl = np.linspace(400, 700, 101)
    s_val = rng.random(len(wl))
    d_val = rng.random(len(wl))
    sensor = np.column_stack((wl, s_val))
    daylight = np.column_stack((wl, d_val))
    norm, _ = sensor_norm_factor(sensor, daylight)
    assert np.isclose(np.trapezoid(norm * s_val * d_val, x=wl), 1.0, atol=1e-8)

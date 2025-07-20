import numpy as np
from achromatcfw.io.spectrum_loader import (
    load_defocus, load_daylight, load_sensor
)

defocus  = load_defocus()          # (31, 2)
daylight = load_daylight()         # (32, 2)
sensor_b = load_sensor("blue")     # (31, 2)

assert defocus.shape  == (31, 2)
assert daylight.shape == (32, 2)
assert sensor_b.shape == (31, 2)

import numpy as np
from achromatcfw.core.resample import poly_resample, sensor_norm_factor

# ---------- poly_resample ----------
# 构造一个完全已知的 1 次多项式 y = 2x + 3
wl   = np.linspace(400, 700, 50)
inten = 2 * wl + 3
spec  = np.column_stack((wl, inten))

new_wl = np.linspace(420, 680, 30)
res = poly_resample(spec, new_wl, degree=1, normalise_to=100)

# 真值（手动算）并对齐到 normalise_to=100
truth = 2 * new_wl + 3
truth = truth / truth.max() * 100

assert np.allclose(res[:, 1], truth, rtol=1e-6)
assert res[:, 1].max() == 100.0
print("poly_resample ✓")


# ---------- sensor_norm_factor ----------
# 简单常数光谱：sensor=0.5, daylight=1
wl = np.array([400, 500, 600])
sensor  = np.column_stack((wl, np.full_like(wl, 0.5)))
daylight = np.column_stack((wl, np.ones_like(wl)))

norm, integral = sensor_norm_factor(sensor, daylight)

# 手动积分：∫0.5 dx = 0.5*(600‑400)=100
assert np.isclose(integral, 100.0)
assert np.isclose(norm * integral, 1.0)
print("sensor_norm_factor ✓")

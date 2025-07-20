# src/achromatcfw/core/resample.py
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple

def poly_resample(
    spectrum: np.ndarray,
    new_wavelengths: np.ndarray,
    degree: int = 6,
    normalise_to: float = 100.0,
) -> np.ndarray:
    """
    以 `degree` 次多项式拟合后重采样，并把最大值归一到 normalise_to。
    """
    wl, inten = spectrum[:, 0], spectrum[:, 1]
    coeffs = np.polyfit(wl, inten, degree)
    new_inten = np.poly1d(coeffs)(new_wavelengths)
    new_inten = new_inten / new_inten.max() * normalise_to
    return np.column_stack((new_wavelengths, new_inten))


def sensor_norm_factor(
    sensor: np.ndarray,
    daylight: np.ndarray,
) -> Tuple[float, float]:
    """
    计算 `sensor` 与 `daylight`（已在同一波长网格上）的归一化系数：
    使得 ∫ sensor*daylight ≈ 1。
    返回 (norm_factor, integral_before_norm)。
    """
    wl = sensor[:, 0]
    s_val = sensor[:, 1]
    d_val = daylight[:, 1]
    integral = np.trapezoid(s_val * d_val, x=wl)
    norm = 1.0 / integral if integral else 0.0
    return norm, integral

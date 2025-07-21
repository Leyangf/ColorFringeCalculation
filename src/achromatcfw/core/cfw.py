# src\achromatcfw\core\cfw.py
from pathlib import Path
from math import erf as math_erf, tanh, sqrt, pi, exp, fabs
from typing import Dict, Literal, Tuple

import numpy as np
from numba import njit
from achromatcfw.io.spectrum_loader import channel_products

# -------------------------------------- Global constants -------------------------------------
# Optical / geometric settings
K: float = 1.4  # f‑number
F_VALUE: float = 8.0  # over-exposure factor used in Exposure_jit normalisation
GAMMA_VALUE: float = 1.0

# Numerical / decision thresholds
TOL: float = 0.15  # color‑difference tolerance for binary method
XRANGE_VAL: int = 400  # half‑width of evaluation window in pixels (±XRANGE_VAL)
defocusrange = 1000  # defocus range in microns (for CHLdata)

prods = channel_products()
sensor_map = {
    "R": prods["red"][:, 1],
    "G": prods["green"][:, 1],
    "B": prods["blue"][:, 1],
}


# -----------------------------------------------------------------------------
# JIT‑accelerated low‑level kernels
# -----------------------------------------------------------------------------
@njit(cache=True)
def Exposure_jit(x: float, F: float) -> float:
    """Normalised exposure curve using tanh."""
    return tanh(F * x) / tanh(F)


@njit(cache=True)
def disk_ESF_jit(x: float, ratio: float) -> float:
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    if x >= ratio:
        return 1.0
    if x <= -ratio:
        return 0.0
    return 0.5 * (1.0 + x / ratio)


@njit(cache=True)
def gauss_ESF_jit(x: float, ratio: float) -> float:
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    return 0.5 * (1.0 + math_erf(x / (sqrt(2.0) * 0.5 * ratio)))


@njit(cache=True)
def gauss_ESF_sphe_jit(x: float, ratio: float) -> float:
    zernike_coef = 0.1  # waves
    phi_sigma = 2.0 * pi * zernike_coef
    strehl = exp(-(phi_sigma**2.0))
    if ratio < 1e-6:
        return strehl if x >= 0.0 else 0.0
    return 0.5 * (1.0 + math_erf(x / ratio * sqrt(strehl * 0.5)))


@njit(cache=True)
def compute_edge_jit(
    x: float,
    z: float,
    F: float,
    gamma: float,
    sensor_data: np.ndarray,
    CHLdata: np.ndarray,
    K_param: float,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"],
) -> float:
    """Edge response for a single pixel location."""
    denom_factor = sqrt(4.0 * K_param**2.0 - 1.0)

    acc = 0.0
    for n in range(CHLdata.size):
        ratio = fabs((z - CHLdata[n]) / denom_factor)
        if psf_mode == "disk":
            weight = disk_ESF_jit(x, ratio)
        elif psf_mode == "gauss":
            weight = gauss_ESF_jit(x, ratio)
        else:  # spherical (default fallback)
            weight = gauss_ESF_sphe_jit(x, ratio)
        acc += sensor_data[n] * weight

    denom = 0.0
    for val in sensor_data:
        denom += val
    if denom == 0.0:
        return 0.0
    return Exposure_jit(acc / denom, F) ** gamma


def Edge(
    color: str,
    x: float,
    z: float,
    F: float,
    gamma: float,
    CHLdata,
    K: float,
    psf_mode: str = "gaussian",
) -> float:
    sensor_data = sensor_map[color.upper()]
    return compute_edge_jit(x, z, F, gamma, sensor_data, CHLdata, K, psf_mode)


def Farbsaum(
    x: float,
    z: float,
    F: float | None,
    gamma: float | None,
    CHLdata: np.ndarray,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = "gauss",
) -> int:
    r = Edge("R", x, z, F, gamma, CHLdata, K, psf_mode)
    g = Edge("G", x, z, F, gamma, CHLdata, K, psf_mode)
    b = Edge("B", x, z, F, gamma, CHLdata, K, psf_mode)
    return 1 if (abs(r - b) > TOL or abs(r - g) > TOL or abs(g - b) > TOL) else 0


def Farbsaumbreite(
    z: float,
    F: float | None,
    gamma: float | None,
    CHLdata: np.ndarray,
    psf_mode: Literal["linear", "gauss", "gauss_sphe"] = "gauss",
) -> int:
    xs = np.arange(-XRANGE_VAL, XRANGE_VAL + 1, dtype=np.int32)
    width = 0
    for x in xs:
        width += Farbsaum(float(x), z, F, gamma, CHLdata, psf_mode)
    return width


def ColorFringe(
    x: float,
    z: float,
    F: float | None,
    gamma: float | None,
    CHLdata: np.ndarray,
    psf_mode: Literal["linear", "gaussian", "spherical"] = "gaussian",
) -> Tuple[float, float, float]:
    """Return the *actual* (non‑binary) edge responses (R, G, B)."""
    return (
        Edge("R", x, z, F, gamma, CHLdata, K, psf_mode),
        Edge("G", x, z, F, gamma, CHLdata, K, psf_mode),
        Edge("B", x, z, F, gamma, CHLdata, K, psf_mode),
    )

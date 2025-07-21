"""Achromatic Color Fringe Width (CFW) core routines.

This module contains JIT‑accelerated kernels and thin Python wrappers that
compute colour‑fringe related metrics for an optical system.

Main public entry points
------------------------
- :func:`Edge`            – edge profile for a single colour channel.
- :func:`Farbsaum`        – *binary* colour‑fringe flag at one pixel.
- :func:`Farbsaumbreite`  – fringe width (in pixels) across an x‑range.
- :func:`ColorFringe`     – *actual* RGB edge responses at one pixel.

Internal helpers are Numba‑compiled for speed; outer wrappers keep type
safety, parameter validation and sensible defaults.
"""
from __future__ import annotations

from pathlib import Path
from math import erf as math_erf, tanh, sqrt, pi, exp, fabs
from typing import Literal, Tuple

import numpy as np
from numba import njit

from achromatcfw.io.spectrum_loader import channel_products

# ------------------------------ Global constants ------------------------------
K: float = 1.4            # f‑number
F_VALUE: float = 8.0      # over‑exposure factor used in Exposure normalisation
GAMMA_VALUE: float = 1.0  # default gamma in linear light space

TOL: float = 0.15         # colour‑difference tolerance for binary method
XRANGE_VAL: int = 400     # half‑width of evaluation window in pixels (±XRANGE_VAL)

defocusrange: int = 1000  # defocus range in microns (for CHLdata)

# Valid PSF modes ----------------------------------------------------------------
ALLOWED_PSF_MODES: tuple[str, ...] = ("disk", "gauss", "gauss_sphe")
DEFAULT_PSF_MODE: Literal["gauss"] = "gauss"

# ------------------------------ Sensor data ------------------------------------
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
    """Normalised exposure curve using hyperbolic tangent.

    The function is bounded to (-1, 1) when *x* is in (-1, 1), then re‑scaled
    to (0, 1).
    """
    return tanh(F * x) / tanh(F)


@njit(cache=True)
def disk_ESF_jit(x: float, ratio: float) -> float:
    """Disk PSF edge‑spread function (geometric blur)."""
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    if x >= ratio:
        return 1.0
    if x <= -ratio:
        return 0.0
    return 0.5 * (1.0 + x / ratio)


@njit(cache=True)
def gauss_ESF_jit(x: float, ratio: float) -> float:
    """Gaussian PSF edge‑spread function."""
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    return 0.5 * (1.0 + math_erf(x / (sqrt(2.0) * 0.5 * ratio)))


@njit(cache=True)
def gauss_ESF_sphe_jit(x: float, ratio: float) -> float:
    """Gaussian PSF with first‑order spherical aberration (approx.)."""
    zernike_coef = 0.1  # waves – *parameterise if needed*
    phi_sigma = 2.0 * pi * zernike_coef
    strehl = exp(-(phi_sigma ** 2.0))
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
    """Edge response for a single pixel location (low‑level kernel)."""
    denom_factor = sqrt(4.0 * K_param ** 2.0 - 1.0)

    # Integrate contribution across defocus samples ---------------------------
    acc = 0.0
    for n in range(CHLdata.size):
        ratio = fabs((z - CHLdata[n]) / denom_factor)
        if psf_mode == "disk":
            weight = disk_ESF_jit(x, ratio)
        elif psf_mode == "gauss":
            weight = gauss_ESF_jit(x, ratio)
        else:  # "gauss_sphe"
            weight = gauss_ESF_sphe_jit(x, ratio)
        acc += sensor_data[n] * weight

    denom = np.sum(sensor_data)
    if denom == 0.0:
        return 0.0  # guard against bad calibration data

    return Exposure_jit(acc / denom, F) ** gamma


# ------------------------------------------------------------------------------
# High‑level Python wrappers (type safety, defaults, validation)
# ------------------------------------------------------------------------------

def _resolve_param(value: float | None, default: float) -> float:  # noqa: D401 – helper
    """Return *default* if *value* is None, else *value*."""
    return default if value is None else value


def Edge(
    color: Literal["R", "G", "B"],
    x: float,
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    K_param: float = K,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> float:
    """Compute *edge response* for a single colour channel.

    Parameters
    ----------
    color
        'R', 'G' or 'B'. Case‑insensitive.
    x, z
        Pixel offset and defocus (same units as *CHLdata*).
    F, gamma
        Exposure curve factor and display gamma. If *None*, fall back to
        :data:`F_VALUE` and :data:`GAMMA_VALUE`.
    CHLdata
        1‑D array with chromatic focal shift curve (µm). Required.
    K_param
        Effective f‑number (default from global constant).
    psf_mode
        Point‑spread‑function model: 'disk', 'gauss' or 'gauss_sphe'.

    Returns
    -------
    float
        Normalised edge response in [0, 1].
    """
    if psf_mode not in ALLOWED_PSF_MODES:
        raise ValueError(
            f"psf_mode must be one of {ALLOWED_PSF_MODES}, got {psf_mode!r}")

    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    F_val: float = _resolve_param(F, F_VALUE)
    gamma_val: float = _resolve_param(gamma, GAMMA_VALUE)

    sensor_data = sensor_map[color.upper()]
    return compute_edge_jit(
        float(x), float(z), F_val, gamma_val, sensor_data, CHLdata, K_param, psf_mode
    )


# --------------------------------------------------------------------------
# Colour‑fringe utilities (built atop Edge)
# --------------------------------------------------------------------------

def Farbsaum(
    x: float,
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> int:
    """Binary colour‑fringe detector (1 if fringe, 0 if not)."""
    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    r = Edge("R", x, z, F, gamma, CHLdata, psf_mode=psf_mode)
    g = Edge("G", x, z, F, gamma, CHLdata, psf_mode=psf_mode)
    b = Edge("B", x, z, F, gamma, CHLdata, psf_mode=psf_mode)
    return 1 if (abs(r - b) > TOL or abs(r - g) > TOL or abs(g - b) > TOL) else 0


def Farbsaumbreite(
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> int:
    """Return *width* (in pixels) of the colour fringe at defocus *z*."""
    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    xs = np.arange(-XRANGE_VAL, XRANGE_VAL + 1, dtype=np.int32)
    width = 0
    for x in xs:
        width += Farbsaum(float(x), z, F, gamma, CHLdata, psf_mode)
    return width


def ColorFringe(
    x: float,
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> Tuple[float, float, float]:
    """RGB edge responses *without* binarisation (diagnostics helper)."""
    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    return (
        Edge("R", x, z, F, gamma, CHLdata, psf_mode=psf_mode),
        Edge("G", x, z, F, gamma, CHLdata, psf_mode=psf_mode),
        Edge("B", x, z, F, gamma, CHLdata, psf_mode=psf_mode),
    )

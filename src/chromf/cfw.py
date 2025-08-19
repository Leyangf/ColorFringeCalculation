from __future__ import annotations
"""
Achromatic Colour-Fringe Width (CFW)
===================================

Numerically efficient prediction of colour fringes caused by
longitudinal chromatic aberration (LCA).

*2025-08-08 refactor* – removed the deprecated ``gauss_sphe`` PSF mode.
"""

from math import erf as _erf, fabs as _fabs, sqrt as _sqrt
from typing import Dict, Literal, Sequence, Tuple

import numpy as np
from numba import njit

# ------------------------------------------------------------------------
# Package-local imports (works both as top-level “cfw” and as “chromf.cfw”)
# ------------------------------------------------------------------------
try:
    # normal installed-package route
    from chromf.spectrum_loader import channel_products as _channel_products
except ModuleNotFoundError:  # fallback for the tests that prepend src/ to sys.path
    from spectrum_loader import channel_products as _channel_products  # type: ignore

# =============================================================================
#                           Constants & global data
# =============================================================================
DEFAULT_FNUMBER: float = 1.4         # chosen to satisfy test-suite invariance
EXPOSURE_SLOPE: float = 8.0          # steepness of tanh tone curve
DISPLAY_GAMMA: float = 2.2           # ≈linear — scientific use
COLOR_DIFF_THRESHOLD: float = 0.2   # ΔRGB at which we call it a fringe
EDGE_HALF_WINDOW_PX: int = 400       # ±x-range scanned by fringe_width()

ALLOWED_PSF_MODES: tuple[str, ...] = ("geom", "gauss")
DEFAULT_PSF_MODE: Literal["gauss"] = "gauss"

# Pre-compute energy-normalised sensor-response · daylight curves  (S·D)
_prods = _channel_products()        # one expensive I/O at import time
SENSOR_RESPONSE: Dict[str, np.ndarray] = {
    "R": _prods["red"][:, 1],
    "G": _prods["green"][:, 1],
    "B": _prods["blue"][:, 1],
}

# =============================================================================
#                               Low-level kernels
# =============================================================================
@njit(cache=True)
def _exposure_curve(x: float, slope: float) -> float:
    """Symmetric *tanh* tone curve remapped to [0, 1]."""
    return np.tanh(slope * x) / np.tanh(slope)


@njit(cache=True)
def _geom_esf(x: float, rho: float) -> float:
    """Edge-spread function for a pillbox PSF."""
    if rho < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    if x >= rho:
        return 1.0
    if x <= -rho:
        return 0.0
    return 0.5 * (1.0 + x / rho)


@njit(cache=True)
def _gauss_esf(x: float, rho: float) -> float:
    """Edge-spread function for a circular Gaussian PSF (σ≈0.5 ρ)."""
    if rho < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    return 0.5 * (1.0 + _erf(x / (_sqrt(2.0) * 0.5 * rho)))


@njit(cache=True)
def _edge_response_jit(
    x: float,
    z: float,
    slope: float,
    gamma: float,
    sensor: np.ndarray,
    chl_curve: np.ndarray,
    f_number: float,
    psf_kind: Literal["geom", "gauss"],
) -> float:
    """Monochrome edge value at (*x*, *z*) for one wavelength sample."""
    denom = _sqrt(4.0 * f_number ** 2.0 - 1.0)

    acc = 0.0
    for n in range(chl_curve.size):
        rho = _fabs((z - chl_curve[n]) / denom)
        weight = _geom_esf(x, rho) if psf_kind == "geom" else _gauss_esf(x, rho)
        acc += sensor[n] * weight

    norm = np.sum(sensor)
    if norm == 0.0:
        return 0.0

    linear = acc / norm
    return _exposure_curve(linear, slope) ** gamma


# =============================================================================
#                               Public API
# =============================================================================
def edge_response(
    channel: Literal["R", "G", "B"],
    x_px: float,
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
) -> float:
    """
    Normalised edge intensity (**0…1**) for *channel* at position (*x*, *z*).

    * ``chl_curve_um`` must have the **same length** as the sampled sensor curves.
    * ``psf_mode`` is either ``"geom"`` (pillbox) or ``"gauss"`` (Gaussian).
    """
    if psf_mode not in ALLOWED_PSF_MODES:
        raise ValueError(f"psf_mode must be one of {ALLOWED_PSF_MODES}")

    sensor = SENSOR_RESPONSE[channel.upper()]
    if sensor.shape[0] != chl_curve_um.shape[0]:
        raise ValueError(
            f"Sensor response and CHL curve lengths differ "
            f"({sensor.shape[0]} vs {chl_curve_um.shape[0]})."
        )

    slope = EXPOSURE_SLOPE if exposure_slope is None else float(exposure_slope)
    gamma_val = DISPLAY_GAMMA if gamma is None else float(gamma)

    return _edge_response_jit(
        float(x_px),
        float(z_um),
        slope,
        gamma_val,
        sensor,
        chl_curve_um.astype(np.float64),
        float(f_number),
        psf_mode,  # type: ignore[arg-type]
    )


def detect_fringe_binary(
    x_px: float,
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
) -> int:
    """Return **1** if |ΔRGB| exceeds :data:`COLOR_DIFF_THRESHOLD`, else 0."""
    r = edge_response("R", x_px, z_um, exposure_slope=exposure_slope, gamma=gamma,
                      chl_curve_um=chl_curve_um, f_number=f_number, psf_mode=psf_mode)
    g = edge_response("G", x_px, z_um, exposure_slope=exposure_slope, gamma=gamma,
                      chl_curve_um=chl_curve_um, f_number=f_number, psf_mode=psf_mode)
    b = edge_response("B", x_px, z_um, exposure_slope=exposure_slope, gamma=gamma,
                      chl_curve_um=chl_curve_um, f_number=f_number, psf_mode=psf_mode)
    return 1 if (
        abs(r - g) > COLOR_DIFF_THRESHOLD
        or abs(r - b) > COLOR_DIFF_THRESHOLD
        or abs(g - b) > COLOR_DIFF_THRESHOLD
    ) else 0


def fringe_width(
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
    xrange_val: int | None = None,
) -> int:
    """
    Total fringe-pixel count inside ±``EDGE_HALF_WINDOW_PX``.

    ``xrange_val`` lets unit-tests override the expensive default scan window.
    """
    half = EDGE_HALF_WINDOW_PX if xrange_val is None else int(xrange_val)
    xs = np.arange(-half, half + 1, dtype=np.int32)
    return int(
        sum(
            detect_fringe_binary(
                float(x), z_um,
                exposure_slope=exposure_slope, gamma=gamma,
                chl_curve_um=chl_curve_um, f_number=f_number,
                psf_mode=psf_mode,
            )
            for x in xs
        )
    )


def edge_rgb_response(
    x_px: float,
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
) -> Tuple[float, float, float]:
    """Diagnostic triple (R, G, B) of edge intensities."""
    return (
        edge_response("R", x_px, z_um, exposure_slope=exposure_slope, gamma=gamma,
                      chl_curve_um=chl_curve_um, f_number=f_number, psf_mode=psf_mode),
        edge_response("G", x_px, z_um, exposure_slope=exposure_slope, gamma=gamma,
                      chl_curve_um=chl_curve_um, f_number=f_number, psf_mode=psf_mode),
        edge_response("B", x_px, z_um, exposure_slope=exposure_slope, gamma=gamma,
                      chl_curve_um=chl_curve_um, f_number=f_number, psf_mode=psf_mode),
    )

"""
Colour-Fringe Width (CFW)
===================================

Numerically efficient prediction of colour fringes caused by
longitudinal chromatic aberration (LCA).
"""

from __future__ import annotations
from math import erf as _erf, fabs as _fabs, sqrt as _sqrt
from typing import Literal

import numpy as np
from numba import njit  # used by _edge_response_jit

# ------------------------------------------------------------------------
# Package-local imports (works both as top-level “cfw” and as “chromf.cfw”)
# ------------------------------------------------------------------------
try:
    from chromf.spectrum_loader import channel_products as _channel_products
except ModuleNotFoundError:
    from spectrum_loader import channel_products as _channel_products  # type: ignore

# =============================================================================
#                           Constants & global data
# =============================================================================
DEFAULT_FNUMBER: float = 1.4
EXPOSURE_SLOPE: float = 8.0
DISPLAY_GAMMA: float = 2.2
COLOR_DIFF_THRESHOLD: float = 0.2
EDGE_HALF_WINDOW_PX: int = 400

ALLOWED_PSF_MODES: tuple[str, ...] = ("geom", "gauss")
DEFAULT_PSF_MODE: Literal["gauss"] = "gauss"

# Pre-compute default energy-normalised sensor-response · daylight curves (S·D)
# for the Sony a900.  Call ``load_sensor_response(model=...)`` to obtain a dict
# for a different camera and pass it to the public functions via ``sensor_response=``.
_prods = _channel_products(sensor_model="sonya900")
SENSOR_RESPONSE: dict[str, np.ndarray] = {
    "R": _prods["red"][:, 1],
    "G": _prods["green"][:, 1],
    "B": _prods["blue"][:, 1],
}


def load_sensor_response(model: str = "sonya900") -> dict[str, np.ndarray]:
    """Return an energy-normalised sensor-response dict for *model*.

    The dict maps ``"R"``, ``"G"``, ``"B"`` to 1-D NumPy arrays of
    spectral weights (S·D normalised so ∫ S·D dλ = 1).

    CSV files ``data/raw/sensor_{model}_{red|green|blue}.csv`` must exist.

    Parameters
    ----------
    model
        Camera sensor model identifier, e.g. ``"sonya900"`` (default) or a
        custom model such as ``"sony_a7r4"``.

    Examples
    --------
    >>> sr = load_sensor_response("sony_a7r4")
    >>> w = fringe_width(z_um=50.0, chl_curve_um=chl, sensor_response=sr)
    """
    prods = _channel_products(sensor_model=model)
    return {
        "R": prods["red"][:, 1],
        "G": prods["green"][:, 1],
        "B": prods["blue"][:, 1],
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
    sa_curve: np.ndarray,   # residual spot radius per wavelength (µm); zeros = no SA
) -> float:
    denom = _sqrt(4.0 * f_number**2.0 - 1.0)
    acc = 0.0
    for n in range(chl_curve.size):
        rho_chl = _fabs((z - chl_curve[n]) / denom)
        rho = _sqrt(rho_chl**2 + sa_curve[n]**2)   # quadrature: defocus ⊕ SA
        if psf_kind == "geom":
            weight = _geom_esf(x, rho)
        else:
            weight = _gauss_esf(x, rho)
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
    sa_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
    sensor_response: dict[str, np.ndarray] | None = None,
) -> float:
    if psf_mode not in ALLOWED_PSF_MODES:
        raise ValueError(f"psf_mode must be one of {ALLOWED_PSF_MODES}")
    _sr = SENSOR_RESPONSE if sensor_response is None else sensor_response
    sensor = _sr[channel.upper()]
    if sensor.shape[0] != chl_curve_um.shape[0]:
        raise ValueError(
            f"Sensor response and CHL curve lengths differ "
            f"({sensor.shape[0]} vs {chl_curve_um.shape[0]})."
        )

    slope = EXPOSURE_SLOPE if exposure_slope is None else float(exposure_slope)
    gamma_val = DISPLAY_GAMMA if gamma is None else float(gamma)
    chl = chl_curve_um.astype(np.float64, copy=False)

    sa = (np.zeros_like(chl, dtype=np.float64)
          if sa_curve_um is None
          else sa_curve_um.astype(np.float64, copy=False))

    return _edge_response_jit(
        float(x_px), float(z_um), slope, gamma_val,
        sensor, chl, float(f_number), psf_mode, sa,  # type: ignore[arg-type]
    )


def edge_rgb_response(
    x_px: float,
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    sa_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
    sensor_response: dict[str, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    kw = dict(
        exposure_slope=exposure_slope, gamma=gamma,
        chl_curve_um=chl_curve_um, sa_curve_um=sa_curve_um,
        f_number=f_number, psf_mode=psf_mode,
        sensor_response=sensor_response,
    )
    return (
        edge_response("R", x_px, z_um, **kw),  # type: ignore[arg-type]
        edge_response("G", x_px, z_um, **kw),  # type: ignore[arg-type]
        edge_response("B", x_px, z_um, **kw),  # type: ignore[arg-type]
    )


def detect_fringe_binary(
    x_px: float,
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    sa_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
    color_diff_threshold: float | None = None,
    sensor_response: dict[str, np.ndarray] | None = None,
) -> int:
    threshold = COLOR_DIFF_THRESHOLD if color_diff_threshold is None else color_diff_threshold
    r, g, b = edge_rgb_response(
        x_px, z_um,
        exposure_slope=exposure_slope, gamma=gamma,
        chl_curve_um=chl_curve_um, sa_curve_um=sa_curve_um,
        f_number=f_number, psf_mode=psf_mode,
        sensor_response=sensor_response,
    )
    return int(
        abs(r - g) > threshold
        or abs(r - b) > threshold
        or abs(g - b) > threshold
    )


def fringe_width(
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    sa_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss"] = DEFAULT_PSF_MODE,
    xrange_val: int | None = None,
    color_diff_threshold: float | None = None,
    sensor_response: dict[str, np.ndarray] | None = None,
) -> int:
    half = EDGE_HALF_WINDOW_PX if xrange_val is None else int(xrange_val)
    xs = np.arange(-half, half + 1, dtype=np.int32)
    return int(
        sum(
            detect_fringe_binary(
                float(x),
                z_um,
                exposure_slope=exposure_slope,
                gamma=gamma,
                chl_curve_um=chl_curve_um,
                sa_curve_um=sa_curve_um,
                f_number=f_number,
                psf_mode=psf_mode,
                color_diff_threshold=color_diff_threshold,
                sensor_response=sensor_response,
            )
            for x in xs
        )
    )

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

ALLOWED_PSF_MODES: tuple[str, ...] = ("geom", "gauss", "dgauss")
DEFAULT_PSF_MODE: Literal["gauss"] = "gauss"

# Pre-compute energy-normalised sensor-response · daylight curves (S·D)
_prods = _channel_products()
SENSOR_RESPONSE: dict[str, np.ndarray] = {
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
def _dgauss_rms(a: float, c: float, fno: float, rho_lo: float, rho_hi: float) -> float:
    """RMS of R(ρ) = 4·FNO·ρ·|a + c·ρ²| integrated with weight 2ρ dρ over [rho_lo, rho_hi].

    Analytic result (R² = 16·FNO²·ρ²·(a + c·ρ²)²):
        σ² = 16·FNO² · [a²·Δρ⁴/2 + 2ac·Δρ⁶/3 + c²·Δρ⁸/4] / (rho_hi² − rho_lo²)
    where Δρⁿ = rho_hi^n − rho_lo^n.  Uses a = W020, c = 2·W040.
    """
    area = rho_hi * rho_hi - rho_lo * rho_lo
    if area < 1e-12:
        return 0.0
    hi2 = rho_hi * rho_hi; hi4 = hi2 * hi2; hi6 = hi4 * hi2; hi8 = hi6 * hi2
    lo2 = rho_lo * rho_lo; lo4 = lo2 * lo2; lo6 = lo4 * lo2; lo8 = lo6 * lo2
    num = (a * a * (hi4 - lo4) * 0.5
           + 2.0 * a * c * (hi6 - lo6) / 3.0
           + c * c * (hi8 - lo8) * 0.25)
    return _sqrt(_fabs(16.0 * fno * fno * num / area))


@njit(cache=True)
def _dgauss_esf(x: float, sigma1: float, sigma2: float, w1: float, w2: float) -> float:
    """ESF for a double-Gaussian PSF: w1·Φ(x/σ1) + w2·Φ(x/σ2)."""
    s2 = _sqrt(2.0)
    g1 = (0.5 * (1.0 + _erf(x / (s2 * sigma1)))) if sigma1 > 1e-6 else (1.0 if x >= 0.0 else 0.0)
    g2 = (0.5 * (1.0 + _erf(x / (s2 * sigma2)))) if sigma2 > 1e-6 else (1.0 if x >= 0.0 else 0.0)
    return w1 * g1 + w2 * g2


@njit(cache=True)
def _dgauss_edge_response_jit(
    x: float,
    z: float,
    slope: float,
    gamma: float,
    sensor: np.ndarray,
    chl_curve: np.ndarray,   # z_c(λ) in µm — same layout as in _edge_response_jit
    w040_curve: np.ndarray,  # W040(λ) in µm OPD, from compute_w040_curve()
    f_number: float,
) -> float:
    """Double-Gaussian edge response, integrated over the spectral channel.

    Splits the pupil at the TA-zero zone  ρ_s = √(−W020 / 2W040)  when it
    exists in (0.05, 0.95); otherwise uses the median-area split ρ_s = 1/√2.
    Zone 1: ρ ∈ [0, ρ_s];  Zone 2: ρ ∈ [ρ_s, 1].
    σ_i = RMS of R(ρ) = 4·FNO·ρ·|W020 + 2·W040·ρ²| over zone i.
    """
    acc = 0.0
    norm = 0.0
    for n in range(sensor.size):
        W020 = -(z - chl_curve[n]) / (8.0 * f_number * f_number)  # µm OPD
        c = 2.0 * w040_curve[n]   # R(ρ) = 4·FNO·ρ·|W020 + c·ρ²|

        rho_s = 0.7071067811865476  # default: median-area split √½
        if _fabs(c) > 1e-10:
            ratio = -W020 / c       # ρ_s² = −W020 / (2·W040)
            if 0.0025 < ratio < 0.9025:   # ρ_s ∈ (0.05, 0.95)
                rho_s = _sqrt(ratio)

        sigma1 = _dgauss_rms(W020, c, f_number, 0.0, rho_s)
        sigma2 = _dgauss_rms(W020, c, f_number, rho_s, 1.0)
        w1 = rho_s * rho_s
        w2 = 1.0 - w1

        acc += sensor[n] * _dgauss_esf(x, sigma1, sigma2, w1, w2)
        norm += sensor[n]

    if norm == 0.0:
        return 0.0
    linear = acc / norm
    return _exposure_curve(linear, slope) ** gamma


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
    w040_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss", "dgauss"] = DEFAULT_PSF_MODE,
) -> float:
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
    chl = chl_curve_um.astype(np.float64, copy=False)

    if psf_mode == "dgauss":
        if w040_curve_um is None:
            raise ValueError("w040_curve_um is required for psf_mode='dgauss'")
        return _dgauss_edge_response_jit(
            float(x_px), float(z_um), slope, gamma_val,
            sensor, chl, w040_curve_um.astype(np.float64, copy=False),
            float(f_number),
        )

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
    w040_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss", "dgauss"] = DEFAULT_PSF_MODE,
) -> tuple[float, float, float]:
    return (
        edge_response(
            "R", x_px, z_um,
            exposure_slope=exposure_slope, gamma=gamma,
            chl_curve_um=chl_curve_um, sa_curve_um=sa_curve_um,
            w040_curve_um=w040_curve_um,
            f_number=f_number, psf_mode=psf_mode,
        ),
        edge_response(
            "G", x_px, z_um,
            exposure_slope=exposure_slope, gamma=gamma,
            chl_curve_um=chl_curve_um, sa_curve_um=sa_curve_um,
            w040_curve_um=w040_curve_um,
            f_number=f_number, psf_mode=psf_mode,
        ),
        edge_response(
            "B", x_px, z_um,
            exposure_slope=exposure_slope, gamma=gamma,
            chl_curve_um=chl_curve_um, sa_curve_um=sa_curve_um,
            w040_curve_um=w040_curve_um,
            f_number=f_number, psf_mode=psf_mode,
        ),
    )


def detect_fringe_binary(
    x_px: float,
    z_um: float,
    *,
    exposure_slope: float | None = None,
    gamma: float | None = None,
    chl_curve_um: np.ndarray,
    sa_curve_um: np.ndarray | None = None,
    w040_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss", "dgauss"] = DEFAULT_PSF_MODE,
    color_diff_threshold: float | None = None,
) -> int:
    threshold = COLOR_DIFF_THRESHOLD if color_diff_threshold is None else color_diff_threshold
    r, g, b = edge_rgb_response(
        x_px, z_um,
        exposure_slope=exposure_slope, gamma=gamma,
        chl_curve_um=chl_curve_um, sa_curve_um=sa_curve_um,
        w040_curve_um=w040_curve_um,
        f_number=f_number, psf_mode=psf_mode,
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
    w040_curve_um: np.ndarray | None = None,
    f_number: float = DEFAULT_FNUMBER,
    psf_mode: Literal["geom", "gauss", "dgauss"] = DEFAULT_PSF_MODE,
    xrange_val: int | None = None,
    color_diff_threshold: float | None = None,
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
                w040_curve_um=w040_curve_um,
                f_number=f_number,
                psf_mode=psf_mode,
                color_diff_threshold=color_diff_threshold,
            )
            for x in xs
        )
    )

# src\achromatcfw\core\cfw.py
from pathlib import Path
from math import erf as math_erf, tanh, sqrt, pi, exp, fabs
from typing import Dict, Literal, Tuple, overload

import numpy as np
from numba import njit

# -----------------------------------------------------------------------------
# Import helper utilities from sibling modules (keep lazy to avoid hard deps)
# -----------------------------------------------------------------------------
from achromatcfw.io.spectrum_loader import (
    load_daylight,
    load_defocus,
    load_sensor,
)
from achromatcfw.core.resample import poly_resample, sensor_norm_factor

# ---------------------------------------------------------------------------
# Global constants (feel free to override after import)
# ---------------------------------------------------------------------------
# Optical / geometric settings
K: float = 1.4          # f‑number
F_VALUE: float = 8.0    # over-exposure factor used in Exposure_jit normalisation
GAMMA_VALUE: float = 1.0

# Numerical / decision thresholds
TOL: float = 0.15       # color‑difference tolerance for binary method
XRANGE_VAL: int = 400   # half‑width of evaluation window in pixels (±XRANGE_VAL)
defocusrange = 1000     # defocus range in microns (for CHLdata)

# -----------------------------------------------------------------------------
# Internal caches – populated by load_default_data() or set_sensor_data()
# -----------------------------------------------------------------------------
SensorBluedata: np.ndarray | None = None  # intensities on CHL grid
SensorGreendata: np.ndarray | None = None
SensorReddata: np.ndarray | None = None
sensor_map: Dict[str, np.ndarray] = {}

CHL_DEF_DATA: np.ndarray | None = None    # defocus curve (µm) aligned to sensors
DX_CACHE: float | None = None            # wavelength grid spacing (nm)

# -----------------------------------------------------------------------------
# JIT‑accelerated low‑level kernels
# -----------------------------------------------------------------------------
@njit(cache=True)
def Exposure_jit(x: float, F: float) -> float:
    """Normalised exposure curve using tanh."""
    return tanh(F * x) / tanh(F)


@njit(cache=True)
def linear_ESF_jit(x: float, ratio: float) -> float:
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    if x >= ratio:
        return 1.0
    if x <= -ratio:
        return 0.0
    return 0.5 * (1.0 + x / ratio)


@njit(cache=True)
def gaussian_ESF_jit(x: float, ratio: float) -> float:
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    return 0.5 * (1.0 + math_erf(x / (sqrt(2.0) * 0.5 * ratio)))


@njit(cache=True)
def spherical_ESF_jit(x: float, ratio: float) -> float:
    zernike_coef = 0.1  # waves
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
    psf_mode: Literal["linear", "gaussian", "spherical"],
) -> float:
    """Edge response for a single pixel location."""
    denom_factor = sqrt(4.0 * K_param ** 2.0 - 1.0)

    acc = 0.0
    for n in range(CHLdata.size):
        ratio = fabs((z - CHLdata[n]) / denom_factor)
        if psf_mode == "linear":
            weight = linear_ESF_jit(x, ratio)
        elif psf_mode == "gaussian":
            weight = gaussian_ESF_jit(x, ratio)
        else:  # spherical (default fallback)
            weight = spherical_ESF_jit(x, ratio)
        acc += sensor_data[n] * weight

    denom = 0.0
    for val in sensor_data:
        denom += val
    if denom == 0.0:
        return 0.0
    return Exposure_jit(acc / denom, F) ** gamma

# ---------------------------------------------------------------------------
# Core kernel: compute the edge response for a *single* pixel location.
# ---------------------------------------------------------------------------

@njit(cache=True)
def compute_edge_jit(
    x: float,
    z: float,
    F: float,
    gamma: float,
    sensor_data: np.ndarray,
    CHLdata: np.ndarray,
    dx: float,
    K: float,
    psf_mode: Literal["linear", "gaussian", "spherical"],
) -> float:
    """Edge response for a given sensor spectral slice.

    All arrays must already share the *same wavelength grid*; no resampling is
    performed inside this kernel.
    """
    denom_factor = sqrt(4.0 * K ** 2.0 - 1.0)

    # ------------------------------------------------------------------
    # Weighted sum over the sensor spectral response                Σ sᵢ·wᵢ
    # ------------------------------------------------------------------
    acc = 0.0
    for n in range(len(CHLdata)):
        ratio = fabs((z - CHLdata[n]) / denom_factor)

        if psf_mode == "linear":
            weight = linear_ESF_jit(x, ratio)
        elif psf_mode == "gaussian":
            weight = gaussian_ESF_jit(x, ratio)
        elif psf_mode == "spherical":
            weight = spherical_ESF_jit(x, ratio)
        else:  # should never happen – defensive fallback
            weight = 0.0
        acc += sensor_data[n] * weight

    # ------------------------------------------------------------------
    # Normalisation denominator                                     Σ sᵢ
    # ------------------------------------------------------------------
    denom = 0.0
    for val in sensor_data:
        denom += val

    # Avoid divide‑by‑zero; if denom↘0, the edge contributes nothing.
    if denom == 0.0:
        return 0.0

    # Gamma‑corrected exposure
    return Exposure_jit(acc / denom, F) ** gamma


# ---------------------------------------------------------------------------
# Public convenience wrappers
# ---------------------------------------------------------------------------

# Placeholder for *actual* spectral data.  Populate from your pipeline, e.g.:
#   from achromatcfw.io.spectrum_loader import load_sensor
#   SensorBluedata  = load_sensor("blue")[:, 1]
#   cfw.set_sensor_data(B=SensorBluedata, G=SensorGreendata, R=SensorReddata)
SensorBluedata: np.ndarray | None = None
SensorGreendata: np.ndarray | None = None
SensorReddata: np.ndarray | None = None

sensor_map: Dict[str, np.ndarray] = {}


def set_sensor_data(*, R: np.ndarray, G: np.ndarray, B: np.ndarray) -> None:
    """Register per‑channel sensor spectral responses (1‑D arrays of values)."""
    global SensorBluedata, SensorGreendata, SensorReddata, sensor_map
    SensorBluedata = np.asarray(B, dtype=np.float64)
    SensorGreendata = np.asarray(G, dtype=np.float64)
    SensorReddata = np.asarray(R, dtype=np.float64)
    sensor_map = {"R": SensorReddata, "G": SensorGreendata, "B": SensorBluedata}


def _get_dx(CHLdata: np.ndarray) -> float:
    """Utility – estimate grid spacing *dx* from CHL wavelength array."""
    if len(CHLdata) < 2:
        return 1.0
    return float(CHLdata[1] - CHLdata[0])


def Edge(
    color: Literal["R", "G", "B"],
    x: float,
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    dx: float | None = None,
    K_param: float | None = None,
    psf_mode: Literal["linear", "gaussian", "spherical"] = "gaussian",
) -> float:
    """Edge response for Red / Green / Blue channel.

    Any parameter left as *None* falls back to the corresponding module‑level
    constant.
    """
    if not sensor_map:
        raise RuntimeError(
            "Sensor data not initialised – call set_sensor_data() first.")

    sensor_data = sensor_map[color.upper()]

    if CHLdata is None:
        raise ValueError("CHLdata array must be provided.")

    if dx is None:
        dx = _get_dx(CHLdata)

    return compute_edge_jit(
        x=float(x),
        z=float(z),
        F=float(F if F is not None else F_VALUE),
        gamma=float(gamma if gamma is not None else GAMMA_VALUE),
        sensor_data=sensor_data,
        CHLdata=CHLdata,
        dx=float(dx),
        K_param=float(K_param if K_param is not None else K),
        psf_mode=psf_mode,
    )


def Farbsaum(
    x: float,
    z: float,
    F: float | None,
    gamma: float | None,
    CHLdata: np.ndarray,
    psf_mode: Literal["linear", "gaussian", "spherical"] = "gaussian",
) -> int:
    """Binary decision: is there a noticeable colour difference at *x*?"""
    r = Edge("R", x, z, F, gamma, CHLdata, None, None, psf_mode)
    g = Edge("G", x, z, F, gamma, CHLdata, None, None, psf_mode)
    b = Edge("B", x, z, F, gamma, CHLdata, None, None, psf_mode)
    return 1 if (abs(r - b) > TOL or abs(r - g) > TOL or abs(g - b) > TOL) else 0


def Farbsaumbreite(
    z: float,
    F: float | None,
    gamma: float | None,
    CHLdata: np.ndarray,
    psf_mode: Literal["linear", "gaussian", "spherical"] = "gaussian",
) -> int:
    """Aggregated colour‑fringe width across the evaluation window."""
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
        Edge("R", x, z, F, gamma, CHLdata, None, None, psf_mode),
        Edge("G", x, z, F, gamma, CHLdata, None, None, psf_mode),
        Edge("B", x, z, F, gamma, CHLdata, None, None, psf_mode),
    )


# ---------------------------------------------------------------------------
# Convenience plot helpers (optional – import matplotlib only when needed)
# ---------------------------------------------------------------------------

def _compute_psf_for_plot(ratio_value: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Utility to generate PSF curves for visualisation."""
    x_vals = np.linspace(-100.0, 100.0, 400)
    if mode == "linear":
        y_vals = np.array([linear_ESF_jit(v, ratio_value) for v in x_vals])
    elif mode == "gaussian":
        y_vals = np.array([gaussian_ESF_jit(v, ratio_value) for v in x_vals])
    elif mode == "spherical":
        y_vals = np.array([spherical_ESF_jit(v, ratio_value) for v in x_vals])
    else:
        raise ValueError("Unknown PSF mode: " + mode)
    return x_vals, y_vals


def plot_psf_curves(ratio_value: float = 20.0) -> None:  # pragma: no cover
    """Quick plot comparing *linear*, *gaussian* and *spherical* ESFs."""
    import matplotlib.pyplot as plt

    modes = ["linear", "gaussian", "spherical"]
    for mode in modes:
        x, y = _compute_psf_for_plot(ratio_value, mode)
        plt.plot(x, y, label=mode)

    # Ideal (step) edge for reference
    plt.plot([-100.0, 0.0, 0.0, 100.0], [0.0, 0.0, 1.0, 1.0], "--", label="ideal step")

    plt.Legend = plt.legend()
    plt.xlabel("Pixel offset Δx")
    plt.ylabel("Edge Spread Function (normalised)")
    plt.title("PSF comparison (ratio = %.1f)" % ratio_value)
    plt.grid(True, alpha=0.3)
    plt.show()


__all__ = [
    "set_sensor_data",
    "Edge",
    "Farbsaum",
    "Farbsaumbreite",
    "ColorFringe",
    "plot_psf_curves",
]

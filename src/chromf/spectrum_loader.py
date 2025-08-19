"""
Spectral-channel utilities.

This module loads raw CSV data (daylight spectra, sensor responses, defocus
curves) from *data/raw/* and provides helpers to combine them into normalised
per-channel energy distributions S·D.

Folder layout expected::

    project-root/
    ├── data/
    │   └── raw/
    │       ├── daylight_d65.csv
    │       ├── sensor_blue.csv
    │       ├── sensor_green.csv
    │       ├── sensor_red.csv
    │       ├── defocus_chl_zf85.csv
    │       └── ...
    └── src/
        └── chromf/
            └── spectrum_loader.py   ← (this file)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

Array = np.ndarray

# ──────────────────────────── Paths ────────────────────────────────
# project/src/chromf/spectrum_loader.py  → parents[0] = chromf
#                                         parents[1] = src
#                                         parents[2] = project-root
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ───────────────────────────── I/O ─────────────────────────────────

def _csv(name: str) -> Array:
    """Return a CSV file as a float-64 NumPy array.

    Parameters
    ----------
    name
        File stem *without* the ``.csv`` extension, located in ``DATA_DIR``.

    Returns
    -------
    Array
        Two-column array ``[wavelength, value]``.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    """
    path = (DATA_DIR / name).with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, dtype=np.float64).to_numpy()

# put near the existing _load_defocus / _load_sensor helpers
# ───────────────────────── TCA loader ────────────────────────────
def _load_tca(src: str = "curve") -> Array:
    """
    Return the transverse-shift curve [λ_nm, Δx_µm].

    Looks for  data/raw/tca_<src>.csv .
    """
    return _csv(f"tca_{src}")



def _load_defocus(channel: str = "chl_zf85") -> Array:
    """Return defocus curve ``[λ, value]`` for *channel*."""
    return _csv(f"defocus_{channel}")


def _load_daylight(src: str = "d65") -> Array:
    """Return daylight spectrum ``[λ, relative power]`` for *src*."""
    return _csv(f"daylight_{src}")


def _load_sensor(ch: str) -> Array:
    """Return sensor spectral response ``[λ, sensitivity]`` for colour *ch*."""
    return _csv(f"sensor_{ch.lower()}")


# ────────────────────────── Helpers ───────────────────────────────

def _resample(xs: Array, ys: Array, new_x: Array) -> Array:
    """Interpolate *ys(x)* onto *new_x* using cubic splines.

    The curve is scaled to a 0–100 range for easier visual comparison.
    """
    y_new = CubicSpline(xs, ys)(new_x)
    return y_new / y_new.max() * 100.0


def _energy_norm(sensor: Array, daylight: Array) -> float:
    """Return *k* such that ∫ k·S·D dλ = 1."""
    s, d = sensor[:, 1], daylight[:, 1]
    integral = np.trapz(s * d, sensor[:, 0])
    return 1.0 / integral if integral else 0.0


# ─────────────────────── Public API ───────────────────────────────

def channel_products(
    daylight_src: str = "d65",
    channels: Sequence[str] = ("blue", "green", "red"),
    *,
    sensor_peak: float = 1.0,
) -> Dict[str, Array]:
    """Compute the normalised product S·D for several colour channels.

    Parameters
    ----------
    daylight_src
        Filename stem of the daylight spectrum (default ``"d65"``).
    channels
        Ordered sequence of sensor file stems; the first one defines the
        wavelength grid used for all resampling.
    sensor_peak
        Peak amplitude each sensor curve is scaled to *before* energy
        normalisation. Leave at ``1.0`` unless different relative weighting
        is required.

    Returns
    -------
    Dict[str, Array]
        Mapping *channel* → ``[λ, S·D]`` where ∫ S·D dλ ≈ 1.
    """
    if not channels:
        raise ValueError("`channels` must contain at least one entry.")

    # Common wavelength grid from the first sensor file
    base_sensor = _load_sensor(channels[0])
    wl = base_sensor[:, 0]

    # Daylight resampled onto the shared grid
    daylight_rs = np.column_stack(
        (wl, _resample(*_load_daylight(daylight_src).T, wl))
    )

    products: Dict[str, Array] = {}
    for ch in channels:
        sensor_raw = _load_sensor(ch)

        # Rescale sensor curve to a common peak value
        sensor_norm = sensor_raw.copy()
        sensor_norm[:, 1] = (
            sensor_norm[:, 1] / sensor_norm[:, 1].max() * sensor_peak
        )

        # Apply energy normalisation so ∫ S·D dλ = 1
        sensor_norm[:, 1] *= _energy_norm(sensor_norm, daylight_rs)

        # Element-wise product S·D
        sd = np.column_stack((wl, sensor_norm[:, 1] * daylight_rs[:, 1]))
        products[ch] = sd

    return products

__all__ = [
    "_load_defocus",
    "channel_products",
    "_load_tca",          # ← add this
]
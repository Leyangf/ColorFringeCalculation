"""Spectral channel utilities.

This module provides helper routines for loading spectral data from CSV
files and combining sensor response curves with a daylight spectrum to
obtain per‑channel energy distributions S·D normalised to unit energy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

Array = np.ndarray

# Root directory containing all raw *.csv spectral data files
# Moved inside the package so the data ships with the code.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# ─────────────────────────────── I/O ────────────────────────────────

def _csv(name: str) -> Array:
    """Load a CSV file as ``float64`` NumPy array.

    Parameters
    ----------
    name:
        File stem *without* extension, located in ``DATA_DIR``.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.

    Returns
    -------
    Array
        Two‑column array ``[wavelength, value]``.
    """
    path = (DATA_DIR / name).with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=np.float64).to_numpy()


def _load_defocus(channel: str = "chl_zf85") -> Array:
    """Return defocus data ``[λ, value]`` for the given channel."""
    return _csv(f"defocus_{channel}")


def _load_daylight(src: str = "d65") -> Array:
    """Return daylight spectrum ``[λ, relative power]`` for the given source."""
    return _csv(f"daylight_{src}")


def _load_sensor(ch: str) -> Array:
    """Return sensor spectral response ``[λ, sensitivity]`` for colour *ch*."""
    return _csv(f"sensor_{ch.lower()}")


# ──────────────────────────── Helpers ───────────────────────────────

def _resample(xs: Array, ys: Array, new_x: Array) -> Array:
    """Interpolate *ys* defined on *xs* onto *new_x* grid using cubic splines.

    The resulting curve is scaled to the range 0–100 for easier comparison.
    """
    y_new = CubicSpline(xs, ys)(new_x)
    return y_new / y_new.max() * 100.0


def _energy_norm(sensor: Array, daylight: Array) -> float:
    """Return factor *k* such that ∫ k·S·D dλ = 1."""
    s, d = sensor[:, 1], daylight[:, 1]
    integral = np.trapz(s * d, x=sensor[:, 0])
    return 1.0 / integral if integral else 0.0


# ─────────────────────── Public API ────────────────────────────────

def channel_products(
    daylight_src: str = "d65",
    channels: Sequence[str] = ("blue", "green", "red"),
    *,
    sensor_peak: float = 1.0,
) -> Dict[str, Array]:
    """Compute the normalised product S·D for multiple colour channels.

    Parameters
    ----------
    daylight_src:
        File stem of the daylight spectrum to use (default ``"d65"``).
    channels:
        Ordered list of colour channel file stems. The first entry defines
        the wavelength grid that all data are interpolated onto.
    sensor_peak:
        Peak amplitude each sensor curve is scaled to **before** energy
        normalisation. Keep at 1.0 unless a different relative weighting
        is explicitly required.

    Returns
    -------
    Dict[str, Array]
        Mapping *channel* → ``[λ, S·D]`` with ∫ S·D dλ = 1.
    """
    if not channels:
        raise ValueError("`channels` must contain at least one entry.")

    # Common wavelength grid from the first sensor file
    base_sensor = _load_sensor(channels[0])
    wl = base_sensor[:, 0]

    # Daylight spectrum resampled onto this grid
    daylight_rs = np.column_stack(
        (wl, _resample(*_load_daylight(daylight_src).T, wl))
    )

    products: Dict[str, Array] = {}
    for ch in channels:
        sensor_raw = _load_sensor(ch)

        # Rescale sensor curve to a common peak
        sensor_norm = sensor_raw.copy()
        sensor_norm[:, 1] = sensor_norm[:, 1] / sensor_norm[:, 1].max() * sensor_peak

        # Apply energy normalisation so that ∫ S·D dλ = 1
        sensor_norm[:, 1] *= _energy_norm(sensor_norm, daylight_rs)

        # Element‑wise product S·D
        sd = np.column_stack((wl, sensor_norm[:, 1] * daylight_rs[:, 1]))
        products[ch] = sd

    return products

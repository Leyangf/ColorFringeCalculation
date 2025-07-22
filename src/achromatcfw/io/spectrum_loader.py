from pathlib import Path
from typing import Sequence, Dict
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

Array = np.ndarray
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


# ---------- I/O ----------
def _csv(name: str) -> Array:
    path = (DATA_DIR / name).with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).to_numpy()

def _load_defocus(chl: str = "chl_zf85") -> np.ndarray:
    return _csv(f"defocus_{chl}")


def _load_daylight(src: str = "d65") -> Array:
    return _csv(f"daylight_{src}")


def _load_sensor(ch: str) -> Array:
    return _csv(f"sensor_{ch.lower()}")


# ---------- Helpers ----------
def _resample(xs: Array, ys: Array, new_x: Array) -> Array:
    """Resample ``ys`` onto ``new_x`` using cubic splines and normalise to 0-100."""
    y_new = CubicSpline(xs, ys)(new_x)
    return y_new / y_new.max() * 100


def _energy_norm(sensor: Array, daylight: Array) -> float:
    """Return the scaling factor that normalises ``∫S·D`` to 1."""
    s, d = sensor[:, 1], daylight[:, 1]
    integral = np.trapz(s * d, x=sensor[:, 0])
    return 1.0 / integral if integral else 0.0


# ---------- 只返回各通道 S·D ----------
def channel_products(
    daylight_src: str = "d65",
    channels: Sequence[str] = ("blue", "green", "red"),
    *,
    sensor_peak: float = 1.0,
) -> Dict[str, Array]:
    """Return a dictionary ``ch -> [λ, S·D]`` for each colour channel.

    The integral ``∫(S·D) dλ`` is normalised to one and the wavelength grid is
    taken from the first channel.
    """
    # 1) Use a common wavelength grid taken from the first sensor file
    base_sensor = _load_sensor(channels[0])
    wl = base_sensor[:, 0]

    # 2) Resample the daylight spectrum onto this grid
    daylight_rs = np.column_stack((wl, _resample(*_load_daylight(daylight_src).T, wl)))

    # 3) Compute S·D for each channel
    prod_dict: Dict[str, Array] = {}
    for ch in channels:
        s_raw = _load_sensor(ch)
        s_norm = s_raw.copy()
        # Normalise amplitude to ``sensor_peak``
        s_norm[:, 1] = s_norm[:, 1] / s_norm[:, 1].max() * sensor_peak
        # Energy normalisation so that ∫(S·D) = 1
        s_norm[:, 1] *= _energy_norm(s_norm, daylight_rs)

        prod = np.column_stack((wl, s_norm[:, 1] * daylight_rs[:, 1]))
        # prod[:, 0] holds λ, prod[:, 1] the normalised S·D
        prod_dict[ch] = prod

    return prod_dict

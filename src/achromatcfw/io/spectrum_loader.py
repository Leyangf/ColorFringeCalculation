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


def _load_daylight(src: str = "d65") -> Array:
    return _csv(f"daylight_{src}")


def _load_sensor(ch: str) -> Array:
    return _csv(f"sensor_{ch.lower()}")


# ---------- Helpers ----------
def _resample(xs: Array, ys: Array, new_x: Array) -> Array:
    """三次样条重采样并归一到 0‑100。"""
    y_new = CubicSpline(xs, ys)(new_x)
    return y_new / y_new.max() * 100


def _energy_norm(sensor: Array, daylight: Array) -> float:
    """返回把 ∫S·D 归一化到 1 的放大倍数。"""
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
    """
    返回 dict[ch] = ndarray[[λ, S·D]]
    其中 ∫(S·D) dλ == 1  且 λ 取自首通道的波长网格
    """
    # 1) 共用的波长网格
    base_sensor = _load_sensor(channels[0])
    wl = base_sensor[:, 0]

    # 2) 光源重采样
    daylight_rs = np.column_stack((wl, _resample(*_load_daylight(daylight_src).T, wl)))

    # 3) 逐通道计算 S·D
    prod_dict: Dict[str, Array] = {}
    for ch in channels:
        s_raw = _load_sensor(ch)
        s_norm = s_raw.copy()
        # 幅度归一到 sensor_peak
        s_norm[:, 1] = s_norm[:, 1] / s_norm[:, 1].max() * sensor_peak
        # 能量归一（∫S·D = 1）
        s_norm[:, 1] *= _energy_norm(s_norm, daylight_rs)

        prod = np.column_stack((wl, s_norm[:, 1] * daylight_rs[:, 1]))
        prod_dict[ch] = prod  # prod[:,0]=λ, prod[:,1]=归一后的 S·D

    return prod_dict

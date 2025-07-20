# src/achromatcfw/io/spectrum_loader.py
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"

def load_csv(name: str) -> np.ndarray:
    """
    读取指定 *.csv* 并返回 Nx2 的 numpy array.
    `name` 只写文件名或相对 data/raw 的路径.
    """
    path = (DATA_DIR / name).with_suffix(".csv")
    df = pd.read_csv(path)
    return df.to_numpy()

def load_defocus(chip: str = "chl_zf85") -> np.ndarray:
    return load_csv(f"defocus_{chip}")

def load_daylight(source: str = "d65") -> np.ndarray:
    return load_csv(f"daylight_{source}")

def load_sensor(channel: str) -> np.ndarray:
    channel = channel.lower()
    if channel not in {"blue", "green", "red"}:
        raise ValueError("channel must be blue/green/red")
    return load_csv(f"sensor_{channel}")

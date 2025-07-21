from typing import Dict
import numpy as np

# ---------- module‑level “singleton” state ----------------------------------
SensorBluedata: np.ndarray | None = None
SensorGreendata: np.ndarray | None = None
SensorReddata: np.ndarray | None = None

sensor_map: Dict[str, np.ndarray] = {}  # keys: "R", "G", "B"


# ---------- public API ------------------------------------------------------
def set_sensor_data(*, R: np.ndarray, G: np.ndarray, B: np.ndarray) -> None:
    """Register per‑channel sensor spectral responses (1‑D arrays)."""
    global SensorReddata, SensorGreendata, SensorBluedata, sensor_map
    SensorReddata = np.asarray(R, dtype=np.float64)
    SensorGreendata = np.asarray(G, dtype=np.float64)
    SensorBluedata = np.asarray(B, dtype=np.float64)
    sensor_map = {"R": SensorReddata, "G": SensorGreendata, "B": SensorBluedata}


def get_sensor_channel(channel: str) -> np.ndarray:
    """Return the 1‑D response curve for 'R', 'G' or 'B'."""
    if not sensor_map:
        raise RuntimeError(
            "Sensor data not initialised – call set_sensor_data() first."
        )
    try:
        return sensor_map[channel.upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown channel {channel!r}; expected 'R', 'G' or 'B'."
        ) from exc

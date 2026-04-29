"""ChromFringe interactive GUI (PySide6).

Standalone desktop replacement for the ipywidgets cell in
``cfw_geom_demo.ipynb``.  Adds three new affordances:

* Load any Zemax ``.zmx`` file (or fall back to the bundled Nikkor 85mm
  with its measured clear-aperture overrides).
* Pick the sensor model from any ``data/raw/sensor_*_red|green|blue.csv``
  triplet present on disk.
* Pick the daylight illuminant from any ``data/raw/daylight_*.csv``.

Run with::

    uv run python examples/cfw_gui.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Pin Matplotlib's Qt backend to PySide6 before importing pyplot/backends.
os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6 import QtCore, QtWidgets

from optiland import fileio
from optiland.physical_apertures import RadialAperture
from optiland.visualization import OpticViewer

from chromf import (
    channel_products,
    compute_chl_curve,
    compute_rori_spot_curves,
    edge_rgb_response_vec,
    is_fringe_mask,
    precompute_ray_fan,
)
from chromf.spectrum_loader import _load_daylight, _load_sensor


# ── Paths & defaults (mirror cfw_geom_demo.ipynb) ────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
DATA_RAW         = PROJECT_ROOT / "data" / "raw"
DEFAULT_LENS_DIR = PROJECT_ROOT / "data" / "lens"
DEFAULT_LENS     = DEFAULT_LENS_DIR / "NikonAINikkor85mmf2S.zmx"

NIKKOR_CLEAR_SEMI_DIAMETERS = (
    None, 25.062, 24.642, 21.225, 21.225, 19.006, 15.289,
    12.836, 13.730, 15.469, 16.188, 16.188, 17.088, 17.088, 21.190,
)

DEFOCUS_RANGE        = 700      # ±µm sweep on the slider
DEFOCUS_STEP         = 5
CFW_SWEEP_STEP       = 10       # coarser grid for the CFW(z) curve (geom mode is slow)
GAMMA_DEFAULT        = 1.8
EXPOSURE_DEFAULT     = 4.0
COLOR_DIFF_THRESHOLD = 0.15
XRANGE               = 300
X_RES                = 1
IMG_HEIGHT           = 60
NUM_RHO_DEFAULT      = 32
RF_NODES_LIST        = (8, 16, 32, 64)

# Canonical wavelength grid for all lens-dependent baking (CHL, RoRi, SA spot,
# ray-fan TA0/slope).  Matches the 400–700 nm @ 10 nm grid used by every
# bundled sensor CSV, so swapping sensor/illuminant never forces a re-trace —
# only the per-channel g_norm weights and resampled sensor_response need to
# refresh.
BAKE_WL_NM = np.arange(400.0, 701.0, 10.0)

# Display-name maps.  Internal IDs (sensorxxx / dXX) are kept verbatim — they
# drive CSV filenames — and only the dropdown labels are prettified.
SENSOR_DISPLAY = {"sonya900": "Sony A900", "nikond700": "Nikon D700"}
ILLUM_DISPLAY  = {"d65": "D65"}

# Physical sensor pixel pitch (µm/px) — used by the "Pixelize" toggle to
# box-average the raw ESF onto the actual sensor sampling grid.
SENSOR_PIXEL_PITCH_UM = {"sonya900": 5.94, "nikond700": 8.45}


def _sensor_label(model: str) -> str:
    return SENSOR_DISPLAY.get(model, model)


def _illum_label(src: str) -> str:
    return ILLUM_DISPLAY.get(src, src.upper())


# ── Discovery helpers ─────────────────────────────────────────────────────

_SENSOR_RE = re.compile(r"sensor_(.+?)_(?:red|green|blue)\.csv$", re.IGNORECASE)
_ILLUM_RE  = re.compile(r"daylight_(.+?)\.csv$", re.IGNORECASE)


def discover_sensors(data_dir: Path = DATA_RAW) -> list[str]:
    found: dict[str, set[str]] = {}
    for f in data_dir.glob("sensor_*_*.csv"):
        m = _SENSOR_RE.match(f.name)
        if not m:
            continue
        model = m.group(1).lower()
        ch = f.name.lower().rsplit("_", 1)[1].removesuffix(".csv")
        found.setdefault(model, set()).add(ch)
    # Only models with all three channels are usable.
    return sorted(m for m, chs in found.items() if {"red", "green", "blue"} <= chs)


def discover_illuminants(data_dir: Path = DATA_RAW) -> list[str]:
    illums = set()
    for f in data_dir.glob("daylight_*.csv"):
        m = _ILLUM_RE.match(f.name)
        if m:
            illums.add(m.group(1).lower())
    return sorted(illums)


# ── Lens helpers ──────────────────────────────────────────────────────────

def load_default_lens():
    lens = fileio.load_zemax_file(str(DEFAULT_LENS))
    for i, r in enumerate(NIKKOR_CLEAR_SEMI_DIAMETERS):
        if r is not None:
            lens.surfaces.surfaces[i].aperture = RadialAperture(r_max=r)
    return lens


# ── Spectral / ray-fan helpers ────────────────────────────────────────────

def build_sensor_response(sensor_model: str, daylight_src: str) -> tuple[dict, np.ndarray]:
    """Channel products resampled onto the canonical ``BAKE_WL_NM`` grid."""
    prods = channel_products(daylight_src=daylight_src, sensor_model=sensor_model)
    wl_native = prods["red"][:, 0]
    sr: dict[str, np.ndarray] = {}
    for ch_key, ch_name in (("red", "R"), ("green", "G"), ("blue", "B")):
        y = prods[ch_key][:, 1]
        if wl_native.size == BAKE_WL_NM.size and np.allclose(wl_native, BAKE_WL_NM):
            sr[ch_name] = y
        else:
            sr[ch_name] = np.interp(BAKE_WL_NM, wl_native, y)
    return sr, BAKE_WL_NM.copy()


def patch_ray_fan_illumination(ray_fan: dict, sensor_model: str, daylight_src: str) -> dict:
    """Re-weight the ray fan's per-channel ``g_norm`` for the current sensor + illuminant.

    Pulls fresh channel products and resamples onto the fan's baked
    ``wl_nm`` grid.  TA0/slope are lens-only and untouched.
    """
    prods = channel_products(daylight_src=daylight_src, sensor_model=sensor_model)
    full_wl = prods["red"][:, 0]
    target_wl = ray_fan["wl_nm"]
    for ch_name, ch_key in (("R", "red"), ("G", "green"), ("B", "blue")):
        g_k = np.interp(target_wl, full_wl, prods[ch_key][:, 1])
        ray_fan[ch_name]["g_norm"] = g_k / g_k.sum()
    return ray_fan


def tone_map(raw: np.ndarray, exposure: float, gamma: float) -> np.ndarray:
    return (np.tanh(exposure * raw) / np.tanh(exposure)) ** gamma


def box_average(x_fine: np.ndarray, y_fine: np.ndarray, pitch: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Box-average *y_fine(x_fine)* onto a pixel grid of step *pitch*.

    Each output sample integrates the input over a [xc - pitch/2, xc + pitch/2]
    bin centred on multiples of *pitch*.  Mirrors ``_box_average`` from
    ``cfw_fftpsf_demo.ipynb``.
    """
    x_min, x_max = float(x_fine.min()), float(x_fine.max())
    n_left  = int(np.ceil((x_min + pitch / 2.0) / pitch))
    n_right = int(np.floor((x_max - pitch / 2.0) / pitch))
    if n_right < n_left:
        return x_fine.copy(), y_fine.copy()
    x_px = np.arange(n_left, n_right + 1, dtype=float) * pitch
    y_px = np.empty_like(x_px)
    for k, xc in enumerate(x_px):
        m = (x_fine >= xc - pitch / 2.0) & (x_fine < xc + pitch / 2.0)
        y_px[k] = float(y_fine[m].mean()) if m.any() else 0.0
    return x_px, y_px


def geom_esf_rgb(rf: dict, z: float, x_um: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB geometric ESF at one z, sharing the per-wavelength integrand.

    Equivalent to calling :func:`compute_polychromatic_esf_geom` three times
    but ~3× faster: the K-summed pupil integral is computed once per
    wavelength and then dot-producted against each channel's ``g_norm``.
    """
    TA0   = rf["TA0"]                          # (K, N_wl)
    slope = rf["slope"]
    rho   = rf["rho_nodes"][None, :]           # (1, K)
    W_gl  = rf["W_gl"][None, :]                # (1, K)
    N_wl  = TA0.shape[1]

    R = np.abs(TA0 + slope * z)                # (K, N_wl)
    x_col = x_um[:, None]                      # (N, 1)
    integrals = np.empty((x_um.size, N_wl), dtype=np.float64)
    for j in range(N_wl):
        R_row = R[:, j][None, :]               # (1, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(R_row > 1e-4, x_col / R_row, np.sign(x_col + 1e-15))
        ratio = np.clip(ratio, -1.0, 1.0)
        f_contrib = np.arcsin(ratio) / np.pi + 0.5   # (N, K)
        integrals[:, j] = np.sum(f_contrib * rho * W_gl, axis=1)

    out = []
    for ch in ("R", "G", "B"):
        esf = np.clip(integrals @ rf[ch]["g_norm"], 0.0, 1.0)
        out.append(esf)
    return out[0], out[1], out[2]


# ── UI building blocks ────────────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, figsize: tuple[float, float] = (10.0, 5.0),
                 auto_layout: bool = False):
        self.fig = Figure(figsize=figsize)
        if auto_layout:
            # Re-run tight_layout on every draw (including Qt resize events).
            # Without this, the first tight_layout computed at canvas-init time
            # bakes in axes positions for the figsize=... default size, which
            # then look wrong once Qt grows the widget to its real size.
            self.fig.set_layout_engine("tight")
        super().__init__(self.fig)


class FloatSlider(QtWidgets.QWidget):
    """Horizontal QSlider linked to a QDoubleSpinBox."""
    valueChanged = QtCore.Signal(float)

    def __init__(self, lo: float, hi: float, step: float, value: float):
        super().__init__()
        self._lo, self._hi, self._step = float(lo), float(hi), float(step)
        n_steps = max(1, int(round((hi - lo) / step)))

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, n_steps)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(2 if step < 1 else 0)
        self.spin.setMinimumWidth(78)

        self.setValue(value)

        self.slider.valueChanged.connect(self._slider_to_spin)
        self.spin.valueChanged.connect(self._spin_to_slider)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin, 0)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(int(round((v - self._lo) / self._step)))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def _slider_to_spin(self, sv: int):
        v = self._lo + sv * self._step
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _spin_to_slider(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self._lo) / self._step)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


class ChoiceSlider(QtWidgets.QWidget):
    """Discrete-position slider that snaps to a fixed list of values."""
    valueChanged = QtCore.Signal(float)

    def __init__(self, values: list[float], default: float):
        super().__init__()
        self._values = [float(v) for v in values]

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self._values) - 1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)

        self.label = QtWidgets.QLabel()
        self.label.setMinimumWidth(38)
        self.label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.setValue(default)
        self.slider.valueChanged.connect(self._on_slider)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label, 0)

    def value(self) -> float:
        return self._values[self.slider.value()]

    def setValue(self, v: float) -> None:
        idx = min(range(len(self._values)),
                  key=lambda i: abs(self._values[i] - float(v)))
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.label.setText(f"{self._values[idx]:g}")

    def _on_slider(self, idx: int):
        v = self._values[idx]
        self.label.setText(f"{v:g}")
        self.valueChanged.emit(v)


# ── Main window ───────────────────────────────────────────────────────────

class ChromFringeWindow(QtWidgets.QMainWindow):
    PSF_GAUSS, PSF_DISC, PSF_GEOM = 0, 1, 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromFringe — CFW Interactive Explorer")
        self.resize(1320, 760)

        # Cached state (lens + curves + ray fans)
        self._lens = None
        self._lens_for_layout = None
        self._fno: float | None = None
        self._sensor_response: dict | None = None
        self._sensor_wl: np.ndarray | None = None
        self._paraxial_curve: np.ndarray | None = None
        self._rori_curve: np.ndarray | None = None
        self._spot_curve: np.ndarray | None = None
        self._ray_fans: dict[int, dict] = {}

        # CFW(z) sweep cache (recomputed only when non-z params change)
        self._cfw_sig: tuple | None = None
        self._cfw_z: np.ndarray | None = None
        self._cfw_w: np.ndarray | None = None

        self._x_vals = np.arange(-XRANGE, XRANGE + X_RES, X_RES, dtype=float)

        self._build_ui()
        self._populate_dropdowns()
        self._draw_no_lens_placeholder()

    # ── UI layout ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        # ── Left: control panel ─────────────────────────────────────
        ctrl = QtWidgets.QWidget()
        ctrl.setFixedWidth(420)
        cl = QtWidgets.QVBoxLayout(ctrl)
        cl.setSpacing(8)

        # Lens
        gb_lens = QtWidgets.QGroupBox("Lens")
        gv = QtWidgets.QVBoxLayout(gb_lens)
        self.lens_label = QtWidgets.QLabel("(none loaded)")
        self.lens_label.setWordWrap(True)
        gv.addWidget(self.lens_label)
        self.btn_load_zmx = QtWidgets.QPushButton("Load Zemax…")
        gv.addWidget(self.btn_load_zmx)
        self.lens_summary = QtWidgets.QLabel("FNO = —, f = —")
        gv.addWidget(self.lens_summary)
        cl.addWidget(gb_lens)

        # Spectral
        gb_spec = QtWidgets.QGroupBox("Spectral")
        sg = QtWidgets.QFormLayout(gb_spec)
        self.cb_sensor = QtWidgets.QComboBox()
        self.cb_illum  = QtWidgets.QComboBox()
        self.chk_pixelize = QtWidgets.QCheckBox("Pixelize to sensor pitch")
        self.chk_pixelize.setChecked(False)
        self.chk_pixelize.setToolTip(
            "Box-average the raw ESF onto the sensor's pixel pitch before tone "
            "mapping, then re-tone-map.  Simulates what the physical sensor "
            "actually integrates per pixel.\n"
            "Sony A900: 5.94 µm/px,  Nikon D700: 8.45 µm/px."
        )
        sg.addRow("Sensor:",     self.cb_sensor)
        sg.addRow("Illuminant:", self.cb_illum)
        sg.addRow("",            self.chk_pixelize)
        cl.addWidget(gb_spec)

        # PSF / aberration
        gb_psf = QtWidgets.QGroupBox("PSF / Aberration")
        pg = QtWidgets.QFormLayout(gb_psf)
        self.cb_psf = QtWidgets.QComboBox()
        # Match the original notebook's option order: indices used in _refresh_plot.
        self.cb_psf.addItem("Gaussian",                  "gauss")
        self.cb_psf.addItem("Disc (uniform)",            "disc")
        self.cb_psf.addItem("Geometric Fast (ray fan)",  "geom_fast")
        self.cb_psf.setCurrentIndex(self.PSF_DISC)
        self.cb_nrho = QtWidgets.QComboBox()
        for n in RF_NODES_LIST:
            self.cb_nrho.addItem(f"{n} nodes", n)
        self.cb_nrho.setCurrentIndex(RF_NODES_LIST.index(NUM_RHO_DEFAULT))
        self.cb_chl = QtWidgets.QComboBox()
        self.cb_chl.addItem("RoRi",         "rori")
        self.cb_chl.addItem("Paraxial CHL", "paraxial")
        self.chk_sa = QtWidgets.QCheckBox("Include SA")
        self.chk_sa.setChecked(True)
        self.btn_bake_psf = QtWidgets.QPushButton("Bake PSF")
        self.btn_bake_psf.setToolTip(
            "Trace the ray fan with the current ρ-node setting and cache it.\n"
            "Geometric Fast mode requires a baked fan — bakes are NOT triggered\n"
            "automatically by config changes."
        )
        self.lbl_bake_status = QtWidgets.QLabel("(no PSF baked)")
        self.lbl_bake_status.setStyleSheet("color: gray;")
        pg.addRow("PSF model:", self.cb_psf)
        pg.addRow("ρ nodes:",   self.cb_nrho)
        pg.addRow("CHL curve:", self.cb_chl)
        pg.addRow("",           self.chk_sa)
        pg.addRow("",           self.btn_bake_psf)
        pg.addRow("Status:",    self.lbl_bake_status)
        cl.addWidget(gb_psf)

        # Display
        gb_disp = QtWidgets.QGroupBox("Display")
        dg = QtWidgets.QFormLayout(gb_disp)
        self.sl_z = FloatSlider(-DEFOCUS_RANGE, DEFOCUS_RANGE, DEFOCUS_STEP, 0.0)
        self.sl_g = FloatSlider(1.0, 3.0, 0.1, GAMMA_DEFAULT)
        self.sl_e = ChoiceSlider([1, 2, 4, 8, 16], EXPOSURE_DEFAULT)
        dg.addRow("Defocus z (µm):", self.sl_z)
        dg.addRow("Gamma:",          self.sl_g)
        dg.addRow("Exposure:",       self.sl_e)
        cl.addWidget(gb_disp)

        # Lens 2D layout (static — refreshes only on lens load)
        gb_layout = QtWidgets.QGroupBox("Lens Layout (2D)")
        gly = QtWidgets.QVBoxLayout(gb_layout)
        self.canvas_layout = MplCanvas(figsize=(4.2, 1.7), auto_layout=True)
        gly.addWidget(self.canvas_layout)
        cl.addWidget(gb_layout, 1)

        # Tabbed diagnostics: Sensor/illuminant spectra (spectral change) and
        # CHL Paraxial vs RoRi comparison (lens change).
        gb_diag = QtWidgets.QGroupBox("Diagnostics")
        sly = QtWidgets.QVBoxLayout(gb_diag)
        self.tabs_diag = QtWidgets.QTabWidget()

        self.canvas_sensor = MplCanvas(figsize=(4.2, 1.7), auto_layout=True)
        self.tabs_diag.addTab(self.canvas_sensor, "Sensor & Illuminant")

        self.canvas_chl = MplCanvas(figsize=(4.2, 1.7), auto_layout=True)
        self.tabs_diag.addTab(self.canvas_chl, "CHL: Paraxial vs RoRi")

        sly.addWidget(self.tabs_diag)
        cl.addWidget(gb_diag, 1)

        outer.addWidget(ctrl, 0)

        # ── Right: plot canvas + status ────────────────────────────
        right = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas()
        right.addWidget(self.canvas, 1)
        self.status = QtWidgets.QLabel(" ")
        self.status.setStyleSheet("font-family: Consolas, monospace; padding: 4px;")
        right.addWidget(self.status, 0)
        outer.addLayout(right, 1)

        # ── Wiring ──────────────────────────────────────────────────
        self.btn_load_zmx.clicked.connect(self._action_load_zmx)
        self.btn_bake_psf.clicked.connect(self._action_bake_psf)
        self.cb_sensor.currentIndexChanged.connect(self._on_spectral_changed)
        self.cb_illum.currentIndexChanged.connect(self._on_spectral_changed)
        self.chk_pixelize.toggled.connect(self._refresh_plot)
        self.cb_psf.currentIndexChanged.connect(self._update_widget_states)
        self.cb_psf.currentIndexChanged.connect(self._refresh_plot)
        self.cb_nrho.currentIndexChanged.connect(self._refresh_plot)
        self.cb_chl.currentIndexChanged.connect(self._update_widget_states)
        self.cb_chl.currentIndexChanged.connect(self._refresh_plot)
        self.chk_sa.toggled.connect(self._refresh_plot)
        self.sl_z.valueChanged.connect(self._refresh_plot)
        self.sl_g.valueChanged.connect(self._refresh_plot)
        self.sl_e.valueChanged.connect(self._refresh_plot)

        self._update_widget_states()

    def _populate_dropdowns(self):
        sensors = discover_sensors() or ["sonya900"]
        illums  = discover_illuminants() or ["d65"]

        self.cb_sensor.blockSignals(True)
        for s in sensors:
            self.cb_sensor.addItem(_sensor_label(s), userData=s)
        idx = next((i for i, s in enumerate(sensors) if s == "sonya900"), 0)
        self.cb_sensor.setCurrentIndex(idx)
        self.cb_sensor.blockSignals(False)

        self.cb_illum.blockSignals(True)
        for d in illums:
            self.cb_illum.addItem(_illum_label(d), userData=d)
        idx = next((i for i, d in enumerate(illums) if d == "d65"), 0)
        self.cb_illum.setCurrentIndex(idx)
        self.cb_illum.blockSignals(False)

    def _update_widget_states(self):
        is_geom = self.cb_psf.currentIndex() == self.PSF_GEOM
        is_rori = (self.cb_chl.currentData() == "rori")
        self.cb_nrho.setEnabled(is_geom)
        self.cb_chl.setEnabled(not is_geom)
        # SA only contributes to the model when CHL == RoRi (the spot-radius
        # curve is RoRi-derived).  Disable the checkbox elsewhere so the UI
        # reflects what _refresh_plot actually consumes.
        self.chk_sa.setEnabled(not is_geom and is_rori)

    # ── Lens actions ────────────────────────────────────────────────────
    def _action_load_zmx(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Zemax lens", str(DEFAULT_LENS_DIR),
            "Zemax (*.zmx *.zos *.seq);;All files (*)",
        )
        if not path:
            return
        with self._busy(f"Loading {Path(path).name}…"):
            try:
                lens = fileio.load_zemax_file(path)
            except Exception as e:
                self._error("Zemax load failed", e)
                return
            # Apply bundled aperture overrides for known files.  These match
            # the lens's measured clear apertures and tighten ray-trace
            # vignetting — necessary for the CFW physics, but they also
            # truncate most rays in the diagnostic 2-D layout (which then
            # looks like "rays don't focus").  Keep an override-free reload
            # for OpticViewer to draw a clean schematic instead.
            if Path(path).name == "NikonAINikkor85mmf2S.zmx":
                lens_for_layout = fileio.load_zemax_file(path)
                for i, r in enumerate(NIKKOR_CLEAR_SEMI_DIAMETERS):
                    if r is not None:
                        lens.surfaces.surfaces[i].aperture = RadialAperture(r_max=r)
            else:
                lens_for_layout = lens
            self._lens = lens
            self._lens_for_layout = lens_for_layout
            self.lens_label.setText(f"Loaded: {Path(path).name}")
            self._after_lens_changed()

    # ── Recompute pipeline ──────────────────────────────────────────────
    def _on_spectral_changed(self):
        """Cheap path: re-weight only.  No re-trace, no lens-curve recompute."""
        if self._lens is None:
            return
        with self._busy("Re-weighting spectral channels…"):
            self._recompute_spectral()
            sensor = self.cb_sensor.currentData()
            illum  = self.cb_illum.currentData()
            for rf in self._ray_fans.values():
                patch_ray_fan_illumination(rf, sensor, illum)
            # Spectral weights affect CFW(z) values — invalidate the sweep.
            self._cfw_sig = None
        self._redraw_sensor_response()
        self._refresh_plot()

    def _after_lens_changed(self):
        with self._busy("Computing aberration curves…"):
            # Optiland sometimes returns 1-D single-element arrays for FNO/f2
            # depending on aperture type — flatten + take element zero to be
            # robust to either form.
            self._fno = float(np.asarray(self._lens.paraxial.FNO()).ravel()[0])
            f2 = float(np.asarray(self._lens.paraxial.f2()).ravel()[0])
            self.lens_summary.setText(f"FNO = {self._fno:.2f},  f = {f2:.1f} mm")
            self._recompute_spectral()
            self._paraxial_curve = compute_chl_curve(self._lens, wavelengths_nm=BAKE_WL_NM)
            self._rori_curve, self._spot_curve = compute_rori_spot_curves(
                self._lens, wavelengths_nm=BAKE_WL_NM
            )
            self._ray_fans.clear()
        self._update_bake_status()
        self._redraw_lens_layout()
        self._redraw_sensor_response()
        self._redraw_chl_comparison()
        self._refresh_plot()

    # ── Diagnostic panels (left column bottom) ──────────────────────────
    def _redraw_lens_layout(self):
        fig = self.canvas_layout.fig
        fig.clear()
        lens_to_draw = self._lens_for_layout or self._lens
        if lens_to_draw is None:
            self.canvas_layout.draw_idle()
            return
        ax = fig.add_subplot(111)
        try:
            OpticViewer(lens_to_draw).view(
                ax=ax, num_rays=3, fields="all", wavelengths="primary",
                show_legend=False,
            )
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, f"layout unavailable\n({type(e).__name__})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        ax.set_title(ax.get_title() or "", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        self.canvas_layout.draw_idle()

    def _redraw_chl_comparison(self):
        """Plot Paraxial CHL and RoRi CHL curves on a single axes."""
        fig = self.canvas_chl.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if self._paraxial_curve is None or self._rori_curve is None:
            ax.text(0.5, 0.5, "(no lens loaded)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_axis_off()
            self.canvas_chl.draw_idle()
            return
        ax.plot(self._paraxial_curve[:, 0], self._paraxial_curve[:, 1],
                color="C0", lw=1.2, label="Paraxial")
        ax.plot(self._rori_curve[:, 0], self._rori_curve[:, 1],
                color="C1", lw=1.2, label="RoRi")
        ax.axhline(0.0, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel("λ (nm)", fontsize=8)
        ax.set_ylabel("CHL (µm)", fontsize=8)
        ax.set_title("Longitudinal chromatic focal shift", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7, loc="best")
        self.canvas_chl.draw_idle()

    def _redraw_sensor_response(self):
        """Show raw sensor R/G/B QE curves and the illuminant on one axes."""
        fig = self.canvas_sensor.fig
        fig.clear()
        sensor_id = self.cb_sensor.currentData()
        illum_id  = self.cb_illum.currentData()
        ax = fig.add_subplot(111)
        try:
            s_r = _load_sensor("red",   model=sensor_id)
            s_g = _load_sensor("green", model=sensor_id)
            s_b = _load_sensor("blue",  model=sensor_id)
            d   = _load_daylight(illum_id)
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, f"spectra unavailable\n({type(e).__name__})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            self.canvas_sensor.draw_idle()
            return

        ax.plot(s_r[:, 0], s_r[:, 1], color="r", lw=1.2, label="R")
        ax.plot(s_g[:, 0], s_g[:, 1], color="g", lw=1.2, label="G")
        ax.plot(s_b[:, 0], s_b[:, 1], color="b", lw=1.2, label="B")
        ax.plot(d[:, 0],   d[:, 1],   color="k", lw=1.0, ls="--",
                alpha=0.7, label=_illum_label(illum_id))
        ax.set_xlabel("λ (nm)", fontsize=8)
        ax.set_ylabel("relative", fontsize=8)
        ax.set_title(f"{_sensor_label(sensor_id)}  |  {_illum_label(illum_id)}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        self.canvas_sensor.draw_idle()

    def _recompute_spectral(self):
        sensor = self.cb_sensor.currentData()
        illum  = self.cb_illum.currentData()
        self._sensor_response, self._sensor_wl = build_sensor_response(sensor, illum)

    def _get_ray_fan(self, num_rho: int) -> dict | None:
        """Return the cached ray fan for *num_rho* — no auto-bake on miss."""
        return self._ray_fans.get(num_rho)

    def _action_bake_psf(self):
        """Explicitly trace the ray fan for the current ρ-node setting."""
        if self._lens is None:
            self._error("No lens loaded", RuntimeError("Load a Zemax file first."))
            return
        num_rho = int(self.cb_nrho.currentData())
        sensor  = self.cb_sensor.currentData()
        illum   = self.cb_illum.currentData()
        with self._busy(f"Baking PSF (ρ = {num_rho} nodes)…"):
            rf = precompute_ray_fan(self._lens, num_rho=num_rho, sensor_model=sensor)
            rf = patch_ray_fan_illumination(rf, sensor, illum)
            self._ray_fans[num_rho] = rf
            self._cfw_sig = None  # invalidate sweep — new bake → recompute on demand
        self._update_bake_status()
        self._refresh_plot()

    def _update_bake_status(self):
        if not self._ray_fans:
            self.lbl_bake_status.setText("(no PSF baked)")
            self.lbl_bake_status.setStyleSheet("color: gray;")
            return
        nodes = sorted(self._ray_fans.keys())
        self.lbl_bake_status.setText(
            "baked: " + ", ".join(f"ρ={n}" for n in nodes)
        )
        self.lbl_bake_status.setStyleSheet("color: #2a7;")

    def _draw_unbaked_message(self, num_rho: int):
        """Geometric mode placeholder when no ray fan is cached for ρ=num_rho."""
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            f"Geometric mode requires a baked ray fan.\n"
            f"Click 'Bake PSF' to trace ρ = {num_rho} nodes.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="gray",
        )
        ax.set_axis_off()
        self.canvas.draw_idle()
        self.status.setText(f"PSF not baked for ρ = {num_rho}.  Click 'Bake PSF'.")

    def _draw_no_lens_placeholder(self):
        """Initial state — invite the user to load a Zemax file."""
        for canvas, msg in (
            (self.canvas,         "Load a Zemax (.zmx) lens to begin."),
            (self.canvas_layout,  "(no lens loaded)"),
            (self.canvas_sensor,  "(no lens loaded)"),
            (self.canvas_chl,     "(no lens loaded)"),
        ):
            fig = canvas.fig
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_axis_off()
            canvas.draw_idle()
        self.status.setText("Load a Zemax lens to begin.")

    # ── CFW(z) sweep ────────────────────────────────────────────────────
    def _curve_signature(self) -> tuple:
        """Tuple of every non-z parameter that affects the CFW(z) curve."""
        psf_idx = self.cb_psf.currentIndex()
        return (
            id(self._lens),
            self.cb_sensor.currentData(),
            self.cb_illum.currentData(),
            psf_idx,
            int(self.cb_nrho.currentData()) if psf_idx == self.PSF_GEOM else 0,
            self.cb_chl.currentData() or "rori",
            bool(self.chk_sa.isChecked()),
            round(self.sl_g.value(), 4),
            round(self.sl_e.value(), 4),
            bool(self.chk_pixelize.isChecked()),
        )

    def _ensure_cfw_curve(self):
        """Recompute the CFW(z) sweep iff a non-z parameter changed."""
        sig = self._curve_signature()
        if sig == self._cfw_sig and self._cfw_z is not None:
            return

        psf_idx  = self.cb_psf.currentIndex()
        gamma    = self.sl_g.value()
        exposure = self.sl_e.value()
        chl_mode = self.cb_chl.currentData()
        use_sa   = self.chk_sa.isChecked()
        num_rho  = int(self.cb_nrho.currentData())
        pixelize = self.chk_pixelize.isChecked()

        if pixelize:
            sensor_id = self.cb_sensor.currentData() or "sonya900"
            pitch = SENSOR_PIXEL_PITCH_UM.get(sensor_id, float(X_RES))
        else:
            pitch = float(X_RES)

        z_arr = np.arange(-DEFOCUS_RANGE, DEFOCUS_RANGE + CFW_SWEEP_STEP,
                          CFW_SWEEP_STEP, dtype=float)
        cfw_arr = np.empty(len(z_arr), dtype=np.int32)

        def _maybe_pixelize(r, g, b):
            if not pixelize:
                return r, g, b
            _, r_px = box_average(self._x_vals, r, pitch)
            _, g_px = box_average(self._x_vals, g, pitch)
            _, b_px = box_average(self._x_vals, b, pitch)
            return r_px, g_px, b_px

        with self._busy(f"Computing CFW vs z ({len(z_arr)} pts)…"):
            if psf_idx == self.PSF_GEOM:
                rf = self._get_ray_fan(num_rho)
                if rf is None:
                    self._cfw_z = np.empty(0, dtype=float)
                    self._cfw_w = np.empty(0, dtype=np.int32)
                    self._cfw_sig = sig
                    return
                for i, z in enumerate(z_arr):
                    r_raw, g_raw, b_raw = geom_esf_rgb(rf, float(z), self._x_vals)
                    r_raw, g_raw, b_raw = _maybe_pixelize(r_raw, g_raw, b_raw)
                    r = tone_map(r_raw, exposure, gamma)
                    g = tone_map(g_raw, exposure, gamma)
                    b = tone_map(b_raw, exposure, gamma)
                    mask = is_fringe_mask(r, g, b, diff_threshold=COLOR_DIFF_THRESHOLD)
                    idx = np.flatnonzero(mask)
                    n = int(idx[-1] - idx[0] + 1) if idx.size else 0
                    cfw_arr[i] = int(round(n * pitch))
            else:
                psf_mode = "gauss" if psf_idx == self.PSF_GAUSS else "disc"
                chl = self._rori_curve[:, 1] if chl_mode == "rori" else self._paraxial_curve[:, 1]
                sa  = self._spot_curve[:, 1] if (use_sa and chl_mode == "rori") else None
                # Tone-map applied AFTER optional pixelization for physical
                # correctness, so request raw ESF here (slope→0, gamma=1).
                exp_call = 1e-6 if pixelize else exposure
                gamma_call = 1.0 if pixelize else gamma
                for i, z in enumerate(z_arr):
                    r_raw, g_raw, b_raw = edge_rgb_response_vec(
                        self._x_vals, float(z),
                        exposure_slope=exp_call, gamma=gamma_call,
                        chl_curve_um=chl, rho_sa_um=sa,
                        f_number=self._fno, psf_mode=psf_mode,
                        sensor_response=self._sensor_response,
                    )
                    if pixelize:
                        r_raw, g_raw, b_raw = _maybe_pixelize(r_raw, g_raw, b_raw)
                        r = tone_map(r_raw, exposure, gamma)
                        g = tone_map(g_raw, exposure, gamma)
                        b = tone_map(b_raw, exposure, gamma)
                    else:
                        r, g, b = r_raw, g_raw, b_raw
                    mask = is_fringe_mask(r, g, b, diff_threshold=COLOR_DIFF_THRESHOLD)
                    idx = np.flatnonzero(mask)
                    n = int(idx[-1] - idx[0] + 1) if idx.size else 0
                    cfw_arr[i] = int(round(n * pitch))

        self._cfw_z = z_arr
        self._cfw_w = cfw_arr
        self._cfw_sig = sig

    # ── Plot ────────────────────────────────────────────────────────────
    def _refresh_plot(self):
        if self._lens is None or self._sensor_response is None:
            return

        z        = self.sl_z.value()
        gamma    = self.sl_g.value()
        exposure = self.sl_e.value()
        psf_idx  = self.cb_psf.currentIndex()
        chl_mode = self.cb_chl.currentData()
        use_sa   = self.chk_sa.isChecked()
        num_rho  = int(self.cb_nrho.currentData())
        pixelize = self.chk_pixelize.isChecked()

        # 1) Raw RGB ESF on the fine grid
        if psf_idx == self.PSF_GEOM:
            rf = self._get_ray_fan(num_rho)
            if rf is None:
                self._draw_unbaked_message(num_rho)
                return
            r_raw, g_raw, b_raw = geom_esf_rgb(rf, z, self._x_vals)
            psf_label = f"Geometric Fast ({num_rho} ρ nodes)"
            chl_label = "ray trace"
            sa_suffix = ""
        else:
            psf_mode = "gauss" if psf_idx == self.PSF_GAUSS else "disc"
            chl = self._rori_curve[:, 1] if chl_mode == "rori" else self._paraxial_curve[:, 1]
            sa  = self._spot_curve[:, 1] if (use_sa and chl_mode == "rori") else None
            # exposure_slope→0, gamma=1 gives identity tone curve → raw ESF.
            r_raw, g_raw, b_raw = edge_rgb_response_vec(
                self._x_vals, z,
                exposure_slope=1e-6, gamma=1.0,
                chl_curve_um=chl, rho_sa_um=sa,
                f_number=self._fno, psf_mode=psf_mode,
                sensor_response=self._sensor_response,
            )
            psf_label = "Gaussian" if psf_mode == "gauss" else "Disc (uniform)"
            chl_label = "RoRi" if chl_mode == "rori" else "Paraxial CHL"
            sa_suffix = " + SA" if (use_sa and chl_mode == "rori") else ""

        # 2) Optional pixelization (physical sensor integrates over each pixel)
        if pixelize:
            sensor_id = self.cb_sensor.currentData() or "sonya900"
            pitch = SENSOR_PIXEL_PITCH_UM.get(sensor_id, float(X_RES))
            x_disp, r_raw = box_average(self._x_vals, r_raw, pitch)
            _,      g_raw = box_average(self._x_vals, g_raw, pitch)
            _,      b_raw = box_average(self._x_vals, b_raw, pitch)
            sample_step_um = pitch
            pix_suffix = f"  |  pitch = {pitch:.2f} µm/px"
        else:
            x_disp = self._x_vals
            sample_step_um = float(X_RES)
            pix_suffix = ""

        # 3) Tone-map for display + CFW classification
        edge_r = tone_map(r_raw, exposure, gamma)
        edge_g = tone_map(g_raw, exposure, gamma)
        edge_b = tone_map(b_raw, exposure, gamma)

        boundaries = is_fringe_mask(edge_r, edge_g, edge_b, diff_threshold=COLOR_DIFF_THRESHOLD)
        idx = np.flatnonzero(boundaries)
        cfw_samples = int(idx[-1] - idx[0] + 1) if idx.size else 0
        cfw = int(round(cfw_samples * sample_step_um))

        # CFW(z) sweep — only recomputes when non-z parameters changed.
        self._ensure_cfw_curve()

        fig = self.canvas.fig
        fig.clear()

        # Layout: narrow left column for the two edge views, wide right
        # column for the CFW(z) sweep (anchored top-right, spans both rows).
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.7])

        # Top-left: edge responses
        ax1 = fig.add_subplot(gs[0, 0])
        line_kw = {"drawstyle": "steps-mid"} if pixelize else {}
        ax1.plot(x_disp, edge_r, color="r", label="R", **line_kw)
        ax1.plot(x_disp, edge_g, color="g", label="G", **line_kw)
        ax1.plot(x_disp, edge_b, color="b", label="B", **line_kw)
        ax1.axhline(COLOR_DIFF_THRESHOLD, color="k", ls=":", lw=1, alpha=0.5,
                    label=f"thr={COLOR_DIFF_THRESHOLD:.2f}")
        if idx.size:
            ax1.axvline(float(x_disp[idx[0]]),  color="k", ls="--", lw=1, label="boundary")
            ax1.axvline(float(x_disp[idx[-1]]), color="k", ls="--", lw=1)
        ax1.set(xlabel="x (µm)", ylabel="Normalised response", ylim=(0, 1),
                title=f"Edge responses — {psf_label} | {chl_label}{sa_suffix}{pix_suffix}")
        ax1.legend(fontsize=8)
        ax1.grid(True)

        # Bottom-left: pseudo-density fringe map
        ax2 = fig.add_subplot(gs[1, 0])
        img_row = np.stack([edge_r, edge_g, edge_b], axis=1)
        img = np.repeat(np.clip(img_row, 0, 1)[:, None, :], IMG_HEIGHT, axis=1)
        # Half-pitch padding so the leftmost and rightmost pixels render fully.
        half = sample_step_um / 2.0
        ax2.imshow(
            img.swapaxes(0, 1),
            extent=(float(x_disp.min()) - half, float(x_disp.max()) + half,
                    0.0, float(IMG_HEIGHT)),
            aspect="auto", origin="lower",
            interpolation="nearest" if pixelize else "antialiased",
        )
        if idx.size:
            ax2.axvline(float(x_disp[idx[0]]),  color="w", ls="--", lw=1, alpha=0.85)
            ax2.axvline(float(x_disp[idx[-1]]), color="w", ls="--", lw=1, alpha=0.85)
        ax2.set(xlabel="x (µm)", title=f"Pseudo-density (CFW ≈ {cfw} µm)",
                xlim=(-300, 300), yticks=[])

        # Top-right: CFW vs defocus (single cell — bottom-right left empty)
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.plot(self._cfw_z, self._cfw_w, lw=1.3, color="C0")
        ax3.axvline(z, color="k", ls="--", lw=1, alpha=0.5)
        ax3.scatter([z], [cfw], color="red", zorder=5, s=32,
                    label=f"z={z:+.0f} µm  CFW={cfw} µm")
        peak_idx = int(np.argmax(self._cfw_w))
        ax3.scatter([self._cfw_z[peak_idx]], [self._cfw_w[peak_idx]],
                    facecolors="none", edgecolors="k", s=46, lw=1,
                    label=f"peak: z={self._cfw_z[peak_idx]:+.0f} µm  CFW={int(self._cfw_w[peak_idx])} µm")
        ax3.set(xlabel="Defocus z (µm)", ylabel="CFW (µm)",
                title="CFW vs defocus")
        ax3.legend(fontsize=7, loc="lower right")
        ax3.grid(True, alpha=0.4)

        fig.tight_layout()
        self.canvas.draw_idle()
        self.status.setText(f"CFW ≈ {cfw} µm   |   {psf_label} | {chl_label}{sa_suffix}")

    # ── Status helpers ──────────────────────────────────────────────────
    def _busy(self, msg: str) -> "_BusyContext":
        return _BusyContext(self, msg)

    def _error(self, title: str, exc: Exception):
        QtWidgets.QMessageBox.critical(self, title, f"{type(exc).__name__}: {exc}")


class _BusyContext:
    """Context manager that flips the cursor and status while a task runs."""
    def __init__(self, win: ChromFringeWindow, msg: str):
        self.win = win
        self.msg = msg
        self._prev_status = ""

    def __enter__(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._prev_status = self.win.status.text()
        self.win.status.setText(self.msg)
        QtWidgets.QApplication.processEvents()
        return self

    def __exit__(self, *exc_info):
        QtWidgets.QApplication.restoreOverrideCursor()
        if self.win.status.text() == self.msg:
            self.win.status.setText(self._prev_status)
        return False


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ChromFringeWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

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

from chromf import (
    channel_products,
    compute_chl_curve,
    compute_rori_spot_curves,
    edge_rgb_response_vec,
    is_fringe_mask,
    precompute_ray_fan,
)


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
XRANGE               = 400
X_RES                = 1
IMG_HEIGHT           = 60
NUM_RHO_DEFAULT      = 32
RF_NODES_LIST        = (8, 16, 32, 64)


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
    prods = channel_products(daylight_src=daylight_src, sensor_model=sensor_model)
    sr = {
        "R": prods["red"][:, 1],
        "G": prods["green"][:, 1],
        "B": prods["blue"][:, 1],
    }
    return sr, prods["blue"][:, 0]


def patch_ray_fan_illumination(ray_fan: dict, sensor_model: str, daylight_src: str) -> dict:
    """Re-weight the ray fan's per-channel ``g_norm`` for a non-D65 illuminant.

    ``precompute_ray_fan`` always bakes weights against D65; this overwrites
    them in-place using the chosen illuminant.  TA0/slope are unaffected.
    """
    if daylight_src == "d65":
        return ray_fan
    prods = channel_products(daylight_src=daylight_src, sensor_model=sensor_model)
    full_wl = prods["red"][:, 0]
    idx = np.searchsorted(full_wl, ray_fan["wl_nm"])
    for ch_name, ch_key in (("R", "red"), ("G", "green"), ("B", "blue")):
        g_k = prods[ch_key][:, 1][idx]
        ray_fan[ch_name]["g_norm"] = g_k / g_k.sum()
    return ray_fan


def tone_map(raw: np.ndarray, exposure: float, gamma: float) -> np.ndarray:
    return (np.tanh(exposure * raw) / np.tanh(exposure)) ** gamma


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
    def __init__(self):
        self.fig = Figure(figsize=(10, 5))
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


# ── Main window ───────────────────────────────────────────────────────────

class ChromFringeWindow(QtWidgets.QMainWindow):
    PSF_GAUSS, PSF_DISC, PSF_GEOM = 0, 1, 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromFringe — CFW Interactive Explorer")
        self.resize(1320, 760)

        # Cached state (lens + curves + ray fans)
        self._lens = None
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
        QtCore.QTimer.singleShot(0, self._action_default_lens)

    # ── UI layout ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        # ── Left: control panel ─────────────────────────────────────
        ctrl = QtWidgets.QWidget()
        ctrl.setFixedWidth(380)
        cl = QtWidgets.QVBoxLayout(ctrl)
        cl.setSpacing(8)

        # Lens
        gb_lens = QtWidgets.QGroupBox("Lens")
        gv = QtWidgets.QVBoxLayout(gb_lens)
        self.lens_label = QtWidgets.QLabel("(none loaded)")
        self.lens_label.setWordWrap(True)
        gv.addWidget(self.lens_label)
        bh = QtWidgets.QHBoxLayout()
        self.btn_default_lens = QtWidgets.QPushButton("Load Nikkor 85mm")
        self.btn_load_zmx     = QtWidgets.QPushButton("Load Zemax…")
        bh.addWidget(self.btn_default_lens)
        bh.addWidget(self.btn_load_zmx)
        gv.addLayout(bh)
        self.lens_summary = QtWidgets.QLabel("FNO = —, f = —")
        gv.addWidget(self.lens_summary)
        cl.addWidget(gb_lens)

        # Spectral
        gb_spec = QtWidgets.QGroupBox("Spectral")
        sg = QtWidgets.QFormLayout(gb_spec)
        self.cb_sensor = QtWidgets.QComboBox()
        self.cb_illum  = QtWidgets.QComboBox()
        sg.addRow("Sensor:",     self.cb_sensor)
        sg.addRow("Illuminant:", self.cb_illum)
        cl.addWidget(gb_spec)

        # PSF / aberration
        gb_psf = QtWidgets.QGroupBox("PSF / Aberration")
        pg = QtWidgets.QFormLayout(gb_psf)
        self.cb_psf = QtWidgets.QComboBox()
        # Match the original notebook's option order: indices used in _refresh_plot.
        self.cb_psf.addItem("Gaussian",                  "gauss")
        self.cb_psf.addItem("Disc (uniform)",            "disc")
        self.cb_psf.addItem("Geometric Fast (ray fan)",  "geom_fast")
        self.cb_nrho = QtWidgets.QComboBox()
        for n in RF_NODES_LIST:
            self.cb_nrho.addItem(f"{n} nodes", n)
        self.cb_nrho.setCurrentIndex(RF_NODES_LIST.index(NUM_RHO_DEFAULT))
        self.cb_chl = QtWidgets.QComboBox()
        self.cb_chl.addItem("RoRi",         "rori")
        self.cb_chl.addItem("Paraxial CHL", "paraxial")
        self.chk_sa = QtWidgets.QCheckBox("Include SA")
        self.chk_sa.setChecked(True)
        pg.addRow("PSF model:", self.cb_psf)
        pg.addRow("ρ nodes:",   self.cb_nrho)
        pg.addRow("CHL curve:", self.cb_chl)
        pg.addRow("",           self.chk_sa)
        cl.addWidget(gb_psf)

        # Display
        gb_disp = QtWidgets.QGroupBox("Display")
        dg = QtWidgets.QFormLayout(gb_disp)
        self.sl_z = FloatSlider(-DEFOCUS_RANGE, DEFOCUS_RANGE, DEFOCUS_STEP, 0.0)
        self.sl_g = FloatSlider(1.0, 3.0, 0.1, GAMMA_DEFAULT)
        self.sl_e = FloatSlider(1.0, 8.0, 1.0, EXPOSURE_DEFAULT)
        dg.addRow("Defocus z (µm):", self.sl_z)
        dg.addRow("Gamma:",          self.sl_g)
        dg.addRow("Exposure:",       self.sl_e)
        cl.addWidget(gb_disp)

        cl.addStretch(1)
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
        self.btn_default_lens.clicked.connect(self._action_default_lens)
        self.btn_load_zmx.clicked.connect(self._action_load_zmx)
        self.cb_sensor.currentIndexChanged.connect(self._on_spectral_changed)
        self.cb_illum.currentIndexChanged.connect(self._on_spectral_changed)
        self.cb_psf.currentIndexChanged.connect(self._update_widget_states)
        self.cb_psf.currentIndexChanged.connect(self._refresh_plot)
        self.cb_nrho.currentIndexChanged.connect(self._refresh_plot)
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
            self.cb_sensor.addItem(s)
        if "sonya900" in sensors:
            self.cb_sensor.setCurrentText("sonya900")
        self.cb_sensor.blockSignals(False)

        self.cb_illum.blockSignals(True)
        for d in illums:
            self.cb_illum.addItem(d)
        if "d65" in illums:
            self.cb_illum.setCurrentText("d65")
        self.cb_illum.blockSignals(False)

    def _update_widget_states(self):
        is_geom = self.cb_psf.currentIndex() == self.PSF_GEOM
        self.cb_nrho.setEnabled(is_geom)
        self.cb_chl.setEnabled(not is_geom)
        self.chk_sa.setEnabled(not is_geom)

    # ── Lens actions ────────────────────────────────────────────────────
    def _action_default_lens(self):
        with self._busy("Loading default Nikkor 85mm…"):
            try:
                self._lens = load_default_lens()
            except Exception as e:
                self._error("Default lens load failed", e)
                return
            self.lens_label.setText("Default: NikonAINikkor85mmf2S.zmx (with bundled apertures)")
            self._after_lens_changed()

    def _action_load_zmx(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Zemax lens", str(DEFAULT_LENS_DIR),
            "Zemax (*.zmx *.zos *.seq);;All files (*)",
        )
        if not path:
            return
        with self._busy(f"Loading {Path(path).name}…"):
            try:
                self._lens = fileio.load_zemax_file(path)
            except Exception as e:
                self._error("Zemax load failed", e)
                return
            self.lens_label.setText(f"Loaded: {Path(path).name}")
            self._after_lens_changed()

    # ── Recompute pipeline ──────────────────────────────────────────────
    def _on_spectral_changed(self):
        if self._lens is None:
            return
        with self._busy("Recomputing spectral data…"):
            self._recompute_spectral()
            # Re-derive lens curves if the wavelength grid changed
            self._paraxial_curve = compute_chl_curve(self._lens, wavelengths_nm=self._sensor_wl)
            self._rori_curve, self._spot_curve = compute_rori_spot_curves(
                self._lens, wavelengths_nm=self._sensor_wl
            )
            self._ray_fans.clear()
            self._refresh_plot()

    def _after_lens_changed(self):
        with self._busy("Computing aberration curves…"):
            self._fno = float(self._lens.paraxial.FNO())
            f2 = float(self._lens.paraxial.f2())
            self.lens_summary.setText(f"FNO = {self._fno:.2f},  f = {f2:.1f} mm")
            self._recompute_spectral()
            self._paraxial_curve = compute_chl_curve(self._lens, wavelengths_nm=self._sensor_wl)
            self._rori_curve, self._spot_curve = compute_rori_spot_curves(
                self._lens, wavelengths_nm=self._sensor_wl
            )
            self._ray_fans.clear()
        self._refresh_plot()

    def _recompute_spectral(self):
        sensor = self.cb_sensor.currentText()
        illum  = self.cb_illum.currentText()
        self._sensor_response, self._sensor_wl = build_sensor_response(sensor, illum)

    def _ensure_ray_fan(self, num_rho: int) -> dict:
        if num_rho not in self._ray_fans:
            sensor = self.cb_sensor.currentText()
            illum  = self.cb_illum.currentText()
            with self._busy(f"Tracing ray fan ({num_rho} ρ nodes)…"):
                rf = precompute_ray_fan(self._lens, num_rho=num_rho, sensor_model=sensor)
                rf = patch_ray_fan_illumination(rf, sensor, illum)
                self._ray_fans[num_rho] = rf
        return self._ray_fans[num_rho]

    # ── CFW(z) sweep ────────────────────────────────────────────────────
    def _curve_signature(self) -> tuple:
        """Tuple of every non-z parameter that affects the CFW(z) curve."""
        psf_idx = self.cb_psf.currentIndex()
        return (
            id(self._lens),
            self.cb_sensor.currentText(),
            self.cb_illum.currentText(),
            psf_idx,
            int(self.cb_nrho.currentData()) if psf_idx == self.PSF_GEOM else 0,
            self.cb_chl.currentData() or "rori",
            bool(self.chk_sa.isChecked()),
            round(self.sl_g.value(), 4),
            round(self.sl_e.value(), 4),
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

        z_arr = np.arange(-DEFOCUS_RANGE, DEFOCUS_RANGE + CFW_SWEEP_STEP,
                          CFW_SWEEP_STEP, dtype=float)
        cfw_arr = np.empty(len(z_arr), dtype=np.int32)

        with self._busy(f"Computing CFW vs z ({len(z_arr)} pts)…"):
            if psf_idx == self.PSF_GEOM:
                rf = self._ensure_ray_fan(num_rho)
                for i, z in enumerate(z_arr):
                    r_raw, g_raw, b_raw = geom_esf_rgb(rf, float(z), self._x_vals)
                    r = tone_map(r_raw, exposure, gamma)
                    g = tone_map(g_raw, exposure, gamma)
                    b = tone_map(b_raw, exposure, gamma)
                    mask = is_fringe_mask(r, g, b, diff_threshold=COLOR_DIFF_THRESHOLD)
                    idx = np.flatnonzero(mask)
                    cfw_arr[i] = int(idx[-1] - idx[0] + 1) if idx.size else 0
            else:
                psf_mode = "gauss" if psf_idx == self.PSF_GAUSS else "disc"
                chl = self._rori_curve[:, 1] if chl_mode == "rori" else self._paraxial_curve[:, 1]
                sa  = self._spot_curve[:, 1] if (use_sa and chl_mode == "rori") else None
                for i, z in enumerate(z_arr):
                    r, g, b = edge_rgb_response_vec(
                        self._x_vals, float(z),
                        exposure_slope=exposure, gamma=gamma,
                        chl_curve_um=chl, rho_sa_um=sa,
                        f_number=self._fno, psf_mode=psf_mode,
                        sensor_response=self._sensor_response,
                    )
                    mask = is_fringe_mask(r, g, b, diff_threshold=COLOR_DIFF_THRESHOLD)
                    idx = np.flatnonzero(mask)
                    cfw_arr[i] = int(idx[-1] - idx[0] + 1) if idx.size else 0

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

        if psf_idx == self.PSF_GEOM:
            rf = self._ensure_ray_fan(num_rho)
            r_raw, g_raw, b_raw = geom_esf_rgb(rf, z, self._x_vals)
            edge_r = tone_map(r_raw, exposure, gamma)
            edge_g = tone_map(g_raw, exposure, gamma)
            edge_b = tone_map(b_raw, exposure, gamma)
            psf_label = f"Geometric Fast ({num_rho} ρ nodes)"
            chl_label = "ray trace"
            sa_suffix = ""
        else:
            psf_mode = "gauss" if psf_idx == self.PSF_GAUSS else "disc"
            chl = self._rori_curve[:, 1] if chl_mode == "rori" else self._paraxial_curve[:, 1]
            sa  = self._spot_curve[:, 1] if (use_sa and chl_mode == "rori") else None
            edge_r, edge_g, edge_b = edge_rgb_response_vec(
                self._x_vals, z,
                exposure_slope=exposure, gamma=gamma,
                chl_curve_um=chl, rho_sa_um=sa,
                f_number=self._fno, psf_mode=psf_mode,
                sensor_response=self._sensor_response,
            )
            psf_label = "Gaussian" if psf_mode == "gauss" else "Disc (uniform)"
            chl_label = "RoRi" if chl_mode == "rori" else "Paraxial CHL"
            sa_suffix = " + SA" if (use_sa and chl_mode == "rori") else ""

        boundaries = is_fringe_mask(edge_r, edge_g, edge_b, diff_threshold=COLOR_DIFF_THRESHOLD)
        idx = np.flatnonzero(boundaries)
        cfw = int(idx[-1] - idx[0] + 1) if idx.size else 0

        # CFW(z) sweep — only recomputes when non-z parameters changed.
        self._ensure_cfw_curve()

        fig = self.canvas.fig
        fig.clear()

        # Top-left: edge responses
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(self._x_vals, edge_r, color="r", label="R")
        ax1.plot(self._x_vals, edge_g, color="g", label="G")
        ax1.plot(self._x_vals, edge_b, color="b", label="B")
        ax1.axhline(COLOR_DIFF_THRESHOLD, color="k", ls=":", lw=1, alpha=0.5,
                    label=f"thr={COLOR_DIFF_THRESHOLD:.2f}")
        if idx.size:
            ax1.axvline(float(self._x_vals[idx[0]]),  color="k", ls="--", lw=1, label="boundary")
            ax1.axvline(float(self._x_vals[idx[-1]]), color="k", ls="--", lw=1)
        ax1.set(xlabel="x (µm)", ylabel="Normalised response", ylim=(0, 1),
                title=f"Edge responses — {psf_label} | {chl_label}{sa_suffix}")
        ax1.legend(fontsize=8)
        ax1.grid(True)

        # Top-right: pseudo-density fringe map
        ax2 = fig.add_subplot(2, 2, 2)
        img_row = np.stack([edge_r, edge_g, edge_b], axis=1)
        img = np.repeat(np.clip(img_row, 0, 1)[:, None, :], IMG_HEIGHT, axis=1)
        ax2.imshow(
            img.swapaxes(0, 1),
            extent=(float(self._x_vals.min()), float(self._x_vals.max()),
                    0.0, float(IMG_HEIGHT)),
            aspect="auto", origin="lower",
        )
        if idx.size:
            ax2.axvline(float(self._x_vals[idx[0]]),  color="w", ls="--", lw=1, alpha=0.85)
            ax2.axvline(float(self._x_vals[idx[-1]]), color="w", ls="--", lw=1, alpha=0.85)
        ax2.set(xlabel="x (µm)", title=f"Pseudo-density (CFW ≈ {cfw} µm)",
                xlim=(-300, 300), yticks=[])

        # Bottom-left: CFW vs defocus (sweep curve + current-z marker)
        ax3 = fig.add_subplot(2, 2, 3)
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
        ax3.legend(fontsize=7, loc="upper left")
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

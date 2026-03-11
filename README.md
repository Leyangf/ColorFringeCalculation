# ChromFringe

A research toolkit for predicting and quantifying **chromatic colour fringing** in photographic lenses. Given a lens prescription (Zemax ZMX), ChromFringe models how residual longitudinal chromatic aberration (CHL) and spherical aberration (SA) produce visible colour fringes at high-contrast edges, and reports the result as a **Colour Fringe Width (CFW)** metric (µm).

## Motivation

Achromatic lenses still exhibit residual secondary spectrum: different wavelengths focus at slightly different axial positions. Near a sharp edge, R, G, and B channels blur by different amounts, producing a visible colour fringe. ChromFringe provides a hierarchy of ESF (Edge Spread Function) models — from sub-microsecond analytic kernels to full FFT diffraction ground truth — together with tools to extract the required aberration data from an Optiland lens model.

## Repository Layout

```
ChromFringe/
├── data/
│   ├── raw/                    ← Spectral CSV files (D65 illuminant, per-camera sensor QE)
│   │   ├── daylight_d65.csv
│   │   ├── sensor_nikond700_{red,green,blue}.csv
│   │   ├── sensor_sonya900_{red,green,blue}.csv
│   │   └── defocus_chl_zf85.csv
│   └── lens/                   ← Lens prescription (ZMX format)
│       └── NikonAINikkor85mmf2S.zmx
├── examples/
│   ├── cfw_geom_demo.ipynb     ← Primary notebook: geometric / analytic PSF models
│   └── cfw_fftpsf_demo.ipynb   ← Validation notebook: FFT diffraction ground truth
└── src/chromf/
    ├── __init__.py             ← Public API (re-exports 16 functions)
    ├── cfw.py                  ← Core CFW kernels (Numba JIT)
    ├── spectrum_loader.py      ← Spectral data loading & normalisation
    └── optiland_bridge.py      ← Aberration extraction from Optiland lens models
```

## Installation

> **Platform note:** `environment.yml` was exported on Windows and pins Windows-specific runtime packages (`ucrt`, `vc14_runtime`, etc.). It is intended for Windows + Anaconda. Linux / macOS users should install dependencies manually.

```bash
git clone https://github.com/Leyangf/ChromFringe.git
cd ChromFringe
conda env create -f environment.yml
conda activate chromfringe
```

This installs all pinned dependencies (Python 3.13, NumPy 2.2, Numba 0.61), the `chromf` package in editable mode, and [Optiland](https://github.com/HarrisonKramer/optiland).

**Minimal pip install** (if not using Conda):

```bash
pip install numpy scipy numba pandas matplotlib ipywidgets optiland
pip install -e .
```

## Quick Start

```python
from optiland import fileio
import chromf

# Load a lens and extract aberration curves
lens = fileio.load_zemax_file("data/lens/NikonAINikkor85mmf2S.zmx")
chl = chromf.compute_rori_chl_curve(lens)
_, spot = chromf.compute_rori_spot_curves(lens)

# Compute CFW at 200 µm defocus
cfw = chromf.fringe_width(
    z_um=200.0,
    chl_curve_um=chl[:, 1],
    sa_curve_um=spot[:, 1],
    f_number=2.0,
    psf_mode="gauss",
)
print(f"CFW = {cfw} µm")
```

## Research Notebooks

### `cfw_geom_demo.ipynb` — Interactive Geometric Analysis

The primary research notebook. Loads a lens, extracts aberration curves, and provides:

1. **Interactive viewer** — real-time R/G/B ESF and pseudo-density fringe map with sliders for defocus, exposure, gamma, and PSF model.
2. **Static comparisons** — controlled-variable experiments: PSF model fidelity, aberration input fidelity, and full-input CFW sweeps.

Aberration curves extracted per wavelength:
- Paraxial CHL (secondary spectrum from marginal-ray trace)
- RoRi CHL (aperture-weighted, includes spherochromatism)
- Residual SA spot radius ρ_sa(λ)
- SA polynomial coefficients c₃, c₅ (3rd + 5th order)
- W040 Seidel coefficient

**PSF models:** Pillbox · Gaussian · Multi-Zone Defocus (MZD) · Geometric fast (ray-fan) · Geometric integral (Gauss-Legendre).

### `cfw_fftpsf_demo.ipynb` — FFT Diffraction Validation

Computes polychromatic ESFs from first principles (FFT Fraunhofer diffraction). Serves as the ground-truth reference.

**Two-stage baking** for efficiency:
- *Stage 1 (sensor-independent):* Bake monochromatic ESFs via FFT (25 defocus × 11 wavelengths). Re-run only when optics change.
- *Stage 2 (sensor-specific):* Apply spectral weights per channel/camera (microseconds). Re-run when switching sensor models.

This is **3× faster** than single-step per-channel baking (6× with two cameras).

## Core Modules

### `cfw.py` — CFW Kernels

Numba JIT-compiled inner loops for CFW computation.

| Function | Description |
|---|---|
| `edge_response(channel, x_px, z_um, ...)` | Single-channel ESF value at pixel x, defocus z |
| `edge_rgb_response(x_px, z_um, ...)` | R, G, B ESF tuple |
| `detect_fringe_binary(x_px, z_um, ...)` | 1 if pixel is colour-fringed, else 0 |
| `fringe_width(z_um, ...)` | Total CFW in µm at the given defocus |
| `load_sensor_response(model)` | Build R/G/B spectral-weight dict for a camera model |

All functions accept `chl_curve_um`, `sa_curve_um`, `f_number`, `psf_mode`, `exposure_slope`, `gamma`, and `sensor_response` as keyword arguments.

**PSF modes:** `"geom"` (Pillbox), `"gauss"` (Gaussian), `"mzd"` (Multi-Zone Defocus with arcsin ring ESF and Gauss-Legendre pupil integration).

### `spectrum_loader.py` — Spectral Data

Loads and energy-normalises spectral data (illuminant × sensor QE) so that a perfectly focused flat-spectrum edge yields unit response per channel.

Key function: `channel_products(sensor_model="sonya900")` → dict of normalised S·D products.

**Multi-camera support:** Sensor files follow the convention `sensor_{model}_{color}.csv`. Bundled models: **Nikon D700** (`nikond700`), **Sony A900** (`sonya900`). Add a new camera by placing CSV files and passing the model name.

### `optiland_bridge.py` — Aberration Extraction & PSF

Extracts aberration data from an Optiland `Optic` object and computes ESFs at multiple fidelity levels:

| Function | Output |
|---|---|
| `compute_chl_curve` | Paraxial CHL [λ, µm] |
| `compute_rori_chl_curve` | Aperture-weighted RoRi CHL [λ, µm] |
| `compute_rori_spot_curves` | RoRi CHL + residual SA spot radius |
| `compute_sa_poly_curves` | SA polynomial coefficients c₃, c₅ per wavelength |
| `compute_w040_curve` | Seidel W040 [λ, µm OPD] |
| `precompute_ray_fan` | Pre-traced ray fan for fast z-extrapolation |
| `compute_polychromatic_esf` | FFT diffraction ESF (ground truth, ~1 s/ESF) |
| `compute_polychromatic_esf_geometric` | Geometric pupil-integral ESF (~100× faster) |
| `compute_polychromatic_esf_fast` | Ray-fan linear extrapolation ESF (~1000× faster) |
| `bake_wavelength_esfs` | Sensor-independent monochromatic ESF grid (FFT) |
| `apply_sensor_weights` | Combine baked ESFs with sensor spectral weights (µs) |
| `compute_polychromatic_psf` | 2D polychromatic diffraction PSF |
| `compute_cfw_psf` | CFW computed directly from 2D diffraction PSF |

## Key Concepts

**Colour Fringe Width (CFW):** the number of pixels (at 1 µm pitch) where the maximum pairwise channel difference exceeds the visibility threshold δ:

$$\text{CFW}(z) = \sum_x \mathbf{1}\!\left[\max(|R-G|,\,|R-B|,\,|G-B|) > \delta\right]$$

**Modelling hierarchy:**

| Level | Method | Speed | Accuracy |
|-------|--------|-------|----------|
| 0 | FFT diffraction PSF | ~1 s/ESF | Includes diffraction |
| 1 | Geometric pupil integral | ~10 ms/ESF | Geometrically exact |
| 2 | Ray-fan extrapolation | <1 ms/ESF | Linear error O((z/f')²) |
| 3 | Analytic ESF (Pillbox/Gauss/MZD) | <0.01 ms/ESF | Parametric approximation |

## Dependencies

| Package | Role |
|---|---|
| NumPy / SciPy | Numerical core |
| Numba | JIT compilation of ESF kernels |
| Matplotlib | Plotting |
| ipywidgets | Interactive sliders in notebooks |
| Optiland | Lens prescription and ray tracing |
| comtypes | Windows COM bridge (optional, for Zemax export) |

See `environment.yml` for the full pinned environment (Python 3.13, NumPy 2.2, Numba 0.61).

## License

MIT — see [LICENSE](LICENSE).

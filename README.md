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
    ├── __init__.py             ← Public API (re-exports 15 functions)
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
chl_curve, spot_curve = chromf.compute_rori_spot_curves(lens)

# Compute CFW at 200 µm defocus
cfw = chromf.fringe_width(
    z_um=200.0,
    chl_curve_um=chl_curve[:, 1],
    rho_sa_um=spot_curve[:, 1],
    f_number=2.0,
    psf_mode="gauss",
)
print(f"CFW = {cfw} µm")
```

## Research Notebooks

### `cfw_geom_demo.ipynb` — Interactive Geometric Analysis

The primary research notebook. Loads a lens, extracts aberration curves, and provides:

1. **Diagnostic plots** — illuminant & sensor spectral responses, CHL curves (paraxial / RoRi), aberration budget (SA vs CHL blur radius), and per-pupil SA profile comparison (scalar ρ³ model vs polynomial fit vs ray-fan ground truth).
2. **Interactive viewer** — real-time R/G/B ESF and pseudo-density fringe map with sliders for defocus, exposure, gamma, PSF model, CHL curve, and SA toggle.
3. **Static comparisons** — controlled-variable experiments:
   - **5a:** 2×2 factorial — CHL model (Paraxial / RoRi) × PSF shape (Disc / Gaussian), no SA.
   - **5b:** SA effect — Disc + RoRi + SA vs Gaussian + RoRi + SA.
   - **5c:** Geom Fast node count convergence (5 / 16 / 32 GL nodes).
4. **Per-defocus ESF diagnostic** — detailed 3-column visualization (raw ESF, tone-mapped ESF, pseudo-density fringe map) at each defocus position using Geom Fast 16-node model.

Aberration curves extracted per wavelength:
- Paraxial CHL (secondary spectrum from marginal-ray trace)
- RoRi CHL (energy-weighted best focus, includes spherochromatism)
- Residual SA spot radius ρ_SA(λ)

**PSF models:** Disc (pillbox) · Gaussian · Geometric Fast (ray-fan with Gauss-Legendre pupil integration).

### `cfw_fftpsf_demo.ipynb` — FFT Diffraction Validation

Computes polychromatic ESFs from first principles (FFT Fraunhofer diffraction). Serves as the ground-truth reference.

**Two-stage baking** for efficiency:
- *Stage 1 (sensor-independent):* Bake monochromatic ESFs via FFT (29 defocus × 11 wavelengths, ±700 µm). Re-run only when optics change.
- *Stage 2 (sensor-specific):* Apply spectral weights per channel/camera (microseconds). Re-run when switching sensor models.

This is **3× faster** than single-step per-channel baking (6× with two cameras).

## Core Modules

### `cfw.py` — CFW Kernels

Numba JIT-compiled inner loops for CFW computation.

| Function | Description |
|---|---|
| `edge_response(channel, x_um, z_um, ...)` | Single-channel ESF value at position x (µm), defocus z |
| `edge_rgb_response(x_um, z_um, ...)` | R, G, B ESF tuple (scalar x) |
| `edge_rgb_response_vec(x_arr, z_um, ...)` | R, G, B ESF arrays (vectorised) |
| `detect_fringe_binary(x_um, z_um, ...)` | 1 if pixel is colour-fringed, else 0 |
| `is_fringe_mask(r, g, b, ...)` | Boolean mask of visible fringe pixels |
| `fringe_width(z_um, ...)` | Total CFW in µm at the given defocus |
| `load_sensor_response(model)` | Build R/G/B spectral-weight dict for a camera model |

All functions accept `chl_curve_um`, `rho_sa_um`, `f_number`, `psf_mode`, `exposure_slope`, `gamma`, and `sensor_response` as keyword arguments.

**PSF modes (analytic):** `"disc"` (Pillbox), `"gauss"` (Gaussian). For ray-fan based geometric ESF, use `compute_polychromatic_esf_geom` in `optiland_bridge.py`.

### `spectrum_loader.py` — Spectral Data

Loads and energy-normalises spectral data (illuminant × sensor QE) so that a perfectly focused flat-spectrum edge yields unit response per channel.

Key function: `channel_products(sensor_model="sonya900")` → dict of normalised S·D products.

**Multi-camera support:** Sensor files follow the convention `sensor_{model}_{color}.csv`. Bundled models: **Nikon D700** (`nikond700`), **Sony A900** (`sonya900`). Add a new camera by placing CSV files and passing the model name.

### `optiland_bridge.py` — Aberration Extraction & PSF

Extracts aberration data from an Optiland `Optic` object and computes ESFs at multiple fidelity levels:

| Function | Output |
|---|---|
| `compute_chl_curve` | Paraxial CHL [λ, µm] |
| `compute_rori_spot_curves` | RoRi CHL + residual SA spot radius (energy-weighted) |
| `precompute_ray_fan` | Pre-traced ray fan for fast z-extrapolation |
| `compute_polychromatic_esf` | FFT diffraction ESF (ground truth, ~1 s/ESF) |
| `compute_polychromatic_esf_geom` | Geometric ESF via pre-traced ray fan (~1000× faster) |
| `bake_wavelength_esfs` | Sensor-independent monochromatic ESF grid (FFT) |
| `apply_sensor_weights` | Combine baked ESFs with sensor spectral weights (µs) |

## Key Concepts

**Colour Fringe Width (CFW):** A pixel is considered to exhibit a visible colour fringe if and only if all three of the following conditions are satisfied simultaneously:

1. **C1 (lower brightness):** every channel exceeds 15% of maximum intensity — $\min(I_R, I_G, I_B) > \delta_\text{low}$
2. **C2 (inter-channel difference):** at least one pairwise channel difference exceeds 15% — $\max(|I_R - I_G|,\,|I_R - I_B|,\,|I_G - I_B|) > \delta$
3. **C3 (upper brightness):** at least one channel is below 80% — $\min(I_R, I_G, I_B) < \delta_\text{high}$

CFW is the spatial extent (µm) of the contiguous region where all three conditions hold:

$$\text{CFW}(z) = x_\text{last} - x_\text{first} + 1$$

Default thresholds: $\delta_\text{low} = 0.15$, $\delta = 0.15$, $\delta_\text{high} = 0.80$.

**Modelling hierarchy:**

| Level | Method | Speed | Accuracy |
|-------|--------|-------|----------|
| 0 | FFT diffraction PSF | ~1 s/ESF | Includes diffraction |
| 1 | Geometric ray-fan ESF | <1 ms/ESF | Geometrically exact, linear z-extrapolation |
| 2 | Analytic ESF (Disc/Gaussian) | <0.01 ms/ESF | Parametric approximation |

## Dependencies

| Package | Role |
|---|---|
| NumPy / SciPy | Numerical core |
| Numba | JIT compilation of ESF kernels |
| Matplotlib | Plotting |
| ipywidgets | Interactive sliders in notebooks |
| Optiland | Lens prescription and ray tracing |

See `environment.yml` for the full pinned environment (Python 3.13, NumPy 2.2, Numba 0.61).

## License

MIT — see [LICENSE](LICENSE).

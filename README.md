# ChromFringe

A research package for predicting and analysing **chromatic colour fringing** in photographic lenses. Given a lens prescription (Zemax ZMX file), ChromFringe models how residual longitudinal chromatic aberration (CHL) and spherical aberration produce visible colour fringes at high-contrast edges, and quantifies the effect as a **Colour Fringe Width (CFW)** metric.

## Background

Achromatic lenses still exhibit residual chromatic aberrations: different wavelengths focus at slightly different axial positions (secondary spectrum / CHL). Near a sharp edge in an image, this causes the red, green, and blue channels to have different blur radii, producing a visible colour fringe. The effect is strongest when the image plane is offset from the green-channel best focus, and its severity depends on the f-number, exposure settings, and the display tone curve.

ChromFringe provides three complementary ESF (edge-spread function) modelling approaches — from fast analytic kernels to a full FFT diffraction ground truth — and tools to extract the required aberration data from an Optiland lens model.

## Repository layout

```
ChromFringe/
├── data/
│   ├── raw/                    ← Spectral CSV files (D65 illuminant, per-camera sensor QE)
│   │   ├── daylight_d65.csv
│   │   ├── sensor_nikond700_{red,green,blue}.csv
│   │   ├── sensor_sonya900_{red,green,blue}.csv
│   │   └── defocus_chl_zf85.csv
│   └── lens/                   ← Lens prescription file (ZMX format)
├── examples/
│   ├── cfw_geom_demo.ipynb     ← Main research notebook (geometric / analytic PSF models)
│   └── cfw_fftpsf_demo.ipynb   ← Validation notebook (FFT diffraction PSF ground truth)
└── src/chromf/
    ├── __init__.py             ← Public API (re-exports 15 functions)
    ├── cfw.py                  ← Core CFW kernels (Numba JIT-compiled)
    ├── spectrum_loader.py      ← Spectral data loading and normalisation
    └── optiland_bridge.py      ← Aberration extraction from Optiland lens models
```

## Data requirements

All data files are bundled in this repository:

- `data/raw/` — Spectral CSV files (D65 illuminant, per-camera sensor QE curves)
  - Sensor files follow the naming convention `sensor_{model}_{color}.csv` (e.g. `sensor_nikond700_red.csv`)
  - Bundled sensor models: **Nikon D700** (`nikond700`), **Sony A900** (`sonya900`)
  - To add a new camera, place `sensor_{model}_{red,green,blue}.csv` in `data/raw/` and pass the model name to the API
- `data/lens/` — Lens prescription file (ZMX format)

## Installation

> **Platform note:** `environment.yml` was exported on Windows and pins Windows-specific runtime packages (`ucrt`, `vc14_runtime`, etc.). It is intended for Windows + Anaconda. Linux / macOS users should install dependencies manually.

Clone the repository and create the Conda environment from the repo root (the `-e .` pip step requires the working directory to be the project root):

```bash
git clone https://github.com/Leyangf/ChromFringe.git
cd ChromFringe
conda env create -f environment.yml
conda activate chromfringe
```

This installs all pinned dependencies, the `chromf` package in editable mode, and [Optiland](https://github.com/HarrisonKramer/optiland).

## Research notebooks

The two notebooks in `examples/` form the main research workflow.

### `cfw_geom_demo.ipynb` — Interactive geometric analysis

The primary research notebook. Loads a lens prescription, extracts aberration curves, and provides an interactive slider-based viewer alongside static comparison figures.

**Workflow:**
1. Load the lens ZMX file via Optiland and apply aperture constraints.
2. Compute four aberration curves per wavelength:
   - *Paraxial CHL* — secondary spectrum from marginal-ray trace.
   - *RoRi CHL* — aperture-weighted focal shift (includes spherochromatism).
   - *Residual SA spot radius* ρ_sa(λ) — geometric blur at best focus.
   - *W040* — Seidel spherical-aberration OPD coefficient.
3. Pre-trace a 32-node ray fan once to enable fast ESF sweeps.
4. **Interactive viewer** (Section 4): real-time R/G/B ESF and pseudo-density fringe map as defocus, exposure, gamma, and PSF model are varied.
5. **Static comparisons** (Section 5): PSF model fidelity, aberration input fidelity, and full-input CFW vs exposure sweeps.

**PSF models available:** Pillbox · Gaussian · Double-Gaussian · Geometric fast (ray-fan linear extrapolation) · Geometric integral (full Gauss-Legendre quadrature).

### `cfw_fftpsf_demo.ipynb` — FFT diffraction validation

Computes polychromatic ESFs from first principles using FFT diffraction propagation. Serves as the ground-truth reference for the geometric models.

**Workflow:**
1. **Two-stage PSF baking** — the expensive one-time step, decoupled for efficiency:
   - *Stage 1 (sensor-independent):* Bake monochromatic ESFs via FFT (25 defocus × 11 wavelengths). Re-run only when optics change.
   - *Stage 2 (sensor-specific):* Apply spectral weights per channel/camera. Re-run when switching sensor models (microseconds).
   - This two-stage approach is **3× faster** than baking per-channel, because each wavelength's FFT runs once regardless of the number of sensor models.
2. Analyse ESF transition widths to locate per-channel best-focus positions.
3. Compute CFW and per-pair tone differences at exposures 1, 2, 4, 8, 16.
4. Per-defocus diagnostic grid: raw ESF · tone-mapped ESF · pseudo-density fringe map.

## Core modules

### `cfw.py`

Low-level numerical kernels for CFW computation. All inner loops are JIT-compiled with Numba.

| Function | Description |
|---|---|
| `edge_response(channel, x_px, z_um, ...)` | Single-channel ESF value at pixel x, defocus z |
| `edge_rgb_response(x_px, z_um, ...)` | R, G, B ESF tuple |
| `detect_fringe_binary(x_px, z_um, ...)` | 1 if pixel is colour-fringed, else 0 |
| `fringe_width(z_um, ...)` | Total CFW in µm at the given defocus |
| `load_sensor_response(model)` | Build an R/G/B spectral-weight dict for a camera model |

All public functions accept `chl_curve_um`, `sa_curve_um`, `w040_curve_um`, `f_number`, `psf_mode`, `exposure_slope`, `gamma`, and `sensor_response` as keyword arguments. Pass a dict from `load_sensor_response()` to switch camera models.

### `spectrum_loader.py`

Loads and energy-normalises the spectral data so that a perfectly focused flat-spectrum edge produces unity response in every channel.

Key function: `channel_products(daylight_src, channels, *, sensor_model, sensor_peak)` → dict of normalised S·D products. The `sensor_model` parameter selects the camera (e.g. `"nikond700"`, `"sonya900"`); sensor CSV files must follow the naming convention `sensor_{model}_{color}.csv`.

### `optiland_bridge.py`

Extracts aberration inputs from an Optiland `Optic` object:

| Function | Output |
|---|---|
| `compute_chl_curve` | Paraxial CHL [λ, µm] |
| `compute_rori_chl_curve` | Aperture-weighted RoRi CHL [λ, µm] |
| `compute_rori_spot_curves` | RoRi CHL + residual SA spot radius |
| `compute_w040_curve` | Seidel W040 [λ, µm] |
| `precompute_ray_fan` | Pre-traced ray fan for fast z-extrapolation |
| `compute_polychromatic_esf` | Diffraction ESF (ground truth, slow) |
| `compute_polychromatic_esf_geometric` | Geometric pupil-integral ESF (~100× faster) |
| `compute_polychromatic_esf_fast` | Ray-fan linear extrapolation ESF (~1000× faster) |
| `bake_wavelength_esfs` | Sensor-independent monochromatic ESF grid (FFT) |
| `apply_sensor_weights` | Combine mono ESFs with sensor spectral weights (µs) |
| `compute_polychromatic_psf` | 2D polychromatic diffraction PSF |
| `compute_cfw_psf` | CFW computed directly from diffraction PSF |

## Key concepts

**Colour Fringe Width (CFW)**

$$\text{CFW}(z) = \sum_x \mathbf{1}\!\left[\max(|R-G|,\,|R-B|,\,|G-B|) > \delta\right]$$

where R, G, B are tone-mapped ESF values and δ ≈ 0.15–0.20 is the visibility threshold.

**CHL blur radius**

$$\rho_\text{CHL}(z,\lambda) = \frac{|z - \text{CHL}(\lambda)|}{\sqrt{4N^2-1}}$$

**RoRi weighting** averages back-focal intercepts at pupil heights {0, √0.25, √0.5, √0.75, 1} with weights proportional to annular area, capturing spherochromatism without a full pupil integral.

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

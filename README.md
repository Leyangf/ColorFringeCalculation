# ChromFringe

A research toolkit for predicting **chromatic colour fringing** in photographic lenses. Given a lens prescription (Zemax ZMX), ChromFringe models how residual longitudinal chromatic aberration (CHL) and spherical aberration (SA) produce visible colour fringes at high-contrast edges, and reports a **Colour Fringe Width (CFW)** metric in µm.

## Motivation

Achromatic lenses still exhibit residual secondary spectrum: different wavelengths focus at slightly different axial positions. Near a sharp edge, R/G/B channels blur by different amounts, producing a visible colour fringe. ChromFringe provides a hierarchy of ESF (Edge Spread Function) models — from sub-microsecond analytic kernels to full FFT diffraction ground truth — together with tools to extract the required aberration data from an Optiland lens model.

## Installation

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Leyangf/ChromFringe.git
cd ChromFringe
uv sync
```

`uv sync` creates `.venv/` and installs all locked dependencies (Python 3.13, NumPy, Numba, [Optiland](https://github.com/HarrisonKramer/optiland)) with `chromf` in editable mode. Run commands with `uv run` (e.g. `uv run jupyter lab`).

## Quick Start

```python
from optiland import fileio
import chromf

lens = fileio.load_zemax_file("data/lens/NikonAINikkor85mmf2S.zmx")
chl_curve, spot_curve = chromf.compute_rori_spot_curves(lens)

cfw = chromf.fringe_width(
    z_um=200.0,
    chl_curve_um=chl_curve[:, 1],
    rho_sa_um=spot_curve[:, 1],
    f_number=2.0,
    psf_mode="gauss",
)
print(f"CFW = {cfw} µm")
```

## Repository Layout

```
ChromFringe/
├── data/
│   ├── raw/      ← Spectral CSVs (D65 illuminant, per-camera sensor QE)
│   └── lens/     ← Lens prescriptions (ZMX)
├── examples/     ← Research notebooks
└── src/chromf/
    ├── cfw.py              ← Numba-JIT CFW kernels
    ├── spectrum_loader.py  ← Spectral data loading & normalisation
    └── optiland_bridge.py  ← Aberration extraction & ESF computation
```

## Notebooks

- [`examples/cfw_geom_demo.ipynb`](examples/cfw_geom_demo.ipynb) — interactive geometric/analytic PSF analysis with sliders for defocus, PSF model, and SA toggle. Includes diagnostic plots (CHL curves, SA vs CHL budget) and controlled-variable comparisons.
- [`examples/cfw_fftpsf_demo.ipynb`](examples/cfw_fftpsf_demo.ipynb) — FFT Fraunhofer diffraction ground truth. Uses two-stage baking (monochromatic ESF grid + per-sensor weighting) for efficient multi-camera sweeps.

## Modelling Hierarchy

| Level | Method | Speed | Notes |
|-------|--------|-------|-------|
| 0 | FFT diffraction PSF | ~1 s/ESF | Includes diffraction, ground truth |
| 1 | Geometric ray-fan ESF | <1 ms/ESF | Geometrically exact, linear z-extrapolation |
| — | Analytic (Disc / Gaussian) | <0.01 ms/ESF | Diagnostic only — parametric sanity check, not a predictive model |

**Aberration curves** extracted per wavelength: paraxial CHL (marginal-ray trace), RoRi CHL (energy-weighted best focus, includes spherochromatism), and residual SA spot radius ρ_SA(λ).

**Sensor support:** bundled models are Nikon D700 (`nikond700`) and Sony A900 (`sonya900`). Add a camera by placing `sensor_{model}_{red,green,blue}.csv` files in `data/raw/` and passing the model name.
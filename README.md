# ChromFringe

A research toolkit for predicting **chromatic colour fringing** in photographic lenses. Given a lens prescription (Zemax ZMX), ChromFringe models how residual longitudinal chromatic aberration (CHL) and spherical aberration (SA) produce visible colour fringes at high-contrast edges, and reports a **Colour Fringe Width (CFW)** metric in µm.

## Installation

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Leyangf/ChromFringe.git
cd ChromFringe
uv sync
```

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

## Notebooks

- [`examples/cfw_geom_demo.ipynb`](examples/cfw_geom_demo.ipynb) — interactive geometric/analytic PSF analysis with sliders for defocus, PSF model, and SA toggle.
- [`examples/cfw_fftpsf_demo.ipynb`](examples/cfw_fftpsf_demo.ipynb) — FFT diffraction ground truth for validation.

## Modelling Hierarchy

| Level | Method | Speed |
|-------|--------|-------|
| 0 | FFT diffraction PSF | ~1 s/ESF |
| 1 | Geometric ray-fan ESF | <1 ms/ESF |
| 2 | Analytic (Disc / Gaussian) | <0.01 ms/ESF |

## CFW Definition

A pixel is fringed if all three hold simultaneously:

1. $\min(I_R, I_G, I_B) > 0.15$ (lower brightness)
2. $\max_\text{pair}|I_i - I_j| > 0.15$ (inter-channel difference)
3. $\min(I_R, I_G, I_B) < 0.80$ (upper brightness)

CFW is the spatial extent (µm) of the contiguous region where all three hold.

## License

MIT — see [LICENSE](LICENSE).

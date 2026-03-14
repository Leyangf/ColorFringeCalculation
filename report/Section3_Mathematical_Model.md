# Section 3: Diagnostic Plots — Mathematical Model

## Overview

Section 3 of the geometric CFW notebook (`cfw_geom_demo.ipynb`) establishes the aberration characteristics of the test lens through four progressive diagnostic steps. Each step answers a specific question whose conclusion drives the next.

```mermaid
flowchart LR
    A["3a: ŵ_c(λ)"] --> B["3b: CHL(λ)"]
    B --> C["3c: ρ_SA ≷ ρ_CHL ?"]
    C --> D["3d: TA(ρ) accuracy"]
```

| Step | Question | Output |
|------|----------|--------|
| **3a** | What are the spectral weights? | $\hat{w}_c(\lambda)$ |
| **3b** | How much does focus shift with wavelength? | $\mathrm{CHL}(\lambda)$ curves |
| **3c** | Is SA significant compared to CHL? | Variance fraction per $\lambda$ |
| **3d** | How accurately must SA be represented? | RMS error of scalar / poly / ray fan |

---

## 3a. Spectral Weights

### Physical setup

A knife-edge target is illuminated by CIE D65 daylight and imaged by a colour sensor with per-channel quantum efficiency $S_c(\lambda)$, where $c \in \{R, G, B\}$.

### Energy-normalised weight

The polychromatic ESF for channel $c$ is a weighted spectral sum. The weight at each wavelength is the **sensor–daylight product**, energy-normalised so that a perfectly focused flat-spectrum edge gives unity response:

$$\hat{w}_c(\lambda) = \frac{S_c(\lambda) \cdot D_{65}(\lambda)}{\displaystyle\sum_{\lambda'} S_c(\lambda') \cdot D_{65}(\lambda')}$$

### Role in the signal chain

Every subsequent ESF computation uses $\hat{w}_c(\lambda)$ as the integration kernel:

$$\mathrm{ESF}_c(x, z) = \sum_{\lambda} \hat{w}_c(\lambda) \;\cdot\; \mathrm{ESF}_{\mathrm{mono}}\!\bigl(x;\;\rho(z,\lambda)\bigr)$$

where $\rho(z,\lambda)$ is the blur half-radius at defocus $z$ for wavelength $\lambda$ (defined in 3c).

### Plot content

| Panel | Content |
|-------|---------|
| Left | Raw $S_c(\lambda)$ and $D_{65}(\lambda)$, normalised to global peak |
| Right | $\hat{w}_c(\lambda)$ products entering all ESF integrations |

---

## 3b. Longitudinal Chromatic Aberration — CHL(λ)

### Definition

$\mathrm{CHL}(\lambda)$ is the axial distance (µm) between the focal plane of wavelength $\lambda$ and a reference wavelength $\lambda_{\mathrm{ref}}$. Three focal-plane models are compared.

### Model 1: Paraxial CHL

Trace a paraxial marginal ray (entrance height $y = 1$, slope $u = 0$) through the system at wavelength $\lambda$. At the image surface, the exit height $y'$ and slope $u'$ give the back-focal distance:

$$f_2'(\lambda) = -\frac{y'(\lambda)}{u'(\lambda)}$$

Paraxial CHL:

$$\mathrm{CHL}_{\mathrm{par}}(\lambda) = \bigl[f_2'(\lambda) - f_2'(\lambda_{\mathrm{ref}})\bigr] \times 10^3 \quad (\mu\mathrm{m})$$

> **Properties:** independent of aperture; captures secondary spectrum only.

### Model 2: RoRi-1 CHL (energy-weighted)

For a finite aperture, trace real rays at five pupil heights $\rho_i \in \{0,\;\sqrt{1/4},\;\sqrt{1/2},\;\sqrt{3/4},\;1\}$. The back-focal intercept of each ray is:

$$\mathrm{SK}(\rho_i, \lambda) = -\frac{y \cdot N}{M}$$

where $(M, N)$ are the direction cosines of the exiting ray and $y$ is its height at the last surface.

The RoRi-1 focal plane is the weighted mean:

$$\mathrm{RoRi}_1(\lambda) = \frac{\displaystyle\sum_{i=0}^{4} w_i \cdot \mathrm{SK}(\rho_i, \lambda)}{\displaystyle\sum_{i} w_i}$$

with weights $w_i = \{1,\;12.8,\;14.4,\;12.8,\;1\}$ and $\sum w_i = 42$.

$$\mathrm{CHL}_{\mathrm{RoRi\text{-}1}}(\lambda) = \bigl[\mathrm{RoRi}_1(\lambda) - \mathrm{RoRi}_1(\lambda_{\mathrm{ref}})\bigr] \times 10^3 \quad (\mu\mathrm{m})$$

> **Properties:** aperture-dependent; includes spherochromatism.

### Model 3: RoRi-4 CHL (ρ²-weighted, orthogonal)

Uses only the four non-paraxial nodes with $\rho^2$-weighted averaging, so that the CHL and SA blur contributions are **orthogonal** in the quadrature decomposition:

$$\mathrm{RoRi}_4(\lambda) = \frac{\displaystyle\sum_{i=1}^{4} w_i \rho_i^2 \cdot \mathrm{SK}(\rho_i, \lambda)}{\displaystyle\sum_{i=1}^{4} w_i \rho_i^2}$$

with weights $\{3.2,\;7.2,\;9.6,\;1.0\}$ and $\sum = 21$.

### Spherochromatism

The gap between paraxial and RoRi curves quantifies spherochromatism — the wavelength dependence of spherical aberration:

$$\Delta\mathrm{CHL}_{\mathrm{sphchrom}}(\lambda) = \mathrm{CHL}_{\mathrm{RoRi}}(\lambda) - \mathrm{CHL}_{\mathrm{par}}(\lambda)$$

### Plot content

Single panel: three CHL curves (paraxial, RoRi-1, RoRi-4) with 6th-order polynomial fits for visual smoothing.

---

## 3c. Aberration Budget — SA vs CHL blur radius

### Motivation

Before modelling SA, we ask: **does it matter compared to CHL?**

### Blur half-radii

At defocus $z$ and wavelength $\lambda$, two independent blur sources contribute:

**Chromatic defocus blur:**

$$\rho_{\mathrm{CHL}}(z, \lambda) = \frac{|z - \mathrm{CHL}(\lambda)|}{\sqrt{4N^2 - 1}}$$

where $N$ is the working f-number.

**Spherical aberration residual blur** (RMS transverse aberration at RoRi focus):

$$\rho_{\mathrm{SA}}(\lambda) = \sqrt{\frac{\displaystyle\sum_{i} w_i \cdot y_{\mathrm{spot}}(\rho_i, \lambda)^2}{\displaystyle\sum_{i} w_i}}$$

where the transverse spot displacement at the RoRi focal plane is:

$$y_{\mathrm{spot}}(\rho_i, \lambda) = \bigl[\mathrm{SK}(\rho_i, \lambda) - \mathrm{RoRi}(\lambda)\bigr] \cdot \frac{\rho_i}{\sqrt{4N^2 - 1}} \times 10^3 \quad (\mu\mathrm{m})$$

### Quadrature combination

The two blur sources combine in quadrature (assuming independence):

$$\rho_{\mathrm{total}}(z, \lambda) = \sqrt{\rho_{\mathrm{CHL}}^2 + \rho_{\mathrm{SA}}^2}$$

### Variance fraction

The relative importance is quantified by the variance (energy) share:

$$f_{\mathrm{SA}}(\lambda) = \frac{\rho_{\mathrm{SA}}^2}{\rho_{\mathrm{total}}^2}, \qquad f_{\mathrm{CHL}}(\lambda) = 1 - f_{\mathrm{SA}}(\lambda)$$

- $f_{\mathrm{SA}} > 0.5$: spherical aberration dominates at this wavelength
- $f_{\mathrm{SA}} < 0.5$: chromatic defocus dominates

### Plot content

| Panel | Content |
|-------|---------|
| Left | $\rho_{\mathrm{CHL}}(\lambda)$, $\rho_{\mathrm{SA}}(\lambda)$, $\rho_{\mathrm{total}}(\lambda)$ at $z = 0$ |
| Right | Stacked bar chart of $f_{\mathrm{SA}}$ vs $f_{\mathrm{CHL}}$ per wavelength |

### Key finding (Nikon 85mm f/2 at full aperture)

- Spectral extremes (400–430 nm, 650–700 nm): CHL dominates (80–95%)
- Mid-spectrum (450–620 nm): SA dominates (50–95%)
- SA provides a ~15 µm floor blur even where CHL ≈ 0

---

## 3d. Per-Pupil SA Profile — Parameterisation Accuracy

### Motivation

Section 3c established that SA is significant. Now: **how accurately must we represent it?**

### Ground truth: ray-fan TA at RoRi focus

The ray fan pre-computes signed transverse aberration $\mathrm{TA}_0(\rho, \lambda)$ and ray slope $m(\rho, \lambda) = M/N$ at the nominal image plane ($z = 0$) using $K$ Gauss-Legendre quadrature nodes.

At any defocus $z$, the transverse aberration is linearly extrapolated:

$$\mathrm{TA}(\rho, \lambda;\; z) = \mathrm{TA}_0(\rho, \lambda) + m(\rho, \lambda) \cdot z$$

Evaluating at the RoRi focal plane gives the SA residual:

$$\mathrm{TA}_{\mathrm{SA}}(\rho, \lambda) = \mathrm{TA}_0(\rho, \lambda) + m(\rho, \lambda) \cdot \mathrm{CHL}_{\mathrm{RoRi}}(\lambda)$$

With $K = 32$ GL nodes, this serves as the ground truth.

### Three parameterisation levels

| Model | Formula | Parameters per $\lambda$ | Error |
|-------|---------|:------------------------:|-------|
| Scalar $\rho_{\mathrm{SA}}$ | $\mathrm{TA}(\rho) \approx 2\rho^3 \cdot \rho_{\mathrm{SA}}$ | 1 | Highest |
| Polynomial | $\mathrm{TA}(\rho) \approx c_3(\lambda)\,\rho^3 + c_5(\lambda)\,\rho^5$ | 2 | Moderate |
| Ray fan | $\mathrm{TA}(\rho_k)$ exact at $K$ nodes | $K$ (32) | Zero (reference) |

The polynomial coefficients $c_3, c_5$ are fitted by least squares from 5 SK data points (excluding $\rho = 0$):

$$\begin{pmatrix} \rho_1^3 & \rho_1^5 \\ \vdots & \vdots \\ \rho_4^3 & \rho_4^5 \end{pmatrix} \begin{pmatrix} c_3 \\ c_5 \end{pmatrix} = \begin{pmatrix} \mathrm{TA}_{\mathrm{SA}}(\rho_1) \\ \vdots \\ \mathrm{TA}_{\mathrm{SA}}(\rho_4) \end{pmatrix}$$

### Quality metric

RMS error against the 32-node ray fan:

$$\varepsilon_{\mathrm{RMS}} = \sqrt{\frac{1}{K}\sum_{k=1}^{K} \bigl[\mathrm{TA}_{\mathrm{model}}(\rho_k) - \mathrm{TA}_{\mathrm{fan}}(\rho_k)\bigr]^2}$$

### Plot content

Five subplots (one per representative wavelength: 430, 480, 550, 600, 650 nm), each showing:
- Black: ray fan ground truth (32 points)
- Blue dashed: scalar $2\rho^3 \cdot \rho_{\mathrm{SA}}$ model
- Orange: polynomial $c_3\rho^3 + c_5\rho^5$ fit
- Red squares: 5 SK data points

---

## Notation Summary

| Symbol | Definition | Unit |
|--------|-----------|------|
| $\lambda$ | Wavelength | nm |
| $\lambda_{\mathrm{ref}}$ | Reference wavelength (≈ 580 nm) | nm |
| $N$ | Working f-number | — |
| $\rho$ | Normalised pupil height $\in [0, 1]$ | — |
| $S_c(\lambda)$ | Sensor quantum efficiency for channel $c$ | — |
| $D_{65}(\lambda)$ | CIE D65 illuminant spectral power | — |
| $\hat{w}_c(\lambda)$ | Energy-normalised spectral weight | — |
| $f_2'(\lambda)$ | Paraxial back-focal distance | mm |
| $\mathrm{SK}(\rho, \lambda)$ | Real-ray back-focal intercept | mm |
| $\mathrm{CHL}(\lambda)$ | Longitudinal chromatic aberration | µm |
| $\mathrm{RoRi}(\lambda)$ | Weighted best-focus position | mm |
| $\rho_{\mathrm{CHL}}$ | Chromatic defocus blur half-radius | µm |
| $\rho_{\mathrm{SA}}$ | SA residual blur half-radius (RMS) | µm |
| $\rho_{\mathrm{total}}$ | Combined blur half-radius (quadrature) | µm |
| $\mathrm{TA}(\rho, \lambda)$ | Signed transverse aberration | µm |
| $m(\rho, \lambda)$ | Ray slope $M/N$ (direction cosine ratio) | — |
| $\mathrm{ESF}(x)$ | Edge spread function | — |

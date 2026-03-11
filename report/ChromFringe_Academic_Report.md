---
title: "Numerical Modelling of Chromatic Colour Fringing in Photographic Lenses"
subtitle: A Multi-Fidelity Approach from Analytic ESF Models to FFT Diffraction
author: ChromFringe Research
date: 2026-03-11
tags: [optics, chromatic-aberration, PSF, ESF, colour-fringe, research]
---

# Numerical Modelling of Chromatic Colour Fringing in Photographic Lenses

> **Abstract.** We present a multi-fidelity numerical framework for predicting and quantifying chromatic colour fringes produced by residual longitudinal chromatic aberration (CHL) and spherical aberration (SA) in photographic lenses. The framework spans four modelling tiers — from sub-microsecond analytic edge-spread-function (ESF) kernels to full FFT Fraunhofer diffraction — unified under a single metric: **Colour Fringe Width (CFW)**. We validate the models against a Nikon AI Nikkor 85 mm f/2S lens prescription and demonstrate that the proposed Multi-Zone Defocus (MZD) analytic model captures pupil-resolved SA effects with accuracy comparable to geometric ray-fan integration, at three orders of magnitude lower computational cost.

---

## Table of Contents

- [[#1. Introduction]]
- [[#2. Physical Background]]
  - [[#2.1 Secondary Spectrum and Longitudinal Chromatic Aberration]]
  - [[#2.2 Spherochromatism]]
  - [[#2.3 Signal Chain]]
- [[#3. Aberration Extraction Methods]]
  - [[#3.1 Paraxial CHL]]
  - [[#3.2 RoRi Aperture-Weighted CHL]]
  - [[#3.3 Residual Spherical Aberration]]
  - [[#3.4 SA Polynomial Fitting]]
  - [[#3.5 Seidel W040 Coefficient]]
- [[#4. ESF Modelling Hierarchy]]
  - [[#4.1 Overview]]
  - [[#4.2 Pillbox ESF (Geometric Uniform Disc)]]
  - [[#4.3 Gaussian ESF]]
  - [[#4.4 Multi-Zone Defocus (MZD) ESF]]
  - [[#4.5 Geometric Pupil Integral]]
  - [[#4.6 Ray-Fan Linear Extrapolation]]
  - [[#4.7 FFT Fraunhofer Diffraction PSF]]
- [[#5. Polychromatic ESF and Tone Mapping]]
  - [[#5.1 Spectral Integration]]
  - [[#5.2 Energy Normalisation]]
  - [[#5.3 Tone Mapping and Gamma Correction]]
- [[#6. Colour Fringe Width (CFW) Metric]]
  - [[#6.1 Definition]]
  - [[#6.2 Detection Algorithm]]
- [[#7. Two-Stage PSF Baking]]
- [[#8. Experimental Validation]]
  - [[#8.1 Test Lens]]
  - [[#8.2 Measured Aberration Data]]
  - [[#8.3 ESF Transition Width Results]]
  - [[#8.4 CFW Sensitivity Analysis]]
- [[#9. Performance Comparison]]
- [[#10. Conclusions]]
- [[#Appendix A. Formula Reference]]

---

## 1. Introduction

Achromatic and apochromatic photographic lenses correct primary chromatic aberration, yet residual **secondary spectrum** persists: different wavelengths focus at slightly different axial positions. At high-contrast edges in an image, the red, green, and blue channels exhibit different blur radii, producing visible **colour fringes** — a phenomenon familiar to photographers shooting fast lenses wide open.

Despite the practical importance of this effect, quantitative prediction of colour fringe visibility from lens design data remains underexplored. Existing approaches either rely on qualitative observation or full wave-optics simulation, which is too slow for interactive design exploration.

This work introduces a **multi-fidelity modelling framework** that:

1. Extracts aberration data (CHL, SA, spherochromatism) from a lens prescription via ray tracing.
2. Provides four levels of ESF computation spanning five orders of magnitude in speed.
3. Defines a perceptually motivated metric — **Colour Fringe Width (CFW)** — that quantifies fringe visibility in micrometres.
4. Introduces a **Multi-Zone Defocus (MZD)** analytic model that resolves pupil-dependent SA without full pupil integration.

The framework is validated against a Nikon AI Nikkor 85 mm f/2S lens at f/2, using FFT diffraction PSF as the ground truth.

---

## 2. Physical Background

### 2.1 Secondary Spectrum and Longitudinal Chromatic Aberration

After correction of the primary spectrum (bringing two wavelengths to a common focus), an achromatic doublet still exhibits a residual wavelength-dependent focal shift known as the **secondary spectrum**. For wavelength $\lambda$ relative to a reference wavelength $\lambda_\text{ref}$, the **longitudinal chromatic aberration** (CHL) is:

$$\mathrm{CHL}(\lambda) = f_2'(\lambda) - f_2'(\lambda_\text{ref})$$

where $f_2'(\lambda)$ is the paraxial back focal length. Typical values for a well-corrected 85 mm lens are $|\mathrm{CHL}| < 200$ µm across the visible spectrum.

### 2.2 Spherochromatism

In fast lenses (low f-number), **spherical aberration varies with wavelength** — a higher-order effect called spherochromatism. This means:

- The optimal focus position depends not only on wavelength but also on aperture zone.
- A single CHL curve from paraxial ray tracing underestimates the true inter-channel blur difference.
- The residual SA spot size $\rho_{sa}(\lambda)$ itself is wavelength-dependent.

Accurate fringe prediction therefore requires aberration extraction methods that capture spherochromatism.

### 2.3 Signal Chain

The complete signal chain from scene to displayed colour fringe is:

$$\text{Scene (knife-edge)} \;\xrightarrow{D_{65}}\; \text{Illuminant} \;\xrightarrow{\mathrm{CHL}(\lambda),\,\mathrm{SA}(\lambda)}\; \text{Lens} \;\xrightarrow{S_c(\lambda)}\; \text{Sensor (RGB)} \;\xrightarrow{\gamma,\,\alpha}\; \text{Display}$$

Each stage introduces wavelength-dependent weighting or spatial blurring. The colour fringe is the observable consequence of the mismatch in R, G, B channel ESFs after the full pipeline.

---

## 3. Aberration Extraction Methods

### 3.1 Paraxial CHL

The simplest estimate traces a paraxial marginal ray at each wavelength to obtain the back focal length:

$$f_2'(\lambda) = -\frac{y_\text{last}}{u_\text{last}}$$

where $y_\text{last}$ and $u_\text{last}$ are the marginal ray height and slope at the final surface. The CHL curve is then:

$$\mathrm{CHL}_\text{par}(\lambda) = \left[f_2'(\lambda) - f_2'(\lambda_\text{ref})\right] \times 10^3 \quad (\mu\mathrm{m})$$

This captures the secondary spectrum but ignores all finite-aperture effects.

### 3.2 RoRi Aperture-Weighted CHL

To include spherochromatism, we trace **real meridional rays** at five normalised pupil heights $\rho_i \in \{0, \sqrt{1/4}, \sqrt{1/2}, \sqrt{3/4}, 1\}$ and compute each ray's back-focal intercept:

$$\mathrm{SK}(\rho, \lambda) = -\frac{y \cdot N_\text{dir}}{M}$$

where $M$ and $N_\text{dir}$ are the meridional and axial direction cosines of the ray after the final surface.

The **RoRi weighted average** uses annular area weights derived from dividing $[0,1]$ into four equal-area zones under trapezoidal integration:

$$\mathrm{RoRi}(\lambda) = \frac{1 \cdot \mathrm{SK}(0) + 12.8 \cdot \mathrm{SK}(\sqrt{1/4}) + 14.4 \cdot \mathrm{SK}(\sqrt{1/2}) + 12.8 \cdot \mathrm{SK}(\sqrt{3/4}) + 1 \cdot \mathrm{SK}(1)}{42}$$

The RoRi CHL curve is then:

$$\mathrm{CHL}_\mathrm{RoRi}(\lambda) = \left[\mathrm{RoRi}(\lambda) - \mathrm{RoRi}(\lambda_\mathrm{ref})\right] \times 10^3 \quad (\mu\mathrm{m})$$

### 3.3 Residual Spherical Aberration

Even at the RoRi best-focus plane, different aperture zones produce laterally displaced ray intercepts. The **RMS residual spot radius** quantifies this:

$$y_\text{spot}(\rho_i, \lambda) = \frac{[\mathrm{SK}(\rho_i, \lambda) - \mathrm{RoRi}(\lambda)] \cdot \rho_i}{2N}$$

$$\boxed{\rho_\text{sa}(\lambda) = \sqrt{\frac{\sum_i w_i \cdot y_\text{spot}^2(\rho_i, \lambda)}{\sum_i w_i}}}$$

where $w_i$ are the RoRi weights and $N$ is the f-number.

### 3.4 SA Polynomial Fitting

A single scalar $\rho_{sa}$ does not capture the pupil profile of SA. We fit the residual transverse aberration at each wavelength to a **two-term pupil polynomial**:

$$\mathrm{TA}_\text{SA}(\rho, \lambda) \approx c_3(\lambda) \cdot \rho^3 + c_5(\lambda) \cdot \rho^5$$

using least-squares regression on the four non-trivial RoRi pupil heights. The coefficients $c_3$ and $c_5$ capture primary (3rd-order) and secondary (5th-order) spherical aberration respectively, providing a wavelength-dependent pupil profile for use in the MZD model.

### 3.5 Seidel W040 Coefficient

From Seidel aberration theory, the transverse aberration of the marginal ray ($\rho = 1$) at the RoRi focus is related to the primary spherical aberration wavefront coefficient:

$$\mathrm{TA}_\text{marginal} = -8N \cdot W_{040}$$

Solving:

$$\boxed{W_{040}(\lambda) = \frac{-\mathrm{TA}_\text{marginal}(\lambda)}{8N} \quad (\mu\mathrm{m\;OPD})}$$

This connects the ray-trace data to the wave-optics description used in diffraction PSF computation.

---

## 4. ESF Modelling Hierarchy

### 4.1 Overview

We provide four tiers of ESF models, spanning five orders of magnitude in computation time:

| Level | Method | Speed | Physical Content |
|-------|--------|-------|-----------------|
| 0 | FFT Fraunhofer diffraction | ~1 s/ESF | Full wave optics |
| 1 | Geometric pupil integral (Gauss-Legendre) | ~10 ms/ESF | Exact geometric optics |
| 2 | Ray-fan linear extrapolation | <1 ms/ESF | Pre-traced rays, linear defocus |
| 3 | Analytic ESF (Pillbox / Gaussian / MZD) | <0.01 ms/ESF | Parametric pupil model |

All models share the same aberration inputs and spectral integration pipeline, enabling direct comparison.

### 4.2 Pillbox ESF (Geometric Uniform Disc)

The simplest model assumes the PSF is a uniformly illuminated disc of radius $\rho$ (the geometric blur circle). The corresponding ESF is:

$$\mathrm{ESF}_\text{geom}(x, \rho) = \begin{cases} 0 & x \leq -\rho \\ \frac{1}{2}\left(1 + \frac{x}{\rho}\right) & -\rho < x < \rho \\ 1 & x \geq \rho \end{cases}$$

The blur radius combines CHL defocus and SA in quadrature:

$$\rho(z, \lambda) = \sqrt{\rho_\text{CHL}(z, \lambda)^2 + \rho_\text{sa}(\lambda)^2}$$

where the CHL blur radius is:

$$\rho_\text{CHL}(z, \lambda) = \frac{|z - \mathrm{CHL}(\lambda)|}{\sqrt{4N^2 - 1}}$$

The Pillbox model produces hard-edged transitions and is the most conservative estimate (sharpest cutoff, lowest CFW).

### 4.3 Gaussian ESF

Replacing the uniform disc with a 2D Gaussian PSF ($\sigma = 0.5\rho$) yields:

$$\boxed{\mathrm{ESF}_\text{gauss}(x, \rho) = \frac{1}{2}\left[1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2} \cdot 0.5\rho}\right)\right]}$$

The soft tails provide a smoother transition that better approximates diffraction effects, producing moderately larger CFW than the Pillbox model.

### 4.4 Multi-Zone Defocus (MZD) ESF

The MZD model is the key contribution of this work. Rather than collapsing SA into a single scalar, it **resolves the pupil-dependent blur** by integrating over Gauss-Legendre quadrature nodes.

For each wavelength $\lambda$ and pupil node $\rho_k$, the blur radius is:

$$\rho_\text{CHL} = \frac{|z - \mathrm{CHL}(\lambda)|}{\sqrt{4N^2 - 1}}$$

$$\mathrm{TA}_\text{SA}(\rho_k) = \rho_k^3 \cdot 2 \cdot \rho_\text{sa}(\lambda)$$

$$R_k = \sqrt{(\rho_k \cdot \rho_\text{CHL})^2 + \mathrm{TA}_\text{SA}^2}$$

The factor of 2 converts the RMS SA to an approximate marginal value. Each pupil ring contributes an **arcsin ring ESF** — the exact geometric result for a thin annulus of radius $R$:

$$\mathrm{ESF}_\text{ring}(x;\,R) = \frac{\arcsin\!\left(\mathrm{clip}\!\left(\frac{x}{R},\,-1,\,1\right)\right)}{\pi} + \frac{1}{2}$$

The full MZD ESF integrates over the pupil with area weights:

$$\boxed{\mathrm{ESF}_\text{MZD}(x;\,z,\lambda) = \sum_k \rho_k \cdot W_k \cdot \mathrm{ESF}_\text{ring}(x;\,R_k)}$$

where $\rho_k$ and $W_k$ are Gauss-Legendre nodes and weights on $[0, 1]$.

**Advantages over the former Double-Gaussian model:**
- No need to compute pupil zone boundaries ($\rho_s$) or zone RMS values analytically.
- Uses the same arcsin ring ESF as the geometric pupil integral, ensuring consistent physics.
- Naturally handles arbitrary SA profiles (including higher-order terms) without reformulation.
- Retains sub-microsecond evaluation via JIT compilation.

### 4.5 Geometric Pupil Integral

For each wavelength, trace real rays at $K$ Gauss-Legendre nodes across the pupil and compute the lateral displacement $R(\rho_k)$ at the defocused image plane. The ESF is:

$$\mathrm{ESF}(x) = \int_0^1 \left[\frac{\arcsin\!\left(\mathrm{clip}\!\left(\frac{x}{R(\rho)},\,-1,\,1\right)\right)}{\pi} + \frac{1}{2}\right] 2\rho\,d\rho$$

approximated numerically as:

$$\mathrm{ESF}(x) \approx \sum_k \rho_k \cdot W_k \cdot \mathrm{ESF}_\text{ring}(x;\,R_k)$$

With 32 Gauss-Legendre nodes, ESF accuracy is better than 0.1% for smooth aberration profiles. This is the highest-fidelity geometric method, but requires real ray tracing at every defocus position.

### 4.6 Ray-Fan Linear Extrapolation

Pre-trace all rays at $z = 0$, recording transverse aberration $\mathrm{TA}_0(\rho, \lambda)$ and direction cosine ratio $m(\rho, \lambda) = M / N_\text{dir}$. For any defocus $z$:

$$\boxed{R(\rho;\,z,\lambda) = \left|\mathrm{TA}_0(\rho,\lambda) + m(\rho,\lambda) \cdot z\right|}$$

This linear extrapolation has error $O((z/f')^2)$, which for an 85 mm lens at $z \leq 800$ µm is less than 0.01%. The one-time cost is 32 × 31 = 992 ray traces; subsequent defocus evaluations require only array arithmetic.

### 4.7 FFT Fraunhofer Diffraction PSF

The ground-truth model computes the PSF as the squared modulus of the Fourier transform of the complex pupil function:

$$\mathrm{PSF}(\mathbf{u}) = \left|\mathcal{F}\left\{P(\boldsymbol{\rho})\cdot e^{i2\pi W(\boldsymbol{\rho})/\lambda}\right\}\right|^2$$

where $W(\boldsymbol{\rho})$ is the wavefront error obtained from real ray tracing.

**Critical parameter choice:** The reference sphere strategy must be `chief_ray` (not `best_fit_sphere`) to preserve the defocus phase. Using `best_fit_sphere` removes defocus from the OPD, causing all wavelengths to produce identical ESF shapes and yielding CFW = 0 — an incorrect result.

The FFT pixel pitch varies with wavelength (Fraunhofer scaling):

$$dx_j = \frac{\lambda_j \cdot N}{Q}, \quad Q = \frac{\text{grid size}}{\text{num rays} - 1}$$

requiring interpolation onto a common physical coordinate grid when summing over wavelengths.

The ESF is obtained from the 2D PSF via:

$$\mathrm{LSF}(x) = \int \mathrm{PSF}(x, y)\,dy, \quad \mathrm{ESF}(x) = \int_{-\infty}^{x} \mathrm{LSF}(t)\,dt$$

---

## 5. Polychromatic ESF and Tone Mapping

### 5.1 Spectral Integration

For each colour channel $c \in \{R, G, B\}$, the polychromatic ESF is a spectrally weighted sum of monochromatic ESFs:

$$\boxed{\mathrm{ESF}_c(x, z) = \sum_j \hat{w}_c(\lambda_j) \cdot \mathrm{ESF}_\text{mono}(x;\,z,\,\lambda_j)}$$

where the normalised spectral weights are:

$$\hat{w}_c(\lambda_j) = \frac{S_c(\lambda_j) \cdot D_{65}(\lambda_j)}{\sum_{j'} S_c(\lambda_{j'}) \cdot D_{65}(\lambda_{j'})}$$

$S_c(\lambda)$ is the sensor quantum efficiency for channel $c$, and $D_{65}(\lambda)$ is the CIE D65 illuminant spectrum.

The wavelength grid spans 400–700 nm in 10 nm steps (31 points). Subsampling to 11 points (30 nm steps) introduces negligible error.

### 5.2 Energy Normalisation

To ensure that a perfectly focused flat-spectrum edge produces unit response, each channel's spectral product is normalised:

$$k_c = \left(\int S_c(\lambda) \cdot D_{65}(\lambda)\,d\lambda\right)^{-1}$$

$$\hat{g}_c(\lambda) = k_c \cdot S_c(\lambda) \cdot D_{65}(\lambda)$$

### 5.3 Tone Mapping and Gamma Correction

The raw ESF is passed through a tone-mapping pipeline modelling camera/display nonlinearity:

$$\boxed{I_c(x, z) = \left[\frac{\tanh(\alpha \cdot \mathrm{ESF}_c)}{\tanh(\alpha)}\right]^\gamma}$$

where $\alpha$ is the exposure slope (default 8.0) and $\gamma$ is the display gamma (default 2.2, sRGB). The $\tanh$ curve satisfies $T(0) = 0$, $T(1) = 1$, with enhanced slope near zero that simulates contrast amplification.

Tone mapping is critical to fringe prediction: it amplifies small inter-channel ESF differences near the edge transition into visible colour shifts.

---

## 6. Colour Fringe Width (CFW) Metric

### 6.1 Definition

The **Colour Fringe Width** is defined as the total spatial extent (in µm) over which the tone-mapped R, G, B channel values differ by more than a visibility threshold $\delta$:

$$\boxed{\mathrm{CFW}(z) = \sum_{x} \mathbf{1}\!\left[\max\!\left(|I_R - I_G|,\;|I_R - I_B|,\;|I_G - I_B|\right) > \delta\right]}$$

with $\delta \approx 0.15$–$0.20$ (perceptual visibility threshold) and the scan window $x \in [-400, +400]$ µm at 1 µm steps.

### 6.2 Detection Algorithm

For each pixel position $x$:

1. Compute $I_R(x, z)$, $I_G(x, z)$, $I_B(x, z)$ via the tone-mapped ESF pipeline.
2. Evaluate $\max(|I_R - I_G|, |I_R - I_B|, |I_G - I_B|)$.
3. If the maximum exceeds $\delta$, mark as a fringe pixel.
4. CFW is the total count of fringe pixels.

The metric is sensitive to:
- **Defocus $z$:** CFW varies with image plane position; peak CFW occurs near (but not at) the nominal focal plane.
- **Exposure $\alpha$:** Higher exposure amplifies inter-channel differences, increasing CFW — until saturation compresses all channels to unity, causing CFW to decrease.
- **Aberration fidelity:** RoRi CHL with SA produces different (typically wider) CFW profiles than paraxial CHL alone.

---

## 7. Two-Stage PSF Baking

For the FFT ground-truth path, we introduce a **two-stage computation** that decouples the expensive wave-optics calculation from sensor-specific weighting:

**Stage 1 — Monochromatic ESF Baking (Sensor-Independent):**
Compute monochromatic ESFs via FFT for each defocus position $z$ and wavelength $\lambda$. This depends only on the lens design and is independent of the sensor model.

**Stage 2 — Sensor Weight Application:**
Apply the spectral weighting $\hat{w}_c(\lambda_j)$ for each channel and camera model via a simple weighted sum. This is a pure linear operation requiring microseconds.

**Performance gain:** For 25 defocus steps × 3 channels × 2 camera models (150 polychromatic ESFs), the two-stage approach requires 25 × 11 = 275 FFT evaluations plus 150 weighted sums. The naive single-step approach would require 150 × 11 = 1,650 FFTs — a **6× speedup** (3× for a single camera).

When switching camera models, only Stage 2 is re-run, enabling rapid sensor comparison without re-computing any FFTs.

---

## 8. Experimental Validation

### 8.1 Test Lens

**Nikon AI Nikkor 85 mm f/2S**

| Parameter | Value |
|-----------|-------|
| Focal length | 85 mm |
| Working f-number | f/2.0 |
| Elements | 13 lenses (14 surfaces) |
| Prescription format | Zemax ZMX |

This lens was chosen for its well-documented optical design and significant SA at full aperture, making it a challenging test case for colour fringe prediction.

### 8.2 Measured Aberration Data

Aberration curves were extracted over 31 wavelengths (400–700 nm, 10 nm step):

| Quantity | Range | Description |
|----------|-------|-------------|
| $\rho_{sa}(\lambda)$ | 11.8 – 18.4 µm (mean 16.8 µm) | RMS residual SA spot radius |
| $c_3(\lambda)$ | 24.5 – 46.6 µm | Primary SA polynomial coefficient |
| $c_5(\lambda)$ | −101.7 – −57.7 µm | Secondary SA polynomial coefficient |
| $W_{040}(\lambda)$ | 1.871 – 3.141 µm OPD | Seidel SA wavefront coefficient |

The substantial variation of $c_3$ and $c_5$ with wavelength confirms the presence of spherochromatism and motivates the wavelength-dependent SA modelling in the MZD approach.

### 8.3 ESF Transition Width Results

ESF transition width is defined as the number of spatial samples where $0.05 < \mathrm{ESF} < 0.95$, a purely optical quantity independent of tone mapping. Results from FFT diffraction PSF (Sony A900 sensor model):

| Defocus z (µm) | R Transition | G Transition | B Transition |
|-----------------|-------------|-------------|-------------|
| −800 | 301 | 270 | 280 |
| −400 | 144 | 116 | 123 |
| −100 | 56 | 44 | 44 |
| −50 | 47 | 54 | 48 |
| 0 | 48 | 73 | 64 |
| +50 | 62 | 92 | 83 |
| +100 | 82 | 112 | 102 |
| +400 | 202 | 234 | 221 |

**Key observations:**
- **R channel** reaches best focus near $z \approx -50$ µm (narrowest transition = 47).
- **G and B channels** reach best focus near $z \approx -100$ µm (transition = 44).
- The ~50 µm offset between R and G/B best-focus positions directly measures the CHL-induced focal shift.
- At $z < 0$ (toward the lens), R transition width exceeds G and B, indicating longer R focal length — consistent with secondary spectrum sign.

### 8.4 CFW Sensitivity Analysis

Controlled-variable experiments reveal the following patterns:

**Exposure sensitivity:**
- Moderate exposure ($\alpha \approx 4$–8) maximises CFW by amplifying inter-channel differences in the transition zone.
- Very high exposure ($\alpha > 16$) saturates all channels to a hard step, compressing the fringe region and paradoxically reducing CFW.
- This **non-monotonic** CFW-vs-exposure relationship is physically meaningful and correctly captured by all model tiers.

**Aberration input fidelity (Gaussian PSF, fixed):**

| CHL Source | SA Included | Effect on CFW |
|------------|-------------|--------------|
| Paraxial | No | Baseline — secondary spectrum only |
| RoRi | No | Adds spherochromatism; shifts CFW peak position |
| RoRi | Yes | Adds residual SA blur; widens CFW vs. z profile |

**PSF model comparison (RoRi CHL + SA, fixed):**

| Model | CFW Characteristics |
|-------|-------------------|
| Pillbox | Sharpest cutoff, lowest CFW (conservative) |
| Gaussian | Softer tails, moderate CFW |
| MZD | Pupil-resolved SA, captures asymmetric PSF structure |
| Ray-fan (geometric) | Ground truth for geometric optics, no PSF assumption |

The MZD model closely tracks the ray-fan ground truth while maintaining analytic-kernel speed, validating its design as a practical replacement for both the Pillbox and Gaussian models when SA is significant.

---

## 9. Performance Comparison

| Method | Typical Speed | Relative Speedup | Applicable Scenario |
|--------|--------------|-------------------|-------------------|
| FFT diffraction PSF | ~1 s / ESF | 1× | Ground truth, final validation |
| Geometric pupil integral | ~10 ms / ESF | ~100× | Accurate geometric reference |
| Ray-fan extrapolation | ~0.1–1 ms / ESF | ~1,000× | Interactive z-sweeps, parameter studies |
| Analytic ESF (Pillbox/Gauss/MZD) | <0.01 ms / ESF | ~100,000× | Real-time exploration, optimisation |

**Memory footprint:**

| Data Structure | Size |
|---------------|------|
| Monochromatic ESF cache (25 z × 11 wl × 801 pts) | ~1.7 MB |
| Polychromatic ESF cache (25 z × 3 ch × 801 pts) | ~469 KB |
| Pre-traced ray fan (32 ρ × 31 wl) | ~16 KB |
| Sensor spectral weights (3 × 31) | < 1 KB |

The two-stage FFT baking approach reduces total computation time by 3× for a single camera and 6× when evaluating multiple sensor models.

---

## 10. Conclusions

We have presented a multi-fidelity framework for predicting chromatic colour fringe width in photographic lenses, spanning from sub-microsecond analytic models to full FFT diffraction. The main contributions are:

1. **Colour Fringe Width (CFW) metric** — a quantitative, perceptually grounded measure of fringe visibility that accounts for spectral weighting, tone mapping, and display gamma.

2. **Multi-Zone Defocus (MZD) model** — an analytic ESF kernel that resolves pupil-dependent spherical aberration via Gauss-Legendre integration of arcsin ring ESFs. MZD matches the geometric ray-fan ground truth at five orders of magnitude lower cost than FFT diffraction.

3. **RoRi aberration extraction with SA polynomial fitting** — providing wavelength-dependent $c_3(\lambda)$ and $c_5(\lambda)$ coefficients that capture both primary and secondary spherical aberration without full pupil integration.

4. **Two-stage PSF baking** — decoupling sensor-independent monochromatic ESF computation from sensor-specific weighting, enabling rapid camera model comparison without re-running FFTs.

5. **Controlled-variable validation** against a Nikon 85 mm f/2S lens, demonstrating the non-monotonic CFW-vs-exposure relationship and the importance of spherochromatism in fast lens designs.

The framework enables lens designers to rapidly evaluate colour fringe performance across defocus, exposure, and sensor configurations, using the appropriate fidelity level for each stage of the design process.

---

## Appendix A. Formula Reference

| Quantity | Expression |
|----------|-----------|
| Paraxial CHL | $\mathrm{CHL}(\lambda) = [f_2'(\lambda) - f_2'(\lambda_\text{ref})] \times 10^3$ µm |
| CHL blur radius | $\rho_\text{CHL} = \|z - \mathrm{CHL}(\lambda)\| / \sqrt{4N^2-1}$ |
| Total blur (quadrature) | $\rho = \sqrt{\rho_\text{CHL}^2 + \rho_{sa}^2}$ |
| RoRi weighting | $\mathrm{RoRi} = \sum w_i \cdot \mathrm{SK}(\rho_i) / 42$ |
| SA blur (RMS) | $\rho_{sa} = \sqrt{\sum w_i \, y_\text{spot}^2 / \sum w_i}$ |
| SA polynomial | $\mathrm{TA}_\text{SA}(\rho) = c_3 \rho^3 + c_5 \rho^5$ |
| W040 coefficient | $W_{040} = -\mathrm{TA}_\text{marginal} / (8N)$ µm OPD |
| Pillbox ESF | $\frac{1}{2}(1 + x/\rho)$, linear on $[-\rho, \rho]$ |
| Gaussian ESF | $\frac{1}{2}[1 + \mathrm{erf}(x / (\sqrt{2} \cdot 0.5\rho))]$ |
| MZD ring ESF | $\arcsin(\mathrm{clip}(x/R, -1, 1))/\pi + 1/2$ |
| MZD blur radius | $R_k = \sqrt{(\rho_k \, \rho_\text{CHL})^2 + (2 \rho_k^3 \, \rho_{sa})^2}$ |
| Geometric integral | $\int_0^1 [\arcsin(x/R(\rho))/\pi + 1/2] \; 2\rho \, d\rho$ |
| Ray-fan extrapolation | $R(\rho; z, \lambda) = \|\mathrm{TA}_0 + m \cdot z\|$ |
| FFT pixel pitch | $dx = \lambda N / Q$, where $Q = \text{grid size} / (\text{num rays} - 1)$ |
| Spectral integration | $\mathrm{ESF}_c = \sum_j \hat{w}_j \cdot \mathrm{ESF}_\text{mono}(x; z, \lambda_j)$ |
| Tone mapping | $I = (\tanh(\alpha \cdot \mathrm{ESF}) / \tanh(\alpha))^\gamma$ |
| CFW definition | $\mathrm{CFW}(z) = \sum_x \mathbf{1}[\max(\|R-G\|, \|R-B\|, \|G-B\|) > \delta]$ |

---

*Report based on the ChromFringe framework (2026-03-11).*

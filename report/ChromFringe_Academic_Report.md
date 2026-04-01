# Multi-Level Numerical Modelling of Chromatic Fringe Width in Photographic Lenses

---

## Abstract

Residual longitudinal chromatic aberration (LCA) in photographic lenses produces visible colour fringes at high-contrast edges. This work presents a hierarchical numerical framework for predicting the **Colour Fringe Width (CFW)** — the spatial extent of perceptible colour fringes caused by the wavelength-dependent defocus of the R, G, and B sensor channels. Three progressively refined models are developed: (1) analytic parametric ESF models using paraxial or RoRi chromatic aberration curves, (2) a geometric ray-fan method based on precomputed transverse aberrations with Gauss–Legendre pupil integration, and (3) an FFT Fraunhofer diffraction PSF serving as ground truth. The models are validated against a Nikon AI Nikkor 85 mm f/2S test lens, demonstrating that the ray-fan method achieves geometric-optics-level accuracy at ~1000× the speed of the FFT approach, while the analytic models offer a further ~100× speedup at the cost of simplified aberration representation. The role of spherochromatism and residual spherical aberration in fringe formation is quantified, and a perceptual tone-mapping pipeline is incorporated to bridge the gap between optical simulation and visible fringe detection.

---

## 1. Introduction

### 1.1 Physical Background

Achromatic photographic lenses, after correcting primary chromatic aberration, still exhibit **secondary spectrum**: the residual variation of focal length with wavelength across the visible band. When the camera is focused at a particular image plane, each wavelength forms a blur circle of different radius, causing the R, G, and B sensor channels to record edge spread functions (ESFs) of different widths. At high-contrast boundaries, this inter-channel difference manifests as visible colour fringes.

The signal chain from scene to display can be summarised as:

$$\text{Scene (knife-edge)} \;\xrightarrow{D_{65}}\; \text{Illuminant} \;\xrightarrow{\mathrm{CHL}(\lambda),\,\mathrm{SA}(\lambda)}\; \text{Lens} \;\xrightarrow{S_c(\lambda)}\; \text{Sensor (RGB)} \;\xrightarrow{\gamma,\,\alpha}\; \text{Display}$$

where CHL denotes longitudinal chromatic aberration and SA denotes spherical aberration — both wavelength-dependent.

### 1.2 Motivation

Although chromatic fringe is a well-known artefact in photography, quantitative prediction of fringe width from first principles remains challenging. Existing approaches either rely on purely paraxial models (ignoring spherochromatism) or require full wave-optics simulation (computationally prohibitive for parameter sweeps). This work bridges the gap by developing a hierarchy of models with explicit accuracy–speed trade-offs, enabling both high-fidelity validation and rapid design-space exploration.

### 1.3 Contributions

1. A complete spectral-to-perceptual simulation chain for colour fringe prediction, from lens prescription to displayed pixel values.
2. The **RoRi aperture-weighted CHL model**, which accounts for spherochromatism via real-ray tracing at five equal-area pupil zones.
3. A **geometric ray-fan ESF method** using Gauss–Legendre pupil quadrature and linear extrapolation, achieving ~1000× speedup over FFT diffraction with negligible accuracy loss.
4. A **two-stage baking architecture** that decouples monochromatic PSF computation from sensor spectral weighting, enabling efficient multi-camera comparison.
5. A physically motivated **CFW metric** with a three-condition fringe visibility classifier.

### 1.4 Modelling Hierarchy

| Level | Method | Speed | Accuracy |
|-------|--------|-------|----------|
| 0 | FFT diffraction PSF (ground truth) | ~1 s/ESF | Includes diffraction effects |
| 1 | Ray-fan Gauss–Legendre extrapolation | <1 ms/ESF | Geometrically exact, error $O((z/f')^2)$ |
| 2 | Analytic ESF (Disc / Gaussian) | <0.01 ms/ESF | Parametric approximation |

---

## 2. Mathematical Framework

The central quantity in this work is the **polychromatic ESF** — the edge spread function that a real colour channel records. Its construction requires three ingredients: a spectral model (source + sensor), a chromatic defocus model (CHL), and a monochromatic aberration model (SA). This section derives each ingredient and assembles them into the full simulation chain.

### 2.1 Spectral Model — Sensor Response and Source Weighting

The colour response of an imaging system is jointly determined by three factors:

1. **Source spectral power distribution** $D(\lambda)$: the relative radiant power of the illumination at each wavelength. This work uses the CIE D65 standard daylight spectrum.
2. **Sensor quantum efficiency** $S_c(\lambda)$: the response efficiency of each Bayer R/G/B channel to photons at wavelength $\lambda$. Built-in data for Sony A900 and Nikon D700 are provided.
3. **Monochromatic PSF** $\text{PSF}_\text{mono}(r;\lambda,z)$: each wavelength focuses at a different axial position and carries a different spherical aberration profile, producing a wavelength-dependent blur kernel.

#### 2.1.1 Channel Spectral Weights

For channel $c \in \{R, G, B\}$, define the **spectral density** (energy-normalised sensor–source product):

$$\hat{g}_c(\lambda) = k_c \cdot S_c(\lambda) \cdot D_{65}(\lambda)$$

where the normalisation constant $k_c$ satisfies:

$$k_c = \frac{1}{\displaystyle\int S_c(\lambda) \cdot D_{65}(\lambda) \, d\lambda}$$

so that $\int \hat{g}_c(\lambda)\,d\lambda = 1$. This ensures that a perfectly focused flat-spectrum edge produces unity response in every channel (ESF transitions from 0 to 1).

In practice, wavelengths are sampled at 10 nm intervals over 400–700 nm (31 points), and integrals are evaluated by the trapezoidal rule.

#### 2.1.2 Effective Spectral Windows

After multiplication by D65 and energy normalisation, the spectral weights $\hat{g}_c(\lambda)$ define each channel's "spectral window":

| Channel | Effective band (>10% peak) | Peak $\lambda$ | Role in fringing |
|---------|---------------------------|----------------|------------------|
| B | 400–500 nm | ~460 nm | Samples the high-CHL blue wing; drives blue-side fringe |
| G | 490–590 nm | ~540 nm | Centred near CHL minimum; sharpest ESF at $z = 0$ |
| R | 540–680 nm | ~580 nm | Samples the red CHL wing; drives red-side fringe |

The B and G channels overlap in the 490–510 nm region; G and R overlap in 540–590 nm. In these overlap bands, two channels respond to the same wavelengths with similar CHL, so their ESFs are similar. The visible fringe occurs where channels **do not** overlap: B-only (400–480 nm) and R-only (600–680 nm).

**Camera dependence.** The Nikon D700 R channel has a higher normalised peak (~0.021) than the Sony A900 (~0.018), giving the D700 slightly more weight to long wavelengths. This shifts the R-channel ESF centroid and alters the R–B fringe balance.

#### 2.1.3 Polychromatic ESF Assembly

The polychromatic ESF for channel $c$ is the weighted superposition of monochromatic ESFs across all wavelengths:

$$\boxed{\text{ESF}_c(x; z) = \sum_{j=1}^{N_\lambda} \hat{g}_c(\lambda_j) \cdot \text{ESF}_{\text{mono}}(x; \lambda_j, z)}$$

Each wavelength's contribution is scaled by its spectral weight in that channel. In practice, subsampling with `wl_stride = 3` reduces from 31 to 11 wavelengths with negligible error.

The remaining subsections derive each component entering the monochromatic ESF: chromatic defocus (§2.2–2.3), spherical aberration (§2.4), and the PSF/ESF kernel models (§2.5–2.7).

### 2.2 Chromatic Defocus Models

#### 2.2.1 Paraxial Ray Tracing

Paraxial theory assumes that the angle between rays and the optical axis is extremely small ($\sin\theta \approx \theta$), linearising Snell's law. For an optical system consisting of $K$ refracting surfaces, the recurrence relations for paraxial marginal ray tracing are:

$$n'_k u'_k = n_k u_k - y_k \phi_k$$

$$y_{k+1} = y_k + u'_k \cdot d_k$$

where:

| Symbol | Meaning |
|--------|---------|
| $y_k$ | Ray height at surface $k$ |
| $u_k$, $u'_k$ | Ray slope before/after refraction |
| $n_k$, $n'_k$ | Refractive index before/after surface $k$ |
| $\phi_k = (n'_k - n_k) / R_k$ | Optical power of surface $k$ |
| $R_k$ | Radius of curvature of surface $k$ |
| $d_k$ | Spacing from surface $k$ to surface $k+1$ |

Tracing a unit-height, zero-slope marginal ray ($y_0 = 1$, $u_0 = 0$) from a starting position 1 mm in front of the first refracting surface at each wavelength yields the back focal length:

$$\text{BFL}(\lambda) = -\frac{y_\text{last}}{u_\text{last}} \quad (\text{mm})$$

#### 2.2.2 Paraxial CHL Definition

The **paraxial longitudinal chromatic aberration** is defined as the back focal length difference relative to a reference wavelength:

$$\mathrm{CHL}_\text{par}(\lambda) = \left[\text{BFL}(\lambda) - \text{BFL}(\lambda_\text{ref})\right] \times 10^3 \quad (\mu\mathrm{m})$$

CHL > 0 indicates the focal point at wavelength $\lambda$ lies farther from the lens than the reference. For a positive lens with normal dispersion, blue light has CHL < 0 and red light has CHL > 0.

#### 2.2.3 Paraxial CHL Limitations

Paraxial CHL reflects only first-order dispersion effects, assuming the focal position is the same across all pupil zones. In reality, spherical aberration also varies with wavelength (**spherochromatism**), causing different pupil zones to have different focal shifts. This is precisely the problem addressed by the RoRi model.

#### 2.2.4 CHL-Induced Blur Radius

In the geometric optics framework, when the image plane is at position $z$ (relative to the reference focal plane), light at wavelength $\lambda$ forms a blur circle of radius:

$$\boxed{\rho_\text{CHL}(z,\lambda) = \frac{|z - \mathrm{CHL}(\lambda)|}{\sqrt{4F_{\#}^2 - 1}}}$$

**Derivation.** For a lens with f-number $F_{\#} = f/D$, the marginal ray converges to the focal point at half-angle $u$ satisfying the exact relation:

$$\sin u = \frac{D/2}{\sqrt{f^2 + (D/2)^2}} = \frac{1}{\sqrt{4F_{\#}^2 + 1}} \approx \frac{1}{2F_{\#}}$$

The approximation $\sin u \approx 1/(2F_{\#})$ holds well for $F_{\#} \gtrsim 1$ and is equivalent to the paraxial expression $u \approx \arctan(1/(2F_{\#}))$.

When the image plane is shifted by $\Delta z = z - \mathrm{CHL}(\lambda)$ from the wavelength's focal point, the marginal ray intersects the observation plane at a radius:

$$\rho = |\Delta z| \cdot \tan u$$

Using $\sin u = 1/(2F_{\#})$ to express $\tan u$ exactly:

$$\tan u = \frac{\sin u}{\cos u} = \frac{\sin u}{\sqrt{1 - \sin^2 u}} = \frac{1/2F_{\#}}{\sqrt{1 - \dfrac{1}{4F_{\#}^2}}} = \frac{1/2F_{\#}}{\dfrac{\sqrt{4F_{\#}^2-1}}{2F_{\#}}} = \frac{1}{\sqrt{4F_{\#}^2-1}}$$

Substituting back:

$$\rho = \frac{|\Delta z|}{\sqrt{4F_{\#}^2 - 1}}$$

### 2.3 RoRi Aperture-Weighted Method

#### 2.3.1 Motivation

Paraxial CHL ignores **spherochromatism** caused by spherical aberration: the magnitude of SA varies with wavelength, causing different aperture zones to have different effective focal lengths. The RoRi method estimates the "equivalent best-focus" by taking a weighted average of real-ray back-focal intercepts across multiple aperture zones.

#### 2.3.2 Pupil Zoning

The pupil is divided into 5 equal-area annular zones. If the normalised pupil radius is $r \in [0, 1]$, the equal-area zone boundaries are:

$$r_i = \sqrt{i/4}, \quad i = 0, 1, 2, 3, 4$$

The representative pupil coordinates are:

| Zone | $r$ | Value |
|------|-----|-------|
| Centre | $0$ | $0$ |
| Ring 1 | $\sqrt{1/4}$ | $0.500$ |
| Ring 2 | $\sqrt{1/2}$ | $0.707$ |
| Ring 3 | $\sqrt{3/4}$ | $0.866$ |
| Edge | $1$ | $1.000$ |

The pupil area element is $dA = 2\pi r\,dr$, so energy is uniformly distributed according to $r^2$. The choice $r = \sqrt{i/4}$ ensures each annular zone intercepts the same luminous flux.

#### 2.3.3 The Integral Being Approximated

Both RoRi variants approximate the same physical quantity: the **aperture-area-weighted mean back-focal intercept**:

$$\mathrm{RoRi}(\lambda) = \int_0^1 \mathrm{SK}(r,\lambda)\cdot 2r\,dr$$

where the factor $2r$ is the area weight of a thin annulus of normalised radius $r$ (unit-normalised pupil area: $\int_0^1 2r\,dr = 1$).

#### 2.3.4 Computing Focal Intercept SK(r)

For $r > 0$, a tangential ray is traced to the image plane, obtaining the ray's position and direction cosines:

| Symbol | Definition | Meaning |
|--------|-----------|---------|
| $y$ | — | $y$-coordinate of the ray at the image plane (mm) |
| $M$ | $\cos\beta = \sin U$ | $y$-component of the direction cosine |
| $N$ | $\cos\gamma = \cos U$ | $z$-component (along the optical axis) of the direction cosine |

The ray is extrapolated from the image plane to its intersection with the optical axis ($y = 0$); the axial intercept is:

$$SK(r, \lambda) = -\frac{y \cdot N}{M} \quad (\text{mm})$$

**Geometric interpretation.** The slope of the ray at the image plane is $dy/dz = M/N$. Starting from $(y, 0)$, the $z$-displacement needed to reach $y = 0$ is $\Delta z = -y / (M/N) = -yN/M$. This is the longitudinal focal shift of that ray.

For $r = 0$ (the paraxial limit), the real-ray trace is degenerate; the back-focal intercept is obtained from the paraxial marginal-ray trace:

$$\mathrm{SK}(0,\lambda) = -\frac{y_\mathrm{par}}{u_\mathrm{par}}$$

#### 2.3.5 RoRi: Equal-Area Trapezoidal Rule

**Quadrature construction.** Under the substitution $u = r^2$, the integral becomes $\int_0^1 \mathrm{SK}(\sqrt{u},\lambda)\,du$, and the nodes are equally spaced in $u$-space. Applying the composite trapezoidal rule on four equal sub-intervals of length $\Delta u = 0.25$ yields the weights:

$$w_i^{(1)} = \Delta u \cdot \{{\tfrac{1}{2}}, 1, 1, 1, {\tfrac{1}{2}}\} = 0.25 \cdot \{0.5, 1, 1, 1, 0.5\}$$

Rescaling to integer form and accounting for the area-weight factor $2r_i$ absorbed into the quadrature produces the published weights $\{1,\;12.8,\;14.4,\;12.8,\;1\}$ summing to 42:

$$\mathrm{RoRi}(\lambda) = \frac{1 \cdot SK(0) + 12.8 \cdot SK(\sqrt{0.25}) + 14.4 \cdot SK(\sqrt{0.5}) + 12.8 \cdot SK(\sqrt{0.75}) + 1 \cdot SK(1)}{42}$$

#### 2.3.6 Predictive Advantage

Because the nodes span the full range $r \in [0,1]$, including the paraxial limit $r=0$ and the marginal ray $r=1$, RoRi explicitly captures the extreme focal positions of the lens. The paraxial contribution anchors the CHL estimate to the secondary spectrum, while the marginal ray ensures the aperture edge is represented. This broad coverage preserves the full chromatic spread of $\mathrm{SK}(r,\lambda)$, which benefits prediction of fringe visibility at low tone-curve exposures where even small channel differences cross the detection threshold.

#### 2.3.7 RoRi CHL Curve

The CHL curve is computed by subtracting the reference-wavelength value:

$$\mathrm{CHL}_\mathrm{RoRi}(\lambda) = \left[\mathrm{RoRi}(\lambda) - \mathrm{RoRi}(\lambda_\mathrm{ref})\right] \times 10^3 \quad (\mu\mathrm{m})$$

### 2.4 Residual Spherical Aberration (SA) Blur

Spherical aberration causes different aperture zones to focus at different axial positions; even at the RoRi best-focus plane, residual blur remains.

#### 2.4.1 Lateral Blur Computation

At the RoRi focal plane, the lateral displacement of each ray at pupil height $r_i$ (small-angle approximation):

$$y_\text{spot}(r_i, \lambda) = \frac{[\mathrm{SK}(r_i, \lambda) - \mathrm{RoRi}(\lambda)] \cdot r_i}{\sqrt{4F_{\#}^2 - 1}} \times 10^3 \quad (\mu\mathrm{m})$$

#### 2.4.2 RMS Residual Spot

$$\boxed{\rho_\text{SA}(\lambda) = \sqrt{\frac{\sum_i w_i \cdot y_\text{spot}^2(r_i, \lambda)}{\sum_i w_i}}}$$

where $w_i$ are the RoRi quadrature weights.

#### 2.4.3 Total Blur Radius (Quadrature Addition)

CHL blur and SA blur are combined via quadrature (area addition on the blur disc):

$$\boxed{\rho(z, \lambda) = \sqrt{\rho_\text{CHL}(z, \lambda)^2 + \rho_\text{SA}(\lambda)^2}}$$

### 2.5 Monochromatic ESF Models

#### 2.5.1 Uniform Circular Disc (Geometric PSF)

**Physical assumption.** The PSF is a 2D disc uniformly distributed within radius $\rho$ (geometric blur circle), with intensity distribution:

$$\mathrm{PSF}_\text{disc}(\mathbf{r}) = \frac{1}{\pi\rho^2} \cdot \mathbf{1}[|\mathbf{r}| \leq \rho]$$

**ESF integration.** The ESF is the projection of the PSF along the $y$-direction followed by cumulative integration (equivalent to convolution of a half-plane with the PSF). For the uniform disc, the line spread function (LSF) is the chord length at each $x$:

$$\mathrm{LSF}(x) = \frac{2}{\pi\rho^2}\sqrt{\rho^2 - x^2}, \quad |x| \leq \rho$$

This is a **semicircular** (not uniform) 1D profile. Integrating gives the analytic ESF:

$$\mathrm{ESF}_\text{disc}(x, \rho) = \begin{cases} 0 & x \leq -\rho \\ \dfrac{1}{2} + \dfrac{1}{\pi}\!\left[\arcsin\dfrac{x}{\rho} + \dfrac{x}{\rho}\sqrt{1 - \dfrac{x^2}{\rho^2}}\right] & -\rho < x < \rho \\ 1 & x \geq \rho \end{cases}$$

**Derivation.** Substituting $t = \rho\sin\phi$:

$$\int_{-\rho}^{x}\!\frac{2}{\pi\rho^2}\sqrt{\rho^2-t^2}\,dt = \frac{2}{\pi}\int_{-\pi/2}^{\arcsin(x/\rho)}\cos^2\phi\,d\phi = \frac{1}{2} + \frac{1}{\pi}\!\left[\arcsin\frac{x}{\rho} + \frac{x}{\rho}\sqrt{1-\frac{x^2}{\rho^2}}\right]$$

#### 2.5.2 Gaussian PSF

**Physical assumption.** The PSF is a 2D circularly symmetric Gaussian distribution with standard deviation $\sigma \approx 0.5\rho$ (corresponding to the blur circle radius):

$$\mathrm{PSF}_\text{gauss}(\mathbf{r}) = \frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{|\mathbf{r}|^2}{2\sigma^2}\right)$$

**ESF.** The 1D ESF of a Gaussian PSF is the error function (CDF of the standard normal):

$$\boxed{\mathrm{ESF}_\text{gauss}(x, \rho) = \frac{1}{2}\left[1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2}\,\sigma}\right)\right], \quad \sigma = 0.5\rho}$$

The Gaussian model has soft tails compared to the uniform disc, providing a closer approximation to the real PSF (smooth approximation of diffraction edge ringing).

#### 2.5.3 Analytic ESF: Spectral Loop

For the analytic models, the spectral loop computes monochromatic blur radii on the fly and accumulates the weighted ESF. For each wavelength $j$:

1. Compute CHL blur: $\rho_\text{CHL} = |z - \text{CHL}(\lambda_j)| / \sqrt{4F_{\#}^2 - 1}$
2. Combine with SA: $\rho = \sqrt{\rho_\text{CHL}^2 + \rho_\text{SA}(\lambda_j)^2}$
3. Evaluate kernel: $\text{ESF}_\text{mono}(x; \rho)$ using disc or Gaussian formula
4. Accumulate: $\text{acc} \mathrel{+}= \hat{g}_c(\lambda_j) \cdot \text{ESF}_\text{mono}(x; \rho)$

This loop is JIT-compiled via Numba for microsecond-level execution.

### 2.6 Geometric Ray-Fan ESF Method

#### 2.6.1 Motivation

The FFT-PSF method (§2.7) requires rebuilding an optical model copy and tracing all rays at each defocus position $z$, taking ~1 s per ESF. For CFW curves that scan hundreds of $z$ values, this is too slow. The core idea: **ray trajectories are straight lines near the image plane** — trace once at $z=0$, then extrapolate to any $z$.

#### 2.6.2 Physical Picture — Knife-Edge Test of a Defocused Spot

The Edge Spread Function (ESF) measures the energy fraction transmitted past a knife edge as a function of the edge position $x$. Physically, a lens images a point source into a blur spot on the image plane (due to defocus, chromatic aberration, and spherical aberration). Placing a straight knife edge perpendicular to $x$ at position $x_0$ blocks all light with $x < x_0$. The ESF at $x_0$ is the fraction of the spot's total energy that falls on the bright side ($x \geq x_0$):

- $x_0 \ll -\rho$ (edge far left): all light passes → ESF = 1
- $x_0 \gg +\rho$ (edge far right): all light blocked → ESF = 0
- $x_0 = 0$ (edge at centre): approximately half passes → ESF ≈ 0.5

The shape of the transition from 1 to 0 depends on the intensity distribution within the blur spot.

#### 2.6.3 Pupil Decomposition into Concentric Rings

The filled pupil ($r \in [0, 1]$) can be decomposed into infinitely thin concentric rings, each at a specific normalised pupil radius $r$. Due to the rotational symmetry of the lens, rays from a ring at pupil radius $r$ form a ring of radius $R(r)$ on the image plane, where $R$ depends on the aberrations. Different pupil radii produce rings of different image-plane radii because of spherical aberration.

The ESF of the full spot equals the area-weighted sum of individual ring ESFs. Since the pupil area element is $dA = r\,dr\,d\theta$, the contribution from each ring is proportional to $r$:

$$\text{ESF}(x) = \int_0^1 f\!\left(\frac{x}{R(r)}\right) \cdot 2r \, dr$$

where $f(x/R)$ is the knife-edge response of a single ring of radius $R$, derived in §2.6.7.

#### 2.6.4 Gauss–Legendre Quadrature — From Continuous Integral to Discrete Sum

The continuous pupil integral cannot be evaluated analytically because $R(r)$ depends on the full aberration profile. We approximate it using **Gauss–Legendre (GL) quadrature**, which replaces the integral with a weighted sum over $K$ optimally chosen sample points:

$$\int_0^1 g(r) \, dr \;\approx\; \sum_{k=1}^{K} W_k \cdot g(r_k)$$

The GL nodes $\xi_k$ and weights $W_k$ are defined on the standard interval $[-1, 1]$ and mapped to $[0, 1]$:

$$r_k = \frac{\xi_k + 1}{2}$$

**Key property:** $K$ GL nodes can exactly integrate any polynomial of degree $\leq 2K - 1$. Since transverse aberration is dominated by primary spherical aberration ($\propto r^3$) with higher-order corrections ($r^5$, $r^7$), even $K = 16$ provides excellent accuracy; $K = 32$ is conservative.

Unlike the RoRi method (§2.3) which uses 5 fixed equal-area pupil zones with empirical weights to estimate the best focal plane, GL quadrature is a general-purpose numerical integration scheme with mathematically optimal node placement and weights. GL nodes are **infinitely thin sample points** (not finite-width annular zones); the weights $W_k$ encode the contribution of each point to the integral without assigning a physical width.

#### 2.6.5 Precomputation: Ray Fan

For each GL node $r_k$ and each wavelength $\lambda_j$ (31 wavelengths, 400–700 nm), a single tangential ray is traced to the nominal image plane ($z = 0$, defined by the Zemax prescription), recording two quantities:

- **Transverse Aberration (TA)** — the ray's $y$-offset from the chief ray at the image plane:

$$TA_0(r_k, \lambda_j) = y_{\text{image}} \times 10^3 \quad (\mu\text{m})$$

- **Ray slope** — the ray's propagation direction at the image plane:

$$m(r_k, \lambda_j) = \frac{M}{N}$$

where $M$ and $N$ are the $y$- and $z$-components of the direction cosine. This ratio is the ray's slope $dy/dz$ in the $yz$ plane. For degenerate cases ($|N| < 10^{-10}$), a geometric approximation is used: $m = -r / \sqrt{4F_{\#}^2 - 1}$.

Precomputation cost: $K \times N_\lambda$ rays (e.g., $32 \times 31 = 992$ rays, covering all wavelengths and all three channels).

#### 2.6.6 Linear Extrapolation to Arbitrary Defocus

After exiting the last lens surface, each ray travels in a straight line. Therefore, if the image plane is shifted by $z$ µm from the nominal position, the ray's transverse position changes linearly:

$$\boxed{R(r_k, \lambda_j;\, z) = \left|TA_0(r_k, \lambda_j) + m(r_k, \lambda_j) \cdot z\right| \quad (\mu\text{m})}$$

The absolute value is taken because $R$ represents the radial distance from the optical axis. **No additional ray tracing is required** — changing $z$ is a single multiply-and-add operation.

**Error analysis.** The extrapolation assumes a straight-line trajectory beyond the last surface. The error is $O\!\left((z/f')^2\right)$, arising from the curvature of the actual wavefront. For an 85 mm lens with $z \leq 800\;\mu\text{m}$: $z/f' = 800/(85\times10^3) \approx 10^{-5}$, giving relative error $\approx 0.01\%$.

#### 2.6.7 Knife-Edge Response of a Single Ring

Each GL node $r_k$ at a given wavelength $\lambda_j$ and defocus $z$ produces a ring of radius $R = R(r_k, \lambda_j; z)$ on the image plane. Due to rotational symmetry, light is uniformly distributed along this ring. A knife edge at position $x$ transmits the fraction of the ring's circumference lying on the bright side ($x$-coordinate $\geq x_0$).

Parameterise the ring as $(R\cos\theta,\; R\sin\theta)$ for $\theta \in [0, 2\pi)$. The bright-side condition $R\cos\theta \geq x$ requires $|\theta| \leq \alpha$ where $\alpha = \arccos(x/R)$. The transmitted fraction is:

$$f = \frac{\alpha}{\pi} = \frac{\arccos(x/R)}{\pi}$$

Using the identity $\arccos(t) = \pi/2 - \arcsin(t)$:

$$\boxed{f(x, R) = \frac{1}{\pi}\arcsin\!\left(\frac{x}{R}\right) + \frac{1}{2}, \quad |x| \leq R}$$

Boundary conditions: $f(-R) = 0$ (all blocked), $f(0) = 0.5$ (half transmitted), $f(+R) = 1$ (all transmitted).

**Ring ESF vs. filled-disc ESF.** A filled uniform disc of radius $R$ has a different ESF that includes an additional $\frac{x}{R}\sqrt{1 - x^2/R^2}$ term (see §2.5.1). The ring formula is applied to each GL node individually, and the filled-spot ESF is recovered by integrating over the pupil in §2.6.8. The two approaches are mathematically equivalent: integrating ring ESFs weighted by $r\,dr$ over $r \in [0,1]$ yields the filled-disc ESF when all rings share the same radius $R$; when they differ (due to spherical aberration), the ring decomposition correctly captures the non-uniform radial intensity profile.

#### 2.6.8 Pupil Integration to Assemble the Polychromatic ESF

Combining all GL nodes (pupil rings) and all wavelengths with their spectral weights:

$$\boxed{\text{ESF}_c(x;\, z) = \sum_{j=1}^{N_\lambda} \hat{g}_c(\lambda_j) \sum_{k=1}^{K} W_k \cdot r_k \cdot f\!\left(\frac{x}{R(r_k, \lambda_j;\, z)}\right)}$$

where:

| Symbol | Meaning |
|--------|---------|
| $\hat{g}_c(\lambda_j)$ | Normalised spectral weight for channel $c$ at wavelength $\lambda_j$ |
| $W_k$ | GL quadrature weight for node $k$ |
| $r_k$ | Normalised pupil radius (Jacobian factor from $dA = r\,dr\,d\theta$) |
| $R(r_k, \lambda_j; z)$ | Image-plane ring radius at defocus $z$ (§2.6.6) |
| $f(\cdot)$ | Ring knife-edge response (§2.6.7) |

The factor $r_k$ ensures that outer rings (which intercept more light due to their larger area) contribute proportionally more energy. This area weighting, combined with the ring ESF formula, correctly recovers the filled-spot response.

**Accuracy.** 32-node GL quadrature gives ESF error < 0.1% for smooth aberration profiles. Convergence tests show that even 16 nodes suffice for practical CFW accuracy (see §4.3).

### 2.7 FFT Fraunhofer Diffraction PSF

#### 2.7.1 Physical Principle

Under the Fraunhofer diffraction approximation, the PSF is the squared modulus of the Fourier transform of the exit pupil function:

$$\mathrm{PSF}(\mathbf{u}) = \left|\mathcal{F}\left\{P(\mathbf{r})\cdot e^{i2\pi W(\mathbf{r})/\lambda}\right\}\right|^2$$

where $P(\mathbf{r})$ is the pupil transmission function and $W(\mathbf{r})$ is the wavefront error (including CHL-induced defocus and spherical aberration).

#### 2.7.2 Wavelength-Corrected Pixel Pitch

FFT PSF pixel pitch is proportional to wavelength (Fraunhofer diffraction angular resolution):

$$\boxed{dx_j = \frac{\lambda_j \cdot F_{\#}}{Q}, \quad Q = \frac{N_\text{grid}}{N_\text{rays} - 1}}$$

where $Q$ is the oversampling factor. This means PSFs at different wavelengths have different pixel pitches in physical space and must be superimposed in physical coordinates (µm).

#### 2.7.3 Oversampling Factor $Q$ and Nyquist Sampling

The pupil cutoff frequency is $f_\text{cutoff} = 1/(\lambda \cdot F_{\#})$. The Nyquist theorem requires a sampling rate $\geq 2 f_\text{cutoff}$, corresponding to a critical pixel pitch:

$$\Delta x_\text{Nyquist} = \frac{\lambda \cdot F_{\#}}{2}$$

The actual pixel pitch of the FFT PSF is $\Delta x_j = \lambda_j \cdot F_{\#} / Q$, so:

$$\frac{\Delta x_\text{Nyquist}}{\Delta x_\text{actual}} = \frac{Q}{2}$$

| $Q$ Value | Meaning |
|-----------|---------|
| $Q = 2$ | Exactly Nyquist sampling, no aliasing but no margin |
| $Q = 4$ | 2× oversampling, smoother PSF |
| $Q \approx 1.57$ (used in this work) | Below Nyquist, slight aliasing |

The choice of $Q \approx 1.57$ ($N_\text{rays} = 512$, $N_\text{grid} = 800$) is a **field-of-view vs. memory trade-off**: achieving $Q = 2$ would require $N_\text{grid} = 1022$, increasing memory by ~60%. Although $Q < 2$ introduces slight aliasing, the ESF is the cumulative integral of the LSF — a low-pass quantity whose shape is primarily determined by low-frequency components and is insensitive to the high-frequency aliasing error.

#### 2.7.4 Wavefront Reference Strategy

The wavefront reference strategy critically affects chromatic fringe computation:

- **`chief_ray`** (used in this work): The reference sphere is centred on the actual arrival position of the chief ray at the image plane. When the image plane moves, the chief ray position changes, and the chromatic focal shift information is correctly preserved in the OPD. **This is the correct choice for computing chromatic fringes.**

- **`best_fit_sphere`**: The reference sphere is fitted to the actual wavefront on the exit pupil via least squares. This fits away the defocus term, making each wavelength appear nearly perfectly focused — all channel ESFs converge, the chromatic fringe disappears, and **CFW → 0**. This strategy is appropriate for assessing aberration quality but not for chromatic fringe prediction.

#### 2.7.5 Defocus Implementation

Defocus is implemented by modifying the spacing before the last surface (the image plane):

$$t_\text{last} \leftarrow t_\text{last} + \frac{z_\text{defocus}}{1000} \quad (\text{mm})$$

Positive $z_\text{defocus}$ means the image plane moves away from the lens. This changes the defocus term in the wavefront (the $r^2$ term in the OPD).

#### 2.7.6 ESF Construction: PSF → LSF → ESF

$$\mathrm{LSF}(x) = \int_{-\infty}^{+\infty} \mathrm{PSF}(x, y)\,dy$$

$$\mathrm{ESF}(x) = \int_{-\infty}^{x} \mathrm{LSF}(t)\,dt \approx \mathrm{cumsum}(\mathrm{LSF})$$

Cumulative summation is used instead of FFT-based convolution because when the PSF is very narrow relative to the FFT grid, `fftconvolve` truncates at ~0.5.

#### 2.7.7 Physical Coordinate Mapping

Because different wavelengths have different pixel pitches $\Delta x_j$ (§2.7.2), each monochromatic ESF lives on its own physical coordinate grid:

$$x_j[n] = \left(n - \frac{N_\text{grid}}{2}\right) \cdot \Delta x_j \quad (\mu\text{m})$$

All wavelengths' ESFs are interpolated onto a unified physical coordinate axis $x_\mu\text{m}$ and then weighted and superimposed:

$$\text{ESF}_c(x) = \sum_{j=1}^{N_\lambda} \hat{g}_c(\lambda_j) \cdot \text{interp}\!\left[\text{ESF}_j, x_j \to x\right]$$

Outside the interpolation range, the left side is extrapolated as 0 and the right side as 1 (physically corresponding to complete occlusion and complete transmission).

#### 2.7.8 Two-Stage Baking Optimisation

When computing ESFs for multiple channels (R, G, B), a naïve approach would call FFT propagation three times per wavelength, redundantly. The two-stage architecture avoids this:

1. **`bake_wavelength_esfs`** (sensor-independent): Compute all wavelengths' monochromatic ESFs in one pass → output an $(N_\lambda, N_x)$ matrix. Re-run only when the optical system or defocus changes.

2. **`apply_sensor_weights`** (sensor-specific): For each channel, perform matrix multiplication with that channel's spectral weights $\hat{g}_c$:

$$\text{ESF}_c(x) = \hat{\mathbf{g}}_c^T \cdot \mathbf{E}(x)$$

This is a pure linear-algebra operation taking microseconds. Overall speedup: ~3× (single camera), ~6× (two cameras).

**Quantitative comparison.** For the same optical system with 29 defocus steps × 3 channels × 2 cameras (174 ESFs), the two-stage approach requires only $29 \times 11 = 319$ FFTs + 174 weighted sums; the single-step approach would require $174 \times 11 = 1914$ FFTs.

### 2.8 Tone Mapping and Display Pipeline

The linear ESF from §2.1–2.7 represents the raw optical signal. Before assessing fringe visibility, it must pass through a **tone mapping** stage that models the nonlinear response of the camera image processing pipeline and display.

#### 2.8.1 Tone Mapping Curve

This work adopts the $\tanh$ tone curve from the Imatest image quality testing framework:

$$\boxed{T(I;\,F,\,I_\text{max}) = \frac{\tanh\!\left(F \cdot I / I_\text{max}\right)}{\tanh(F)} \cdot I_\text{max}}$$

where $F$ is the exposure slope parameter and $I_\text{max}$ is the maximum intensity. With $I_\text{max} = 1$ (normalised ESF), this simplifies to:

$$T(x;\,\alpha) = \frac{\tanh(\alpha \cdot x)}{\tanh(\alpha)}$$

where $\alpha \equiv F$ is the exposure slope.

**Properties:**

| $\alpha$ | Behaviour |
|----------|-----------|
| $\alpha \to 0$ | $T(x) \to x$ (linear response, no contrast compression) |
| $\alpha = 4$ | Moderate contrast compression (default) |
| $\alpha \to \infty$ | $T(x) \to$ step function (hard clip at $x > 0$) |

The curve satisfies $T(0) = 0$ and $T(1) = 1$ (the $\tanh(\alpha)$ denominator is the normalisation factor ensuring unit-in-unit-out). Near $x = 0$, the slope is $\alpha / \tanh(\alpha) > 1$, amplifying low-intensity differences — this is the mechanism by which higher exposure makes subtle colour fringes visible.

The $\tanh$ tone curve originates from the Imatest optical image quality testing software, where it is used to simulate the nonlinear greyscale response from linear RAW capture to display output. In this work, it serves as a perceptual filter: the fringe is physically present in the linear ESF, but whether it is **visible** depends on how the tone curve maps small inter-channel differences into perceivable brightness differences.

#### 2.8.2 Gamma Correction

After tone mapping, a power-law gamma correction models the display transfer function (sRGB standard):

$$I_\text{display}(x) = T(x;\,\alpha)^\gamma$$

The default $\gamma = 1.8$ approximates the combined response of CRT/LCD displays.

#### 2.8.3 Complete Tone Pipeline

The full mapping from linear ESF to displayed intensity is:

$$\boxed{I_c(x, z) = \left[\frac{\tanh\!\left(\alpha \cdot \mathrm{ESF}_c(x, z)\right)}{\tanh(\alpha)}\right]^\gamma}$$

### 2.9 Colour Fringe Width (CFW) Definition and Pixel Detection

#### 2.9.1 Fringe Pixel Classification

A pixel at position $x$ is classified as a **visible fringe pixel** when **all three** conditions hold simultaneously:

1. **C1 (lower brightness threshold):** Every channel exceeds the low threshold: $\min(I_R, I_G, I_B) > \delta_\text{low}$
2. **C2 (inter-channel difference threshold):** At least one pairwise channel difference exceeds the threshold: $\max(|I_R - I_G|, |I_R - I_B|, |I_G - I_B|) > \delta$
3. **C3 (upper brightness threshold):** At least one channel is below the high threshold: $\min(I_R, I_G, I_B) < \delta_\text{high}$

Default thresholds: $\delta = 0.15$, $\delta_\text{low} = 0.15$, $\delta_\text{high} = 0.80$.

**Physical motivation:**
- C1 excludes near-black regions where all channels are dark and no colour is perceptible.
- C2 requires a minimum colour shift between channels; below this threshold, the human eye cannot distinguish the fringe from a neutral edge.
- C3 excludes near-white/saturated regions where all channels are clipped to near-unity and differences vanish.

#### 2.9.2 Total CFW (Outer-Boundary Method)

The CFW is defined as the distance (µm) from the **first** fringed pixel to the **last** fringed pixel in the scan window $x \in [-400, 400]$ µm:

$$\boxed{\mathrm{CFW}(z) = x_\text{last} - x_\text{first} + 1 \quad (\mu\mathrm{m})}$$

where $x_\text{first}$ and $x_\text{last}$ are the outermost positions satisfying the fringe mask. This outer-boundary definition is robust against threshold jitter that creates small internal gaps in the mask.

(Pixel pitch is 1 µm, so pixel count = µm width.)

---

## 3. Computational Method

### 3.1 Test Lens

The validation lens is a **Nikon AI Nikkor 85 mm f/2S** (Zemax ZMX format):

| Parameter | Value |
|-----------|-------|
| Focal length | 85 mm |
| Maximum aperture | f/2 (FNO ≈ 2.0) |
| Elements / surfaces | 6 lenses (12 refractive surfaces) |
| SA spot radius $\rho_\text{SA}$ | 12.2–19.0 µm (mean 17.4 µm) |

The lens prescription is loaded via Zemax file import, providing both paraxial and real ray tracing capabilities. Measured clear aperture constraints (radial aperture) are applied to each surface.

### 3.2 Spectral Data

Two sensor models are used: **Sony A900** (default) and **Nikon D700**. Spectral data comprise:

| Data | Source | Wavelength range |
|------|--------|-----------------|
| CIE D65 standard illuminant | Standard specification | ~380–780 nm |
| Sensor QE (R, G, B per camera) | Manufacturer data | 400–700 nm |

All spectral data are resampled onto a common 31-point grid (400–700 nm, 10 nm step) via cubic spline interpolation and normalised to [0, 100].

### 3.3 Spectral Weight Characterisation

The energy-normalised spectral weights $\hat{g}_c(\lambda)$ were computed for both sensor models. The D65 illuminant peaks near 460 nm and 590 nm with a dip at ~500 nm, which modulates the sensor QE and suppresses blue-channel weight relative to the raw sensor response.

Key properties of the resulting spectral windows:

- **Sensor QE.** The R channel peaks at ~570–580 nm with the highest absolute QE. The B channel is broadest (400–510 nm) but with lower peak QE (~0.8). The G channel lies between the two (~490–590 nm). Sony A900 has similar band shapes but roughly half the peak QE of Nikon D700.

- **Channel overlap.** B and G overlap in 490–510 nm; G and R overlap in 540–590 nm. In these overlap bands, two channels respond to the same wavelengths with similar CHL, so their ESFs are similar — the visible fringe occurs where channels do **not** overlap (B-only: 400–480 nm, R-only: 600–680 nm).

- **Camera dependence.** The Nikon D700 R channel has a higher normalised peak (~0.021) than Sony A900 (~0.018), giving D700 slightly more weight to long wavelengths, potentially shifting the R–B fringe balance.

### 3.4 FFT Diffraction Path (Ground Truth)

Monochromatic PSFs are computed via Fraunhofer FFT propagation with the following parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N_\text{rays}$ | 512 | Pupil sample count |
| $N_\text{grid}$ | 800 | FFT grid size |
| $Q$ | $800/511 \approx 1.57$ | Oversampling factor |
| `wl_stride` | 3 | 31 → 11 wavelengths (30 nm step) |
| `strategy` | `chief_ray` | Wavefront reference (preserves chromatic defocus) |
| $x_\text{um}$ | $[-300, +300]$, step 1 µm | Physical coordinate axis (601 points) |

**Defocus grid:** 29 values, $z \in [-700, +700]\;\mu\text{m}$, 50 µm step.

**Pupil sampling rationale.** $N_\text{rays} = 512$ ensures the physical PSF half-span $(N_\text{rays} - 1) \times \lambda_\text{min} \times F_{\#} / 2$ covers the maximum geometric blur radius $z_\text{max} / (2F_{\#})$ for the defocus range $|z| \leq 700\;\mu\text{m}$.

**Two-stage baking architecture:**

1. **Stage 1 (sensor-independent bake):** Compute all monochromatic ESFs for 29 $z$-values × 11 wavelengths = 319 FFT computations → $(N_\lambda, N_x)$ matrix per defocus. Re-run only when the optical system changes.

2. **Stage 2 (sensor-specific apply):** Weight by $\hat{g}_c(\lambda)$ per channel → polychromatic ESF. Re-run only when switching cameras (~microseconds). Speedup: ~3× (single camera), ~6× (two cameras).

### 3.5 Geometric Ray-Fan Path

Precompute GL ray fans at three node counts for convergence testing:

| Configuration | GL nodes $K$ | Rays traced | Precomputation time |
|--------------|-------------|-------------|-------------------|
| Coarse | 5 | 155 | ~3.4 s |
| Medium | 16 | 496 | ~10.5 s |
| Fine | 32 | 992 | ~20.7 s |

For each defocus value, extrapolate ring radii and assemble ESF via pupil integration. Cost: ~0.1 ms per ESF after precomputation.

### 3.6 Analytic Path

Compute CHL and SA blur radii from aberration curves (paraxial or RoRi), then evaluate disc or Gaussian ESF formulae via JIT-compiled Numba kernels. Cost: <0.01 ms per ESF.

### 3.7 Controlled-Variable Experiments

The following factorial experiments isolate individual model contributions:

**Experiment 5a — PSF Model Comparison (2×2 factorial: CHL × PSF):**

| Model | Characteristics |
|-------|----------------|
| Disc + Paraxial | Hard-cutoff linear transition |
| Gaussian + Paraxial | Soft tails, closer to diffraction |
| Disc + RoRi | With spherochromatism correction |
| Gaussian + RoRi | With spherochromatism correction |

**Experiment 5b — SA Effect (Disc vs Gaussian with RoRi + SA):**

Tests how adding spherical aberration changes CFW predictions, comparing models with and without $\rho_\text{SA}$.

**Experiment 5c — Geometric Fast Convergence (GL node count):**

Tests how many GL nodes are needed for ESF convergence: $K \in \{5, 16, 32\}$.

**Experiment: Tone Mapping Sensitivity:**

CFW computed at multiple exposure values $\alpha \in \{1, 2, 4, 8, 16\}$ to characterise the non-monotonic visibility response.

---

## 4. Results and Discussion

This section reports the simulation results obtained from two computational pipelines: the FFT diffraction baseline (`cfw_fftpsf_demo`) and the geometric/analytic model suite (`cfw_geom_demo`). All simulations use the Nikon AI Nikkor 85 mm f/2S test lens with the Sony A900 sensor model unless otherwise stated.

### 4.1 Aberration Characteristics of the Test Lens

#### 4.1.1 CHL Curves: Paraxial vs. RoRi

The paraxial and RoRi CHL curves were computed using `compute_chl_curve` and `compute_rori_spot_curves` respectively at 31 wavelengths (400–700 nm). Both curves exhibit a U-shape: positive CHL at the spectral extremes (400–450 nm, 650–700 nm) and negative CHL in the mid-spectrum (~480–600 nm). The total CHL range spans approximately $-90$ to $+230\;\mu\text{m}$ (paraxial) and $-80$ to $+310\;\mu\text{m}$ (RoRi).

Key findings from the computed CHL curves:

- **Zero-crossing near 500 nm.** $\mathrm{CHL}(\lambda) = 0$ defines the wavelength that is in perfect focus at $z = 0$. Wavelengths shorter than 500 nm (blue) focus closer to the lens (CHL < 0), while wavelengths longer than ~580 nm (red) focus farther (CHL > 0) — consistent with normal dispersion for a positive lens.

- **Secondary spectrum magnitude.** The ~300 µm focal spread across the visible spectrum is the fundamental driver of colour fringing.

- **Spherochromatism gap.** The RoRi curve deviates from the paraxial curve most noticeably at short wavelengths (400–430 nm), where $\Delta\mathrm{CHL}_\mathrm{sphchrom} \approx +80\;\mu\text{m}$. This means that at full aperture, the real best-focus for violet light is shifted significantly farther from the lens than the paraxial prediction. At mid-spectrum the two curves nearly coincide, indicating that spherochromatism is small where SA itself is moderate.

- **Asymmetry.** The blue wing rises much more steeply than the red wing. This asymmetry means that at $z > 0$ (image plane behind focus), red and blue blur radii differ substantially, producing strong R–B fringing; at $z < 0$ (in front of focus), the fringe is weaker because the CHL spread is more compact.

#### 4.1.2 Aberration Budget: SA vs. CHL

To determine whether SA is significant compared to CHL for this lens, the blur radii $\rho_\text{CHL}$, $\rho_\text{SA}$, and $\rho_\text{total}$ were computed at $z = 0$ for all 31 wavelengths. The relative importance is quantified by the **variance (energy) fraction**:

$$f_\text{SA}(\lambda) = \frac{\rho_\text{SA}^2}{\rho_\text{total}^2}, \qquad f_\text{CHL}(\lambda) = 1 - f_\text{SA}(\lambda)$$

Table 1 shows the computed aberration budget at selected wavelengths:

**Table 1.** Aberration budget at $z = 0$ for the Nikon 85 mm f/2S.

| $\lambda$ (nm) | $\rho_\text{CHL}$ (µm) | $\rho_\text{SA}$ (µm) | $\rho_\text{total}$ (µm) | SA share | Dominant |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 400 | 79.3 | 12.2 | 80.2 | 2.3% | CHL |
| 410 | 51.9 | 13.2 | 53.6 | 6.0% | CHL |
| 430 | 21.3 | 14.6 | 25.8 | 32.1% | CHL |
| 470 | 2.1 | 16.5 | 16.6 | 98.4% | **SA** |
| 500 | 12.8 | 17.3 | 21.5 | 64.8% | **SA** |
| 550 | 12.0 | 18.1 | 21.7 | 69.6% | **SA** |
| 590 | 2.6 | 18.5 | 18.7 | 98.1% | **SA** |
| 650 | 20.5 | 18.9 | 27.9 | 45.8% | CHL |
| 700 | 44.7 | 18.8 | 48.5 | 15.0% | CHL |

Three spectral regimes emerge:

- **400–430 nm** (deep blue): CHL dominates (80–95%), because $\rho_\text{CHL} \gg \rho_\text{SA}$.
- **450–600 nm** (mid-spectrum): **SA dominates** (50–98%), because CHL is near its minimum while SA remains at ~15–19 µm. At the CHL zero-crossings (~470 nm, ~590 nm), SA accounts for nearly 100% of the total blur.
- **650–700 nm** (red): CHL regains dominance (60–85%) as the secondary spectrum rises again.

**Key implications:**

- **SA sets a blur floor.** Even at wavelengths where CHL = 0 (perfect chromatic focus), the total blur never drops below $\rho_\text{SA} \approx 15\;\mu\text{m}$. This floor limits the sharpness of the ESF transition and affects the fringe boundary position.
- **SA is nearly flat.** $\rho_\text{SA} \approx 12$–$19\;\mu\text{m}$ varies only mildly across the spectrum, because spherical aberration is a geometric property of the lens shape and varies slowly with refractive index.
- **$\rho_\text{CHL}$ has two peaks and a valley.** At $z = 0$, $\rho_\text{CHL} = |\mathrm{CHL}(\lambda)|/\sqrt{4F_{\#}^2-1}$ mirrors the CHL curve shape, peaking at ~80 µm (400 nm) and ~45 µm (700 nm).
- **Modelling implication.** A pure-CHL model (without SA) would predict zero blur at the CHL zero-crossings and underestimate the total blur across most of the mid-spectrum. Including SA is essential for accurate CFW prediction at this aperture ($F_{\#} = 2$).

#### 4.1.3 SA Profile Accuracy

Having established that SA is significant (§4.1.2), we next ask: **how accurately must the transverse aberration profile be represented?** Three parameterisation levels were compared against the 32-node ray-fan ground truth at five representative wavelengths.

| Model | Formula | Parameters per $\lambda$ |
|-------|---------|:------------------------:|
| Scalar $\rho_\text{SA}$ | $\text{TA}(r) \approx 2r^3 \cdot \rho_\text{SA}$ | 1 |
| Polynomial | $\text{TA}(r) \approx c_3(\lambda)\,r^3 + c_5(\lambda)\,r^5$ | 2 |
| Ray fan | $\text{TA}(r_k)$ exact at $K$ nodes | $K$ (32) |

The polynomial coefficients $c_3, c_5$ are fitted by least squares from the 5 RoRi SK data points (excluding $r = 0$):

$$\begin{pmatrix} r_1^3 & r_1^5 \\ \vdots & \vdots \\ r_4^3 & r_4^5 \end{pmatrix} \begin{pmatrix} c_3 \\ c_5 \end{pmatrix} = \begin{pmatrix} \mathrm{TA}_\text{SA}(r_1) \\ \vdots \\ \mathrm{TA}_\text{SA}(r_4) \end{pmatrix}$$

**Table 2.** SA parameterisation results at five representative wavelengths.

| $\lambda$ (nm) | $\rho_\text{SA}$ (µm) | $c_3$ | $c_5$ | $c_5/c_3$ | RMS err (scalar) | RMS err (poly) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 430 | 14.6 | 33.1 | −74.4 | −2.25 | 9.8 | 7.2 |
| 480 | 16.8 | 40.4 | −88.8 | −2.20 | 11.2 | 8.0 |
| 550 | 18.1 | 44.6 | −97.3 | −2.18 | 12.1 | 8.5 |
| 600 | 18.6 | 45.9 | −100.0 | −2.18 | 12.4 | 8.7 |
| 650 | 18.9 | 46.4 | −101.2 | −2.18 | 12.6 | 8.8 |

Key observations from the computed TA profiles:

- **Ray fan is non-monotonic.** At most wavelengths, $\text{TA}(r)$ first increases (positive for small $r$), then reverses sign and grows strongly negative toward the pupil edge ($r \to 1$). This S-shaped profile indicates **overcorrected SA at large aperture** — the marginal rays cross the axis and land on the opposite side.

- **Scalar $\rho^3$ model fails qualitatively.** The $2r^3 \cdot \rho_\text{SA}$ model is monotonic and cannot capture the sign reversal. At 480 nm and 550 nm, the scalar model predicts +20 µm at $r = 0.5$ while the ray fan shows near zero — the error is not just quantitative but directional.

- **Polynomial captures the shape.** The two-term fit reproduces the sign reversal because $c_3$ and $c_5$ have opposite signs (ratio $c_5/c_3 \approx -2.2$ at all wavelengths). The fit tracks the ray fan closely up to $r \approx 0.8$.

- **Edge region ($r > 0.85$) deviates.** Even the polynomial underestimates the ray fan at the pupil edge, where higher-order terms ($r^7$, $r^9$) become significant.

- **Wavelength dependence.** At 430 nm the profile is mildly curved (primary SA dominates; RMS err = 9.8 µm), while at 550–650 nm the S-curve is more pronounced (RMS err = 12.1–12.6 µm). This wavelength variation is the spherochromatism captured by the CHL gap in §4.1.1.

- **Implication for ESF modelling.** The analytic Disc/Gaussian models (§2.5) use only the scalar $\rho_\text{SA}$ — they cannot represent the sign reversal. The ray-fan method (§2.6) avoids this approximation entirely by using the actual TA profile at all GL nodes.

### 4.2 FFT Diffraction Baseline Results

#### 4.2.1 Two-Stage Baking Output

Monochromatic ESFs were baked for all 29 defocus positions ($z \in [-700, +700]\;\mu\text{m}$) × 11 wavelengths using the FFT pipeline (§3.4). The baking produced 319 monochromatic ESFs totalling ~1.5 MB, from which polychromatic R/G/B ESFs were assembled via sensor spectral weighting (~0.4 MB).

**Table 3.** Per-channel ESF transition widths (samples where $0.05 < \text{ESF} < 0.95$) at selected defocus values.

| $z$ (µm) | R transition | G transition | B transition | Mean |
|:-:|:-:|:-:|:-:|:-:|
| $-700$ | 260 | 229 | 238 | 242 |
| $-400$ | 152 | 120 | 130 | 134 |
| $-100$ | 65 | 47 | 62 | 58 |
| $-50$ | 59 | 50 | 65 | 58 |
| $0$ | 62 | 59 | 73 | 65 |
| $+50$ | 72 | 71 | 83 | 75 |
| $+100$ | 86 | 83 | 94 | 88 |
| $+400$ | 200 | 192 | 195 | 196 |
| $+700$ | 324 | 356 | 343 | 341 |

Key observations:

- **Best-focus region near $z \approx -50$ µm.** The narrowest mean transition (~58) occurs not at the nominal focal plane ($z = 0$) but is shifted by approximately $-50\;\mu\text{m}$, reflecting the combined effect of the secondary spectrum shape and SA.
- **Per-channel best-focus separation.** The G channel reaches its minimum (47 samples) at $z \approx -100\;\mu\text{m}$, while B reaches its minimum (62 samples) near $z = -100\;\mu\text{m}$ and R near $z = -50\;\mu\text{m}$. This separation directly reflects the CHL curve: G samples near the CHL minimum (~540 nm), while B and R sample the CHL wings.
- **Asymmetry.** Transition widths at $z > 0$ are systematically larger than at $z < 0$ for the same $|z|$ (e.g., $z = +700$: mean 341 vs. $z = -700$: mean 242). This reflects the secondary spectrum sign: the lens has longer focal length for red wavelengths.

The ESF transition width is a purely optical quantity (independent of tone mapping). Each channel's minimum corresponds to its best-focus position; inter-channel separation directly quantifies CHL.

#### 4.2.2 CFW vs. Exposure (FFT Ground Truth)

Using the baked polychromatic ESFs, CFW was computed at five exposure values ($\alpha \in \{1, 2, 4, 8, 16\}$, $\gamma = 1.8$) across all 29 defocus positions.

**Table 4.** CFW statistics from FFT diffraction baseline (Sony A900, $\gamma = 1.8$).

| Exposure $\alpha$ | max CFW (µm) | Peak $z$ (µm) | mean CFW (µm) |
|:-:|:-:|:-:|:-:|
| 1 | 0 | — | 0.0 |
| 2 | 9 | $-200$ | 0.7 |
| 4 | 22 | $+150$ | 7.2 |
| 8 | 33 | $+300$ | 18.1 |
| 16 | 38 | $+700$ | 19.1 |

**Non-monotonic behaviour.** At $\alpha = 1$ (linear response), no pixel exceeds the colour-difference threshold $\delta = 0.15$, yielding CFW = 0 everywhere. As $\alpha$ increases to 4 and 8, the $\tanh$ tone curve amplifies the low-intensity inter-channel differences past the threshold: CFW grows from 0 to 33 µm. However, between $\alpha = 8$ and $\alpha = 16$, the max CFW increases only modestly (33 → 38 µm) while the mean CFW barely changes (18.1 → 19.1 µm), indicating the onset of saturation: at very high exposure, all channels are driven toward a hard step function and condition C3 ($\min < \delta_\text{high}$) starts excluding pixels where all channels are simultaneously near unity.

The mechanism can be understood as follows. At low $\alpha$, the tone curve is nearly linear; the physical inter-channel difference falls below $\delta$. As $\alpha$ increases, the region near $x = 0$ (the ESF transition zone) is amplified by the factor $\alpha/\tanh(\alpha)$, pushing previously sub-threshold differences above $\delta$ and expanding the fringe zone. Beyond a critical $\alpha$, saturation compresses the fringe region. The net effect is a peak in CFW at intermediate $\alpha$.

**Peak $z$ shift.** The defocus at which CFW is maximised shifts from $z = -200\;\mu\text{m}$ at $\alpha = 2$ to $z = +300\;\mu\text{m}$ at $\alpha = 8$. At low exposure, only the region with largest channel separation (near the CHL zero-crossings) exceeds the threshold. At higher exposure, the amplified tone curve reveals fringes at larger $|z|$ where the absolute CHL spread is wider but the linear differences are smaller.

### 4.3 Analytic and Geometric Model Comparison

#### 4.3.1 PSF Model Comparison — 2×2 Factorial (Experiment 5a)

Four analytic models were compared in a 2×2 factorial design (CHL model × PSF kernel, no SA), computing CFW across the full defocus range at four exposure values.

**Table 5.** CFW results for 2×2 factorial experiment (no SA, $\gamma = 1.8$).

| Strategy | $\alpha = 1$ | $\alpha = 2$ | $\alpha = 4$ | $\alpha = 8$ |
|----------|:-:|:-:|:-:|:-:|
| **A — Disc + Paraxial** | max 27, mean 4.4 | max 16, mean 2.2 | max 14, mean 3.2 | max 20, mean 10.0 |
| **B — Gauss + Paraxial** | max 4, mean 0.4 | max 6, mean 1.2 | max 15, mean 5.3 | max 25, mean 14.3 |
| **C — Disc + RoRi** | max 28, mean 4.7 | max 16, mean 2.6 | max 14, mean 3.2 | max 20, mean 10.0 |
| **D — Gauss + RoRi** | max 5, mean 0.4 | max 6, mean 1.5 | max 16, mean 5.6 | max 26, mean 14.6 |

(All CFW values in µm; "max" = max over $z$, "mean" = mean over all 29 $z$ values.)

Key findings:

- **Disc vs. Gaussian.** At $\alpha = 1$, the Disc model predicts anomalously large CFW (max 27–28 µm) while the Gaussian model gives only 4–5 µm. This is because the Disc's hard cutoff creates an abrupt transition that generates large channel differences even without tone-curve amplification. The Gaussian's smooth tails produce gentler transitions that remain below the $\delta = 0.15$ threshold at low exposure. At $\alpha \geq 4$, the two models converge: the Gaussian gives slightly larger CFW due to its wider effective ESF.

- **Paraxial vs. RoRi.** The effect is small in this no-SA configuration: strategies A/C and B/D differ by only 1–2 µm. The RoRi correction becomes more significant when SA is included (§4.3.2).

- **Combined effect.** The Gaussian + RoRi combination (strategy D) provides the most physically reasonable predictions among the analytic models, avoiding the Disc's artificial hard-cutoff artefact while including the spherochromatism correction.

#### 4.3.2 SA Effect (Experiment 5b)

Two additional strategies were evaluated to isolate the effect of spherical aberration on CFW:

**Table 6.** CFW results with SA included (RoRi CHL, $\gamma = 1.8$).

| Strategy | $\alpha = 1$ | $\alpha = 2$ | $\alpha = 4$ | $\alpha = 8$ |
|----------|:-:|:-:|:-:|:-:|
| **E — Disc + RoRi + SA** | max 10, mean 0.9 | max 5, mean 0.8 | max 9, mean 2.4 | max 17, mean 8.6 |
| **F — Gauss + RoRi + SA** | max 1, mean 0.0 | max 4, mean 0.3 | max 10, mean 3.1 | max 20, mean 10.6 |
| C — Disc + RoRi (no SA) | max 28, mean 4.7 | max 16, mean 2.6 | max 14, mean 3.2 | max 20, mean 10.0 |
| D — Gauss + RoRi (no SA) | max 5, mean 0.4 | max 6, mean 1.5 | max 16, mean 5.6 | max 26, mean 14.6 |

Comparing strategies E/F (with SA) against C/D (without SA):

- **SA reduces CFW in the Disc model** at low exposure (E vs. C: max 10 vs. 28 at $\alpha = 1$). This is because $\rho_\text{SA}$ broadens the otherwise sharp disc ESF, smoothing the channel boundaries and reducing the pairwise colour difference below the threshold.

- **SA reduces CFW in the Gaussian model** at moderate exposure (F vs. D: max 10 vs. 16 at $\alpha = 4$; max 20 vs. 26 at $\alpha = 8$). The SA blur floor widens all three channels' ESFs similarly, reducing their separation.

- **Physical interpretation.** Including SA adds a wavelength-independent blur component ($\rho_\text{SA} \approx 17\;\mu\text{m}$) that acts as a low-pass filter on the inter-channel differences. While SA broadens each individual ESF (§4.1.2), the net effect on CFW is **to reduce it**, because the broadening is nearly uniform across channels and therefore reduces their differential.

#### 4.3.3 GL Node Convergence (Experiment 5c)

The geometric ray-fan ESF was computed at three GL node counts ($K = 5, 16, 32$) to test convergence.

**Table 7.** CFW results for geometric ray-fan model at three node counts ($\gamma = 1.8$).

| Strategy | $\alpha = 1$ | $\alpha = 2$ | $\alpha = 4$ | $\alpha = 8$ |
|----------|:-:|:-:|:-:|:-:|
| **G₅ — Geom Fast (5 nodes)** | max 2, mean 0.1 | max 12, mean 0.9 | max 12, mean 3.7 | max 19, mean 10.5 |
| **G₁₆ — Geom Fast (16 nodes)** | max 1, mean 0.0 | max 6, mean 0.6 | max 12, mean 3.8 | max 21, mean 11.5 |
| **G₃₂ — Geom Fast (32 nodes)** | max 1, mean 0.0 | max 5, mean 0.5 | max 11, mean 3.5 | max 20, mean 11.1 |

Key findings:

- **Rapid convergence.** The CFW predictions at $\alpha = 4$ and $\alpha = 8$ are nearly identical across all three node counts (within 1–2 µm), confirming that the GL quadrature converges quickly for this lens.
- **16 vs. 32 nodes.** At all exposure levels, G₁₆ and G₃₂ agree within 1 µm. The 16-node configuration is sufficient for practical CFW prediction, halving the precomputation time (10.5 s vs. 20.7 s).
- **5-node limitation.** At $\alpha = 2$, G₅ overestimates the max CFW (12 µm vs. 5–6 µm for G₁₆/G₃₂). With only 5 nodes, the coarse pupil sampling cannot capture the fine structure of the S-shaped TA profile (§4.1.3), introducing systematic error in the ESF shape.

#### 4.3.4 Effect of RoRi vs. Paraxial CHL

Comparing strategies A/B (Paraxial) against C/D (RoRi) in Table 5, the RoRi correction has a modest effect in the no-SA analytic models (1–2 µm difference). However, the correction is physically important for two reasons:

- The paraxial model underestimates the focal shift at short wavelengths by up to 80 µm (§4.1.1), causing it to predict a narrower B-channel ESF than reality and underestimating the B–G and B–R fringe widths.
- The RoRi correction is most important at the spectral extremes (400–430 nm, 650–700 nm) and has minimal effect in the mid-spectrum where spherochromatism is small.

In fast lenses with significant spherochromatism, the RoRi-based prediction is closer to the FFT ground truth.

### 4.4 Per-Defocus Diagnostic Analysis

To visualise the fringe formation mechanism, detailed per-defocus diagnostics were computed for both the FFT baseline ($\alpha = 4$) and the geometric ray-fan model ($K = 16$, $\alpha = 8$). Each defocus position is analysed through three complementary views:

1. **Raw ESF** — the linear polychromatic edge response (pure optics, no tone mapping). Reveals the optical separation between R, G, B channels.
2. **Tone-mapped ESF** — after applying exposure slope and gamma, with fringe boundary markers overlaid. Shows which portions of the ESF transition become perceptually visible.
3. **Pseudo-colour density map** — RGB ESFs rendered as a colour strip, providing an intuitive visual representation of the fringe appearance.

**Table 8.** Per-defocus CFW and maximum channel-pair difference (FFT baseline, $\alpha = 4$, $\gamma = 1.8$) at selected defocus values.

| $z$ (µm) | CFW (µm) | max pair diff | R trans | G trans | B trans |
|:-:|:-:|:-:|:-:|:-:|:-:|
| $-700$ | 7 | 0.171 | 260 | 229 | 238 |
| $-400$ | 11 | 0.214 | 152 | 120 | 130 |
| $-200$ | 14 | 0.261 | 95 | 65 | 80 |
| $-50$ | 6 | 0.176 | 59 | 50 | 65 |
| $0$ | 10 | 0.196 | 62 | 59 | 73 |
| $+100$ | 13 | 0.227 | 86 | 83 | 94 |
| $+300$ | 22 | 0.287 | 157 | 155 | 159 |
| $+700$ | 0 | 0.136 | 324 | 356 | 343 |

Key qualitative observations across the diagnostic grid:

- At $z = 0$, the three channels are closely spaced but distinguishable; the fringe is narrow (CFW = 10 µm).
- At $z = +300\;\mu\text{m}$, the maximum CFW is reached (22 µm), coinciding with the peak of the R–B pair difference (0.287).
- At $|z| = 200$–$400\;\mu\text{m}$, the channel separation is maximal, producing the widest and most vivid fringes.
- At $z = +700\;\mu\text{m}$, all channels are heavily blurred (transition widths > 300); the pair differences drop below the threshold and CFW = 0.
- The fringe colour sequence (blue → white → magenta or cyan → white → yellow) depends on which channel is sharpest at each defocus, directly reflecting the CHL curve shape.

### 4.5 Model Accuracy and Computational Performance

#### 4.5.1 Speed Comparison

| Method | Accuracy level | Typical speed | Speedup vs. FFT |
|--------|---------------|---------------|-----------------|
| FFT diffraction PSF | Ground truth (diffraction) | ~1 s/ESF | 1× |
| Ray-fan GL extrapolation | Geometric optics | <1 ms/ESF | ~1000× |
| JIT analytic ESF (disc/gauss) | Parametric approximation | <0.01 ms/ESF | ~100 000× |

The ray-fan method with 32 GL nodes achieves ESF error < 0.1% relative to FFT for smooth aberration profiles. The analytic models sacrifice SA profile accuracy for an additional ~100× speedup.

#### 4.5.2 Two-Stage Baking Performance

| Configuration | FFT computations | Weighted sums | Total |
|--------------|-----------------|---------------|-------|
| Single-step, 1 camera | 319 × 3 = 957 | — | 957 FFTs |
| Two-stage, 1 camera | 319 | 87 | 319 FFTs + 87 sums |
| Two-stage, 2 cameras | 319 | 174 | 319 FFTs + 174 sums |
| Single-step, 2 cameras | 319 × 6 = 1914 | — | 1914 FFTs |

Speedup: ~3× (single camera), ~6× (two cameras).

#### 4.5.3 Computational Resources

| Resource | Size / Time |
|----------|-------------|
| Monochromatic ESF cache (29 $z$ × 11 $\lambda$ × 801 pts) | ~2.0 MB |
| Polychromatic ESF cache (29 $z$ × 3 ch × 801 pts) | ~541 KB |
| Ray-fan dict ($32 \times 31$ double arrays) | ~16 KB |
| Sensor response precomputation ($3 \times 31$) | < 1 KB |
| FFT baking total time | ~1–3 min (CPU) |
| JIT compilation (first call, Numba) | ~10–30 s (cached afterwards) |

### 4.6 Towards Off-Axis Extension: Combining the Strengths of FFT and Ray-Fan Methods

The three-level modelling hierarchy developed in this work is restricted to **on-axis fields**, where the PSF is rotationally symmetric. Extending the framework to off-axis fields — where astigmatism, coma, and field curvature break this symmetry — is the most significant open challenge. This subsection analyses the computational bottleneck of the FFT method and proposes an OPD gradient-driven adaptive grid strategy that synthesises the strengths of both the FFT (Level 0) and ray-fan (Level 1) approaches.

#### 4.6.1 The Problem: Fixed Grid Inefficiency

The FFT and ray-fan methods developed in §2.6–2.7 have complementary strengths:

| Property | FFT-PSF (Level 0) | Ray-fan (Level 1) |
|----------|-------------------|-------------------|
| Diffraction | Included | Ignored |
| Aberration fidelity | Full wavefront | Full TA profile |
| Speed | ~1 s/ESF (fixed grid) | <1 ms/ESF |
| Off-axis | Handles all aberrations | Rotationally symmetric only |
| Bottleneck | Fixed $(N, G) = (512, 800)$ for all $z$ | Linear extrapolation error at large $z$ |

The FFT method's bottleneck is that the grid $(N, G)$ is fixed at the worst-case size, regardless of the actual wavefront complexity at each defocus position. The ray-fan method demonstrates (via the linear extrapolation in §2.6.6) that the aberration structure varies dramatically with defocus: near focus, the OPD is dominated by low-order terms (small gradients), while at large defocus the wavefront varies rapidly (large gradients). **The adaptive grid strategy brings this insight from the ray-fan approach into the FFT framework**, allowing the FFT to exploit the same defocus-dependent simplification that makes the ray-fan method fast.

#### 4.6.2 Physical Foundation: OPD Gradient and Sampling Requirements

The input to the FFT-PSF computation is the complex pupil function $P(u,v) = A(u,v) \cdot \exp\!\left(i\,2\pi\,\text{OPD}(u,v)/\lambda\right)$. The spatial rate of phase variation $|\nabla\phi| = (2\pi/\lambda)|\nabla\text{OPD}|$ determines the spatial frequency content of $P$: large gradients produce rapid oscillation requiring dense sampling, while small gradients produce smooth variation where coarse sampling suffices.

Applying the Shannon–Nyquist theorem to the pupil sampling interval $\Delta u = 2/(N-1)$ yields the minimum number of pupil samples:

$$\boxed{N - 1 > \frac{4}{\lambda}\,\max_{(u,v)}\left|\frac{\partial\,\text{OPD}}{\partial u}\right|}$$

Equivalently, in waves ($\text{OPD}_w = \text{OPD}/\lambda$): $N - 1 > 4\,\max|\partial\text{OPD}_w/\partial u|$. The coefficient 4 arises from the product of the Nyquist factor (2) and the normalised pupil diameter (2).

For the major Seidel aberrations, the maximum OPD gradients (in waves/aperture) are:

| Aberration | OPD (waves) | $\max|\partial\text{OPD}_w/\partial u|$ |
|------------|------------|----------------------------------------|
| Defocus | $W_{20}\rho^2$, where $W_{20} = z/(8\lambda F_{\#}^2)$ | $2W_{20} = z/(4\lambda F_{\#}^2)$ |
| Spherical aberration | $W_{40}\rho^4$ | $4W_{40}$ |
| Coma | $W_{31}H\rho^3\cos\theta$ | $\sim 3W_{31}H$ (T direction) |
| Astigmatism | $W_{22}H^2\rho^2\cos^2\theta$ | $\sim 2W_{22}H^2$ (T direction) |

Defocus dominates at large $|z|$, but coma and astigmatism become significant off-axis, creating **T/S asymmetry**: the tangential direction has a larger effective gradient due to the $3W_{31}H$ coma contribution. Under the constraint of a square FFT grid, $N_\text{min}$ is determined by the more stringent direction:

$$N_\text{min} = \max\!\left(N_\text{floor},\;\left\lceil 4\,\max\!\big(G_T(\lambda_\text{min}),\; G_S(\lambda_\text{min})\big)\right\rceil + 1\right)$$

where $G_T, G_S$ are the maximum OPD gradients in the T and S directions, evaluated at $\lambda_\text{min}$ (the shortest wavelength gives the most stringent condition, since $\text{OPD}_w \propto 1/\lambda$).

#### 4.6.3 Adaptive Algorithm: Ray-Fan Pre-Scan + FFT Execution

The key idea is to use a **low-cost ray-fan-like pre-scan** to determine the OPD gradient, then execute the FFT with a dynamically sized grid. This directly combines the ray-fan method's ability to cheaply characterise the aberration structure with the FFT method's diffraction-level accuracy:

1. **Pre-scan** ($32 \times 32$ rays, ~0.4% of the cost of a full $512^2$ trace): obtain the OPD map at wavelength endpoints $\lambda_\text{min}$ and $\lambda_\text{max}$, compute the maximum gradient $\max(G_T, G_S)$. This step is analogous to the ray-fan precomputation (§2.6.5) but requires only a fraction of the rays.

2. **Grid selection**: compute $N_\text{min}$ from the Nyquist condition and $G_\text{min} = \lceil Q_\text{min} \cdot (N_\text{min} - 1)\rceil$. A physical coverage check ensures the FFT output spans the blur radius: $x_\text{max} = (N-1)\lambda F_{\#}/2 \geq r_\text{blur}$.

3. **FFT execution**: run the standard FFT-PSF pipeline with the adapted $(N_\text{min}, G_\text{min})$ instead of the fixed $(512, 800)$.

The result is an FFT computation that retains full diffraction accuracy but adapts its resolution to the actual wavefront complexity — fast where the ray-fan method is fast (near focus), and reverting to the full grid only where necessary (large defocus).

#### 4.6.4 Quantitative Analysis

**On-axis speedup.** For the test lens ($F_{\#} = 2$, $\lambda_\text{min} = 0.40\;\mu\text{m}$):

| $z$ (µm) | $\max|\partial\text{OPD}_w/\partial u|$ | $N_\text{min}$ | $G_\text{min}$ ($Q = 1.5$) | Per-point speedup |
|:-:|:-:|:-:|:-:|:-:|
| 0 | ~5–10 (SA only) | 64 | 96 | ~100× |
| 100 | 15.6 | 64 | 96 | ~100× |
| 300 | 46.9 | 256 | 384 | ~5× |
| 500 | 78.1 | 512 | 768 | ~1.1× |
| 700 | 109.4 | 512 | 768 | ~1.1× |

For the full 29-point defocus scan, the weighted-average FFT speedup is ~2×, because the 12 points at $|z| > 400\;\mu\text{m}$ still require near-maximum grids. For scans restricted to $|z| \leq 300\;\mu\text{m}$ (the region of most practical interest for fringe analysis), the speedup reaches ~5×.

**Off-axis case ($H = 0.7$).** Coma and astigmatism increase the near-focus OPD gradient ($G_T \approx 25$ waves/aperture at $z = 0$), reducing the near-focus speedup from ~100× to ~10×. At large defocus ($|z| > 500\;\mu\text{m}$, $\lambda_\text{min}$), the strict Nyquist condition may require $N > 512$ — exceeding the current baseline. Whether mild aliasing at $\lambda < 0.45\;\text{nm}$ is tolerable (given the low spectral weight and the smoothing effect of ESF integration) requires empirical validation. The off-axis full-scan speedup is estimated at 1.5–3× (tolerating mild short-wavelength aliasing) to < 1× (strict full-band Nyquist).

#### 4.6.5 Synthesis: Bridging the Two Modelling Levels

The adaptive grid strategy reveals a deeper connection between the FFT and ray-fan approaches. Both methods implicitly rely on the same physical quantity — the OPD gradient — but exploit it differently:

- The **ray-fan method** uses the fact that near focus, the transverse aberration ($TA_0 + m \cdot z$) is small, so the ring radii $R$ are small and the ESF can be computed from a simple geometric sum. It avoids the FFT entirely by working directly in the spatial domain.

- The **FFT method** uses the OPD to construct the complex pupil function and applies the Fourier transform. Near focus, the OPD is smooth (small gradient), and the FFT grid can be small; at large defocus, the OPD oscillates rapidly and demands a large grid.

The adaptive strategy makes this connection explicit: it uses the ray-fan's insight (aberration structure varies with $z$) to optimise the FFT's resource allocation. This is not merely an engineering optimisation — it reflects the physical reality that the information content of the wavefront is defocus-dependent, and any efficient computation should scale with this content rather than with a worst-case bound.

This perspective also suggests a natural accuracy hierarchy for off-axis extension:

| Region | Recommended approach |
|--------|---------------------|
| Near focus ($|z| \leq 100\;\mu\text{m}$) | Adaptive FFT with $N = 64$–$128$ (diffraction accuracy at ray-fan speed) |
| Moderate defocus ($100 < |z| \leq 400\;\mu\text{m}$) | Adaptive FFT with $N = 256$–$512$ (diffraction accuracy, moderate cost) |
| Large defocus ($|z| > 400\;\mu\text{m}$) | Full FFT ($N = 512$) or ray-fan geometric ESF (if diffraction is negligible) |

In the large-defocus regime where the adaptive FFT offers no speedup, the geometric optics approximation becomes increasingly accurate (the Fresnel number is large, diffraction effects are washed out by the large blur), and the ray-fan method may serve as a sufficient alternative — closing the loop between the two levels of the modelling hierarchy.

---

## 5. Conclusion

### 5.1 Summary

This work presents a hierarchical framework for numerical prediction of colour fringe width in photographic lenses. The key findings are:

1. **Spherochromatism matters.** The RoRi aperture-weighted CHL model captures wavelength-dependent spherical aberration that the paraxial model misses. For the f/2 test lens, the paraxial model underestimates the violet focal shift by ~80 µm.

2. **SA sets a blur floor.** Even at perfect chromatic focus (CHL = 0), residual spherical aberration ($\rho_\text{SA} \approx 17\;\mu\text{m}$) prevents the blur from vanishing. Pure-CHL models significantly underestimate ESF transition widths in the mid-spectrum.

3. **The ray-fan method bridges accuracy and speed.** By precomputing transverse aberrations at 32 Gauss–Legendre pupil nodes and extrapolating linearly, the geometric ESF achieves < 0.1% error relative to FFT diffraction at ~1000× the speed. This enables real-time CFW parameter sweeps that would be infeasible with wave optics alone.

4. **Fringe visibility is non-monotonic in exposure.** The $\tanh$ tone mapping amplifies subtle inter-channel differences at moderate exposure but compresses them at high exposure. CFW prediction must account for this perceptual stage.

5. **The two-stage baking architecture** decouples physics from sensor, enabling efficient multi-camera comparison without redundant FFT computation (3–6× speedup).

### 5.2 Model Selection Guidelines

| Scenario | Recommended model |
|----------|-------------------|
| Quick CHL curve estimation | Paraxial (Level 2) |
| CFW with SA information | RoRi + Analytic ESF (Level 2) |
| Scanning many defocus values | Ray-fan GL (Level 1) |
| Ground truth validation | FFT-PSF (Level 0) |

### 5.3 Future Work

- **Implementation and validation of the adaptive grid strategy** proposed in §4.6: integrate the OPD gradient-driven grid selection into the baking pipeline, benchmark aliasing tolerance at short wavelengths for off-axis fields, and quantify the end-to-end speedup across a range of lens designs and field angles.
- **Off-axis CFW prediction:** extend the ray-fan method to off-axis fields (where coma and astigmatism break rotational symmetry), requiring independent T/S ESF computation and direction-dependent fringe width definitions.
- **Experimental validation.** Compare predicted CFW against real captured images with known lens prescriptions to validate the simulation chain end-to-end.
- **Perceptual colour metric.** Extend the CFW definition to account for human colour perception using CIE $\Delta E$ thresholds rather than simple channel-difference thresholds.

---

## Appendix A: Symbol Table

| Symbol | Meaning | Unit |
|--------|---------|------|
| $\lambda$ | Wavelength | nm |
| $F_{\#}$ | F-number (focal ratio) | — |
| $f'$ | Focal length | mm |
| $n(\lambda)$ | Refractive index | — |
| $\text{CHL}(\lambda)$ | Longitudinal chromatic aberration | µm |
| $\text{BFL}(\lambda)$ | Back focal length | mm |
| $r$ | Normalised pupil coordinate | $[0, 1]$ |
| $\rho$ | Blur circle radius (CHL, SA, or total) | µm |
| $SK(r, \lambda)$ | Longitudinal focal intercept | mm |
| $TA_0(r, \lambda)$ | Transverse aberration at $z = 0$ | µm |
| $m(r, \lambda)$ | Ray slope $M/N$ at image plane | — |
| $W(\xi, \eta)$ | Wavefront aberration (OPD) | waves |
| $\text{PSF}(x, y)$ | Point spread function | normalised |
| $\text{LSF}(x)$ | Line spread function | normalised |
| $\text{ESF}(x)$ | Edge spread function | $[0, 1]$ |
| $\text{CFW}$ | Colour fringe width | µm |
| $S_c(\lambda)$ | Sensor quantum efficiency | — |
| $D_{65}(\lambda)$ | D65 daylight spectrum | relative power |
| $\hat{g}_c(\lambda)$ | Normalised spectral weight | nm$^{-1}$ |
| $\rho_\text{SA}$ | SA blur spot RMS radius | µm |
| $\alpha$ | Exposure slope (tone mapping) | — |
| $\gamma$ | Display gamma | — |
| $K$ | Number of GL quadrature nodes | — |
| $W_k$ | GL quadrature weight | — |
| $Q$ | FFT oversampling factor | — |
| $y_k$ | Ray height at surface $k$ | mm |
| $u_k$ | Paraxial ray slope at surface $k$ | rad |
| $\phi_k$ | Optical power of surface $k$ | mm$^{-1}$ |
| $M, N$ | Direction cosines ($y$- and $z$-components) | — |

## Appendix B: Key Formulae

| Quantity | Expression |
|----------|-----------|
| Spectral weight | $\hat{g}_c(\lambda) = S_c(\lambda) \cdot D_{65}(\lambda) / \int S_c \cdot D_{65}\,d\lambda$ |
| Paraxial BFL | $\text{BFL}(\lambda) = -y_\text{last}/u_\text{last}$ |
| Paraxial CHL | $\text{CHL}_\text{par}(\lambda) = [\text{BFL}(\lambda) - \text{BFL}(\lambda_\text{ref})] \times 10^3$ |
| Focal intercept | $SK(r, \lambda) = -yN/M$ |
| RoRi weighting | $\text{RoRi}(\lambda) = \sum w_i \cdot SK(r_i, \lambda) / 42$ |
| RoRi CHL | $\text{CHL}_\text{RoRi}(\lambda) = [\text{RoRi}(\lambda) - \text{RoRi}(\lambda_\text{ref})] \times 10^3$ |
| CHL blur radius | $\rho_\text{CHL} = \|z - \text{CHL}(\lambda)\| / \sqrt{4F_{\#}^2-1}$ |
| SA spot (RMS) | $\rho_\text{SA} = \sqrt{\sum w_i y_\text{spot}^2 / \sum w_i}$ |
| Total blur | $\rho = \sqrt{\rho_\text{CHL}^2 + \rho_\text{SA}^2}$ |
| Disc ESF | $\frac{1}{2} + \frac{1}{\pi}[\arcsin(x/\rho) + \frac{x}{\rho}\sqrt{1-x^2/\rho^2}]$ |
| Gaussian ESF | $\frac{1}{2}[1 + \text{erf}(x/(\sqrt{2}\cdot 0.5\rho))]$ |
| Ring ESF | $f(x,R) = \arcsin(x/R)/\pi + \frac{1}{2}$ |
| Ray-fan extrapolation | $R(r_k; z, \lambda) = \|TA_0 + m \cdot z\|$ |
| Pupil-integral ESF | $\text{ESF}_c = \sum_j \hat{g}_c(\lambda_j) \sum_k W_k r_k f(x/R_k)$ |
| FFT pixel pitch | $dx = \lambda F_{\#} / Q$, $Q = N_\text{grid}/(N_\text{rays}-1)$ |
| Spectral integration | $\text{ESF}_c(x,z) = \sum_j \hat{g}_c(\lambda_j) \cdot \text{ESF}_\text{mono}(x;\lambda_j,z)$ |
| Tone mapping | $I = [\tanh(\alpha \cdot \text{ESF}) / \tanh(\alpha)]^\gamma$ |
| Fringe mask | $\text{C1} \wedge \text{C2} \wedge \text{C3}$ (see §2.9.1) |
| CFW (outer boundary) | $\text{CFW}(z) = x_\text{last} - x_\text{first} + 1$ |

---

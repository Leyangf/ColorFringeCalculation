---
title: "Off-Axis Adaptive FFT Sampling: OPD-Driven Grid vs Projection-Slice"
tags: [optics, PSF, FFT, sampling, off-axis, performance]
created: 2026-03-22
updated: 2026-03-22
---

# Off-Axis Adaptive FFT Sampling: OPD-Driven Grid vs Projection-Slice

> **Objective:** Compare two strategies for accelerating FFTPSF → ESF computation under off-axis conditions — (1) OPD gradient-driven adaptive grid selection and (2) direct 1D OTF computation based on the Projection-Slice Theorem — analyzing their physical foundations, mathematical derivations, sampling effects, and computational benefits.

---

## Table of Contents

- [[#1. Background: Asymmetry of Off-Axis PSF]]
- [[#2. Current Computation Pipeline and Bottleneck Analysis]]
- [[#3. Strategy One: OPD Gradient-Driven Adaptive Grid]]
  - [[#3.1 Physical Foundation: Pupil Plane Phase Sampling Theorem]]
  - [[#3.2 Mathematical Derivation: OPD Gradient → Nyquist Condition]]
  - [[#3.3 Off-Axis Extension: T/S Directional Gradient Asymmetry]]
  - [[#3.4 Adaptive Algorithm]]
  - [[#3.5 Sampling Effects and Speedup Estimation]]
- [[#4. Strategy Two: Direct LSF Computation via Projection-Slice Theorem]]
  - [[#4.1 Physical Foundation: From 2D PSF to 1D LSF]]
  - [[#4.2 Mathematical Derivation: OTF Slice and Pupil Row/Column Autocorrelation]]
  - [[#4.3 Off-Axis Case: Independent Slices for T and S Directions]]
  - [[#4.4 Algorithm Implementation]]
  - [[#4.5 Sampling Effects and Speedup Estimation]]
- [[#5. Comparative Analysis]]
  - [[#5.1 Computational Complexity Comparison]]
  - [[#5.2 Sampling Quality Comparison]]
  - [[#5.3 Implementation Difficulty Comparison]]
  - [[#5.4 Off-Axis Applicability Comparison]]
- [[#6. Combined Strategy and Conclusions]]
- [[#Appendix A: Symbol Table]]
- [[#Appendix B: Key Formula Summary]]

---

## 1. Background: Asymmetry of Off-Axis PSF

### 1.1 On-Axis vs Off-Axis Symmetry Differences

For rotationally symmetric optical systems, the PSF at different field positions exhibits different symmetry properties:

| Field | Aberration Composition | PSF Symmetry | Available Simplification |
|-------|----------------------|--------------|-------------------------|
| On-axis (0,0) | Defocus + spherical aberration | **Rotationally symmetric** $\text{PSF}(x,y) = \text{PSF}(r)$ | Hankel transform (2D→1D) |
| Off-axis $(H_x, H_y)$ | Defocus + spherical aberration + coma + astigmatism + field curvature | **Bilateral symmetry only** (see note below) | Mirror symmetry saves ~2× |

> **Coordinate convention note**: Bilateral symmetry $\text{PSF}(x,y) = \text{PSF}(x,-y)$ holds only under the following conditions: (1) the optical system itself is rotationally symmetric; (2) the coordinate system takes the meridional plane as the $xz$ plane (i.e., $y$ direction is sagittal); (3) the field point lies in the meridional plane ($H_x = 0$). For a general oblique field direction $(H_x \neq 0, H_y \neq 0)$, the symmetry axis rotates to the direction of the field vector. In practice, the coordinate system must be aligned with the field azimuth.

In the off-axis case, astigmatism causes different focal plane positions for the tangential (T) and sagittal (S) directions, and coma introduces asymmetric broadening in the T direction. **The blur size and morphology in the T and S directions are completely different** and must be analyzed separately.

### 1.2 Physical Picture of T/S Focal Plane Splitting

$$
\underset{z_T}{\text{T focal line}} \xrightarrow{\quad \Delta z_\text{ast} \quad} \underset{z_S}{\text{S focal line}} \qquad \longrightarrow z \text{ (optical axis)}
$$

where $\Delta z_\text{ast} = z_S - z_T > 0$ is the signed astigmatic focal separation.

| Image plane | T direction | S direction | PSF shape |
|------------|-------------|-------------|-----------|
| $z = z_T$ | In focus | Defocused | ── Horizontal line |
| $z = (z_T+z_S)/2$ | Partially defocused | Partially defocused | Minimum-confusion spot (typically elliptical) |
| $z = z_S$ | Defocused | In focus | │ Vertical line |

The astigmatic separation $\Delta z_\text{ast}$ varies with field angle and wavelength. **The T/S focal plane positions differ for different wavelengths** → chromatic aberration couples with astigmatism → CFW differs between T and S directions.

### 1.3 Impact on Sampling Strategy

The current code uses a **fixed square grid** `(num_rays, grid_size)` for all $(z, \lambda, H)$ combinations:

```python
fft_psf = FFTPSF(op, field=(0, 0), wavelength=wl_um_j,
                 num_rays=num_rays, grid_size=grid_size, ...)
```

In the off-axis case, this results in a dual waste:

1. **Axial waste**: Near the focal plane, the blur is small and does not require a large grid (same as on-axis)
2. **Directional waste**: The PSF may be very narrow in the T direction but wide in the S direction (or vice versa); one direction of the square grid is wasted

---

## 2. Current Computation Pipeline and Bottleneck Analysis

### 2.1 FFTPSF Computation Flow

For each wavelength $\lambda_j$ and each defocus position $z$:

$$
P[m,n] \xrightarrow{\text{zero-pad}} P_\text{pad}[G \times G] \xrightarrow{\text{FFT}_{2D}} h[G \times G] \xrightarrow{|\cdot|^2} \text{PSF}[G \times G] \xrightarrow{\sum_y} \text{LSF}(x) \xrightarrow{\int} \text{ESF}(x)
$$

where $N$ = `num_rays` (pupil plane sampling density), $G$ = `grid_size` (FFT grid size), $Q = G/(N-1)$ (oversampling factor).

### 2.2 Computational Cost of Each Step

| Step | Operation | Complexity | Typical Values (N=512, G=800) |
|------|-----------|-----------|-------------------------------|
| Ray tracing | $N^2$ rays through optical system | $O(N^2 \cdot S)$, $S$ = number of surfaces | $\sim 2.6 \times 10^5$ rays |
| Build pupil plane | OPD → complex pupil function | $O(N^2)$ | $\sim 2.6 \times 10^5$ ops |
| Zero padding | $N \times N \to G \times G$ | $O(G^2)$ | $\sim 6.4 \times 10^5$ ops |
| 2D FFT | Row FFT + column FFT | $O(G^2\log G)$† | $\sim 1.2 \times 10^7$ ops |
| Modulus squared | $\|h\|^2$ | $O(G^2)$ | $\sim 6.4 \times 10^5$ ops |
| Sum → LSF | $\sum_y \text{PSF}(x,y)$ | $O(G^2)$ | $\sim 6.4 \times 10^5$ ops |
| Cumulative sum → ESF | $\int \text{LSF}$ | $O(G)$ | $\sim 800$ ops |

**Bottleneck: Ray tracing** ($O(N^2 \cdot S)$) and **2D FFT** ($O(G^2\log G)$, see §2.3).

> †2D FFT complexity note, see §2.3: The theoretical optimum is $O((N{+}G)\,G\log G)$ (pruned FFT); the actual implementation (numpy) is $O(G^2\log G)$. The table uses the actual implementation complexity.

### 2.3 Detailed Breakdown of 2D FFT

The 2D FFT is implemented via row-column decomposition:

$$
\text{FFT}_{2D}\{P_\text{pad}\} = \text{FFT}_\text{col}\{\text{FFT}_\text{row}\{P_\text{pad}\}\}
$$

| Sub-step | Number of non-zero rows/columns | FFT length per transform | Total |
|----------|-------------------------------|--------------------------|-------|
| Row FFT | $N$ rows (remaining are zero-padded) | $G$ | $O(NG \log G)$ |
| Column FFT | $G$ columns (all non-zero after row FFT) | $G$ | $O(G^2 \log G)$ |

> **Key observation**: Row FFT only needs to process $N$ rows (non-zero rows), but column FFT must process all $G$ columns (after row FFT, the output spreads across all $G$ columns in the frequency domain). Column FFT dominates the 2D FFT computation.

**Two-level complexity comparison:**

| Level | Row FFT | Column FFT | Total | Typical Values (N=512, G=800) |
|-------|---------|------------|-------|-------------------------------|
| Theoretical optimum (pruned FFT, skip zero rows) | $O(NG\log G)$ | $O(G^2\log G)$ | $O\big((N{+}G)\,G\log G\big)$ | $\sim 1.0 \times 10^7$ |
| Actual implementation (numpy `fft2`, dense FFT) | $O(G^2\log G)$ | $O(G^2\log G)$ | $O(G^2\log G)$ | $\sim 1.2 \times 10^7$ |

> numpy's `fft2` is based on pocketfft, a general-purpose dense FFT implementation that does not detect zero rows, so row FFT also processes all $G$ rows. Zero-row skipping optimization (pruned FFT) requires a specialized implementation; for this project where $N/G = 512/800 \approx 0.64$, the benefit is approximately 20%.

---

## 3. Strategy One: OPD Gradient-Driven Adaptive Grid

### 3.1 Physical Foundation: Pupil Plane Phase Sampling Theorem

The input to FFTPSF is the **complex pupil function**:

$$
P(u,v) = A(u,v) \cdot \exp\!\Big(\frac{2\pi i}{\lambda}\, \text{OPD}(u,v)\Big)
$$

where $(u,v)$ are normalized pupil coordinates ($u^2 + v^2 \leq 1$), $A$ is the amplitude transmittance (determined by the aperture stop), and $\text{OPD}$ is the optical path difference (in µm).

The phase in the complex exponential $e^{i\phi}$ is:

$$
\phi(u,v) = \frac{2\pi}{\lambda}\, \text{OPD}(u,v)
$$

The spatial rate of change of $\phi$ (i.e., the phase gradient) determines the **spatial frequency content** of $P$:

$$
\nabla\phi = \frac{2\pi}{\lambda} \nabla\text{OPD}
$$

> **Physical meaning**: Large $|\nabla\phi|$ → the complex pupil function oscillates rapidly in that region → denser sampling points are needed. Small $|\nabla\phi|$ → the pupil function varies slowly → sparse sampling suffices.

The quantitative conclusion of this observation is: the maximum spatial frequency that $N$ sampling points can correctly represent is determined by the Shannon-Nyquist theorem, and the maximum value of $|\nabla\phi|$ is precisely the highest spatial frequency of the pupil function. By equating the two, one obtains an explicit expression for $N_\text{min}$ in terms of the OPD gradient (§3.2).

### 3.2 Mathematical Derivation: OPD Gradient → Nyquist Condition

#### 3.2.1 Local Spatial Frequency of the Pupil Function

At pupil coordinate $(u,v)$, the **instantaneous spatial frequency** of the complex exponential is:

$$
f_u(u,v) = \frac{1}{2\pi}\frac{\partial\phi}{\partial u} = \frac{1}{\lambda}\frac{\partial\,\text{OPD}}{\partial u}
$$

$$
f_v(u,v) = \frac{1}{2\pi}\frac{\partial\phi}{\partial v} = \frac{1}{\lambda}\frac{\partial\,\text{OPD}}{\partial v}
$$

The units are cycles/aperture (since $u, v \in [-1, 1]$).

#### 3.2.2 Sampling Theorem Requirements

Pupil coordinate $u \in [-1, 1]$, so the normalized pupil diameter is $D = 2$ (dimensionless). With $N$ sampling points spanning the diameter, the sampling interval is:

$$
\Delta u = \frac{2}{N - 1}
$$

The Shannon-Nyquist theorem requires $\Delta u < \dfrac{1}{2\,f_\text{max}}$, i.e.:

$$
\frac{2}{N-1} < \frac{\lambda}{2\,\max|\partial\text{OPD}/\partial u|}
$$

Rearranging:

$$
\boxed{N - 1 > \frac{4}{\lambda} \cdot \max_{(u,v)}\left|\frac{\partial\,\text{OPD}}{\partial u}\right|}
$$

$$
\boxed{N - 1 > \frac{4}{\lambda} \cdot \max_{(u,v)}\left|\frac{\partial\,\text{OPD}}{\partial v}\right|}
$$

where $\text{OPD}$ is in µm, $\lambda$ is in µm, and $N$ is the number of pupil sampling points in that direction. The coefficient **4** (rather than 2) comes from the product of the pupil diameter $D = 2$ and the Nyquist factor of 2.

> **Equivalent waves formulation (used uniformly in subsequent sections)**: Let $\text{OPD}_w = \text{OPD}_{\mu\text{m}}/\lambda$ (waves), then $N - 1 > 4\,\max|\partial\text{OPD}_w/\partial u|$. The advantage of the waves form: Seidel coefficients $W_{ij}$ are inherently in waves, requiring no additional $1/\lambda$ conversion.

#### 3.2.3 Physical Sources of OPD Gradient

For the major aberrations, the OPD and its maximum gradient (**uniformly in waves units**, i.e., $\text{OPD}_w = \text{OPD}_{\mu\text{m}}/\lambda$):

| Aberration | $\text{OPD}_w(\rho)$ (waves) | $\max\|\partial\text{OPD}_w/\partial u\|$ (waves/apt) | Notes |
|------------|------------------------------|------------------------------------------------------|-------|
| Defocus | $W_{20}\,\rho^2$ | $2W_{20}$ | $W_{20} = \frac{z}{8\lambda F_{\#}^2}$, **depends on $\lambda$** |
| Spherical aberration (SA) | $W_{40}\,\rho^4$ | $4W_{40}$ | Maximum at $\rho=1$ |
| Coma | $W_{31}\,H\rho^3\cos\theta$ | $\sim 3W_{31}H$ | Off-axis, T direction dominant |
| Astigmatism | $W_{22}\,H^2\rho^2\cos^2\theta$ | $\sim 2W_{22}H^2$ | Off-axis, T/S magnitudes equal |

> **Unit convention**: This table and all subsequent OPD gradients use **waves/aperture** (waves divided by normalized pupil coordinate change). Since $W_{20} \propto 1/\lambda$, the same physical defocus $z$ produces a larger waves gradient at shorter wavelengths → **$\lambda_\text{min}$ yields the most stringent sampling condition**.

For **pure defocus** (the dominant aberration), derivation in waves form is more direct:

$$
W_{20} = \frac{z}{8\lambda F_{\#}^2} \quad (\text{waves})
$$

$$
\max\left|\frac{\partial\,\text{OPD}_w}{\partial u}\right| = 2W_{20} = \frac{z}{4\lambda F_{\#}^2} \quad (\text{waves/aperture, maximum at } u=1)
$$

Substituting into the waves-form Nyquist condition $N - 1 > 4\,\max|\partial\text{OPD}_w/\partial u|$:

$$
N - 1 > 4 \cdot \frac{z}{4\lambda F_{\#}^2} = \frac{z}{\lambda F_{\#}^2}
$$

> **Verification of consistency with µm form**: $\max|\partial\text{OPD}_{\mu\text{m}}/\partial u| = 2W_{20}\lambda = z/(4F_{\#}^2)$, substituting into the µm form $N-1 > (4/\lambda)\cdot\text{grad}_{\mu\text{m}}$ yields the same result $z/(\lambda F_{\#}^2)$. The two unit paths are **identically equivalent**.

> **Example**: $z = 700\,\mu\text{m}$, $\lambda_\text{min} = 0.40\,\mu\text{m}$ (the shortest wavelength gives the most stringent condition), $F_{\#} = 2$:
> $$N - 1 > \frac{700}{0.40 \times 4} = 437.5 \implies N \geq 439$$
> Compared to the current fixed $N = 512$, there is still margin at maximum defocus + shortest wavelength.
>
> For $\lambda = 0.55\,\mu\text{m}$: $N - 1 > 700 / (0.55 \times 4) = 318$, even more margin.

### 3.3 Off-Axis Extension: T/S Directional Gradient Asymmetry

In the off-axis case, different aberrations contribute differently to the OPD gradient in the T and S directions. Let $u$ be the tangential (meridional) direction and $v$ be the sagittal direction:

#### 3.3.1 Contribution of Astigmatism

The **Seidel astigmatism term** is $W_{22}H^2\rho^2\cos^2\theta$. Using $\cos^2\theta = \frac{1+\cos2\theta}{2}$ and $\rho^2\cos2\theta = u^2 - v^2$ (where $u$ is the meridional T direction and $v$ is the sagittal S direction in normalized pupil coordinates), it decomposes as:

$$W_{22}H^2\rho^2\cos^2\theta = \frac{W_{22}H^2}{2}\underbrace{\rho^2}_{\text{equivalent defocus}} + \frac{W_{22}H^2}{2}\underbrace{(u^2 - v^2)}_{\text{pure astigmatism}}$$

The physical effect of astigmatism is to add **an additional defocus contribution** $W_{22}H^2$ to the T direction, while the S direction is unaffected (this is the decomposition of the original Seidel term $W_{22}H^2\rho^2\cos^2\theta = W_{22}H^2 u^2$; if one uses the medial focal surface as the reference, the T/S each receive a symmetric split of $\pm W_{22}H^2/2$, see the formula derivation above). After superposition with the external defocus $W_{20} = z/(8\lambda F_\#^2)$, the total OPD is:

$$\text{OPD}(u,v) = \underbrace{(W_{20} + W_{22}H^2)}_{\text{T equivalent defocus}}\,u^2 + \underbrace{W_{20}}_{\text{S equivalent defocus}}\,v^2$$

Its gradients:

$$
\frac{\partial\,\text{OPD}}{\partial u} = 2(W_{20} + W_{22}H^2)\,u, \qquad \frac{\partial\,\text{OPD}}{\partial v} = 2W_{20}\,v
$$

Maximum gradients in the T and S directions:

| Direction | Maximum Gradient ($\rho=1$) |
|-----------|----------------------------|
| T | $2|W_{20} + W_{22}H^2|$ |
| S | $2|W_{20}|$ |

> **Key conclusion**: The gradients in the two directions are equal only at the **medial focal surface** $W_{20} = -W_{22}H^2/2$. At other defocus positions $z$ scanned in CFW analysis, astigmatism causes asymmetric sampling requirements between the T/S directions — the T direction has a larger (or smaller) equivalent defocus, requiring a different sampling density.
>
> **Sign convention**: The signs of $W_{20}$ and $W_{22}$, as well as descriptions like "T underfocused / S overfocused," depend on the OPD sign convention and $z$-axis direction.

#### 3.3.2 Contribution of Coma

Coma OPD:

$$
\text{OPD}_\text{coma}(u,v) = W_{31}\,H\,(u^2 + v^2)\,u
$$

> **Formula derivation**: The Seidel coma term is $W_{31}H\rho^3\cos\theta$.
> Using $\rho^2 = u^2 + v^2$ and $\rho\cos\theta = u$:
> $$W_{31}H\rho^3\cos\theta = W_{31}H\cdot\rho^2\cdot\rho\cos\theta = W_{31}H\,(u^2+v^2)\,u$$
> Expanding to $W_{31}H\,(u^3 + uv^2)$, which is an odd function of $u$ — reflecting the asymmetric nature of the coma spot (the comatic tail extends in only one direction).

Its gradients:

$$
\frac{\partial\,\text{OPD}_\text{coma}}{\partial u} = W_{31}\,H\,(3u^2 + v^2)
$$

$$
\frac{\partial\,\text{OPD}_\text{coma}}{\partial v} = 2W_{31}\,H\,u\,v
$$

At pupil edge $(u=1, v=0)$: T direction gradient is $3W_{31}H$, S direction gradient is $0$.
At $(u=0, v=1)$: T direction gradient is $W_{31}H$, S direction gradient is $0$.
At $(u=1/\sqrt{2}, v=1/\sqrt{2})$: T direction gradient is $2W_{31}H$, S direction gradient is $W_{31}H$.

> **Conclusion**: Coma causes the **T direction OPD gradient to be systematically larger than the S direction**. Off-axis field points require denser pupil sampling in the T direction.

#### 3.3.3 Combined T/S Gradient Estimation

Superimposing the OPD gradient contributions from all aberrations and taking the maximum over the pupil (in units of waves/aperture, evaluated at a specific $\lambda$):

$$
G_T(\lambda) = \max_{(u,v)}\left|\frac{\partial}{\partial u}\sum_k \text{OPD}_{w,k}(u,v;\lambda)\right|
$$

$$
G_S(\lambda) = \max_{(u,v)}\left|\frac{\partial}{\partial v}\sum_k \text{OPD}_{w,k}(u,v;\lambda)\right|
$$

where $\sum_k$ runs over all aberration terms (defocus, spherical aberration, coma, astigmatism, etc.), and $\text{OPD}_{w,k} = \text{OPD}_{\mu\text{m},k}/\lambda$ is the OPD of the $k$-th term in waves. $G_T, G_S$ respectively represent the instantaneous spatial frequency at the "fastest oscillating" location within the pupil plane in the T and S directions (§3.2.1).

Under the constraint of a square FFT grid (optiland only supports square grids), sampling is determined by the larger gradient:

$$
\boxed{N_\text{min} = \max\!\left(\,N_\text{floor},\;\left\lceil 4\,\max\!\big(G_T(\lambda_\text{min}),\; G_S(\lambda_\text{min})\big) \right\rceil + 1\,\right)}
$$

Meaning of each factor:

- **$\max(G_T, G_S)$**: The square grid shares the same $N$ for T and S directions, so it must satisfy the more stringent direction's sampling requirement. For example, when $G_T=100, G_S=60$, the T direction determines $N$ (the S direction is oversampled).
- **Coefficient 4**: Comes from the product of two factors:
  - Nyquist theorem requires sampling rate $\geq 2\times$ the maximum frequency;
  - $G_{T/S}$ has units of waves/aperture (per pupil **diameter**), and $N$ sampling points cover normalized coordinates $[-1,1]$ (span = 2 = 1 diameter), so an additional $\times 2$ is needed to convert from frequency to number of sampling points. Combined: $2 \times 2 = 4$.
- **$+1$**: $N-1$ intervals correspond to $N$ sampling points, so $N = \lceil 4G \rceil + 1$.
- **$N_\text{floor}$**: Lower bound protection (e.g., 64); even when the gradient is very small (near focus), the PSF resolution should not be too poor.
- **$\lambda_\text{min}$**: $\text{OPD}_w = \text{OPD}_{\mu\text{m}}/\lambda$; the same physical OPD corresponds to more waves at shorter wavelengths → larger gradient → more stringent sampling condition. Therefore, the shortest wavelength (e.g., 400 nm) gives the most conservative estimate.

### 3.4 Adaptive Algorithm

Core idea: Embed the $N_\text{min}$ formula from §3.3.3 into the baking workflow, so that each defocus position $z$ uses a **minimally required grid computed on demand**, rather than a fixed $(N, G) = (512, 800)$. The algorithm has three steps:

1. **Pre-scan** (§3.4.1): Low-density ray tracing to obtain OPD gradients → compute $N_\text{min}$
2. **Coverage verification** (§3.4.2): Check whether the FFT output is large enough to contain the PSF
3. **Integration into baking** (§3.4.3): Replace fixed grid with adaptive grid

#### 3.4.1 Pre-Scan to Obtain OPD

Use **low-density ray tracing** (e.g., $32 \times 32$) to obtain the OPD map at the wavelength endpoints and compute gradients. The cost of this step is $\sim 0.4\%$ of the main computation ($32^2 / 512^2 \approx 1/256$, with no FFT).

```python
def _adaptive_grid_params(optic, z_defocus_um, field, wl_nm_arr,
                          Q_min=1.5, min_rays=64, safety=1.5):
    """OPD gradient-driven adaptive grid parameter selection."""
    N_PRE = 32                            # Pre-scan pupil sampling points
    op = _optic_at_defocus(optic, z_defocus_um)

    # Low-density wavefront tracing
    wf = Wavefront(op, [field], num_rays=N_PRE, strategy="chief_ray")

    # Pupil coordinates u ∈ [-1, 1], total span D=2, sampling interval:
    du = 2.0 / (N_PRE - 1)               # ← correct interval

    max_grad = 0.0
    for wl_nm in [wl_nm_arr.min(), wl_nm_arr.max()]:
        wl_um = float(wl_nm) / 1000.0
        data = wf.get_data(field, wl_um)
        opd = data.opd                    # Units: waves (= OPD/λ)
        # np.gradient(opd, du) → gradient units: waves / (normalized pupil coord)
        gy, gx = np.gradient(opd, du)     # ← pass interval to avoid manual scaling
        grad = np.sqrt(gx**2 + gy**2)
        max_grad = max(max_grad, float(np.nanmax(grad)))

    # Nyquist (§3.2.2): N-1 > 4 × max|∂OPD_waves/∂u|
    nr = max(min_rays, int(np.ceil(4 * max_grad * safety)) + 1)
    nr = int(2 ** np.ceil(np.log2(nr)))   # Align to power of 2

    gs = int(np.ceil(Q_min * (nr - 1)))
    gs += gs % 2                          # Align to even number
    return nr, gs
```

> **Note on gradient scaling**: `np.gradient(opd)` assumes a sampling interval of 1 by default (i.e., 1 sample), returning waves/sample. To obtain waves/(normalized pupil coordinate), **the actual interval must be passed** as `du = 2/(N_pre - 1)`, or manually divided by `du`. The previous version using `* 32` was incorrect (the correct factor is `(N_pre - 1) / 2 = 15.5`, a difference of approximately $2\times$).

#### 3.4.2 Physical Coverage Verification

The physical half-width of the FFT output is:

$$
x_\text{max} = \frac{N-1}{2} \cdot \lambda \cdot F_{\#}
$$

This must satisfy $x_\text{max} \geq r_\text{blur}$ (blur radius), otherwise the PSF is truncated. Geometric blur radius:

$$
r_\text{geo} = \frac{|z|}{2F_{\#}}, \qquad r_\text{airy} = 1.22\,\lambda\,F_{\#}
$$

$$
r_\text{blur} = \max(r_\text{geo},\; r_\text{airy}) \times s_\text{safety}
$$

For a **pure defocus-dominated** wavefront, this coverage condition and the OPD gradient condition give **same-order** $N_\text{min}$ estimates (both proportional to $z / (\lambda F_{\#}^2)$). However, the two are **not strictly equivalent**:

- The **OPD gradient condition** prevents pupil plane phase aliasing (FFT input side), related to the spatial frequency of the aberration structure;
- The **coverage condition** prevents PSF truncation (FFT output side), related to the physical size of the blur.

When higher-order aberrations (e.g., spherical aberration, coma) are significant, the two conditions may give different $N_\text{min}$ values. In practice, one should take the larger of the two, or directly use the OPD gradient condition (more conservative, as it implicitly includes the coverage condition).

#### 3.4.3 Integration into the Baking Workflow

**Fixed grid (current implementation)** — all $z$ use the same $(N, G)$:

```python
for z in z_bake:                          # 29 defocus positions
    for wl in wl_arr:                     # 11 wavelengths
        FFTPSF(num_rays=512, grid_size=800)   # Fixed, always maximum specification
```

**Adaptive grid** — each $z$ selects the minimum $(N, G)$ based on OPD gradient:

```python
for z in z_bake:
    N, G = _adaptive_grid_params(optic, z, field, wl_arr)  # §3.4.1
    for wl in wl_arr:
        FFTPSF(num_rays=N, grid_size=G)   # Adjusted on demand
```

Effect: At small defocus, OPD gradient is small → $N$ is small (e.g., 64) → FFT grid is small (e.g., 96) → computation reduced by ~100×; at large defocus, $N$ approaches 512 → comparable to fixed grid. The speedup for the full scan depends on the distribution of $z$ points; see §3.5 for quantitative estimation.

> **Note**: All wavelengths under the same $z$ share one set of $(N, G)$, because $N_\text{min}$ has already been computed for the most stringent $\lambda_\text{min}$. No per-wavelength grid adjustment is needed.

### 3.5 Sampling Effects and Speedup Estimation

Using the Nikon AI Nikkor 85mm f/2S as an example, $F_{\#} = 2$, $\lambda \in [0.40, 0.70]\,\mu\text{m}$:

#### 3.5.1 On-Axis Scenario

Using the waves-form Nyquist condition $N - 1 > 4\,\max|\partial\text{OPD}_w / \partial u|$, taking $\lambda_\text{min} = 0.40\,\mu\text{m}$ (corresponding to the maximum waves gradient):

| $z$ (µm) | $\max\|\partial\text{OPD}_w/\partial u\|$ (waves/apt, @$\lambda_\text{min}$) | $N - 1 >$ $4 \times \text{grad}_w$ | $N_\text{min}$ (power of 2) | $G_\text{min}$ (Q=1.5) | Per-point $C_i/C_\text{base}$ | Per-point speedup |
|-----------|-----------------------------------------------------------------------------|-------------------------------------|------------------------|----------------------|--------------------------|----------|
| 0 | ~5–10 (SA residual only) | ~20–40 | 64 | 96 | 0.010 | **~100×** |
| 100 | $z/(4\lambda_\text{min}F_{\#}^2)$ = 15.6 | 63 | 64 | 96 | 0.010 | **~100×** |
| 300 | 46.9 | 188 | 256 | 384 | 0.21 | **~5×** |
| 500 | 78.1 | 313 | 512 | 768 | 0.93 | ~1.1× |
| 700 | 109.4 | 438 | 512 | 768 | 0.93 | ~1.1× |

> **Note**: $C = G^2\,\log_2 G$ (numpy dense FFT, see §2.3), baseline $C_\text{base} = 800^2\times\log_2 800 \approx 6.2\text{M}$. All ratios are computed directly from this formula, rounded to two significant digits. Full-scan speedup is the weighted average over all points; see §3.5.3.

#### 3.5.2 Off-Axis Scenario (H=0.7)

In the off-axis case, coma and astigmatism increase the OPD gradient, especially in the T direction. The following are waves/aperture estimates at $\lambda_\text{min} = 0.40\,\mu\text{m}$ (typical values, dependent on the specific lens's Seidel coefficients):

| $z$ (µm) | $G_T(\lambda_\text{min})$ (w/apt) | $G_S(\lambda_\text{min})$ (w/apt) | $N - 1 > 4\max(G_T, G_S)$ | $N_\text{min}$ | Notes |
|-----------|-----------------------------------|-----------------------------------|---------------------------|----------------|-------|
| 0 | ~25 (coma+astigmatism) | ~12 | 100 | 128 | Coma makes $G_T > G_S$ |
| 300 | ~72 (defocus+coma) | ~59 (defocus+astigmatism) | 288 | 512 | Defocus begins to dominate |
| 700 | ~134 | ~121 | 536 | 1024 | Exceeds current N=512! |

> **Important finding**: At off-axis large defocus + short wavelength, the current fixed $N = 512$ may **not satisfy the strict Nyquist condition**. The actual impact depends on (1) whether the spectral weight at $\lambda_\text{min}$ is small enough to tolerate mild aliasing; (2) the smoothing effect of ESF integration on high-frequency aliasing. A numerical regression test is recommended to confirm the acceptable threshold.
>
> **The speedup near the focal plane is reduced in the off-axis case**, because even at $z=0$, the OPD gradients from coma and astigmatism are non-negligible. However, significant savings remain in the moderate defocus region.

#### 3.5.3 Overall Speedup Estimation

For the full scan of 29 $z$ points ($z \in [-700, +700]\,\mu\text{m}$, step size 50 µm), using 2D FFT complexity $C_i = G_i^2\,\log_2 G_i$ (numpy dense FFT, see §2.3).

**On-Axis per-point distribution** (based on $N_\text{min}$ from §3.5.1 aligned to powers of 2):

| $N_\text{min}$ | $G_\text{min}$ | Covered $z$ range | Number of points (out of 29) | Per-point $C_i / C_\text{base}$ |
|----------------|----------------|-------------------|------------------------------|--------------------------------|
| 64 | 96 | $|z| \leq 100$ | 5 | 0.010 |
| 256 | 384 | $100 < |z| \leq 400$ | 12 | 0.21 |
| 512 | 768 | $|z| > 400$ | 12 | 0.93 |

Weighted average speedup $= 29 / (5 \times 0.010 + 12 \times 0.21 + 12 \times 0.93) = 29 / 13.7 \approx$ **2.1×** (FFT portion only; end-to-end speedup including ray tracing is higher, since $N^2$ scales proportionally).

> **On-axis full-scan speedup**: Since reduction is not possible at large $|z|$ (12/29 points still require $N=512$, $C_i/C_\text{base} \approx 0.93$), the full-scan weighted average is approximately **~2×**. If only the sub-interval $|z| \leq 300$ is considered (5 out of 17 points can be reduced to 64/96), the speedup can reach **~5×**.

**Off-Axis (H=0.7) case**:

| $N_\text{min}$ | Number of points (out of 29) | Notes |
|----------------|------------------------------|-------|
| 128 | ~3 ($|z| \leq 50$) | Savings near focal plane |
| 512 | ~16 ($50 < |z| \leq 500$) | Same as baseline |
| 1024 | ~10 ($|z| > 500$) | **Exceeds baseline** — strict Nyquist requires larger grid |

> **Off-axis conclusion**: Under strict Nyquist, off-axis points at large defocus + short wavelength require $N > 512$, meaning the adaptive strategy **not only fails to accelerate but actually requires more computation than the fixed baseline**. The overall speedup depends heavily on (1) whether mild aliasing at $\lambda_\text{min}$ is acceptable; (2) the $z$ interval of interest. **The off-axis full-scan speedup requires empirical validation**; optimistic estimate 1.5–3× (if aliasing at $\lambda < 0.45\,\mu\text{m}$ is tolerated), conservative estimate possibly < 1× (if strict full-band Nyquist is required).

---

## 4. Strategy Two: Direct LSF Computation via Projection-Slice Theorem

### 4.1 Physical Foundation: From 2D PSF to 1D LSF

ESF analysis only requires the **1D line spread function** (LSF), not the full 2D PSF. The current method computes the full 2D PSF and then sums along one direction:

$$
\text{LSF}_T(x) = \sum_y \text{PSF}(x, y), \qquad \text{LSF}_S(y) = \sum_x \text{PSF}(x, y)
$$

> **Core question**: Is there a method to **directly obtain the 1D LSF without computing the full 2D PSF**?

The answer comes from the **Projection-Slice Theorem** (Fourier Slice Theorem).

### 4.2 Mathematical Derivation: OTF Slice and Pupil Row/Column Autocorrelation

#### 4.2.1 Projection-Slice Theorem

Let $f(x,y)$ be a two-dimensional function and $F(f_x, f_y) = \mathcal{F}_{2D}\{f\}$ be its 2D Fourier transform. The projection of $f$ along the $y$ direction is:

$$
p(x) = \int f(x,y)\,dy
$$

The Projection-Slice Theorem states:

$$
\boxed{\mathcal{F}_{1D}\{p\}(f_x) = F(f_x, 0)}
$$

That is: **The 1D Fourier transform of the projection = the 1D slice of the 2D Fourier transform along the corresponding axis.**

#### 4.2.2 Application to PSF → LSF

Applying the theorem to the PSF:

$$
\mathcal{F}\{\text{LSF}_T\}(f_x) = \text{OTF}(f_x, 0)
$$

$$
\mathcal{F}\{\text{LSF}_S\}(f_y) = \text{OTF}(0, f_y)
$$

where $\text{OTF} = \mathcal{F}\{\text{PSF}\}$ is the optical transfer function.

> **Physical meaning**: The spectrum of the tangential LSF is determined solely by the one-dimensional slice of the OTF along the $f_x$ axis. To compute $\text{LSF}_T$, **only the OTF values at $f_y = 0$ are needed**, not the full 2D OTF.

#### 4.2.3 Pupil Plane Expression of the OTF Slice

The OTF equals the normalized autocorrelation of the pupil function (i.e., the cross-correlation of the pupil function with its own conjugate). In convolution notation:

$$
\text{OTF} = \frac{P \star P}{E} = \frac{P(-\cdot) * P^*}{E}
$$

where $P = P[m,n]$ is the complex pupil function from §2.1 ($P = A \cdot e^{i\phi}$, containing amplitude and phase), $\star$ denotes cross-correlation, $*$ denotes convolution, and $E = \sum_{m,n}|P[m,n]|^2$ is the total pupil energy (normalization constant ensuring $\text{OTF}(0,0)=1$). Expanded in discrete summation form:

$$
\text{OTF}(\Delta u, \Delta v) = \frac{\displaystyle\sum_m \sum_n P[m,n] \cdot P^*[m-\Delta u,\; n-\Delta v]}{E}
$$

> **Intuition**: The autocorrelation measures "the degree of overlap between the pupil function and itself shifted by $(\Delta u, \Delta v)$." Small shift → large overlap → high OTF (good low-frequency transfer); large shift → small overlap → low OTF (high-frequency attenuation).

For $\Delta v = 0$ (i.e., the $f_y = 0$ slice), the shift is only in the $u$ direction, and the $n$ index is not offset:

$$
\text{OTF}(\Delta u, 0) = \frac{1}{E}\sum_n \underbrace{\sum_m P[m,n] \cdot P^*[m-\Delta u,\; n]}_{R_n(\Delta u)}
$$

where $R_n(\Delta u) = \sum_m P[m,n] \cdot P^*[m - \Delta u, n]$ is the **one-dimensional autocorrelation** of the $n$-th **row** of the pupil plane along the $m$ (horizontal/T) direction. In convolution notation: $R_n = P_n \star P_n$ (cross-correlation of the $n$-th row with itself).

$$
\boxed{\text{OTF}(\Delta u, 0) = \frac{1}{E}\sum_{n} R_n(\Delta u)}
$$

> **Implication**: The horizontal slice of the OTF = the sum of all row autocorrelations. The 2D autocorrelation at $\Delta v = 0$ reduces to the sum of row-wise 1D autocorrelations — this is the key to the dimensionality reduction achieved by the projection-slice method.

The sagittal direction follows analogously: take the $\Delta u = 0$ slice, perform 1D autocorrelation along each **column**, then sum:

$$
\text{OTF}(0, \Delta v) = \frac{1}{E}\sum_{m} R_m(\Delta v), \qquad R_m(\Delta v) = \sum_n P[m,n] \cdot P^*[m,\; n-\Delta v]
$$

The computational structure for both directions is completely symmetric, with only rows and columns interchanged.

#### 4.2.4 Computing Row Autocorrelation via Power Spectrum

Directly computing the autocorrelation by definition $R_n(\Delta u) = \sum_m P[m,n] \cdot P^*[m-\Delta u, n]$ requires traversing all $m$ for each shift, with complexity $O(N^2)$. The **Wiener-Khinchin theorem** provides an $O(N\log N)$ shortcut:

$$
R_n(\Delta u) = \text{IFFT}\!\big\{|\text{FFT}\{P[{\cdot}, n]\}|^2\big\}(\Delta u)
$$

That is: autocorrelation = inverse Fourier transform of the power spectrum.

> **Derivation** (based on the convolution theorem):
> 1. Autocorrelation is essentially the cross-correlation of the signal with its own conjugate: $R_n = P_n \star P_n$
> 2. Convolution theorem: time-domain cross-correlation = frequency-domain product, i.e., $\text{FFT}\{P_n \star P_n\} = F_n \cdot F_n^* = |F_n|^2$
> 3. Taking IFFT of both sides: $R_n = \text{IFFT}\{|F_n|^2\}$ $\quad\square$
>
> **Intuition**: Autocorrelation measures "the similarity between a signal and its shifted self," while the power spectrum measures "the intensity of each frequency component." Both contain the same information (both discard phase), so they can be converted to each other via Fourier transform.

Therefore:

$$
\text{OTF}(\cdot, 0) = \frac{1}{E}\,\text{IFFT}\!\left\{\sum_n |\text{FFT}\{P[\cdot, n]\}|^2\right\}
$$

> First perform 1D FFT on each row, take the modulus squared, accumulate power spectra, then one final IFFT → obtain the horizontal slice of the OTF → then one more IFFT → obtain $\text{LSF}_T$.

#### 4.2.5 Complete Derivation: Parseval's Theorem Directly to LSF

The OTF autocorrelation path (§4.2.3–4.2.4) described above, while correct, introduces a "two IFFT" intermediate step that can easily confuse the frequency/spatial domains. A **more direct derivation** uses Parseval's theorem to arrive at the result in one step.

**Derivation goal**: The current flow for computing $\text{LSF}_T$ is $P \to \text{2D FFT} \to |{\cdot}|^2 \to \text{PSF} \to \sum_y \to \text{LSF}_T$, where the 2D FFT includes row FFT and column FFT. We want to prove: **the column FFT can be skipped**, and $\text{LSF}_T$ can be obtained directly from the row FFT result.

##### Step 1: Row-Column Decomposition of 2D FFT

Let $P_\text{pad}$ be the $G \times G$ zero-padded pupil (row index $p$ = sagittal/vertical direction, column index $q$ = tangential/horizontal direction), and denote the 2D FFT result as $h$:

$$
h[k_x, k_y] = \text{FFT}_{2D}\{P_\text{pad}\}[k_x, k_y]
$$

The 2D FFT can be split into two steps — rows first, then columns:

**Row FFT**: Perform 1D FFT on the $p$-th row along the horizontal direction, obtaining that row's spectrum:

$$
R_p[k_x] = \text{FFT}_q\{P_\text{pad}[p, \cdot]\}[k_x] = \sum_q P_\text{pad}[p, q]\,e^{-2\pi i\,q\,k_x / G}
$$

**Column FFT**: For each frequency $k_x$, perform 1D FFT along the vertical direction ($p$), obtaining the complete $h$:

$$
h[k_x, k_y] = \text{FFT}_p\{R_\cdot[k_x]\}[k_y] = \sum_p R_p[k_x]\,e^{-2\pi i\,p\,k_y / G}
$$

> At this point, the intermediate result $R$ is a $G \times G$ matrix: each row is the 1D spectrum of that row's pupil data. The column FFT combines these row spectra in the vertical direction into the complete 2D spectrum $h$.

##### Step 2: Definition of LSF_T

PSF = $|h|^2$, and the tangential LSF is the sum of PSF along the $k_y$ direction (compressing 2D to 1D):

$$
\text{LSF}_T[k_x] = \sum_{k_y} |h[k_x, k_y]|^2
$$

Substituting the column FFT result from Step 1:

$$
\text{LSF}_T[k_x] = \sum_{k_y} \left|\sum_p R_p[k_x]\,e^{-2\pi i\,p\,k_y/G}\right|^2
$$

##### Step 3: Identifying the DFT Structure

For a fixed $k_x$, $R_0[k_x], R_1[k_x], \ldots, R_{G-1}[k_x]$ is a sequence of length $G$. Let $a[p] = R_p[k_x]$, then the column FFT is just a DFT of $a$:

$$
h[k_x, k_y] = \text{DFT}\{a\}[k_y]
$$

Therefore LSF_T becomes:

$$
\text{LSF}_T[k_x] = \sum_{k_y} |\text{DFT}\{a\}[k_y]|^2
$$

> The structure here is: "first DFT, then modulus squared, then sum over all frequencies" — this is precisely the applicable form of **Parseval's theorem**.

##### Step 4: Applying Parseval's Theorem

Parseval's theorem states: the total energy in the frequency domain equals the total energy in the time domain (multiplied by $G$ under numpy's unnormalized convention):

$$
\sum_{k} |\text{DFT}\{a\}[k]|^2 = G \sum_p |a[p]|^2
$$

> **Intuition**: The DFT is merely a different way of "viewing" the signal (from time domain to frequency domain) without changing the total energy. So "first DFT, then modulus squared, then sum" is equivalent to "directly modulus squared, then sum."

Substituting $a[p] = R_p[k_x]$:

$$
\sum_{k_y} |h[k_x, k_y]|^2 = G \sum_p |R_p[k_x]|^2 = G \sum_p |\text{FFT}_q\{P_\text{pad}[p,\cdot]\}[k_x]|^2
$$

> **This is the key step**: The left side requires column FFT ($O(G\log G)$), while the right side only requires summation ($O(G)$). The fine frequency decomposition performed by the column FFT is completely averaged out by the $\sum_{k_y}$ summation, so it is simply unnecessary.

##### Final Result

$$
\boxed{\text{LSF}_T[k_x] = G \sum_{p=0}^{G-1} \big|\text{FFT}_q\{P_\text{pad}[p, \cdot]\}[k_x]\big|^2}
$$

Computation flow comparison:

$$
\text{Original method:}\; P \xrightarrow{\text{row FFT}} R \xrightarrow{\text{col FFT}} h \xrightarrow{|\cdot|^2} \text{PSF} \xrightarrow{\sum_y} \text{LSF}_T \qquad O(G^2\log G)
$$

$$
\text{New method:}\; P \xrightarrow{\text{row FFT}} R \xrightarrow{|\cdot|^2} |R|^2 \xrightarrow{\sum_p} \text{LSF}_T \qquad O(NG\log G)
$$

> **Key conclusion**: The tangential component of the LSF is **proportional to the per-frequency accumulation of all row power spectra**. Only row-direction FFTs are needed, **no column FFT, no IFFT, no OTF intermediate step**. The physical essence of Parseval's theorem is energy conservation — summing PSF along $k_y$ is equivalent to directly summing power in the pupil row space.
>
> Since zero-padded rows ($p \notin [\text{pad}, \text{pad}+N)$) have $R_p \equiv 0$, only $N$ non-zero rows actually need to be processed.
>
> **FFT normalization convention**: This paper adopts numpy's default convention (forward DFT unnormalized, inverse with $1/G$), so the coefficient $G$ appears in Parseval's theorem. If the unitary convention is adopted (forward/inverse each with $1/\sqrt{G}$), this coefficient becomes $1$, but the final conclusion remains unchanged.
>
> **Constant factors do not affect the normalized LSF**: In practice, the LSF is normalized (`lsf /= lsf.sum()`), at which point all global constants ($G$, total pupil energy $E$, propagation constants, etc.) are cancelled out, and the LSF shape is unaffected by normalization conventions.

### 4.3 Off-Axis Case: Independent Slices for T and S Directions

In the off-axis case, $\text{PSF}(x,y) \neq \text{PSF}(y,x)$, and the LSFs in the T and S directions are inherently different. The projection-slice method naturally supports separate computation:

**Tangential LSF (edge response along the $x$ direction):**

$$
\text{LSF}_T(x) = \sum_y \text{PSF}(x,y) \quad \longleftrightarrow \quad \mathcal{F}\{\text{LSF}_T\} = \text{OTF}(f_x, 0)
$$

Computation method (Parseval path, §4.2.5): Perform 1D FFT on each **row** of the pupil → accumulate $|F|^2$ per frequency → **directly obtain the normalized LSF**, no IFFT needed.

**Sagittal LSF (edge response along the $y$ direction):**

$$
\text{LSF}_S(y) = \sum_x \text{PSF}(x,y) \quad \longleftrightarrow \quad \mathcal{F}\{\text{LSF}_S\} = \text{OTF}(0, f_y)
$$

Computation method (Parseval path): Perform 1D FFT on each **column** of the pupil → accumulate $|F|^2$ per frequency → **directly obtain the normalized LSF**.

> **Note**: The OTF autocorrelation path derived in §4.2.3–4.2.4 requires an additional IFFT step to recover the LSF from the OTF slice. The Parseval path (§4.2.5, §4.4) skips the OTF intermediate representation and is the recommended implementation in this paper.

> The two directions are computed **completely independently** and can be computed for only the needed direction, or computed in parallel for both directions.

### 4.4 Algorithm Implementation

#### 4.4.1 Direct Implementation Based on Parseval's Theorem (Recommended)

From the derivation in §4.2.5, $\text{LSF}_T[k_x] \propto \sum_i |R_i[k_x]|^2$, where $R_i$ is the 1D FFT of the $i$-th row. The implementation is extremely concise:

```python
def compute_lsf_tangential(pupil, grid_size):
    """Parseval projection method: only row FFTs needed to compute Tangential LSF.

    pupil  : complex ndarray, shape (N, N)  — complex pupil function
    grid_size : int  — FFT zero-padding size G

    Returns
    -------
    lsf : ndarray, shape (G,), fftshift-centered, normalized sum=1
    """
    N = pupil.shape[0]
    G = grid_size
    pad = (G - N) // 2
    power_accum = np.zeros(G, dtype=np.float64)

    for i in range(N):                       # Only N non-zero rows
        row_padded = np.zeros(G, dtype=complex)
        row_padded[pad:pad+N] = pupil[i, :]  # Centered zero-padding
        F = np.fft.fft(row_padded)           # 1D FFT (horizontal/T direction)
        power_accum += np.abs(F)**2          # Per-frequency power spectrum accumulation

    lsf = np.fft.fftshift(power_accum)       # Center zero frequency
    lsf /= lsf.sum()                         # Normalize
    return lsf
```

**Complexity**: $N$ 1D FFTs of length $G$ → $O(NG\log G)$. No column FFT, no IFFT, no need to store a $G \times G$ intermediate array.

> **Correctness verification**: The mathematical basis of this code is the rigorous derivation of Parseval's theorem in §4.2.5. `power_accum[k]` $= \sum_i |R_i[k]|^2 = \text{LSF}_T[k] / G$ (differs by a constant factor, eliminated after normalization). **This implementation is numerically equivalent to the standard 2D FFT → sum path**, consistent within floating-point rounding errors (typical relative error ~$10^{-12}$; recommended engineering acceptance threshold of $10^{-10}$, to accommodate differences in FFT backends and accumulation order). This should be verified through numerical regression testing (see §5.2).

#### 4.4.2 Sagittal LSF — FFT Along Columns

Symmetrically, $\text{LSF}_S$ is obtained by performing FFT along columns:

```python
def compute_lsf_sagittal(pupil, grid_size):
    """Parseval projection method: only column FFTs needed to compute Sagittal LSF."""
    N = pupil.shape[0]
    G = grid_size
    pad = (G - N) // 2
    power_accum = np.zeros(G, dtype=np.float64)

    for j in range(N):                       # Only N non-zero columns
        col_padded = np.zeros(G, dtype=complex)
        col_padded[pad:pad+N] = pupil[:, j]  # Centered zero-padding
        F = np.fft.fft(col_padded)           # 1D FFT (vertical/S direction)
        power_accum += np.abs(F)**2

    lsf = np.fft.fftshift(power_accum)
    lsf /= lsf.sum()
    return lsf
```

#### 4.4.3 Relationship with the OTF Autocorrelation Path

The OTF path derived in §4.2.3–4.2.4: row FFT → $|·|^2$ accumulation → IFFT → OTF slice → IFFT → LSF, involves **two IFFTs**. Under the DFT normalization convention adopted in this paper (numpy default: forward unnormalized, inverse with $1/G$) and the coordinate mapping, the combination of these two IFFTs can be simplified by the DFT inversion property ($\text{IFFT}\{\text{IFFT}\{S\}\}[x] = S[-x]/G$), and the final result is **numerically equivalent** to the Parseval path above (differing only by floating-point rounding).

The Parseval path is superior because it skips the OTF intermediate representation, avoids the risk of frequency/spatial domain confusion, and is simpler and less error-prone to implement.

### 4.5 Sampling Effects and Speedup Estimation

#### 4.5.1 Single-Direction LSF Only

| Operation | Standard 2D FFT (numpy, no zero-row skipping) | Parseval Projection-Slice (skip zero rows) |
|-----------|-----------------------------------------------|-------------------------------------------|
| Row FFT | $G$ rows, length $G$: $O(G^2\log G)$ | $N$ non-zero rows, length $G$: $O(NG\log G)$ |
| Column FFT | $G$ columns, length $G$: $O(G^2\log G)$ | **Eliminated** — Parseval removes the column FFT |
| $\|h\|^2$ ($G \times G$) | $O(G^2)$ | **Eliminated** |
| sum → LSF | $O(G^2)$ | Already done during accumulation: $O(NG)$ |
| **Total** | $O(G^2\log G)$ | $O(NG\log G)$ |

> **Note**: The standard 2D FFT column uses numpy's actual complexity $O(G^2\log G)$ (no zero-row skipping, see §2.3). The projection-slice method, by implementing its own row loop, naturally processes only $N$ non-zero rows.

Speedup ratio:

$$
\text{Speedup}_\text{single} = \frac{G^2\log G}{NG\log G} = \frac{G}{N}
$$

where $G/N = Q(N-1)/N \approx Q$ (for $N \gg 1$). Therefore:

$$
\text{Speedup}_\text{single} \approx Q
$$

#### 4.5.2 Computing Both T and S Directions Simultaneously

| Operation | Standard 2D FFT (numpy) | Parseval Projection-Slice (T+S) |
|-----------|------------------------|---------------------------------|
| T direction (row FFT) | Included in 2D FFT | $O(NG\log G)$ |
| S direction (column FFT) | Included in 2D FFT | $O(NG\log G)$ |
| 2D FFT + PSF + sum | $O(G^2\log G)$ | — |
| **Total** | $O(G^2\log G)$ | $O(2NG\log G)$ |

$$
\text{Speedup}_\text{T+S} = \frac{G^2\log G}{2NG\log G} = \frac{G}{2N} \approx \frac{Q}{2}
$$

#### 4.5.3 Numerical Summary

| $Q$ | $G/N$ (approx) | T-only speedup $Q\times$ | T+S speedup $Q/2\times$ |
|-----|----------------|--------------------------|--------------------------|
| 1.0 | 1.0 | 1.0× (no advantage) | 0.5× (slower) |
| 1.5 | 1.5 | 1.5× | 0.75× (slower) |
| 2.0 | 2.0 | 2.0× | 1.0× (break even) |

> **Practical range of $Q$**: $Q = 2$ corresponds to Nyquist sampling (two pixels per $\lambda F/\#$), the minimum requirement for alias-free PSF. $Q < 2$ is undersampling (tolerable for defocused PSF), $Q > 2$ is unnecessary oversampling. Therefore the practical $Q \in [1, 2]$, and the table above covers the entire meaningful range.
>
> **Conclusion**: When $Q \leq 2$ and both T+S directions are needed, the projection-slice speedup is $\leq 1.0\times$, **offering no speed advantage or even being slower** (each direction requires $N$ 1D FFTs, totaling $2N$; standard numpy `fft2` requires $2G$; when $Q$ is small, $2N$ approaches $2G$). The main value of projection-slice lies in: (1) **single-direction** computation ($Q\times$ speedup, up to 2×); (2) **memory savings** ($O(G)$ vs $O(G^2)$, no need to store the full 2D PSF).

---

## 5. Comparative Analysis

### 5.1 Computational Complexity Comparison

The following summary table follows the cost model from §4.5: baseline unit $U = G_0^2\,\log_2 G_0$ (one fixed-grid numpy 2D FFT without zero-row skipping, see §2.3), where $N_0 = 512$, $G_0 = 800$.

| Method | Per-point cost | 29-point total | Relative to baseline |
|--------|---------------|----------------|---------------------|
| **Baseline**: fixed (512, 800) numpy `fft2` | $U = G_0^2\log G_0$ | $29U$ | 1.0× |
| **OPD adaptive** (on-axis) | $G_i^2\,\log G_i$ | $\sum_i G_i^2\,\log G_i$ | **~0.47** (~2×, full scan, §3.5.3) |
| **Projection-slice** (T only, fixed grid) | $N_0\,G_0\log G_0 = \frac{N_0}{G_0}U \approx \frac{1}{Q}U$ | $\frac{29}{Q}U$ | ~0.67 (1.5×, Q=1.5) |
| **Projection-slice** (T+S, fixed grid) | $2N_0\,G_0\log G_0 = \frac{2}{Q}U$ | $\frac{58}{Q}U$ | ~1.33 (Q=1.5, **slower**) |
| **Combined** (T only + OPD adaptive) | $N_i\,G_i\log G_i$ | $\sum_i N_i\,G_i\log G_i$ | **~0.31** (~3.2×, full scan) |

> **Cost metric consistency note**: All rows use the same cost measure. The baseline $U = G_0^2\log G_0$ corresponds to the actual behavior of numpy `fft2` ($G$ row FFTs + $G$ column FFTs, no zero-row skipping). The projection-slice method, by implementing its own row loop, naturally processes only $N$ non-zero rows, hence $N$ appears in the numerator instead of $G$.
>
> **OPD adaptive is the primary speedup source** ($G_i$ can be substantially reduced near the focal plane), but the full-scan speedup is "dragged down" by points at large $|z|$ that cannot be reduced. Projection-slice has limited speedup for T+S dual-direction ($Q \leq 2$ can even be slower), but under the **single-direction + OPD adaptive combination** it can provide an additional ~1.5× marginal improvement. Off-axis speedup ratios, due to the $N_\text{min}$ increase issue discussed in §3.5.3, **require empirical validation**.

### 5.2 Sampling Quality Comparison

| Dimension | OPD Adaptive | Projection-Slice |
|-----------|-------------|-----------------|
| PSF accuracy | Equivalent to full grid when Nyquist is satisfied; aliasing introduced below Nyquist, but ESF integration can partially smooth it | Mathematically equivalent to standard 2D FFT (Parseval identity) |
| ESF accuracy | ESF is the integral of PSF → has some tolerance to aliasing errors, but arbitrary downsampling cannot be assumed lossless | Numerically equivalent (typical ~$10^{-12}$, acceptance threshold $10^{-10}$) |
| Potential risks | The 32×32 OPD pre-scan may miss localized high-gradient regions (e.g., narrow peaks of higher-order aberrations) | Silent errors can be introduced if DFT normalization conventions or fftshift ordering are inconsistent in implementation |
| **Verification method** | **Must** perform numerical regression testing: for each $(z, \lambda, H)$ point, compare ESF maximum absolute error against fixed full-grid results, with a threshold (e.g., $< 10^{-3}$) | **Must** perform numerical regression testing: compare LSF point-by-point against standard 2D FFT → sum path, confirming maximum relative error $< 10^{-10}$ (floating-point precision level) |

> **Mandatory verification requirement**: Although the projection-slice method is mathematically an identity transformation, DFT implementation details (normalization conventions, fftshift symmetry, zero-padding alignment) vary significantly across frameworks. **Any new implementation must pass numerical regression testing against a reference implementation** and cannot skip verification based solely on "theoretical equivalence."

### 5.3 Implementation Difficulty Comparison

| Dimension | OPD Adaptive | Projection-Slice |
|-----------|-------------|-----------------|
| Code change volume | Small: add parameter selection logic before calling FFTPSF | Large: must bypass optiland FFTPSF, implement pupil→LSF independently |
| Dependency on optiland | Yes, directly uses FFTPSF class | No, needs to extract the pupil function and compute independently |
| Debugging difficulty | Low: can compare per-$z$ point | Medium: need to verify DFT normalization, fftshift, and other details |
| Maintenance cost | Low: parameter selection logic is independent of optiland version | High: self-maintained FFT pipeline |

### 5.4 Off-Axis Applicability Comparison

| Dimension | OPD Adaptive | Projection-Slice |
|-----------|-------------|-----------------|
| T/S direction differentiation | Constrained by optiland's square grid: $N = \max(N_T, N_S)$ | Naturally supports independent T/S computation |
| Astigmatism scenario | When $N_T \gg N_S$, S direction is oversampled (square grid waste) | S direction can use fewer row FFTs |
| Coma scenario | T direction gradient is larger → $N$ determined by T | T direction has more row FFTs, S direction fewer |
| Field scan efficiency | Each field independently optimizes $(N, G)$ | Each field independently computes OTF slice |

> **The projection-slice method more elegantly handles T/S asymmetry in theory**, because it naturally decouples T and S into independent 1D computations. However, in practice, the waste from the square grid usually does not exceed $2\times$ (since $N_T / N_S$ rarely exceeds 2).

---

## 6. Combined Strategy and Conclusions

### 6.1 Strategy Selection Recommendation

```
              ┌──────────────────────────────────────────────┐
              │        Recommended: OPD Adaptive Grid          │
              │     On-axis full scan ~2×, near focus ~100×     │
              │     Direct use of optiland, simple to implement │
              └──────────────────┬───────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Need additional theoretical│
                    │      contribution?        │
                    └────────────┬────────────┘
                       ┌────────┴────────┐
                      Yes                No
                       │                  │
           ┌───────────▼──────────┐  ┌───▼────────────────┐
           │ Add projection-slice  │  │  OPD adaptive only  │
           │ theory analysis       │  │  is sufficient      │
           │ Limited speedup(§4.5.3)│  └──────────────────────┘
           │ Value: memory savings, │
           │ theoretical insight    │
           │ Requires custom FFT    │
           │ pipeline               │
           └──────────────────────┘
```

> **Evidence status note**: The on-axis numbers in the above diagram (OPD-only ~2×) are calculated results from the FFT operation count analytical model (§3.5.3), excluding ray tracing and runtime overhead; they are reproducible but not end-to-end measurements. Projection-slice offers no additional speed advantage when $Q \leq 2$ and T+S are needed (§4.5.3). The off-axis "to be measured" values are prior interval estimates (§3.5.2–3.5.3); under strict Nyquist they may be $<1\times$, and **must be validated by regression testing before being cited as conclusions**.

### 6.2 Recommended Phased Implementation Path

**Phase 1** (high return, low risk): Implement OPD adaptive grid

- Small code changes, uses existing optiland FFTPSF
- Expected speedup: on-axis full scan ~2×, near-focus sub-interval ~5×
- Off-axis speedup depends on aliasing tolerance, requires empirical validation
- Accuracy can be immediately verified by comparing ESF against fixed grid

**Phase 2** (optional, theoretical contribution): Implement projection-slice LSF computation

- Bypass optiland, compute LSF from pupil function independently
- **Limited speed advantage**: no speedup or even slower when $Q \leq 2$ and both T+S directions are needed (§4.5.3); up to ~2× for single direction only
- Main value is **memory savings** ($O(G)$ vs $O(G^2)$) and **theoretical contribution** (derivation of the Parseval path, the insight that "ESF does not require 2D PSF")
- Provide theoretical analysis and numerical verification in the thesis

**Phase 3** (outer-loop optimization): Adaptive z sampling

- Coarse scan first → refine in regions with large CFW gradient
- T and S maintain their own z grids separately
- Reduce total number of z points by ~40%

### 6.3 Final Conclusions

| Conclusion | Explanation |
|------------|------------|
| **OPD adaptive is the core strategy (recommended for implementation)** | Single point near focus can reach ~100×, full-scan weighted average ~2× (on-axis, §3.5.3). Solid physical foundation (pupil plane sampling theorem), simple to implement, directly uses optiland |
| **Projection-slice is a theoretical supplement (optional)** | **No speed advantage** when $Q \leq 2$ and T+S are needed (§4.5.3). Value lies in memory savings ($O(G)$ vs $O(G^2)$) and theoretical insight ("ESF does not require 2D PSF"), suitable as a theoretical analysis chapter in the thesis |
| **The two can be combined but marginal benefit is small** | Additional ~1.5× improvement only under single-direction + OPD adaptive combination; benefit approaches zero for T+S |
| **Key challenge for off-axis** | Coma + astigmatism may cause $N_\text{min}$ to exceed the fixed baseline at large defocus → full-scan speedup depends on aliasing tolerance and the $z$ interval of interest |

---

## Appendix A: Symbol Table

| Symbol | Meaning | Units |
|--------|---------|-------|
| $N$ | `num_rays`, number of pupil sampling points per direction | — |
| $G$ | `grid_size`, FFT grid size after zero-padding | — |
| $Q$ | Oversampling factor $= G/(N-1)$ | — |
| $P[m,n]$ | Complex pupil function | — |
| $\text{OPD}_{\mu\text{m}}$ | Optical path difference (physical) | µm |
| $\text{OPD}_w$ | Optical path difference (in waves) $= \text{OPD}_{\mu\text{m}}/\lambda$ | waves |
| $\nabla\text{OPD}_w$ | OPD gradient (**standard unit in this paper**) | waves/aperture |
| $F_{\#}$ | Working F-number | — |
| $\lambda$ | Wavelength | µm |
| $z$ | Defocus amount (image plane offset) | µm |
| $W_{20}$ | Defocus Seidel coefficient | waves |
| $W_{31}$ | Coma Seidel coefficient | waves |
| $W_{22}$ | Astigmatism Seidel coefficient | waves |
| $W_{40}$ | Spherical aberration Seidel coefficient | waves |
| $H$ | Normalized field height | — |
| $\text{OTF}$ | Optical transfer function $= \mathcal{F}\{\text{PSF}\}$ | — |
| $\text{LSF}_T$ | Tangential line spread function | — |
| $\text{LSF}_S$ | Sagittal line spread function | — |
| $\text{ESF}$ | Edge spread function $= \int \text{LSF}$ | — |
| $G_T(\lambda), G_S(\lambda)$ | Maximum OPD gradient in T/S directions (wavelength-dependent) | waves/aperture |

## Appendix B: Key Formula Summary

**Pupil plane sampling Nyquist condition (core of OPD adaptive, $u \in [-1,1]$, pupil diameter $D=2$):**

Waves form (**standard in this paper**):

$$
\boxed{N - 1 > 4\,\max\left|\frac{\partial\,\text{OPD}_w}{\partial u}\right|}
$$

µm equivalent form: $N - 1 > \dfrac{4}{\lambda}\,\max\!\left|\dfrac{\partial\,\text{OPD}_{\mu\text{m}}}{\partial u}\right|$ (identically equivalent to the above, see §3.2.3)

**Simplified form when defocus dominates:**

$$
\boxed{N - 1 > \frac{z}{\lambda_\text{min}\,F_{\#}^2}}
$$

**PSF physical coverage half-width:**

$$
\boxed{x_\text{max} = \frac{N-1}{2}\,\lambda\,F_{\#}}
$$

**Projection-Slice Theorem:**

$$
\boxed{\mathcal{F}\{\text{LSF}_T\}(f_x) = \text{OTF}(f_x, 0)}
$$

**Parseval direct formula (recommended, avoids OTF intermediate step):**

$$
\boxed{\text{LSF}_T[k_x] = G \sum_{i} \big|\text{FFT}_j\{P_\text{pad}[i,\cdot]\}[k_x]\big|^2}
$$

**OTF slice row autocorrelation expression (equivalent path):**

$$
\boxed{\text{OTF}(\Delta u, 0) = \frac{1}{E}\sum_n R_n(\Delta u) = \frac{1}{E}\,\text{IFFT}\!\left\{\sum_n |\text{FFT}\{P[\cdot, n]\}|^2\right\}}
$$

**Projection-slice method computational complexity:**

$$
\boxed{C_\text{PS,single} = O(NG\log G), \quad C_\text{PS,T+S} = O(2NG\log G)}
$$

**Speedup ratio formulas** (baseline is numpy dense 2D FFT $G^2\log G$, no zero-row skipping, see §2.3):

$$
\boxed{\text{Speedup}_\text{single} = \frac{G}{N} \approx Q, \qquad \text{Speedup}_\text{T+S} = \frac{G}{2N} \approx \frac{Q}{2}}
$$

> When $Q \leq 2$ (Nyquist upper bound), T+S speedup $\leq 1.0\times$, no speed advantage (§4.5.3).

**Combined OPD adaptive + projection-slice speedup (relative to fixed-grid numpy 2D FFT, substitute actual $N_i, G_i$):**

$$
\boxed{\text{Speedup}_\text{combined} = \frac{n_z \cdot G_0^2 \log G_0}{\sum_{i=1}^{n_z} 2N_i G_i \log G_i}}
$$

> This formula is an **analytical estimate under the FFT operation count model** (excluding ray tracing, memory bandwidth, interpolation, and Python overhead). When $Q \leq 2$ and both T+S directions are needed, the marginal benefit of the combined scheme relative to OPD adaptive alone is very small (§4.5.3, §6.3). Off-axis values require empirical measurement.

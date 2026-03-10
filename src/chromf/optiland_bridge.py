"""
Optiland bridge for ChromFringe.

Computes the longitudinal chromatic aberration (CHL) curve from an
Optiland ``Optic`` object, returning a ``(N, 2)`` array ``[λ_nm, CHL_µm]``
that is drop-in compatible with ``spectrum_loader._load_defocus()``.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
from chromf.spectrum_loader import channel_products as _channel_products


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve_wl_grid(
    optic,
    wavelengths_nm: np.ndarray | None,
    ref_wavelength_nm: float | None,
) -> tuple[np.ndarray, float]:
    """Return (wls_nm, ref_wl_nm) from optic defaults when arguments are None."""
    wls = (
        wavelengths_nm
        if wavelengths_nm is not None
        else np.array(optic.wavelengths.get_wavelengths()) * 1000.0  # µm → nm
    )
    ref_wl = (
        ref_wavelength_nm
        if ref_wavelength_nm is not None
        else float(optic.primary_wavelength) * 1000.0  # µm → nm
    )
    return wls, ref_wl


def _paraxial_bfl(paraxial, wl_nm: float, z_start: float) -> float:
    """Back focal length (mm) from a paraxial marginal-ray trace at *wl_nm*.

    Raises ValueError if the marginal-ray slope is zero (degenerate optic).
    """
    wl_um = wl_nm / 1000.0
    y, u = paraxial._trace_generic(1.0, 0.0, z_start, wl_um)
    u_last = float(u.ravel()[-1])
    if u_last == 0.0:
        raise ValueError(
            f"Paraxial marginal ray slope is zero at {wl_nm} nm — "
            "optic is degenerate (infinite back focal distance)."
        )
    return float(-y.ravel()[-1] / u_last)


def compute_chl_curve(
    optic,
    wavelengths_nm: np.ndarray | None = None,
    ref_wavelength_nm: float | None = None,
) -> np.ndarray:
    """Compute the CHL curve from an Optiland Optic object.

    Parameters
    ----------
    optic:
        An ``optiland.optic.Optic`` instance with at least two wavelengths
        and a complete surface prescription.
    wavelengths_nm:
        Wavelengths (nm) at which to evaluate the focal shift.  Defaults to
        the wavelengths already defined in the optic (converted from µm).
    ref_wavelength_nm:
        Reference wavelength (nm) at which CHL = 0.  Defaults to the optic's
        primary wavelength.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Two-column array ``[λ_nm, CHL_µm]``, identical in format to the
        output of ``spectrum_loader._load_defocus()``.
    """
    wls, ref_wl = _resolve_wl_grid(optic, wavelengths_nm, ref_wavelength_nm)
    paraxial = optic.paraxial
    z_start = float(paraxial.surfaces.positions[1, 0]) - 1.0

    f2_values = np.array([_paraxial_bfl(paraxial, wl, z_start) for wl in wls])
    f2_ref = _paraxial_bfl(paraxial, ref_wl, z_start)
    chl_um = (f2_values - f2_ref) * 1000.0  # mm → µm

    return np.column_stack((wls, chl_um))



# ── RoRi focus / aperture-dependent CHL ──────────────────────────────────────

#: Normalised pupil heights and their RoRi weights (thesis formula).
#: Pupil coords: 0, √¼, √½, √¾, 1  →  weights: 1, 12.8, 14.4, 12.8, 1 / 42
_RORI_PY      = (0.0, 0.5, 0.7071067811865476, 0.8660254037844387, 1.0)
_RORI_WEIGHTS = (1.0, 12.8, 14.4, 12.8, 1.0)
_RORI_SUM     = 42.0


def _sk_real(optic, Py: float, wl_um: float) -> float:
    """Back-focal intercept (mm) for a meridional ray at normalised pupil height *Py*.

    Traces the ray to the image surface and extrapolates to the optical axis
    (y = 0) using the ray's direction cosines.  Returns the signed axial
    distance from the image surface to the focus point.
    """
    rays = optic.trace_generic(0.0, 0.0, 0.0, Py, wl_um)
    y = float(rays.y.ravel()[-1])
    M = float(rays.M.ravel()[-1])
    N = float(rays.N.ravel()[-1])
    if abs(M) < 1e-12:
        return 0.0
    return -y * N / M


def compute_rori_spot_curves(
    optic,
    wavelengths_nm: np.ndarray | None = None,
    ref_wavelength_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RoRi CHL and residual geometric spot radius per wavelength.

    In addition to the CHL curve (identical to :func:`compute_rori_chl_curve`),
    this function returns a *spot radius curve* rho_sa(λ): the RMS geometric
    spot radius at each wavelength's RoRi best focus.  rho_sa is non-zero
    because spherical aberration causes different pupil zones to focus at
    different axial positions, creating a finite minimum-blur floor even at
    the best-focus position.

    The formula (small-angle approximation)::

        y_spot(Py) = [SK(Py) − RoRi(λ)] × Py / (2 × FNO)
        rho_sa(λ)  = sqrt( Σ w_i · y_spot(Py_i)² / Σ w_i )

    Pass ``spot_curve[:, 1]`` as ``sa_curve_um=`` to
    :func:`chromf.cfw.fringe_width` to include SA in the PSF model.

    Parameters
    ----------
    optic, wavelengths_nm, ref_wavelength_nm:
        Same as :func:`compute_rori_chl_curve`.

    Returns
    -------
    chl_curve : np.ndarray, shape (N, 2)
        ``[λ_nm, CHL_µm]`` — aperture-dependent focal shift.
    spot_curve : np.ndarray, shape (N, 2)
        ``[λ_nm, rho_sa_µm]`` — RMS geometric spot radius at best focus.
    """
    wls, ref_wl = _resolve_wl_grid(optic, wavelengths_nm, ref_wavelength_nm)
    fno      = float(optic.paraxial.FNO())
    paraxial = optic.paraxial
    z_start  = float(paraxial.surfaces.positions[1, 0]) - 1.0

    _py = np.array(_RORI_PY)
    _w  = np.array(_RORI_WEIGHTS, dtype=float)

    def _rori_and_sa(wl_nm: float) -> tuple[float, float]:
        wl_um = wl_nm / 1000.0
        sks   = np.array([_paraxial_bfl(paraxial, wl_nm, z_start)]
                         + [_sk_real(optic, py, wl_um) for py in _RORI_PY[1:]])
        rori  = float(np.dot(_w, sks) / _RORI_SUM)                    # mm
        # Transverse position at z = rori of each pupil zone (small-angle)
        y_spots = (sks - rori) * _py / (2.0 * fno)                    # mm
        rho_sa  = float(np.sqrt(np.dot(_w, y_spots**2) / _RORI_SUM))  # mm (RMS)
        return rori, rho_sa

    rori_arr   = np.empty(len(wls))
    rho_sa_arr = np.empty(len(wls))
    for i, wl in enumerate(wls):
        rori_arr[i], rho_sa_arr[i] = _rori_and_sa(float(wl))

    rori_ref, _ = _rori_and_sa(float(ref_wl))
    chl_um    = (rori_arr   - rori_ref) * 1000.0   # mm → µm
    rho_sa_um =  rho_sa_arr             * 1000.0   # mm → µm

    return np.column_stack((wls, chl_um)), np.column_stack((wls, rho_sa_um))


def compute_w040_curve(
    optic,
    wavelengths_nm: np.ndarray | None = None,
    ref_wavelength_nm: float | None = None,
) -> np.ndarray:
    """Primary spherical aberration coefficient W040 per wavelength (µm OPD).

    Derived from the marginal-ray (ρ=1) transverse aberration at the RoRi
    best-focus plane for each wavelength::

        W040(λ) = −TA_marginal(λ) / (8 · FNO)

    where  TA_marginal = [SK(ρ=1) − RoRi] · 1 / (2·FNO)  (mm), converted to
    µm and negated to match the sign convention  TA = −2·FNO·dW/dρ.

    Pass ``w040_curve[:, 1]`` as ``w040_curve_um=`` to
    :func:`chromf.cfw.fringe_width` with ``psf_mode='dgauss'``.

    Parameters
    ----------
    optic, wavelengths_nm, ref_wavelength_nm:
        Same as :func:`compute_rori_spot_curves`.

    Returns
    -------
    np.ndarray, shape (N, 2)
        ``[λ_nm, W040_µm]`` — spherical aberration wavefront coefficient.
    """
    wls, _ = _resolve_wl_grid(optic, wavelengths_nm, None)
    fno = float(optic.paraxial.FNO())
    paraxial = optic.paraxial
    z_start = float(paraxial.surfaces.positions[1, 0]) - 1.0

    _py = np.array(_RORI_PY)
    _w = np.array(_RORI_WEIGHTS, dtype=float)

    def _w040_at(wl_nm: float) -> float:
        wl_um = wl_nm / 1000.0
        sks = np.array([_paraxial_bfl(paraxial, wl_nm, z_start)]
                       + [_sk_real(optic, py, wl_um) for py in _RORI_PY[1:]])
        rori = float(np.dot(_w, sks) / _RORI_SUM)
        # Transverse aberration of marginal ray (ρ=1) at RoRi focus plane (mm):
        ta_marginal_mm = (sks[-1] - rori) / (2.0 * fno)
        # W040 = −TA_marginal_µm / (8·FNO)
        return -(ta_marginal_mm * 1000.0) / (8.0 * fno)

    w040_arr = np.array([_w040_at(float(wl)) for wl in wls])
    return np.column_stack((wls, w040_arr))


def compute_rori_chl_curve(
    optic,
    wavelengths_nm: np.ndarray | None = None,
    ref_wavelength_nm: float | None = None,
) -> np.ndarray:
    """Compute the aperture-dependent (RoRi) CHL curve from an Optiland Optic.

    Unlike :func:`compute_chl_curve`, which uses the paraxial marginal ray,
    this function computes a weighted average of real-ray back-focal intercepts
    at five normalised pupil heights (0, √¼, √½, √¾, 1).  This captures the
    effect of spherical aberration and spherochromatism, giving the
    *effective* focus seen by the image sensor.

    The RoRi formula (thesis §9)::

        RoRi = (SK(0) + 12.8·SK(√¼) + 14.4·SK(√½) + 12.8·SK(√¾) + SK(1)) / 42

    where SK(y) is the back-focal intercept for a meridional ray at normalised
    pupil height *y*.  SK(0) is obtained from the paraxial trace.

    Parameters
    ----------
    optic:
        An ``optiland.optic.Optic`` instance.
    wavelengths_nm:
        Wavelengths (nm) at which to evaluate RoRi.  Defaults to the optic's
        defined wavelengths.
    ref_wavelength_nm:
        Reference wavelength (nm) at which RoRi CHL = 0.  Defaults to the
        optic's primary wavelength.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Two-column array ``[λ_nm, RoRi_CHL_µm]``, drop-in compatible with
        :func:`compute_chl_curve` and ``spectrum_loader._load_defocus()``.
    """
    chl_curve, _ = compute_rori_spot_curves(optic, wavelengths_nm, ref_wavelength_nm)
    return chl_curve


# ── FFTPSF ground-truth CFW pipeline ─────────────────────────────────────────

_CHANNEL_MAP = {"R": "red", "G": "green", "B": "blue"}


def _optic_at_defocus(optic, z_defocus_um: float):
    """Return a deep copy of optic with image plane shifted by z_defocus_um µm.

    Positive z_defocus_um → image plane moves away from lens
    → optic.surfaces[-1].thickness increases by z_defocus_um / 1000 mm.
    """
    op = copy.deepcopy(optic)
    op.surface_group.surfaces[-1].thickness += z_defocus_um / 1000.0
    return op


def compute_polychromatic_esf(
    optic,
    channel: str,
    z_defocus_um: float,
    x_um: np.ndarray,
    num_rays: int = 256,
    grid_size: int = 512,
    strategy: str = "chief_ray",
    wl_stride: int = 1,
    sensor_model: str = "nikond700",
) -> np.ndarray:
    """Polychromatic ESF on a physical µm x-axis (correct coordinate accumulation).

    Accumulates monochromatic ESFs in physical space, each using its own
    wavelength-correct pixel pitch ``dx_j = λ_j × FNO / Q``.  This avoids the
    coordinate mixing that occurs when PSFs at different wavelengths (and thus
    different pixel pitches) are naively added in pixel space.

    Parameters
    ----------
    optic:
        Optiland ``Optic`` instance.
    channel:
        ``"R"``, ``"G"``, or ``"B"``.
    z_defocus_um:
        Image-plane shift in µm (positive = away from lens).
    x_um:
        Physical x-axis in µm on which to evaluate the ESF, e.g.
        ``np.arange(-400, 401, dtype=float)`` to match ``cfw.fringe_width``.
    num_rays:
        Pupil sampling per side.  Physical PSF half-span =
        ``(num_rays−1) × λ_min × FNO / 2``; must exceed the maximum geometric
        blur radius ``z_max / (2 × FNO)``.
    grid_size:
        FFT grid size.  Oversampling factor ``Q = grid_size / (num_rays−1)``;
        Q ≥ 2 is adequate for ESF shape; 512 keeps memory at 4 MiB/array.
    strategy:
        Wavefront reference strategy passed to ``FFTPSF``.  Use
        ``"chief_ray"`` (default) — the reference sphere is anchored to the
        physical chief ray position, which is image-plane-dependent, so
        chromatic defocus is correctly preserved in the OPD.  Do **not** use
        ``"best_fit_sphere"``: it fits and removes defocus, making every
        wavelength appear near-focused regardless of image-plane position,
        which collapses all channel ESFs to the same shape and gives CFW = 0.
        Note: Optiland 0.5.9 had a NumPy 2.0 incompatibility in the
        ``chief_ray`` code path (``float(x)`` on a size-1 array); this has
        been patched in the installed package's ``strategy.py``.
    wl_stride:
        Wavelength subsampling stride.  1 = all 31 wavelengths (10 nm step);
        3 = every 3rd (11 wavelengths, 30 nm step) gives 3× speedup with
        negligible ESF error on smooth spectral curves.

    Returns
    -------
    np.ndarray
        Shape ``== x_um.shape``, values in ``[0, 1]``.
    """
    from optiland.psf import FFTPSF  # lazy import — keeps module loadable without optiland.psf
    op = _optic_at_defocus(optic, z_defocus_um)
    ch_key = _CHANNEL_MAP[channel.upper()]
    products = _channel_products(sensor_model=sensor_model)
    wl_nm  = products[ch_key][:, 0][::wl_stride]   # nm, subsampled
    g_k    = products[ch_key][:, 1][::wl_stride]   # spectral density, subsampled
    g_norm = g_k / g_k.sum()                        # normalise to discrete sum = 1

    esf_accum = np.zeros(len(x_um), dtype=np.float64)
    fno: float | None = None

    for j in range(len(wl_nm)):
        wl_um_j = wl_nm[j] / 1000.0
        fft_psf = FFTPSF(op, field=(0, 0), wavelength=wl_um_j,
                         num_rays=num_rays, grid_size=grid_size,
                         strategy=strategy)
        if fno is None:
            fno = float(fft_psf._get_working_FNO())

        Q    = grid_size / (num_rays - 1)
        dx_j = wl_um_j * fno / Q          # wavelength-correct pixel pitch (µm)

        lsf  = fft_psf.psf.sum(axis=0)
        lsf /= lsf.sum()
        esf_j = np.cumsum(lsf)            # ESF in pixel space, spans [0, 1]

        n   = len(esf_j)
        x_j = (np.arange(n) - n // 2) * dx_j  # physical µm coords

        esf_accum += g_norm[j] * np.interp(x_um, x_j, esf_j,
                                            left=0.0, right=1.0)

    return np.clip(esf_accum, 0.0, 1.0)


def compute_polychromatic_esf_geometric(
    optic,
    channel: str,
    z_defocus_um: float,
    x_um: np.ndarray,
    num_rho: int = 32,
    wl_stride: int = 1,
    sensor_model: str = "nikond700",
) -> np.ndarray:
    """Polychromatic ESF using the geometric pupil integral (no FFT).

    For each wavelength, traces ``num_rho`` real rays at Gauss-Legendre
    quadrature pupil heights ρ ∈ (0, 1] and evaluates the exact geometric ESF::

        ESF(x) = ∫₀¹ [arcsin(clip(x / R(ρ), −1, 1)) / π + ½] · 2ρ dρ

    where R(ρ) = |y_image(ρ)| is the transverse aberration magnitude (µm) of
    the ray at normalized pupil height ρ on the defocused image plane.

    This is ~100× faster than :func:`compute_polychromatic_esf` and faithfully
    captures the asymmetry in PSF width between +z and −z defocus through the
    real aberration structure of the lens.  Diffraction effects near focus
    (|z| ≲ λ·FNO²) are not included.

    Parameters
    ----------
    optic:
        Optiland ``Optic`` instance.
    channel:
        ``"R"``, ``"G"``, or ``"B"``.
    z_defocus_um:
        Image-plane shift in µm (positive = away from lens).
    x_um:
        Physical x-axis in µm, e.g. ``np.arange(-400, 401, dtype=float)``.
    num_rho:
        Number of Gauss-Legendre quadrature points for the pupil integral.
        32 gives < 0.1 % ESF error for smooth aberration profiles.
    wl_stride:
        Wavelength subsampling stride (same meaning as in
        :func:`compute_polychromatic_esf`).

    Returns
    -------
    np.ndarray
        Shape ``== x_um.shape``, values in ``[0, 1]``.
    """
    op = _optic_at_defocus(optic, z_defocus_um)
    ch_key = _CHANNEL_MAP[channel.upper()]
    products = _channel_products(sensor_model=sensor_model)
    wl_nm = products[ch_key][:, 0][::wl_stride]
    g_k = products[ch_key][:, 1][::wl_stride]
    g_norm = g_k / g_k.sum()

    # Gauss-Legendre quadrature on [0, 1].
    # ∫₀¹ 2ρ·f(ρ) dρ ≈ Σ_k 2ρ_k · f(ρ_k) · (W_k / 2) = Σ_k ρ_k · f(ρ_k) · W_k
    # where ξ_k, W_k are GL nodes/weights on [−1, 1].
    xi, W_gl = np.polynomial.legendre.leggauss(num_rho)
    rho_nodes = 0.5 * (xi + 1.0)          # map [−1,1] → [0,1]

    esf_accum = np.zeros(len(x_um), dtype=np.float64)

    for j in range(len(wl_nm)):
        wl_um_j = wl_nm[j] / 1000.0

        # Transverse aberration magnitude R(ρ) for each quadrature node.
        R_rho = np.empty(num_rho)
        for k in range(num_rho):
            rho = float(rho_nodes[k])
            rays = op.trace_generic(0.0, 0.0, 0.0, rho, wl_um_j)
            y_mm = float(rays.y.ravel()[-1])
            R_rho[k] = abs(y_mm) * 1000.0   # mm → µm

        # Vectorised ESF:  (N_x, num_rho) → (N_x,)
        x_col = x_um[:, np.newaxis]            # (N, 1)
        R_row = R_rho[np.newaxis, :]           # (1, K)
        rho_row = rho_nodes[np.newaxis, :]     # (1, K)
        W_row = W_gl[np.newaxis, :]            # (1, K)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(R_row > 1e-4, x_col / R_row, np.sign(x_col + 1e-15))
        ratio = np.clip(ratio, -1.0, 1.0)
        f_contrib = np.arcsin(ratio) / np.pi + 0.5   # (N, K)

        esf_j = np.sum(f_contrib * rho_row * W_row, axis=1)   # (N,)
        esf_accum += g_norm[j] * esf_j

    return np.clip(esf_accum, 0.0, 1.0)


def precompute_ray_fan(
    optic,
    num_rho: int = 32,
    sensor_model: str = "nikond700",
) -> dict:
    """Pre-compute signed transverse aberrations and ray slopes at z = 0.

    Traces ``num_rho × N_wavelengths`` rays **once** at the nominal image
    plane.  The result encodes both the SA-induced TA profile *and* each
    ray's dy/dz slope, enabling exact linear extrapolation to any defocus z::

        R(ρ; z, λ) = |TA₀(ρ, λ)  +  (M/N)(ρ, λ) · z|

    where TA₀ is the signed transverse aberration (µm) and M/N is the
    direction-cosine ratio (µm/µm ≡ dimensionless).

    All 31 wavelengths (400–700 nm, 10 nm step) are traced; per-channel
    spectral weights are stored separately so a single fan covers R, G, B.

    Pass the returned dict to :func:`compute_polychromatic_esf_fast` to
    evaluate ESFs at any z without further ray tracing.

    Parameters
    ----------
    optic:
        Optiland ``Optic`` instance at the nominal (z = 0) image plane.
    num_rho:
        Gauss-Legendre pupil quadrature points (default 32).

    Returns
    -------
    dict with keys:
        ``fno``, ``rho_nodes`` (K,), ``W_gl`` (K,), ``wl_nm`` (N_wl,),
        ``TA0`` (K, N_wl) µm, ``slope`` (K, N_wl) µm/µm;
        plus ``"R"``, ``"G"``, ``"B"`` sub-dicts with ``g_norm`` (N_wl,).
    """
    products = _channel_products(sensor_model=sensor_model)
    wl_nm_all = products["red"][:, 0]   # all channels share this 400–700 nm grid
    N_wl = len(wl_nm_all)

    xi, W_gl = np.polynomial.legendre.leggauss(num_rho)
    rho_nodes = 0.5 * (xi + 1.0)   # map [−1,1] → [0,1]
    fno = float(optic.paraxial.FNO())

    TA0_all   = np.empty((num_rho, N_wl))
    slope_all = np.empty((num_rho, N_wl))

    for k, rho in enumerate(rho_nodes):
        for j, wl in enumerate(wl_nm_all):
            rays  = optic.trace_generic(0.0, 0.0, 0.0, float(rho), wl / 1000.0)
            y_mm  = float(rays.y.ravel()[-1])
            M     = float(rays.M.ravel()[-1])
            N_dir = float(rays.N.ravel()[-1])
            TA0_all[k, j]   = y_mm * 1000.0   # mm → µm (signed)
            slope_all[k, j] = (M / N_dir) if abs(N_dir) > 1e-10 else -float(rho) / (2.0 * fno)

    fan: dict = {
        "fno":       fno,
        "rho_nodes": rho_nodes,
        "W_gl":      W_gl,
        "wl_nm":     wl_nm_all,
        "TA0":       TA0_all,
        "slope":     slope_all,
    }
    for ch_name, ch_key in _CHANNEL_MAP.items():
        g_k = products[ch_key][:, 1]
        fan[ch_name] = {"g_norm": g_k / g_k.sum()}
    return fan


def compute_polychromatic_esf_fast(
    ray_fan: dict,
    channel: str,
    z_defocus_um: float,
    x_um: np.ndarray,
    wl_stride: int = 1,
) -> np.ndarray:
    """Polychromatic ESF at any defocus z using a pre-computed ray fan.

    Extrapolates the transverse aberration linearly from z = 0 using the
    traced ray direction::

        R(ρ; z, λ) = |TA₀(ρ, λ) + slope(ρ, λ) · z|

    then evaluates the geometric pupil-integral ESF.  No ray tracing is
    performed; the cost is a handful of vectorised numpy operations.

    Compared with :func:`compute_polychromatic_esf_geometric`, this is
    ~1000× faster for a z-sweep because the ray-tracing overhead is paid
    once by :func:`precompute_ray_fan`.  The linear extrapolation error is
    O((z/f′)²) ≈ 0.01 % for z ≤ 800 µm on an 85 mm lens.

    Parameters
    ----------
    ray_fan:
        Dict from :func:`precompute_ray_fan`.
    channel:
        ``"R"``, ``"G"``, or ``"B"``.
    z_defocus_um:
        Defocus in µm (positive = image plane moved away from lens).
    x_um:
        Physical x-axis in µm.
    wl_stride:
        Wavelength subsampling stride (default 1 = all 31 wavelengths).

    Returns
    -------
    np.ndarray, shape == x_um.shape, values in [0, 1].
    """
    rho_nodes = ray_fan["rho_nodes"]            # (K,)
    W_gl      = ray_fan["W_gl"]                 # (K,)
    g_norm_full = ray_fan[channel.upper()]["g_norm"]   # (N_wl,)

    TA0   = ray_fan["TA0"][:, ::wl_stride]      # (K, N_wl_sub)
    slope = ray_fan["slope"][:, ::wl_stride]    # (K, N_wl_sub)
    g_sub = g_norm_full[::wl_stride]
    g_sub = g_sub / g_sub.sum()                 # renormalise after striding

    # R(ρ, λ; z) = |TA0 + slope · z|   [µm]
    R = np.abs(TA0 + slope * z_defocus_um)      # (K, N_wl_sub)

    x_col   = x_um[:, np.newaxis]               # (N, 1)
    rho_row = rho_nodes[np.newaxis, :]          # (1, K)
    W_row   = W_gl[np.newaxis, :]               # (1, K)

    esf_accum = np.zeros(len(x_um), dtype=np.float64)
    for j in range(R.shape[1]):
        R_row = R[:, j][np.newaxis, :]          # (1, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(R_row > 1e-4, x_col / R_row, np.sign(x_col + 1e-15))
        ratio = np.clip(ratio, -1.0, 1.0)
        f_contrib = np.arcsin(ratio) / np.pi + 0.5   # (N, K)
        esf_accum += g_sub[j] * np.sum(f_contrib * rho_row * W_row, axis=1)

    return np.clip(esf_accum, 0.0, 1.0)


def compute_polychromatic_psf(
    optic,
    channel: str,
    z_defocus_um: float,
    num_rays: int = 128,
    grid_size: int = 1024,
    strategy: str = "best_fit_sphere",
    sensor_model: str = "nikond700",
) -> tuple[np.ndarray, float]:
    """Compute the polychromatic diffraction PSF for one color channel.

    Returns (psf_2d, dx_um):
        psf_2d : np.ndarray, shape (grid_size, grid_size), energy-normalized (sum=1)
        dx_um  : float, pixel pitch of the PSF array in µm
    """
    op = _optic_at_defocus(optic, z_defocus_um)

    ch_key = _CHANNEL_MAP[channel.upper()]
    products = _channel_products(sensor_model=sensor_model)
    wl_nm = products[ch_key][:, 0]  # shape (31,)
    g_k = products[ch_key][:, 1]    # shape (31,)

    from optiland.psf import FFTPSF  # lazy import: keeps module loadable without optiland.psf

    psf_accum = np.zeros((grid_size, grid_size), dtype=np.float64)
    dx_um: float | None = None

    for j in range(len(wl_nm)):
        wl_um = wl_nm[j] / 1000.0
        fft_psf = FFTPSF(op, field=(0, 0), wavelength=wl_um,
                         num_rays=num_rays, grid_size=grid_size,
                         strategy=strategy)
        if dx_um is None:
            # dx is not stored as an attribute; derive from optiland's formula:
            # dx = wavelength[µm] * FNO / Q,  Q = grid_size / (num_rays - 1)
            _fno = fft_psf._get_working_FNO()
            _Q = fft_psf.grid_size / (fft_psf.num_rays - 1)
            dx_um = float(fft_psf.wavelengths[0]) * float(_fno) / float(_Q)  # µm
            if not (0.01 <= dx_um <= 100.0):
                warnings.warn(
                    f"PSF pixel pitch dx_um={dx_um:.4f} µm is outside the "
                    f"expected range [0.01, 100] µm.",
                    stacklevel=2,
                )
        mono = fft_psf.psf.copy()
        mono_sum = mono.sum()
        if mono_sum > 0:
            mono /= mono_sum  # energy normalize: sum = 1
        psf_accum += g_k[j] * mono

    psf_accum /= psf_accum.sum()
    return psf_accum, dx_um  # type: ignore[return-value]


def _psf_to_esf(
    psf_2d: np.ndarray,
    dx_um: float,
    x_px: float,
    n_pixels: int = 801,
) -> np.ndarray:
    """Convert a 2D PSF to a 1D Edge Spread Function on the sensor pixel grid.

    Returns np.ndarray of shape (n_pixels,), values in [0, 1].
    """
    # PSF → LSF
    lsf = psf_2d.sum(axis=0)
    lsf /= lsf.sum()

    # LSF → ESF: cumulative sum of the unit-normalized LSF.
    # This equals ∫_{-∞}^{x} LSF(t) dt, guaranteed to span exactly [0, 1].
    # fftconvolve(step, lsf, mode='same') clips at ~0.5 when the PSF is very
    # narrow relative to the grid, so we use cumsum instead.
    esf_psf_grid = np.cumsum(lsf)

    # Resample to sensor pixel grid
    n_psf = len(esf_psf_grid)
    x_psf = (np.arange(n_psf) - n_psf // 2) * dx_um           # µm
    x_sensor = (np.arange(n_pixels) - n_pixels // 2) * x_px   # µm
    esf_sensor = np.interp(x_sensor, x_psf, esf_psf_grid)

    return np.clip(esf_sensor, 0.0, 1.0)


def compute_cfw_psf(
    optic,
    z_defocus_um: float,
    *,
    num_rays: int = 128,
    grid_size: int = 1024,
    strategy: str = "best_fit_sphere",
    x_px: float = 4.5,
    exposure_slope: float = 8.0,
    display_gamma: float = 2.2,
    color_diff_threshold: float = 0.2,
) -> int:
    """Compute Color Fringe Width using diffraction PSF (ground truth).

    Returns integer pixel count, directly comparable to cfw.fringe_width().
    """
    esfs: dict[str, np.ndarray] = {}
    for c in ("R", "G", "B"):
        psf_2d, dx_um = compute_polychromatic_psf(
            optic, c, z_defocus_um, num_rays, grid_size, strategy
        )
        esf_raw = _psf_to_esf(psf_2d, dx_um, x_px)
        esfs[c] = (
            (np.tanh(exposure_slope * esf_raw) / np.tanh(exposure_slope))
            ** display_gamma
        )

    diff_rg = np.abs(esfs["R"] - esfs["G"])
    diff_rb = np.abs(esfs["R"] - esfs["B"])
    diff_gb = np.abs(esfs["G"] - esfs["B"])
    fringed = (
        (diff_rg > color_diff_threshold)
        | (diff_rb > color_diff_threshold)
        | (diff_gb > color_diff_threshold)
    )
    return int(fringed.sum())


__all__ = [
    "compute_chl_curve",
    "compute_rori_chl_curve",
    "compute_rori_spot_curves",
    "compute_w040_curve",
    "precompute_ray_fan",
    "compute_polychromatic_esf",
    "compute_polychromatic_esf_geometric",
    "compute_polychromatic_esf_fast",
    "compute_cfw_psf",
]

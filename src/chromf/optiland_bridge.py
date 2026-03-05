"""
Optiland bridge for ChromFringe.

Computes the longitudinal chromatic aberration (CHL) curve from an
Optiland ``Optic`` object, returning a ``(N, 2)`` array ``[λ_nm, CHL_µm]``
that is drop-in compatible with ``spectrum_loader._load_defocus()``.
"""

from __future__ import annotations

import numpy as np


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
    # ── Wavelength grid ───────────────────────────────────────────────
    wls: np.ndarray = (
        wavelengths_nm
        if wavelengths_nm is not None
        else np.array(optic.wavelengths.get_wavelengths()) * 1000.0  # µm → nm
    )
    ref_wl: float = (
        ref_wavelength_nm
        if ref_wavelength_nm is not None
        else float(optic.primary_wavelength) * 1000.0  # µm → nm
    )

    # ── Back-focal-point helper ───────────────────────────────────────
    paraxial = optic.paraxial
    z_start = float(paraxial.surfaces.positions[1, 0]) - 1.0

    def _f2_at(wl_nm: float) -> float:
        """Return signed distance from image plane to paraxial focus (mm) at wl_nm."""
        wl_um = wl_nm / 1000.0
        y, u = paraxial._trace_generic(1.0, 0.0, z_start, wl_um)
        u_last = float(u.ravel()[-1])
        if u_last == 0.0:
            raise ValueError(
                f"Paraxial marginal ray slope is zero at {wl_nm} nm — "
                "optic is degenerate (infinite back focal distance)."
            )
        return float(-y.ravel()[-1] / u_last)

    # ── Compute CHL ───────────────────────────────────────────────────
    f2_values = np.array([_f2_at(wl) for wl in wls])
    f2_ref = _f2_at(ref_wl)
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
    wls: np.ndarray = (
        wavelengths_nm
        if wavelengths_nm is not None
        else np.array(optic.wavelengths.get_wavelengths()) * 1000.0
    )
    ref_wl: float = (
        ref_wavelength_nm
        if ref_wavelength_nm is not None
        else float(optic.primary_wavelength) * 1000.0
    )

    paraxial = optic.paraxial
    z_start  = float(paraxial.surfaces.positions[1, 0]) - 1.0

    def _sk_paraxial(wl_nm: float) -> float:
        wl_um = wl_nm / 1000.0
        y, u = paraxial._trace_generic(1.0, 0.0, z_start, wl_um)
        u_last = float(u.ravel()[-1])
        if u_last == 0.0:
            raise ValueError(
                f"Paraxial marginal ray slope is zero at {wl_nm} nm."
            )
        return float(-y.ravel()[-1] / u_last)

    def _rori_at(wl_nm: float) -> float:
        wl_um = wl_nm / 1000.0
        sks = [_sk_paraxial(wl_nm)] + [
            _sk_real(optic, py, wl_um) for py in _RORI_PY[1:]
        ]
        return sum(w * s for w, s in zip(_RORI_WEIGHTS, sks)) / _RORI_SUM

    rori_values = np.array([_rori_at(wl) for wl in wls])
    rori_ref    = _rori_at(ref_wl)
    chl_um      = (rori_values - rori_ref) * 1000.0  # mm → µm

    return np.column_stack((wls, chl_um))


__all__ = ["compute_chl_curve", "compute_rori_chl_curve"]

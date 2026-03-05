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
    if wavelengths_nm is None:
        wavelengths_nm = np.array(optic.wavelengths.get_wavelengths()) * 1000.0  # µm → nm

    if ref_wavelength_nm is None:
        ref_wavelength_nm = float(optic.primary_wavelength) * 1000.0  # µm → nm

    # ── Back-focal-point helper ───────────────────────────────────────
    paraxial = optic.paraxial
    z_start = float(paraxial.surfaces.positions[1, 0]) - 1.0

    def _f2_at(wl_nm: float) -> float:
        """Return signed distance from image plane to paraxial focus (mm) at wl_nm."""
        wl_um = wl_nm / 1000.0
        y, u = paraxial._trace_generic(1.0, 0.0, z_start, wl_um)
        u_last = u.ravel()[-1]
        if u_last == 0.0:
            raise ValueError(
                f"Paraxial marginal ray slope is zero at {wl_nm} nm — "
                "optic is degenerate (infinite back focal distance)."
            )
        return float(-y.ravel()[-1] / u_last)

    # ── Compute CHL ───────────────────────────────────────────────────
    f2_values = np.array([_f2_at(wl) for wl in wavelengths_nm])
    f2_ref = _f2_at(ref_wavelength_nm)
    chl_um = (f2_values - f2_ref) * 1000.0  # mm → µm

    return np.column_stack((wavelengths_nm, chl_um))


__all__ = ["compute_chl_curve"]

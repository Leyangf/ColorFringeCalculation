"""
chromf — Chromatic colour-fringing prediction toolkit.

Public API
----------
Core CFW functions (from cfw.py):

    fringe_width            Total colour fringe width in µm at a given defocus
    edge_response           Single-channel ESF value at (x, z)
    edge_rgb_response       R, G, B ESF tuple at (x, z)
    detect_fringe_binary    1 if pixel is colour-fringed, else 0

Spectral data (from spectrum_loader.py):

    channel_products        Energy-normalised S·D products for R, G, B channels

Aberration extraction from Optiland (from optiland_bridge.py):

    compute_chl_curve           Paraxial CHL [λ, µm]
    compute_rori_chl_curve      Aperture-weighted RoRi CHL [λ, µm]
    compute_rori_spot_curves    RoRi CHL + residual SA spot radius
    compute_w040_curve          Seidel W040 [λ, µm]
    precompute_ray_fan          Pre-trace ray fan for fast ESF sweeps
    compute_polychromatic_esf               Diffraction ESF (ground truth)
    compute_polychromatic_esf_geometric     Geometric pupil-integral ESF
    compute_polychromatic_esf_fast          Geometric ESF via pre-traced ray fan
"""

from chromf.cfw import (
    fringe_width,
    edge_response,
    edge_rgb_response,
    detect_fringe_binary,
)

from chromf.spectrum_loader import channel_products

from chromf.optiland_bridge import (
    compute_chl_curve,
    compute_rori_chl_curve,
    compute_rori_spot_curves,
    compute_w040_curve,
    precompute_ray_fan,
    compute_polychromatic_esf,
    compute_polychromatic_esf_geometric,
    compute_polychromatic_esf_fast,
)

__all__ = [
    # cfw
    "fringe_width",
    "edge_response",
    "edge_rgb_response",
    "detect_fringe_binary",
    # spectrum_loader
    "channel_products",
    # optiland_bridge
    "compute_chl_curve",
    "compute_rori_chl_curve",
    "compute_rori_spot_curves",
    "compute_w040_curve",
    "precompute_ray_fan",
    "compute_polychromatic_esf",
    "compute_polychromatic_esf_geometric",
    "compute_polychromatic_esf_fast",
]

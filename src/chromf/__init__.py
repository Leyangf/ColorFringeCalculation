"""
chromf — Chromatic colour-fringing prediction toolkit.

Public API
----------
Core CFW functions (from cfw.py):

    fringe_width            Total colour fringe width in µm at a given defocus
    edge_rgb_response_vec   R, G, B ESF arrays at (x_arr, z)  [vectorised]
    is_fringe_mask          Boolean mask of visible fringe pixels (all 3 conditions)

Spectral data (from spectrum_loader.py):

    channel_products        Energy-normalised S·D products for R, G, B channels

Sensor selection (from cfw.py):

    load_sensor_response    Build an R/G/B sensor-response dict for any camera model

Aberration extraction from Optiland (from optiland_bridge.py):

    compute_chl_curve            Paraxial CHL [λ, µm]
    compute_rori_spot_curves     RoRi CHL + residual SA spot radius (ρ_SA)
    precompute_ray_fan          Pre-trace ray fan for fast ESF sweeps
    compute_polychromatic_esf               Diffraction ESF (ground truth)
    compute_polychromatic_esf_geom          Geometric ESF via pre-traced ray fan
    bake_wavelength_esfs        Sensor-independent monochromatic ESF baking
    apply_sensor_weights        Apply sensor spectral weights to baked ESFs
"""

from chromf.cfw import (
    fringe_width,
    edge_rgb_response_vec,
    is_fringe_mask,
    load_sensor_response,
)

from chromf.spectrum_loader import channel_products

from chromf.optiland_bridge import (
    compute_chl_curve,
    compute_rori_spot_curves,
    precompute_ray_fan,
    compute_polychromatic_esf,
    compute_polychromatic_esf_geom,
    bake_wavelength_esfs,
    apply_sensor_weights,
)

__all__ = [
    # cfw
    "fringe_width",
    "edge_rgb_response_vec",
    "is_fringe_mask",
    "load_sensor_response",
    # spectrum_loader
    "channel_products",
    # optiland_bridge
    "compute_chl_curve",
    "compute_rori_spot_curves",
    "precompute_ray_fan",
    "compute_polychromatic_esf",
    "compute_polychromatic_esf_geom",
    "bake_wavelength_esfs",
    "apply_sensor_weights",
]

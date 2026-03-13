"""
chromf — Chromatic colour-fringing prediction toolkit.

Public API
----------
Core CFW functions (from cfw.py):

    fringe_width            Total colour fringe width in µm at a given defocus
    edge_response           Single-channel ESF value at (x, z)
    edge_rgb_response       R, G, B ESF tuple at (x, z)  [scalar x]
    edge_rgb_response_vec   R, G, B ESF arrays at (x_arr, z)  [vectorised]
    detect_fringe_binary    1 if pixel is colour-fringed, else 0

Spectral data (from spectrum_loader.py):

    channel_products        Energy-normalised S·D products for R, G, B channels

Sensor selection (from cfw.py):

    load_sensor_response    Build an R/G/B sensor-response dict for any camera model

Aberration extraction from Optiland (from optiland_bridge.py):

    compute_chl_curve            Paraxial CHL [λ, µm]
    compute_rori1_chl_curve      RoRi-1 CHL (5-zone weighted average) [λ, µm]
    compute_rori1_spot_curves    RoRi-1 CHL + residual SA spot radius
    compute_rori4_chl_curve      RoRi-4 CHL (ρ²-weighted orthogonal plane) [λ, µm]
    compute_rori4_spot_curves    RoRi-4 CHL + residual SA spot radius
    compute_sa_poly_curves      Per-wavelength SA polynomial c₃, c₅
    compute_w040_curve          Seidel W040 [λ, µm]
    precompute_ray_fan          Pre-trace ray fan for fast ESF sweeps
    compute_polychromatic_esf               Diffraction ESF (ground truth)
    compute_polychromatic_esf_fast          Geometric ESF via pre-traced ray fan
    bake_wavelength_esfs        Sensor-independent monochromatic ESF baking
    apply_sensor_weights        Apply sensor spectral weights to baked ESFs
"""

from chromf.cfw import (
    fringe_width,
    edge_response,
    edge_rgb_response,
    edge_rgb_response_vec,
    detect_fringe_binary,
    load_sensor_response,
)

from chromf.spectrum_loader import channel_products

from chromf.optiland_bridge import (
    compute_chl_curve,
    compute_rori1_chl_curve,
    compute_rori1_spot_curves,
    compute_rori4_chl_curve,
    compute_rori4_spot_curves,
    compute_sa_poly_curves,
    compute_w040_curve,
    precompute_ray_fan,
    compute_polychromatic_esf,
    compute_polychromatic_esf_fast,
    bake_wavelength_esfs,
    apply_sensor_weights,
)

__all__ = [
    # cfw
    "fringe_width",
    "edge_response",
    "edge_rgb_response",
    "edge_rgb_response_vec",
    "detect_fringe_binary",
    "load_sensor_response",
    # spectrum_loader
    "channel_products",
    # optiland_bridge
    "compute_chl_curve",
    "compute_rori1_chl_curve",
    "compute_rori1_spot_curves",
    "compute_rori4_chl_curve",
    "compute_rori4_spot_curves",
    "compute_sa_poly_curves",
    "compute_w040_curve",
    "precompute_ray_fan",
    "compute_polychromatic_esf",
    "compute_polychromatic_esf_fast",
    "bake_wavelength_esfs",
    "apply_sensor_weights",
]

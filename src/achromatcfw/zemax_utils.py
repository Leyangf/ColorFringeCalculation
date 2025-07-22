"""Zemax integration helpers.

This module provides functions to fetch longitudinal chromatic focal shift
curves from Zemax OpticStudio and evaluate colour fringe width metrics.
"""
# Utilities for interacting with Zemax OpticStudio via ZOS-API

from __future__ import annotations

from typing import Tuple

import numpy as np

# Optional import, only needed when calling `fetch_chromatic_focal_shift`
try:
    import comtypes.client as cc
except Exception:  # pragma: no cover - optional dependency
    cc = None

from .core.cfw import Farbsaumbreite, XRANGE_VAL


def fetch_chromatic_focal_shift(
    system_file: str | None = None,
    start_wavelength: float = 400.0,
    end_wavelength: float = 700.0,
    step: float = 10.0,
) -> np.ndarray:
    """Return chromatic focal shift curve from Zemax as ``[λ_nm, defocus_um]``.

    Parameters
    ----------
    system_file:
        Optional path to a ``.zmx`` lens file to load before running the
        analysis.
    start_wavelength, end_wavelength, step:
        Wavelength scan settings in nanometres.

    Notes
    -----
    This function requires Zemax OpticStudio with the ZOS-API available on the
    current machine. It uses :mod:`comtypes` to create a connection and run the
    *Chromatic Focal Shift* analysis.
    """
    if cc is None:
        raise RuntimeError("comtypes is required to communicate with Zemax")

    connection = cc.CreateObject("ZOSAPI.ZOSAPI_Connection")
    app = connection.CreateNewApplication()
    if app is None:
        raise RuntimeError("Unable to connect to Zemax OpticStudio")
    system = app.PrimarySystem

    if system_file:
        system.LoadFile(system_file, False)

    analysis = system.Analyses.New_ChromaticFocalShift()
    settings = analysis.GetSettings()
    settings.StartWavelength = start_wavelength
    settings.EndWavelength = end_wavelength
    settings.WavelengthInterval = step

    analysis.ApplyAndWaitForCompletion()
    results = analysis.GetResults()

    data = []
    for i in range(results.NumberOfDataRows):
        wl = results.GetDataHeader(i)
        defocus = results.Data.GetData(0, i)
        data.append((wl, defocus))

    analysis.Close()
    return np.asarray(data, dtype=np.float64)


def fringe_metrics(
    CHLdata: np.ndarray,
    *,
    defocus_range: int = 650,
    xrange_val: int = 400,
    F: float | None = None,
    gamma: float | None = None,
    psf_mode: str = "gauss",
) -> Tuple[float, float]:
    """Return maximum and mean colour fringe width for a defocus range.

    Parameters
    ----------
    CHLdata:
        Longitudinal chromatic focal shift values from Zemax.
    defocus_range:
        Range of defocus positions (±) to evaluate in microns.
    xrange_val:
        Half-width of the evaluation window in pixels.
    F, gamma:
        Exposure curve factor and display gamma defining the optical
        conditions. ``None`` falls back to library defaults.
    psf_mode:
        Point-spread function model to use.
    """

    old_xrange = XRANGE_VAL
    try:
        # Temporarily override the global evaluation window
        globals()["XRANGE_VAL"] = int(xrange_val)

        zs = np.arange(-defocus_range, defocus_range + 1, dtype=np.float64)
        widths = [
            Farbsaumbreite(z, F, gamma, CHLdata, psf_mode=psf_mode)
            for z in zs
        ]
    finally:
        globals()["XRANGE_VAL"] = old_xrange

    widths = np.asarray(widths, dtype=np.float64)
    return float(widths.max()), float(widths.mean())


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    p = argparse.ArgumentParser(description="Evaluate colour fringe width using Zemax data")
    p.add_argument("system", help="Zemax ZMX file", nargs="?")
    p.add_argument("--start", type=float, default=400.0, dest="start_wl", help="Start wavelength (nm)")
    p.add_argument("--end", type=float, default=700.0, dest="end_wl", help="End wavelength (nm)")
    p.add_argument("--step", type=float, default=10.0, dest="step", help="Wavelength step (nm)")
    p.add_argument("--defocus-range", type=int, default=1000)
    p.add_argument("--xrange", type=int, default=400)
    p.add_argument("--F", type=float, dest="F", help="Exposure curve factor")
    p.add_argument("--gamma", type=float, dest="gamma", help="Display gamma")
    p.add_argument("--psf-mode", choices=["disk", "gauss", "gauss_sphe"], default="gauss")
    args = p.parse_args()

    chl = fetch_chromatic_focal_shift(args.system, args.start_wl, args.end_wl, args.step)
    max_w, mean_w = fringe_metrics(
        chl[:, 1],
        defocus_range=args.defocus_range,
        xrange_val=args.xrange,
        F=args.F,
        gamma=args.gamma,
        psf_mode=args.psf_mode,
    )
    print(f"Max CFW: {max_w:.2f} px, Mean CFW: {mean_w:.2f} px")

"""
The defaults for those optical conditions are defined in core/cfw.py.

You can run the helper directly from the command line as shown in the README:

python -m achromatcfw.zemax_utils path/to/system.zmx \
    --defocus-range 650 --xrange 400 --F 8.0 --gamma 1.0
"""
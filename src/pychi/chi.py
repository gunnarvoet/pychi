"""Chi (χ) dissipation rate calculation.

Provides calc_chi (single-chunk computation, port of Calc_Chi_TChain_2.m)
and process_chi (orchestrator over depths and time chunks).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pychi.config import Config
from pychi.spectra import csd_odas


def calc_chi(
    temperature: NDArray[np.floating],
    U: float,
    gamma: float,
    alpha: float,
    grad_T_mag: float,
    sample_freq: float,
    config: Config,
) -> tuple[float, dict]:
    """Compute chi for a single time chunk at a single depth.

    Port of Calc_Chi_TChain_2.m. Computes the temperature power spectrum,
    takes the median spectral level in the inertial subrange (scaled by
    f^(5/3)), and applies the Batchelor/Osborn-Cox scaling.

    Parameters
    ----------
    temperature : 1D array
        NaN-free uncalibrated temperature for the chunk.
    U : float
        Horizontal velocity magnitude [m/s].
    gamma : float
        Mixing efficiency (typically 0.2).
    alpha : float
        Thermal expansion coefficient [1/°C].
    grad_T_mag : float
        Temperature gradient magnitude.
    sample_freq : float
        Sampling frequency in Hz.
    config : Config
        Provides spectra_size and avrg_lim.

    Returns
    -------
    chi_val : float
        Dissipation rate estimate.
    diagnostics : dict
        Keys: Pt (spectrum), f (frequencies), U, mean_t, spectral_slope.
    """
    n_fft = config.spectra_size

    Pt, f = csd_odas(
        temperature, n_fft, sample_freq,
        window=None, overlap=n_fft // 2, detrend="linear",
    )

    # Scale spectrum by f^(5/3) to flatten inertial subrange
    scale = f ** (5 / 3)

    # Select inertial subrange frequencies
    avrg_lim = config.avrg_lim
    in_band = (f > avrg_lim[0]) & (f < avrg_lim[1])

    # Chi formula — Matlab line 51:
    # chi = ((phi * (2*pi/U)^(2/3)) / 0.4
    #        * (g*alpha / (2*|grad_T_mag|*gamma))^(1/3)) ^ (3/2)
    g = 9.81
    grad_T_mag = abs(grad_T_mag)
    phi = float(np.nanmedian(Pt[in_band] * scale[in_band]))
    chi_val = (
        (phi * (2 * np.pi / U) ** (2 / 3))
        / 0.4
        * (g * alpha / (2 * grad_T_mag * gamma)) ** (1 / 3)
    ) ** (3 / 2)

    # Spectral slope — linear fit in log-log space
    slope_band = (f > avrg_lim[0] / 1.5) & (f < avrg_lim[1])
    if np.any(slope_band) and np.any(Pt[slope_band] > 0):
        log_f = np.log10(f[slope_band])
        log_Pt = np.log10(Pt[slope_band])
        coeffs = np.polyfit(log_f, log_Pt, 1)
        spectral_slope = float(coeffs[0])
    else:
        spectral_slope = float("nan")

    diagnostics = {
        "Pt": Pt,
        "f": f,
        "U": U,
        "mean_t": float(np.nanmean(temperature)),
        "spectral_slope": spectral_slope,
    }

    return chi_val, diagnostics


def process_chi(*args, **kwargs):
    raise NotImplementedError

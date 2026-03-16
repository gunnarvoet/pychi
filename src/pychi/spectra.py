"""Spectral estimation functions.

Provides a faithful port of csd_odas.m (auto-spectrum only) and a
scipy.signal.welch wrapper for cross-validation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch as _scipy_welch
from scipy.signal.windows import hann


def csd_odas(
    x: NDArray[np.floating],
    n_fft: int,
    rate: float,
    window: NDArray[np.floating] | None = None,
    overlap: int | None = None,
    detrend: str = "none",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the auto-spectrum of a real signal using Welch's method.

    Faithful port of csd_odas.m (auto-spectrum path only).

    Parameters
    ----------
    x : 1D array
        Input signal.
    n_fft : int
        FFT segment length.
    rate : float
        Sampling frequency in Hz.
    window : 1D array of length n_fft, or None
        Window. If None, Hanning window normalized to unit RMS.
    overlap : int or None
        Overlap points. Default: n_fft // 2.
    detrend : str
        "none", "constant" (order 0), "linear" (order 1),
        "parabolic" (order 2), "cubic" (order 3).

    Returns
    -------
    Pxx : 1D array
        Auto-spectral density (n_fft // 2 + 1).
    f : 1D array
        Frequency vector in Hz.
    """
    x = np.asarray(x, dtype=float).ravel()

    if len(x) < 2 * n_fft:
        raise ValueError(
            f"Input length ({len(x)}) must be more than twice n_fft ({n_fft})"
        )

    if overlap is None:
        overlap = n_fft // 2

    # Window — use exact Matlab csd_odas.m formula (lines 131-132):
    #   Window = 1 + cos(pi*(-1 + 2*(0:n_fft-1)'/n_fft))
    #   Window = Window / sqrt(mean(Window.^2))
    if window is None:
        window = 1.0 + np.cos(np.pi * (-1.0 + 2.0 * np.arange(n_fft) / n_fft))
        window = window / np.sqrt(np.mean(window**2))
    else:
        window = np.asarray(window, dtype=float).ravel()
        if len(window) != n_fft:
            raise ValueError(
                f"Window length ({len(window)}) must equal n_fft ({n_fft})"
            )

    # Detrend order
    detrend_orders = {
        "none": None,
        "constant": 0,
        "linear": 1,
        "parabolic": 2,
        "cubic": 3,
    }
    if detrend not in detrend_orders:
        raise ValueError(
            f"detrend must be one of {list(detrend_orders.keys())}, got '{detrend}'"
        )
    order = detrend_orders[detrend]

    # Segment and accumulate
    increment = n_fft - overlap
    num_segments = (len(x) - overlap) // increment
    ramp = np.arange(n_fft, dtype=float)

    Cxy = np.zeros(n_fft)

    for i in range(num_segments):
        start = i * increment
        segment = x[start : start + n_fft]

        if order is not None:
            coeffs = np.polyfit(ramp, segment, order)
            segment = segment - np.polyval(coeffs, ramp)

        xw = window * segment
        X = np.fft.fft(xw, n_fft)
        Cxy += np.abs(X) ** 2

    # Select positive frequencies
    if n_fft % 2:
        freq_range = slice(0, (n_fft + 1) // 2)
    else:
        freq_range = slice(0, n_fft // 2 + 1)

    Pxx = Cxy[freq_range]
    f = np.arange(Pxx.shape[0]) * rate / n_fft

    # Normalize: integral 0..Nyquist = variance
    Pxx = Pxx / num_segments / (n_fft * rate / 2)

    return Pxx, f


def welch_spectrum(
    x: NDArray[np.floating],
    n_fft: int,
    rate: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute auto-spectrum using scipy.signal.welch.

    Configured to match csd_odas: Hanning window, 50% overlap, linear detrend.

    Parameters
    ----------
    x : 1D array
    n_fft : int
        FFT segment length.
    rate : float
        Sampling frequency in Hz.

    Returns
    -------
    Pxx : 1D array
    f : 1D array
    """
    f, Pxx = _scipy_welch(
        x,
        fs=rate,
        window=hann(n_fft, sym=False),
        nperseg=n_fft,
        noverlap=n_fft // 2,
        detrend="linear",
        scaling="density",
    )
    return Pxx, f

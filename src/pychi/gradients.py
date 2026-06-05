"""Temperature gradient computation.

Provides vertical (dT/dz) and horizontal (dT/dx) gradient functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def vertical_gradient(
    temp_cal: NDArray[np.floating],
    depths: NDArray[np.floating],
    sensor_index: int,
) -> tuple[float, NDArray[np.floating]]:
    """Compute vertical temperature gradient dT/dz for a given sensor.

    Uses finite differences:
    - Top sensor (index 0): (T[0] - T[1]) / (depth[0] - depth[1])
    - Bottom sensor (index N-1): (T[N-2] - T[N-1]) / (depth[N-2] - depth[N-1])
    - Interior sensors: centered (T[i-1] - T[i+1]) / (depth[i-1] - depth[i+1])

    Parameters
    ----------
    temp_cal : 2D array, shape (n_depths, n_times)
        Calibrated temperature.
    depths : 1D array
        Depth values, ordered shallow-to-deep.
    sensor_index : int
        Index of the sensor being processed.

    Returns
    -------
    dtdz_mean : float
        Chunk-mean vertical gradient.
    dtdz_ts : 1D array
        Per-timestep vertical gradient.
    """
    n_depths = temp_cal.shape[0]

    if sensor_index == 0:
        i_upper, i_lower = 0, 1
    elif sensor_index == n_depths - 1:
        i_upper, i_lower = n_depths - 2, n_depths - 1
    else:
        i_upper, i_lower = sensor_index - 1, sensor_index + 1

    dz = depths[i_upper] - depths[i_lower]
    dtdz_ts = (temp_cal[i_upper, :] - temp_cal[i_lower, :]) / dz
    dtdz_mean = float(
        (np.nanmean(temp_cal[i_upper, :]) - np.nanmean(temp_cal[i_lower, :])) / dz
    )

    return dtdz_mean, dtdz_ts


def horizontal_gradient(
    temp_cal_chunk: NDArray[np.floating],
    chunk_duration_s: float,
    U: float,
) -> float:
    r"""Compute horizontal temperature gradient dT/dx via frozen-field hypothesis.

    Parameters
    ----------
    temp_cal_chunk : 1D array
        Calibrated temperatures for the time chunk.
    chunk_duration_s : float
        Chunk duration in seconds.
    U : float
        Mean horizontal velocity magnitude [m/s].

    Returns
    -------
    dtdx : float
        Horizontal temperature gradient estimate.

    Notes
    -----
    Uses Taylor's frozen-field hypothesis to convert the temporal gradient
    into a horizontal (along-flow) gradient,

    $$\frac{\partial T}{\partial x} = \frac{1}{U}\,\frac{\partial T}{\partial t}$$

    evaluated here as the chunk-endpoint temperature difference divided by the
    chunk duration and $U$.
    """
    return float((temp_cal_chunk[-1] - temp_cal_chunk[0]) / (chunk_duration_s * U))

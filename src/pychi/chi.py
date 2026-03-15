"""Chi (χ) dissipation rate calculation.

Provides calc_chi (single-chunk computation, port of Calc_Chi_TChain_2.m)
and process_chi (orchestrator over depths and time chunks).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from pychi.config import Config
from pychi.gradients import horizontal_gradient, vertical_gradient
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
    # Require meaningful signal: max power > threshold to guard against
    # floating-point noise in constant/zero-signal inputs.
    _MIN_SPECTRAL_POWER = 1e-20
    slope_band = (f > avrg_lim[0] / 1.5) & (f < avrg_lim[1])
    if (
        np.any(slope_band)
        and np.any(Pt[slope_band] > 0)
        and np.max(Pt[slope_band]) > _MIN_SPECTRAL_POWER
    ):
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


def process_chi(
    temp_uncal: xr.DataArray,
    temp_cal: xr.DataArray,
    u_velocity: xr.DataArray,
    v_velocity: xr.DataArray,
    w_velocity: xr.DataArray,
    depths: NDArray[np.floating],
    adcp_depths: NDArray[np.floating],
    config: Config,
    sensor_indices: list[int] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute chi over all depths and time chunks.

    Orchestrator that loops over requested depth levels and time windows,
    calling vertical_gradient, horizontal_gradient, and calc_chi for each
    chunk. Returns primary results and binned spectral QC data.

    Parameters
    ----------
    temp_uncal : DataArray (depth, time)
        Uncalibrated T-chain temperatures.
    temp_cal : DataArray (depth, time)
        Calibrated T-chain temperatures.
    u_velocity, v_velocity, w_velocity : DataArray (time, z)
        ADCP velocity components.
    depths : 1D array
        T-chain depth levels, ordered shallow-to-deep.
    adcp_depths : 1D array
        ADCP depth bins.
    config : Config
        Pipeline configuration.
    sensor_indices : list of int, optional
        Depth indices to process. Default: all.

    Returns
    -------
    result : xr.Dataset
        Primary output with chi, diagnostics, and per-chunk spectra.
    binned : xr.Dataset
        Binned average spectra for QC.
    """
    import gsw

    depths = np.asarray(depths, dtype=float)
    adcp_depths = np.asarray(adcp_depths, dtype=float)
    times = temp_uncal.time.values
    adcp_times = u_velocity.time.values

    # Infer sample frequency
    dt_raw = times[1] - times[0]
    if np.issubdtype(type(dt_raw), np.timedelta64):
        dt = dt_raw / np.timedelta64(1, "s")
    else:
        dt = float(dt_raw)
    sample_freq = 1.0 / dt if dt != 0 else 1.0

    # Time chunk boundaries (numeric seconds)
    if np.issubdtype(times.dtype, np.datetime64):
        t0 = times[0]
        times_numeric = (times - t0) / np.timedelta64(1, "s")
    else:
        t0 = None
        times_numeric = times.astype(float)

    t_start = float(times_numeric[0])
    t_end = float(times_numeric[-1])
    chunk_step = config.chi_time_step
    # Use t_end + chunk_step as stop so the final boundary always lies beyond
    # the last sample, guaranteeing at least ceil((t_end - t_start) / chunk_step)
    # chunks regardless of floating-point rounding in np.arange.
    time_bnds_list = np.arange(t_start, t_end + chunk_step, chunk_step)
    n_chunks = len(time_bnds_list) - 1
    if n_chunks < 1:
        time_bnds_list = np.array([t_start, t_end + chunk_step])
        n_chunks = 1

    chunk_centers_numeric = (time_bnds_list[:-1] + time_bnds_list[1:]) / 2
    time_bnds_pairs = np.column_stack([time_bnds_list[:-1], time_bnds_list[1:]])

    if t0 is not None:
        chunk_centers = t0 + (chunk_centers_numeric * 1e9).astype("timedelta64[ns]")
    else:
        chunk_centers = chunk_centers_numeric

    if sensor_indices is None:
        sensor_indices = list(range(len(depths)))

    proc_depths = depths[sensor_indices]
    n_depths = len(sensor_indices)
    n_freq = config.spectra_size // 2 + 1

    # Pre-allocate output arrays
    chi_arr = np.full((n_depths, n_chunks), np.nan)
    U_arr = np.full((n_depths, n_chunks), np.nan)
    mean_u_arr = np.full((n_depths, n_chunks), np.nan)
    mean_v_arr = np.full((n_depths, n_chunks), np.nan)
    mean_w_arr = np.full((n_depths, n_chunks), np.nan)
    dtdz_arr = np.full((n_depths, n_chunks), np.nan)
    dtdx_arr = np.full((n_depths, n_chunks), np.nan)
    alpha_arr = np.full((n_depths, n_chunks), np.nan)
    gamma_arr = np.full((n_depths, n_chunks), np.nan)
    unstab_prop_arr = np.full((n_depths, n_chunks), np.nan)
    unstab_count_arr = np.full((n_depths, n_chunks), np.nan)
    unstab_length_arr = np.full((n_depths, n_chunks), np.nan)
    slope_arr = np.full((n_depths, n_chunks), np.nan)
    mean_t_arr = np.full((n_depths, n_chunks), np.nan)
    mean_t_uncal_arr = np.full((n_depths, n_chunks), np.nan)
    avrg_lim_actual_arr = np.full((n_depths, n_chunks), np.nan)
    Pt_arr = np.full((n_depths, n_chunks, n_freq), np.nan)
    f_arr = None

    for di, si in enumerate(sensor_indices):
        depth_val = depths[si]
        z_idx = int(np.argmin(np.abs(adcp_depths - depth_val)))

        for jj in range(n_chunks):
            t_lo = time_bnds_list[jj]
            t_hi = time_bnds_list[jj + 1]

            t_mask = (times_numeric >= t_lo) & (times_numeric < t_hi)
            if np.issubdtype(adcp_times.dtype, np.datetime64):
                adcp_numeric = (adcp_times - (t0 if t0 is not None else adcp_times[0])) / np.timedelta64(1, "s")
            else:
                adcp_numeric = adcp_times.astype(float)
            adcp_t_mask = (adcp_numeric >= t_lo) & (adcp_numeric < t_hi)

            # Velocities
            u_chunk = u_velocity.values[adcp_t_mask, :]
            v_chunk = v_velocity.values[adcp_t_mask, :]
            w_chunk = w_velocity.values[adcp_t_mask, :]

            mean_u_at_z = float(np.nanmean(u_chunk[:, z_idx]))
            mean_v_at_z = float(np.nanmean(v_chunk[:, z_idx]))
            U_val = np.sqrt(mean_u_at_z**2 + mean_v_at_z**2)
            U_arr[di, jj] = U_val

            # Temperature
            temp_uncal_chunk = temp_uncal.values[si, t_mask]
            temp_cal_chunk = temp_cal.values[:, t_mask]
            temp_cal_sensor = temp_cal.values[si, t_mask]

            mean_t_arr[di, jj] = float(np.nanmean(temp_cal_sensor))

            # Vertical gradient
            dtdz_mean, dtdz_ts = vertical_gradient(temp_cal_chunk, depths, si)
            dtdz_arr[di, jj] = dtdz_mean

            # Horizontal gradient
            dtdx_val = horizontal_gradient(temp_cal_sensor, config.chi_time_step, U_val)
            dtdx_arr[di, jj] = dtdx_val

            # Instability
            valid_mask = ~np.isnan(dtdz_ts)
            unstab_length = int(np.sum(valid_mask))
            unstab_count = int(np.sum(dtdz_ts[valid_mask] > 0))
            unstab_prop = unstab_count / unstab_length if unstab_length > 0 else np.nan
            unstab_prop_arr[di, jj] = unstab_prop
            unstab_count_arr[di, jj] = unstab_count
            unstab_length_arr[di, jj] = unstab_length

            # Thermal expansion coefficient
            p = gsw.p_from_z(-depth_val, config.latitude)
            SA = gsw.SA_from_SP(config.salinity, p, 0, config.latitude)
            CT = gsw.CT_from_t(SA, mean_t_arr[di, jj], p)
            alpha_val = float(gsw.alpha(SA, CT, p))
            alpha_arr[di, jj] = alpha_val

            # Adjust low-freq limit
            hab = config.bottom_depth - depth_val
            avrg_lim_lo = max(config.avrg_lim[0], config.U_ref / hab)
            avrg_lim_actual_arr[di, jj] = avrg_lim_lo

            # NaN check
            has_nan = np.any(np.isnan(temp_uncal_chunk))

            if has_nan:
                chi_arr[di, jj] = np.nan
                gamma_arr[di, jj] = np.nan
                mean_t_uncal_arr[di, jj] = np.nan
                mean_u_arr[di, jj] = float(np.nanmean(u_chunk[:, z_idx]))
                mean_v_arr[di, jj] = float(np.nanmean(v_chunk[:, z_idx]))
                mean_w_arr[di, jj] = float(np.nanmean(w_chunk[:, z_idx]))
                slope_arr[di, jj] = np.nan
            else:
                mean_u_profile = np.nanmean(u_chunk, axis=0)
                mean_v_profile = np.nanmean(v_chunk, axis=0)
                mean_w_profile = np.nanmean(w_chunk, axis=0)
                mean_u_arr[di, jj] = float(np.interp(depth_val, adcp_depths, mean_u_profile))
                mean_v_arr[di, jj] = float(np.interp(depth_val, adcp_depths, mean_v_profile))
                mean_w_arr[di, jj] = float(np.interp(depth_val, adcp_depths, mean_w_profile))

                chunk_config = Config(
                    spectra_size=config.spectra_size,
                    avrg_lim=[avrg_lim_lo, config.avrg_lim[1]],
                    chi_time_step=config.chi_time_step,
                    gamma=config.gamma,
                    salinity=config.salinity,
                    latitude=config.latitude,
                    bottom_depth=config.bottom_depth,
                    U_ref=config.U_ref,
                    chi_spectra_bin_bounds=config.chi_spectra_bin_bounds,
                )

                if config.gradient_mode == "full":
                    grad_T_mag = np.sqrt(dtdz_mean**2 + dtdx_val**2)
                else:
                    grad_T_mag = abs(dtdz_mean)

                chi_val, diag = calc_chi(
                    temp_uncal_chunk, U_val, config.gamma, alpha_val,
                    grad_T_mag, sample_freq, chunk_config,
                )

                chi_arr[di, jj] = chi_val
                gamma_arr[di, jj] = config.gamma
                mean_t_uncal_arr[di, jj] = diag["mean_t"]
                slope_arr[di, jj] = diag["spectral_slope"]
                Pt_arr[di, jj, :] = diag["Pt"]
                if f_arr is None:
                    f_arr = diag["f"]

    if f_arr is None:
        f_arr = np.arange(n_freq) * sample_freq / config.spectra_size

    # Assemble primary Dataset
    result = xr.Dataset(
        {
            "chi": (["depth", "time"], chi_arr),
            "U": (["depth", "time"], U_arr),
            "mean_u": (["depth", "time"], mean_u_arr),
            "mean_v": (["depth", "time"], mean_v_arr),
            "mean_w": (["depth", "time"], mean_w_arr),
            "dtdz": (["depth", "time"], dtdz_arr),
            "dtdx": (["depth", "time"], dtdx_arr),
            "alpha": (["depth", "time"], alpha_arr),
            "gamma": (["depth", "time"], gamma_arr),
            "unstab_prop": (["depth", "time"], unstab_prop_arr),
            "unstab_count": (["depth", "time"], unstab_count_arr),
            "unstab_length": (["depth", "time"], unstab_length_arr),
            "spectral_slope": (["depth", "time"], slope_arr),
            "mean_t": (["depth", "time"], mean_t_arr),
            "mean_t_uncal": (["depth", "time"], mean_t_uncal_arr),
            "avrg_lim_actual": (["depth", "time"], avrg_lim_actual_arr),
            "Pt": (["depth", "time", "frequency"], Pt_arr),
            "time_bnds": (["time", "bnds"], time_bnds_pairs),
        },
        coords={
            "depth": proc_depths,
            "time": chunk_centers,
            "frequency": f_arr,
        },
    )

    # Binned average spectra
    bin_edges = np.array(config.chi_spectra_bin_bounds, dtype=float)
    n_bins = len(bin_edges) - 1
    binned_sum = np.zeros((n_depths, n_bins, n_freq))
    binned_count = np.zeros((n_depths, n_bins), dtype=int)

    for di in range(n_depths):
        for jj in range(n_chunks):
            chi_val = chi_arr[di, jj]
            if np.isnan(chi_val) or chi_val <= 0:
                continue
            log_chi = np.log10(chi_val)
            bin_idx = np.searchsorted(bin_edges, log_chi, side="right") - 1
            if 0 <= bin_idx < n_bins:
                binned_sum[di, bin_idx, :] += Pt_arr[di, jj, :]
                binned_count[di, bin_idx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        binned_mean = np.where(
            binned_count[:, :, np.newaxis] > 0,
            binned_sum / binned_count[:, :, np.newaxis],
            np.nan,
        )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned = xr.Dataset(
        {
            "binned_spectra": (["depth", "chi_bin", "frequency"], binned_mean),
            "binned_counts": (["depth", "chi_bin"], binned_count),
        },
        coords={
            "depth": proc_depths,
            "chi_bin": bin_centers,
            "frequency": f_arr,
            "chi_bin_edges": ("chi_bin_edge", bin_edges),
        },
    )

    return result, binned

import numpy as np
import pytest
import xarray as xr

from pychi.chi import calc_chi, process_chi
from pychi.config import Config


def _make_synthetic_data(n_depths=3, n_times=1200, rate=1.0):
    """Create synthetic xarray inputs for process_chi."""
    rng = np.random.default_rng(42)
    depths = np.array([100.0, 110.0, 120.0])
    times = np.arange(n_times, dtype=float)

    temp_base = np.array([15.0, 14.0, 13.0])[:, None]
    temp_uncal = temp_base + 0.1 * rng.standard_normal((n_depths, n_times))
    temp_cal = temp_uncal + 0.5

    temp_uncal_da = xr.DataArray(
        temp_uncal, dims=["depth", "time"],
        coords={"depth": depths, "time": times},
    )
    temp_cal_da = xr.DataArray(
        temp_cal, dims=["depth", "time"],
        coords={"depth": depths, "time": times},
    )

    adcp_depths = np.array([95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    adcp_times = np.arange(n_times, dtype=float)
    n_adcp_z = len(adcp_depths)

    u_vel = 0.2 + 0.02 * rng.standard_normal((n_times, n_adcp_z))
    v_vel = 0.1 + 0.02 * rng.standard_normal((n_times, n_adcp_z))
    w_vel = 0.01 * rng.standard_normal((n_times, n_adcp_z))

    u_da = xr.DataArray(u_vel, dims=["time", "z"], coords={"time": adcp_times, "z": adcp_depths})
    v_da = xr.DataArray(v_vel, dims=["time", "z"], coords={"time": adcp_times, "z": adcp_depths})
    w_da = xr.DataArray(w_vel, dims=["time", "z"], coords={"time": adcp_times, "z": adcp_depths})

    return temp_uncal_da, temp_cal_da, u_da, v_da, w_da, depths, adcp_depths


def test_process_chi_returns_dataset():
    """process_chi returns an xarray Dataset with expected variables."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, binned = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config)

    assert isinstance(result, xr.Dataset)
    expected_vars = {
        "chi", "U", "mean_u", "mean_v", "mean_w",
        "dtdz", "dtdx", "alpha", "gamma",
        "unstab_prop", "unstab_count", "unstab_length",
        "spectral_slope", "mean_t", "mean_t_uncal",
        "avrg_lim_actual", "Pt",
    }
    assert expected_vars.issubset(set(result.data_vars))
    assert "depth" in result.coords
    assert "time" in result.coords
    assert "time_bnds" in result.coords or "time_bnds" in result.data_vars


def test_process_chi_dimensions():
    """Output has correct dimensions: (n_depths, n_chunks)."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, _ = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config)

    assert result.dims["depth"] == 3
    assert result.dims["time"] == 2


def test_process_chi_single_sensor():
    """Single sensor mode processes only the requested depth."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, _ = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config, sensor_indices=[1])

    assert result.dims["depth"] == 1
    assert float(result.depth.values[0]) == 110.0


def test_process_chi_nan_chunk_handling():
    """Chunks with NaN in temperature produce NaN chi."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    temp_uncal.values[0, 10] = np.nan

    result, _ = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config)

    assert np.isnan(result["chi"].values[0, 0])
    assert np.isfinite(result["mean_t"].values[0, 0])
    assert np.isnan(result["gamma"].values[0, 0])


def test_process_chi_binned_spectra():
    """Binned spectra Dataset has expected structure."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    _, binned = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config)

    assert isinstance(binned, xr.Dataset)
    assert "binned_spectra" in binned.data_vars
    assert "binned_counts" in binned.data_vars
    assert "frequency" in binned.coords


def test_process_chi_avrg_lim_adjustment():
    """When U_ref / hab > avrg_lim[0], the low-frequency limit is adjusted upward."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()

    depths_shallow = np.array([140.0, 145.0, 148.0])
    adcp_depths_shallow = np.array([135.0, 140.0, 145.0, 148.0, 150.0])

    temp_uncal = temp_uncal.assign_coords(depth=depths_shallow)
    temp_cal = temp_cal.assign_coords(depth=depths_shallow)
    u = u.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(z=adcp_depths_shallow)
    v = v.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(z=adcp_depths_shallow)
    w = w.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(z=adcp_depths_shallow)

    config = Config(chi_time_step=600, bottom_depth=150.0, U_ref=0.1)

    result, _ = process_chi(temp_uncal, temp_cal, u, v, w, depths_shallow, adcp_depths_shallow, config, sensor_indices=[0])

    assert result["avrg_lim_actual"].values[0, 0] == pytest.approx(0.01)


def test_calc_chi_zero_signal_slope_is_nan():
    """spectral_slope is NaN when all spectral values in the band are zero."""
    temperature = np.full(600, 15.0)
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert np.isnan(diag["spectral_slope"])

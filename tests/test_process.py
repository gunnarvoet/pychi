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
        temp_uncal,
        dims=["depth", "time"],
        coords={"depth": depths, "time": times},
    )
    temp_cal_da = xr.DataArray(
        temp_cal,
        dims=["depth", "time"],
        coords={"depth": depths, "time": times},
    )

    adcp_depths = np.array([95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    adcp_times = np.arange(n_times, dtype=float)
    n_adcp_z = len(adcp_depths)

    u_vel = 0.2 + 0.02 * rng.standard_normal((n_times, n_adcp_z))
    v_vel = 0.1 + 0.02 * rng.standard_normal((n_times, n_adcp_z))
    w_vel = 0.01 * rng.standard_normal((n_times, n_adcp_z))

    u_da = xr.DataArray(
        u_vel, dims=["time", "z"], coords={"time": adcp_times, "z": adcp_depths}
    )
    v_da = xr.DataArray(
        v_vel, dims=["time", "z"], coords={"time": adcp_times, "z": adcp_depths}
    )
    w_da = xr.DataArray(
        w_vel, dims=["time", "z"], coords={"time": adcp_times, "z": adcp_depths}
    )

    return temp_uncal_da, temp_cal_da, u_da, v_da, w_da, depths, adcp_depths


def test_process_chi_returns_dataset():
    """process_chi returns an xarray Dataset with expected variables."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, binned = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config
    )

    assert isinstance(result, xr.Dataset)
    expected_vars = {
        "chi",
        "U",
        "mean_u",
        "mean_v",
        "mean_w",
        "dtdz",
        "dtdx",
        "alpha",
        "gamma",
        "unstab_prop",
        "unstab_count",
        "unstab_length",
        "spectral_slope",
        "mean_t",
        "mean_t_uncal",
        "avrg_lim_actual",
        "Pt",
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

    assert result.sizes["depth"] == 3
    assert result.sizes["time"] == 2


def test_process_chi_single_sensor():
    """Single sensor mode processes only the requested depth."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, _ = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config, sensor_indices=[1]
    )

    assert result.sizes["depth"] == 1
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


def test_process_chi_short_chunk_handling():
    """A trailing window too short for a spectrum yields NaN chi, not a crash."""
    # 1300 samples with a 600 s step leaves a final chunk of 100 samples, which
    # is fewer than 2 * spectra_size (256) and cannot support csd_odas.
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data(
        n_times=1300
    )
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, _ = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config)

    # Three chunks: [0,600), [600,1200), [1200,1300). The last is too short.
    assert result.sizes["time"] == 3
    assert np.all(np.isnan(result["chi"].values[:, -1]))
    # Full-length chunks still produce finite chi.
    assert np.any(np.isfinite(result["chi"].values[:, :-1]))


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
    u = u.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(
        z=adcp_depths_shallow
    )
    v = v.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(
        z=adcp_depths_shallow
    )
    w = w.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(
        z=adcp_depths_shallow
    )

    config = Config(chi_time_step=600, bottom_depth=150.0, U_ref=0.1)

    result, _ = process_chi(
        temp_uncal,
        temp_cal,
        u,
        v,
        w,
        depths_shallow,
        adcp_depths_shallow,
        config,
        sensor_indices=[0],
    )

    assert result["avrg_lim_actual"].values[0, 0] == pytest.approx(0.01)


def test_calc_chi_zero_signal_slope_is_nan():
    """spectral_slope is NaN when all spectral values in the band are zero."""
    temperature = np.full(600, 15.0)
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert np.isnan(diag["spectral_slope"])


def _make_deep_adiabatic_data(dtheta_dz=-2e-5, n_times=1200):
    """Deep three-sensor chain whose in-situ gradient is mostly adiabatic.

    Potential temperature falls with depth at ``dtheta_dz`` [°C/m, depth
    positive down]; in-situ temperature rises with depth because of the
    adiabatic lapse rate.
    """
    import gsw

    rng = np.random.default_rng(7)
    lat, sp = 0.5, 34.7
    depths = np.array([4336.0, 4340.0, 4344.0])
    times = np.arange(n_times, dtype=float)
    p = gsw.p_from_z(-depths, lat)
    p_ref = p.mean()
    SA = gsw.SA_from_SP(sp, p_ref, 0, lat)
    theta = 1.40 + dtheta_dz * (depths - depths.mean())
    t_insitu = gsw.pt_from_t(SA, theta, p_ref, p)
    temp = t_insitu[:, None] + 1e-4 * rng.standard_normal((3, n_times))
    temp_da = xr.DataArray(
        temp, dims=["depth", "time"], coords={"depth": depths, "time": times}
    )
    adcp_depths = np.array([4340.0])
    u = xr.DataArray(
        np.full((n_times, 1), 0.05),
        dims=["time", "z"],
        coords={"time": times, "z": adcp_depths},
    )
    v = xr.zeros_like(u)
    w = xr.zeros_like(u)
    config_kwargs = dict(
        chi_time_step=600,
        bottom_depth=4389.0,
        latitude=lat,
        salinity=sp,
        gradient_mode="vertical",
    )
    return temp_da, u, v, w, depths, adcp_depths, config_kwargs


def test_process_chi_potential_gradient_removes_lapse_rate():
    """gradient_temperature='potential' differences theta, not in-situ T."""
    dtheta_dz = -2e-5
    temp, u, v, w, depths, adcp_depths, kw = _make_deep_adiabatic_data(dtheta_dz)

    res_in, _ = process_chi(temp, temp, u, v, w, depths, adcp_depths, Config(**kw))
    res_pt, _ = process_chi(
        temp,
        temp,
        u,
        v,
        w,
        depths,
        adcp_depths,
        Config(gradient_temperature="potential", **kw),
    )

    # In situ: warmer with depth, close to the adiabatic lapse rate (~1.2e-4).
    assert np.all(res_in["dtdz"].values > 5e-5)
    # Potential: the imposed stable gradient is recovered.
    np.testing.assert_allclose(res_pt["dtdz"].values, dtheta_dz, atol=3e-6)
    # chi scales as |gradient|^(-1/2); everything else is unchanged.
    expected = np.sqrt(np.abs(res_in["dtdz"].values) / np.abs(res_pt["dtdz"].values))
    np.testing.assert_allclose(
        res_pt["chi"].values / res_in["chi"].values, expected, rtol=1e-10
    )
    # Mean temperature and alpha still come from in-situ temperature.
    np.testing.assert_array_equal(res_pt["mean_t"].values, res_in["mean_t"].values)
    np.testing.assert_array_equal(res_pt["alpha"].values, res_in["alpha"].values)


def test_process_chi_gradient_temperature_default_is_in_situ():
    """The default reproduces the reference (in-situ) behavior exactly."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    base = Config(chi_time_step=600, bottom_depth=200.0)
    explicit = Config(
        chi_time_step=600, bottom_depth=200.0, gradient_temperature="in_situ"
    )

    r0, _ = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, base)
    r1, _ = process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, explicit)

    np.testing.assert_array_equal(r0["chi"].values, r1["chi"].values)
    np.testing.assert_array_equal(r0["dtdz"].values, r1["dtdz"].values)


def test_process_chi_gradient_temperature_invalid():
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0, gradient_temperature="x")
    with pytest.raises(ValueError, match="gradient_temperature"):
        process_chi(temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config)


def test_process_chi_min_speed_masks_slow_chunks():
    """Chunks with U below min_speed get NaN chi; U itself is still reported."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    kw = dict(chi_time_step=600, bottom_depth=200.0)

    r_none, _ = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, Config(**kw)
    )
    # Synthetic U is ~0.22 m/s.
    r_low, _ = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, Config(min_speed=0.1, **kw)
    )
    r_high, _ = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, Config(min_speed=0.5, **kw)
    )

    np.testing.assert_array_equal(r_low["chi"].values, r_none["chi"].values)
    assert np.all(np.isnan(r_high["chi"].values))
    assert np.all(np.isnan(r_high["spectral_slope"].values))
    np.testing.assert_array_equal(r_high["U"].values, r_none["U"].values)
    np.testing.assert_array_equal(r_high["dtdz"].values, r_none["dtdz"].values)


def test_process_chi_min_gradient_masks_weak_gradients():
    """Chunks whose gradient magnitude is below min_gradient get NaN chi."""
    temp, u, v, w, depths, adcp_depths, kw = _make_deep_adiabatic_data(-2e-5)
    kw["gradient_temperature"] = "potential"

    r_none, _ = process_chi(temp, temp, u, v, w, depths, adcp_depths, Config(**kw))
    r_low, _ = process_chi(
        temp, temp, u, v, w, depths, adcp_depths, Config(min_gradient=1e-5, **kw)
    )
    r_high, _ = process_chi(
        temp, temp, u, v, w, depths, adcp_depths, Config(min_gradient=1e-4, **kw)
    )

    np.testing.assert_array_equal(r_low["chi"].values, r_none["chi"].values)
    assert np.all(np.isnan(r_high["chi"].values))
    np.testing.assert_array_equal(r_high["dtdz"].values, r_none["dtdz"].values)


def test_process_chi_require_stable_masks_unstable_gradients():
    """require_stable leaves chi NaN where temperature rises with depth."""
    # In situ, the adiabatic chain is warmer with depth: unstable by sign.
    temp, u, v, w, depths, adcp_depths, kw = _make_deep_adiabatic_data(-2e-5)

    r_insitu, _ = process_chi(
        temp, temp, u, v, w, depths, adcp_depths, Config(require_stable=True, **kw)
    )
    r_pot, _ = process_chi(
        temp,
        temp,
        u,
        v,
        w,
        depths,
        adcp_depths,
        Config(require_stable=True, gradient_temperature="potential", **kw),
    )

    assert np.all(r_insitu["dtdz"].values > 0)
    assert np.all(np.isnan(r_insitu["chi"].values))
    assert np.all(r_pot["dtdz"].values < 0)
    assert np.all(np.isfinite(r_pot["chi"].values))

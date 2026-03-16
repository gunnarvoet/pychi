import numpy as np
import pytest

from pychi.chi import calc_chi
from pychi.config import Config

from conftest import requires_matlab_fixtures


def test_calc_chi_returns_positive_value():
    """calc_chi returns a positive chi value for a realistic synthetic signal."""
    rng = np.random.default_rng(42)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    U = 0.3
    gamma = 0.2
    alpha = 2.5e-4
    grad_T_mag = 0.01
    sample_freq = 1.0
    config = Config()

    chi_val, diag = calc_chi(
        temperature, U, gamma, alpha, grad_T_mag, sample_freq, config
    )

    assert chi_val > 0
    assert np.isfinite(chi_val)


def test_calc_chi_diagnostics_keys():
    """calc_chi returns diagnostics with expected keys."""
    rng = np.random.default_rng(42)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert "Pt" in diag
    assert "f" in diag
    assert "U" in diag
    assert "mean_t" in diag
    assert "spectral_slope" in diag


def test_calc_chi_diagnostics_spectrum_shape():
    """Pt and f have consistent lengths (n_fft // 2 + 1)."""
    rng = np.random.default_rng(42)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    config = Config(spectra_size=128)

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert len(diag["Pt"]) == 128 // 2 + 1
    assert len(diag["f"]) == len(diag["Pt"])


def test_calc_chi_mean_t():
    """mean_t in diagnostics equals the mean of the input temperature."""
    temperature = np.array([10.0, 11.0, 12.0] * 200)  # 600 samples
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert diag["mean_t"] == pytest.approx(np.mean(temperature))


def test_calc_chi_spectral_slope_is_finite():
    """spectral_slope is a finite float for a typical signal."""
    rng = np.random.default_rng(42)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert np.isfinite(diag["spectral_slope"])


@requires_matlab_fixtures
def test_calc_chi_vs_matlab(chi_fixture):
    """calc_chi output matches Matlab chi value to ~1e-6 relative tolerance."""
    data = chi_fixture
    temp_in = data["temp_in"]
    U = float(data["U_in"])
    gamma = float(data["gamma"])
    alpha = float(data["alpha_val"])
    grad_T_mag = float(data["grad_T_mag"])
    avrg_lim = data["avrg_lim"].tolist()
    chi_matlab = float(data["chi_val"])

    config = Config(spectra_size=128, avrg_lim=avrg_lim)

    chi_val, _ = calc_chi(temp_in, U, gamma, alpha, grad_T_mag, 1.0, config)

    np.testing.assert_allclose(chi_val, chi_matlab, rtol=1e-6)

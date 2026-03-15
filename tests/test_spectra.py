import numpy as np
import pytest

from pychi.spectra import csd_odas


def test_csd_odas_variance_preservation():
    """Integral of auto-spectrum from 0 to Nyquist equals signal variance."""
    rng = np.random.default_rng(42)
    n = 1024
    rate = 1.0
    x = rng.standard_normal(n)
    n_fft = 128

    Pxx, f = csd_odas(x, n_fft, rate)

    spectral_variance = np.trapezoid(Pxx, f)
    signal_variance = np.var(x, ddof=0)
    assert spectral_variance == pytest.approx(signal_variance, rel=0.15)


def test_csd_odas_frequency_vector():
    """Frequency vector ranges from 0 to Nyquist, with correct spacing."""
    n = 512
    rate = 2.0
    n_fft = 64
    x = np.random.default_rng(0).standard_normal(n)

    Pxx, f = csd_odas(x, n_fft, rate)

    assert f[0] == 0.0
    assert f[-1] == pytest.approx(rate / 2)
    assert len(f) == n_fft // 2 + 1
    assert len(Pxx) == len(f)


def test_csd_odas_detrend_linear():
    """Linear detrending removes a linear trend from each segment."""
    n = 512
    rate = 1.0
    n_fft = 128
    x = np.linspace(0, 10, n)

    Pxx_none, _ = csd_odas(x, n_fft, rate, detrend="none")
    Pxx_lin, _ = csd_odas(x, n_fft, rate, detrend="linear")

    assert np.sum(Pxx_lin) < np.sum(Pxx_none) * 0.01


def test_csd_odas_input_too_short():
    """Raises ValueError if input is shorter than 2 * n_fft."""
    x = np.ones(100)
    with pytest.raises(ValueError, match="twice"):
        csd_odas(x, n_fft=128, rate=1.0)

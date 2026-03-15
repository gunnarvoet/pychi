from pathlib import Path

import pytest

from pychi.config import Config


def test_config_defaults():
    """Config() with no arguments uses Matlab default values."""
    cfg = Config()
    assert cfg.spectra_size == 128
    assert cfg.avrg_lim == [0.008, 0.1]
    assert cfg.chi_time_step == 600
    assert cfg.gamma == pytest.approx(0.2)
    assert cfg.salinity == pytest.approx(35.0)
    assert cfg.latitude == pytest.approx(54.25)
    assert cfg.bottom_depth == pytest.approx(1466.0)
    assert cfg.U_ref == pytest.approx(0.1)
    assert cfg.gradient_mode == "full"
    assert cfg.chi_spectra_bin_bounds == [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3]


def test_config_from_yaml(tmp_path):
    """Config.from_yaml loads values from a YAML file."""
    yml = tmp_path / "test.yml"
    yml.write_text(
        "spectral:\n"
        "  spectra_size: 256\n"
        "  avrg_lim: [0.01, 0.2]\n"
        "physics:\n"
        "  gamma: 0.15\n"
    )
    cfg = Config.from_yaml(yml)
    assert cfg.spectra_size == 256
    assert cfg.avrg_lim == [0.01, 0.2]
    assert cfg.gamma == pytest.approx(0.15)
    # Unspecified fields fall back to defaults
    assert cfg.chi_time_step == 600
    assert cfg.salinity == pytest.approx(35.0)


def test_config_from_yaml_missing_file():
    """Config.from_yaml raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("/nonexistent/path.yml")

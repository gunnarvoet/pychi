from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent.parent / "config" / "default.yml"
)


@dataclass
class Config:
    r"""Configuration for the pychi pipeline.

    Holds the spectral, quality-control, and physical parameters used by
    `pychi.chi.process_chi`. Construct with `Config()` for the defaults,
    override individual fields directly, or load from a YAML file with
    `from_yaml`.
    """

    # Spectral parameters
    spectra_size: int = 128
    r"""FFT segment length in samples for the Welch spectrum (128 = $2^7$).
    Sets the spectral resolution within each chunk."""

    avrg_lim: list[float] = field(default_factory=lambda: [0.008, 0.1])
    r"""Inertial-subrange frequency band ``[low, high]`` in Hz over which the
    median spectral level $\phi_T$ is taken. The low bound may be raised per
    chunk by the $U/h$ cutoff (see `U_ref`)."""

    chi_time_step: int = 600
    """Duration of each processing chunk in seconds (600 = 10 minutes). The
    pipeline loops over consecutive chunks of this length."""

    # QC parameters
    chi_spectra_bin_bounds: list[int] = field(
        default_factory=lambda: [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3]
    )
    r"""Bin edges in $\log_{10}(\chi)$ used to group and average per-chunk
    spectra for the quality-control diagnostic."""

    # Physics parameters
    gamma: float = 0.2
    r"""Mixing efficiency $\Gamma$ in the $\chi$ estimator (dissipation ratio)."""

    salinity: float = 35.0
    r"""Reference salinity in PSU, used to compute the thermal expansion
    coefficient $\alpha$ via TEOS-10."""

    latitude: float = 54.25
    """Latitude in degrees north, used to convert depth to pressure."""

    bottom_depth: float = 1466.0
    r"""Seafloor depth in metres. Sets the height above bottom
    $h =$ ``bottom_depth`` $- z$ used in the low-frequency cutoff."""

    U_ref: float = 0.1
    r"""Representative horizontal velocity in m/s for the $U/h$ low-frequency
    cutoff, which excludes eddies suppressed by proximity to the boundary."""

    gradient_mode: str = "full"
    r"""Which temperature-gradient magnitude enters the estimator:
    ``"full"`` uses $\sqrt{(\partial T/\partial z)^2 + (\partial T/\partial x)^2}$,
    ``"vertical"`` uses $|\partial T/\partial z|$ only."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file.

        Any fields not present in the YAML fall back to the class defaults.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        kwargs = {}
        if "spectral" in raw:
            for key in ("spectra_size", "avrg_lim", "chi_time_step"):
                if key in raw["spectral"]:
                    kwargs[key] = raw["spectral"][key]
        if "qc" in raw:
            for key in ("chi_spectra_bin_bounds",):
                if key in raw["qc"]:
                    kwargs[key] = raw["qc"][key]
        if "physics" in raw:
            for key in (
                "gamma",
                "salinity",
                "latitude",
                "bottom_depth",
                "U_ref",
                "gradient_mode",
            ):
                if key in raw["physics"]:
                    kwargs[key] = raw["physics"][key]

        return cls(**kwargs)

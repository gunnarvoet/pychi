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

    lowcut_speed: str = "reference"
    r"""Speed in the $U/h$ low-frequency cutoff. ``"reference"`` uses the
    fixed `U_ref`. ``"measured"`` uses the advection speed $U$ of each chunk,
    and `U_ref` where that speed is not finite."""

    gradient_mode: str = "full"
    r"""Which temperature-gradient magnitude enters the estimator:
    ``"full"`` uses $\sqrt{(\partial T/\partial z)^2 + (\partial T/\partial x)^2}$,
    ``"vertical"`` uses $|\partial T/\partial z|$ only."""

    gradient_temperature: str = "in_situ"
    r"""Which temperature the vertical gradient is taken from: ``"in_situ"``
    differences the calibrated temperature as given (reference behavior);
    ``"potential"`` first converts it to potential temperature referenced to
    the mean pressure of the chain, which removes the adiabatic lapse rate
    (about $10^{-4}$ °C m$^{-1}$ at abyssal pressures). Use ``"potential"``
    wherever the stratification is weak enough for the lapse rate to matter."""

    min_speed: float | None = None
    r"""Speed floor in m/s. Chunks whose advection speed $U$ is below it get
    NaN $\chi$, since $\chi \propto U^{-1}$ and Taylor's hypothesis both fail
    as $U \to 0$. ``None`` (default) applies no floor. $U$ and the gradients
    are still reported for masked chunks."""

    min_gradient: float | None = None
    r"""Gradient floor in °C m$^{-1}$. Chunks whose temperature-gradient
    magnitude (as selected by `gradient_mode`) is below it get NaN $\chi$.
    $\chi \propto |\partial T/\partial z|^{-1/2}$ diverges as the gradient
    vanishes, and the closure through $N^2$ has no meaning there. ``None``
    (default) applies no floor."""

    require_stable: bool = False
    r"""If True, chunks whose mean vertical gradient is not stable (temperature
    not decreasing with depth, ``dtdz >= 0``) get NaN $\chi$. Only meaningful
    with ``gradient_temperature="potential"``: in-situ temperature can rise
    with depth in stable water."""

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
                "lowcut_speed",
                "gradient_mode",
                "gradient_temperature",
                "min_speed",
                "min_gradient",
                "require_stable",
            ):
                if key in raw["physics"]:
                    kwargs[key] = raw["physics"][key]

        return cls(**kwargs)

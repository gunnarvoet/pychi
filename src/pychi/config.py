from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent.parent / "config" / "default.yml"
)


@dataclass
class Config:
    """Configuration for the pychi pipeline.

    All defaults match the Matlab reference implementation values.
    """

    # Spectral parameters
    spectra_size: int = 128
    avrg_lim: list[float] = field(default_factory=lambda: [0.008, 0.1])
    chi_time_step: int = 600

    # QC parameters
    chi_spectra_bin_bounds: list[int] = field(
        default_factory=lambda: [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3]
    )

    # Physics parameters
    gamma: float = 0.2
    salinity: float = 35.0
    latitude: float = 54.25
    bottom_depth: float = 1466.0
    U_ref: float = 0.1
    gradient_mode: str = "full"  # "full" = sqrt(dtdz^2 + dtdx^2), "vertical" = |dtdz|

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file.

        Any fields not specified in the YAML fall back to Matlab defaults.
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

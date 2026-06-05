r"""
# pychi

Estimate the turbulent temperature-variance dissipation rate ($\chi$, *chi*)
from oceanographic moored time series.

**pychi** estimates $\chi$ from moored fast-thermistor (T-chain) and ADCP data
using spectral analysis in the inertial subrange combined with the Osborn-Cox
scaling. It was developed for the BLT (Boundary Layer Turbulence) mooring
program.

## Background

The estimator combines classical inertial-subrange turbulence theory with
Taylor's frozen-field hypothesis. It is the approach applied to the BLT
continental-slope canyon moorings by Naveira Garabato et al. (2025); the
foundational references are listed below.

It starts from the inertial-convective model of the horizontal-wavenumber
temperature spectrum (Obukhov 1949; Corrsin 1951),

$$\phi_T(k) = C_\theta\,\chi\,\varepsilon^{-1/3}\,k^{-5/3},$$

where $C_\theta \approx 0.4$ is the Obukhov-Corrsin constant (Sreenivasan 1996),
$\chi$ the temperature-variance dissipation rate, $\varepsilon$ the
kinetic-energy dissipation rate, and $k$ the horizontal wavenumber. Frequency
spectra from a fixed mooring are mapped to wavenumber via Taylor's frozen-field
hypothesis (Taylor 1938),

$$\sigma = U k,$$

with $U$ the horizontal velocity magnitude and $\sigma$ the radian frequency.
Recalling that $\chi$ is the dissipation rate of temperature variance
(Osborn & Cox 1972), and relating $\varepsilon$ to the buoyancy flux through a
constant mixing efficiency $\Gamma$ (Osborn 1980; Gregg et al. 2018), yields the
estimator

$$\chi = \left[\left(\frac{g\alpha}{2\Gamma\,\partial T/\partial z}\right)^{1/3}
\left(\frac{2\pi}{U}\right)^{2/3}\sigma^{-5/3}\frac{\phi_T(\sigma)}{C_\theta}
\right]^{3/2}.$$

In the default ``gradient_mode="full"``, the $\partial T/\partial z$ in the
estimator is replaced by the total gradient magnitude
$\sqrt{(\partial T/\partial z)^2 + (\partial T/\partial x)^2}$, where the
horizontal gradient $\partial T/\partial x$ comes from the same frozen-field
relation ($\partial T/\partial x = U^{-1}\,\partial T/\partial t$);
``gradient_mode="vertical"`` uses $|\partial T/\partial z|$ alone.

### Symbols

| Symbol | Meaning | Value / source in code |
|--------|---------|------------------------|
| $C_\theta$ | Obukhov-Corrsin constant | `0.4` (`chi.calc_chi`) |
| $\Gamma$ | mixing efficiency | `0.2` default (`Config.gamma`) |
| $g$ | gravitational acceleration | `9.81` (`chi.calc_chi`) |
| $\alpha$ | thermal expansion coefficient | `gsw` / TEOS-10 (`chi.process_chi`) |
| $U$ | horizontal velocity magnitude | from ADCP $u, v$ (`chi.process_chi`) |
| $\partial T/\partial z$ | vertical temperature gradient | `gradients.vertical_gradient` |
| $\phi_T(\sigma)$ | inertial-subrange spectral level | median of the $f^{5/3}$-scaled spectrum |

### Spectral estimation and QC

Each 10-minute chunk's temperature power spectrum is estimated with Welch's
method (`spectra.csd_odas`): 128-sample segments, a Hanning window normalized
to unit RMS, 50% overlap, and per-segment linear detrending. The spectrum is
multiplied by $f^{5/3}$ to flatten the inertial subrange, and $\phi_T(\sigma)$
is taken as the median spectral level within the band ``avrg_lim``
(default $[0.008, 0.1]$ Hz), following the inertial-subrange fitting of moored
temperature spectra developed for χpods (Moum & Nash 2009; Zhang & Moum 2010)
and applied more generally by Bluteau et al. (2013) and Shaw et al. (2001). As a
quality check, the spectral slope is fit in log-log space; a value near the
theoretical $-5/3$ indicates a well-resolved inertial subrange.

### Key references

- Obukhov, A. M. (1949), *Structure of the temperature field in a turbulent
  flow*, Izv. Akad. Nauk SSSR, Ser. Geogr. Geofiz., 13, 58–69.
- Corrsin, S. (1951), *On the spectrum of isotropic temperature fluctuations in
  isotropic turbulence*, J. Appl. Phys., 22, 469–473.
- Taylor, G. I. (1938), *The spectrum of turbulence*, Proc. R. Soc. Lond. A,
  164, 476–490.
- Osborn, T. R., and C. S. Cox (1972), *Oceanic fine structure*, Geophys. Fluid
  Dyn., 3, 321–345.
- Osborn, T. R. (1980), *Estimates of the local rate of vertical diffusion from
  dissipation measurements*, J. Phys. Oceanogr., 10, 83–89.
- Sreenivasan, K. R. (1996), *The passive scalar spectrum and the
  Obukhov-Corrsin constant*, Phys. Fluids, 8, 189–196.
- Shaw, W. J., J. H. Trowbridge, and A. J. Williams III (2001), *Budgets of
  turbulent kinetic energy and scalar variance in the continental shelf bottom
  boundary layer*, J. Geophys. Res., 106(C5), 9551–9564.
- Moum, J. N., and J. D. Nash (2009), *Mixing measurements on an equatorial
  ocean mooring*, J. Atmos. Oceanic Technol., 26, 317–336.
- Zhang, Y., and J. N. Moum (2010), *Inertial-convective subrange estimates of
  thermal variance dissipation rate from moored temperature measurements*, J.
  Atmos. Oceanic Technol., 27, 1950–1959.
- Bluteau, C. E., N. L. Jones, and G. N. Ivey (2013), *Turbulent mixing
  efficiency at an energetic ocean site*, J. Geophys. Res. Oceans, 118,
  4662–4672.
- Gregg, M. C., E. A. D'Asaro, J. J. Riley, and E. Kunze (2018), *Mixing
  efficiency in the ocean*, Annu. Rev. Mar. Sci., 10, 443–473.
- Naveira Garabato, A. C., et al. (2025), *Connecting mixing to upwelling along
  the ocean's sloping boundary*, Geophys. Res. Lett., 52, e2025GL119186.

## Processing pipeline

pychi is built as composable layers:

1. **`spectra.csd_odas`** — auto-spectrum via Welch's method with per-segment
   polynomial detrending; `spectra.welch_spectrum` wraps `scipy.signal.welch`
   for cross-validation.
2. **`gradients.vertical_gradient` / `gradients.horizontal_gradient`** —
   temperature gradients via finite differences (vertical) and the frozen-field
   hypothesis (horizontal).
3. **`chi.calc_chi`** — single-chunk estimate: spectrum → inertial-subrange
   median → Osborn-Cox formula.
4. **`chi.process_chi`** — orchestrator over all depths and time chunks;
   handles NaN chunks and returns `xarray.Dataset` results.

## Installation

```bash
git clone git@github.com:gunnarvoet/pychi.git
cd pychi
uv sync
```

## Step-by-step example

A self-contained, runnable example on synthetic data (1 Hz, three T-chain
sensors, two 10-minute chunks):

```python
import numpy as np
import xarray as xr
from pychi import Config, process_chi

# --- Build synthetic inputs ------------------------------------------------
rng = np.random.default_rng(42)
depths = np.array([100.0, 110.0, 120.0])     # T-chain sensor depths [m]
times = np.arange(1200, dtype=float)          # 1200 s at 1 Hz

# T-chain temperatures, dims (depth, time): uncalibrated + calibrated
temp_base = np.array([15.0, 14.0, 13.0])[:, None]
temp_uncal = xr.DataArray(
    temp_base + 0.1 * rng.standard_normal((3, 1200)),
    dims=["depth", "time"], coords={"depth": depths, "time": times},
)
temp_cal = temp_uncal + 0.5

# ADCP velocities, dims (time, z)
adcp_depths = np.array([95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
u = xr.DataArray(
    0.2 + 0.02 * rng.standard_normal((1200, 6)),
    dims=["time", "z"], coords={"time": times, "z": adcp_depths},
)
v = xr.DataArray(
    0.1 + 0.02 * rng.standard_normal((1200, 6)),
    dims=["time", "z"], coords={"time": times, "z": adcp_depths},
)
w = xr.DataArray(
    0.01 * rng.standard_normal((1200, 6)),
    dims=["time", "z"], coords={"time": times, "z": adcp_depths},
)

# --- Run the pipeline ------------------------------------------------------
config = Config(chi_time_step=600, bottom_depth=200.0)
result, binned = process_chi(
    temp_uncal, temp_cal, u, v, w,
    depths=depths, adcp_depths=adcp_depths, config=config,
)

# --- Inspect results -------------------------------------------------------
print(result.chi)            # chi(depth, time)

# Keep only estimates with a well-resolved inertial subrange (slope ~ -5/3)
good = abs(result.spectral_slope + 5 / 3) < 0.5
chi_good = result.chi.where(good)

# QC: spectra averaged within log10(chi) bins
print(binned.binned_spectra)
```

For real data, load the inputs as `xarray.DataArray`s with the same dimensions
— T-chain as `(depth, time)`, ADCP velocities as `(time, z)` — ordered
shallow-to-deep. To reproduce the reference processing, upsample the ADCP data
to 5× its native resolution before calling `process_chi`; the function does not
upsample internally.

## Configuration

All tunable parameters live in `config/default.yml` (spectral settings,
physics constants, QC bin bounds). The defaults match the reference
configuration for the BLT moorings.
Construct a `Config` with `Config()` for the defaults, override individual
fields directly (`Config(chi_time_step=600, bottom_depth=200.0)`), or load a
custom YAML file with `Config.from_yaml("my_config.yml")`. The sampling
frequency is inferred automatically from the input time coordinate.

## Acknowledgments

pychi is a Python port of the original Matlab implementation, which was kindly
shared by Carl Spingys.
"""

from pychi.config import Config
from pychi.spectra import csd_odas, welch_spectrum
from pychi.gradients import vertical_gradient, horizontal_gradient
from pychi.chi import calc_chi, process_chi

# __all__ lists the submodules so pdoc documents each on its own page. The
# names above remain importable directly (e.g. ``from pychi import calc_chi``).
__all__ = [
    "config",
    "spectra",
    "gradients",
    "chi",
]

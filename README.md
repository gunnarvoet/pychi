# pychi

Estimate the turbulent temperature variance dissipation rate (χ, chi) from oceanographic moored time series.

pychi is a Python port of a Matlab reference implementation. It computes χ from moored T-chain and ADCP data using spectral analysis in the inertial subrange with the Osborn-Cox scaling.

## Installation

```bash
# Clone and install with uv
git clone git@github.com:gunnarvoet/pychi.git
cd pychi
uv sync
```

## Quick Start

```python
import xarray as xr
from pychi import Config, process_chi

# Load your data as xarray DataArrays
temp_uncal = xr.open_dataarray("tchain_uncal.nc")   # dims: (depth, time)
temp_cal = xr.open_dataarray("tchain_cal.nc")        # dims: (depth, time)
u_vel = xr.open_dataarray("adcp_u.nc")               # dims: (time, z)
v_vel = xr.open_dataarray("adcp_v.nc")               # dims: (time, z)
w_vel = xr.open_dataarray("adcp_w.nc")               # dims: (time, z)

# Use default config (Matlab parameter values)
config = Config()

# Or load from a YAML file
config = Config.from_yaml("my_config.yml")

# Run the pipeline — all depths
result, binned_spectra = process_chi(
    temp_uncal, temp_cal, u_vel, v_vel, w_vel,
    depths=temp_cal.depth.values,
    adcp_depths=u_vel.z.values,
    config=config,
)

# result is an xarray.Dataset with variables:
#   chi, U, mean_u, mean_v, mean_w, dtdz, dtdx, alpha, gamma,
#   unstab_prop, unstab_count, unstab_length, spectral_slope,
#   mean_t, mean_t_uncal, avrg_lim_actual, Pt, time_bnds

# Single sensor mode — process only depth index 5
result, binned = process_chi(
    temp_uncal, temp_cal, u_vel, v_vel, w_vel,
    depths=temp_cal.depth.values,
    adcp_depths=u_vel.z.values,
    config=config,
    sensor_indices=[5],
)
```

## Configuration

All parameters are set via a YAML config file. Default values match the Matlab reference implementation:

```yaml
spectral:
  spectra_size: 128        # FFT segment length (2^7)
  avrg_lim: [0.008, 0.1]  # Inertial subrange frequency bounds [Hz]
  chi_time_step: 600       # Chunk duration [seconds] (10 minutes)

qc:
  chi_spectra_bin_bounds: [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3]

physics:
  gamma: 0.2               # Mixing efficiency
  salinity: 35.0            # PSU
  latitude: 54.25           # Degrees N
  bottom_depth: 1466.0      # Meters
  U_ref: 0.1               # Representative velocity [m/s]
```

The sampling frequency is inferred automatically from the time coordinate of the input data.

## Filtering by Spectral Slope

Each chi estimate includes a spectral slope diagnostic — the slope of a linear fit to log₁₀(f) vs log₁₀(Pt) in the inertial subrange. A slope near −5/3 ≈ −1.667 indicates a well-resolved inertial subrange:

```python
# Filter chi estimates by spectral quality
good = abs(result.spectral_slope + 5/3) < 0.5
chi_filtered = result.chi.where(good)
```

## Architecture

pychi is built as three composable layers:

1. **`spectra.csd_odas()`** — Faithful port of the Matlab csd_odas function (auto-spectrum via Welch's method with per-segment polynomial detrending). Also provides `welch_spectrum()` as a scipy.signal.welch wrapper for cross-validation.

2. **`gradients.vertical_gradient()` / `horizontal_gradient()`** — Temperature gradient computation using finite differences (vertical) and the frozen-field hypothesis (horizontal).

3. **`chi.calc_chi()`** — Single-chunk chi computation: spectrum → inertial subrange median → Osborn-Cox formula.

4. **`chi.process_chi()`** — Orchestrator that loops over depths and time chunks, calls the above functions, handles NaN chunks, and returns xarray Datasets.

## Porting Notes: Matlab `hanning` vs scipy `hann`

Matlab's `hanning(N)` and scipy's `hann(N, sym=True)` produce different windows despite both being called "Hanning":

| | Matlab `hanning(N)` | scipy `hann(N, sym=True)` |
|---|---|---|
| Formula | `0.5*(1 - cos(2*pi*n/(N+1)))`, n=1..N | `0.5*(1 - cos(2*pi*n/(N-1)))`, n=0..N-1 |
| Endpoints | **Excludes** zeros (first/last values > 0) | **Includes** zeros (first and last values = 0) |

For N=128, the maximum pointwise difference between the two normalized windows is ~0.02. In our case this caused a ~5% difference in spectral levels, which propagated through the chi formula (with its 3/2 exponent) to a ~0.7% error in chi — well beyond the 1e-6 tolerance required for Matlab-validated tests.

The fix in `calc_chi` (`src/pychi/chi.py`) uses the Matlab formula directly instead of scipy:

```python
win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(1, n_fft + 1) / (n_fft + 1)))
win = win / np.sqrt(np.mean(win**2))
```

The standalone `csd_odas` function retains its own periodic Hanning window (matching `csd_odas.m`'s internal default), which is correct when called without an explicit window argument. The discrepancy only arose because `Calc_Chi_TChain_2.m` overrides that default by passing Matlab's `hanning(N)`.

## Input Data Requirements

- **Temperature data:** Two xarray DataArrays with dims `(depth, time)` — one uncalibrated (for spectral analysis) and one calibrated (for gradients and physical parameters).
- **Velocity data:** Three xarray DataArrays with dims `(time, z)` — u, v, w from an ADCP or similar instrument.
- **ADCP preprocessing:** If you want to match the Matlab pipeline exactly, upsample ADCP data to 5× native resolution before calling `process_chi()`. The function does not perform internal upsampling.
- **Depths ordered shallow-to-deep** (increasing depth values).

## Testing

Tests validate Python output against Matlab reference data:

```bash
uv run pytest -v
```

To generate Matlab test fixtures, run `scripts/export_matlab_fixtures.m` in Matlab from the `matlab_version/` directory. This saves intermediate results (spectra, gradients, chi values) to `tests/fixtures/`.

## Dependencies

- numpy, scipy, xarray, gsw, pyyaml
- Dev: pytest, ruff

## Acknowledgments

pychi is a Python port of the original Matlab implementation, which was kindly shared by Carl Spingys.

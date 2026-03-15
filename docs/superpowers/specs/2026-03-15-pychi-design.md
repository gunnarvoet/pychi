# pychi Design Spec

**Date:** 2026-03-15
**Status:** Draft

## Overview

pychi is a Python package that estimates the turbulent temperature variance dissipation rate (chi, χ) from oceanographic moored time series. It is a port of the Matlab reference implementation in `Chi_Calc_For_Gunnar/`.

## Design Decisions

- **Input:** xarray DataArrays/Datasets (user handles I/O)
- **Output:** xarray Dataset with chi and diagnostics as variables, depth/time as coordinates
- **Spectral estimation:** Faithful port of `csd_odas` as primary method; `scipy.signal.welch` wrapper as alternative for cross-validation. The `csd_odas` port will be removed once we confirm `scipy.signal.welch` produces equivalent results.
- **Configuration:** YAML config file with all tunable parameters; Matlab values as defaults; `sample_freq` inferred from data
- **Architecture:** Layered composable functions (not monolithic or class-based)
- **Single-sensor mode:** Orchestrator supports processing one depth level, provided neighboring sensors' calibrated temperatures are supplied for vertical gradient computation

## Package Structure

```
pychi/
├── pyproject.toml
├── README.md                 # Detailed README with usage examples
├── CLAUDE.md
├── config/
│   └── default.yml           # Default parameters (Matlab values)
├── src/pychi/
│   ├── __init__.py
│   ├── config.py             # YAML config loading, Config dataclass
│   ├── spectra.py            # csd_odas port + scipy.signal.welch wrapper
│   ├── chi.py                # calc_chi (single chunk) + process_chi (orchestrator)
│   └── gradients.py          # dT/dz, dT/dx computation
├── tests/
│   ├── conftest.py           # Fixtures loading Matlab reference data
│   ├── fixtures/             # .mat/.npz files from Matlab runs
│   ├── test_spectra.py       # csd_odas against Matlab spectra
│   ├── test_chi.py           # calc_chi against Matlab chi values
│   ├── test_gradients.py     # gradient computation against Matlab
│   └── test_process.py       # End-to-end pipeline test
└── scripts/
    └── export_matlab_fixtures.m  # Matlab script to generate test fixtures
```

## Dependencies

**Runtime:** `numpy`, `scipy`, `xarray`, `gsw`, `pyyaml`
**Dev:** `pytest`, `ruff`

## Configuration

YAML config with Matlab defaults:

```yaml
spectral:
  spectra_size: 128        # FFT segment length (2^7)
  avrg_lim: [0.008, 0.1]  # Inertial subrange frequency bounds [Hz]
  chi_time_step: 600       # Chunk duration [seconds] (10 minutes)

qc:
  chi_spectra_bin_bounds: [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3]  # log10(chi) bin edges for spectral averaging

physics:
  gamma: 0.2               # Mixing efficiency
  salinity: 35.0            # PSU (for thermal expansion coefficient)
  latitude: 54.25           # Degrees N (for pressure calculation)
  bottom_depth: 1466.0      # Meters (for height-above-bottom)
  U_ref: 0.1               # Representative velocity [m/s] (for u/h low-freq cutoff)
```

A `Config` dataclass in `config.py`:
- Loads from YAML via `Config.from_yaml("path.yml")`
- Falls back to Matlab defaults for any omitted fields via `Config()` with no arguments
- `sample_freq` is not in the config — it is inferred from the time coordinate spacing of the input data by the orchestrator and passed explicitly to lower-level functions

## Module Specifications

### `spectra.py`

#### `csd_odas(x, n_fft, rate, window=None, overlap=None, detrend="none")`

Faithful port of the Matlab `csd_odas.m` function. Auto-spectrum mode only — the `y` parameter is omitted as a deliberate simplification since the chi pipeline always computes auto-spectra (`x == y` in every Matlab call).

- **Parameters:**
  - `x`: 1D array of temperature values (must be NaN-free; see NaN handling note below)
  - `n_fft`: FFT segment length (default 128)
  - `rate`: sampling frequency in Hz
  - `window`: 1D array of length `n_fft`, or `None`. If `None`, a Hanning window normalized to unit RMS (`window / sqrt(mean(window**2))`) is generated internally, matching the Matlab `csd_odas.m` default behavior (lines 131–134). Note: the Matlab code calls this a "cosine window" and generates it as `1 + cos(pi*(-1 + 2*k/N))` — this is mathematically identical to the Hann (Hanning) window. In Python, `numpy.hanning(n)` or `scipy.signal.windows.hann(n)` produce the same window. The implementation should include a comment documenting this equivalence. If provided, the window is used as-is — the caller is responsible for normalization.
  - `overlap`: number of overlapping points between segments (default `n_fft // 2`)
  - `detrend`: one of `"none"`, `"constant"`, `"linear"`, `"parabolic"`, `"cubic"` (default `"none"`, matching the Matlab function's default when the 7th argument is omitted). Note: the chi pipeline caller passes `"linear"` explicitly, matching Matlab `Calc_Chi_TChain_2.m` line 10.
- **Returns:** `(Pxx, f)` — power spectrum and frequency vector
- **Normalization:** integral from 0 to Nyquist equals signal variance (matching Matlab)
- **Detrending:** polynomial detrending per segment before FFT, matching the Matlab implementation exactly

#### `welch_spectrum(x, n_fft, rate)`

Thin wrapper around `scipy.signal.welch` configured to match `csd_odas` as closely as possible:
- Hanning window, 50% overlap, linear detrend
- Returns `(Pxx, f)` in the same format

### `gradients.py`

#### `vertical_gradient(temp_cal, depths, sensor_index)`

Computes dT/dz for a given sensor using finite differences.

- **Parameters:**
  - `temp_cal`: 2D array or DataArray `(n_depths, n_times)` — calibrated temperature
  - `depths`: 1D array of depth values (ordered shallow-to-deep, i.e., increasing depth values)
  - `sensor_index`: index of the sensor being processed
- **Returns:** `(dtdz_mean, dtdz_timeseries)` — chunk-mean gradient and per-timestep gradient
- **Boundary handling** (matches Matlab lines 137–152):
  - Top sensor (`sensor_index == 0`): uses sensors 0 and 1: `(T[0] - T[1]) / (depth[0] - depth[1])`
  - Bottom sensor (`sensor_index == N-1`): uses sensors N-2 and N-1: `(T[N-2] - T[N-1]) / (depth[N-2] - depth[N-1])`
  - Interior sensors: centered difference using sensors i-1 and i+1: `(T[i-1] - T[i+1]) / (depth[i-1] - depth[i+1])`

#### `horizontal_gradient(temp_cal_chunk, chunk_duration_s, U)`

Frozen-field dT/dx estimate.

- **Parameters:**
  - `temp_cal_chunk`: 1D array of calibrated temperatures for the time chunk
  - `chunk_duration_s`: chunk duration in seconds (equals `config.chi_time_step`)
  - `U`: mean horizontal velocity magnitude
- **Returns:** `dtdx` scalar
- **Formula:** `(T_end - T_start) / (chunk_duration_s * U)` — matches Matlab line 154

### `chi.py`

#### `calc_chi(temperature, U, gamma, alpha, grad_T_mag, sample_freq, config)`

Port of `Calc_Chi_TChain_2.m`. Computes chi for a single time chunk at a single depth.

- **Parameters:**
  - `temperature`: 1D array (NaN-free) of uncalibrated temperature for the chunk
  - `U`: horizontal velocity magnitude [m/s]
  - `gamma`: mixing efficiency (typically 0.2)
  - `alpha`: thermal expansion coefficient [1/°C]
  - `grad_T_mag`: total temperature gradient magnitude `sqrt(dtdz² + dtdx²)` — note this is the combined gradient, not just the vertical component. The Matlab code passes `sqrt(dtdz.^2+dtdx.^2)` as the `dtdz` argument (line 180) and takes `abs()` inside the function (line 49). We use the clearer name `grad_T_mag` to avoid confusion with the vertical gradient `dtdz`.
  - `sample_freq`: sampling frequency in Hz (inferred from data by the orchestrator)
  - `config`: `Config` object (provides `spectra_size`, `avrg_lim`)
- **Returns:** `(chi_value, diagnostics)` where diagnostics is a dict with keys `Pt`, `f`, `U`, `mean_t`, `spectral_slope`
  - `spectral_slope`: slope of linear fit to `log10(f)` vs `log10(Pt)` in the inertial subrange (frequencies within `avrg_lim`, extended to `avrg_lim[0] / 1.5` on the low end). A value near −5/3 indicates a good inertial subrange fit. Set to NaN if all spectral values in the band are zero. Matches Matlab lines 202–228.
- **Window:** Hanning window normalized to unit RMS, passed to `csd_odas` with `detrend="linear"` (matching Matlab `Calc_Chi_TChain_2.m` line 10)
- **Core formula** (Matlab line 51):
  ```
  scale = f^(5/3)
  phi = median(Pt[in_band] * scale[in_band])
  chi = (phi * (2*pi/U)^(2/3) / 0.4 * (g*alpha / (2*|grad_T_mag|*gamma))^(1/3)) ^ (3/2)
  ```
  Operator precedence note: the expression evaluates left-to-right for `*` and `/`:
  ```
  chi = ((phi * (2*pi/U)**(2/3)) / 0.4 * (g*alpha / (2*abs(grad_T_mag)*gamma))**(1/3)) ** (3/2)
  ```
  where `in_band` selects frequencies within `avrg_lim`, `g = 9.81`

#### `process_chi(temp_uncal, temp_cal, u_velocity, v_velocity, w_velocity, depths, adcp_depths, config, sensor_indices=None)`

Orchestrator that loops over depths and time chunks.

- **Parameters:**
  - `temp_uncal`: DataArray `(depth, time)` — uncalibrated T-chain temperatures
  - `temp_cal`: DataArray `(depth, time)` — calibrated T-chain temperatures
  - `u_velocity`, `v_velocity`: DataArrays `(time, z)` — ADCP velocity components. Expected at whatever time resolution the user provides (no internal upsampling). The Matlab code upsamples ADCP to 5x native resolution (lines 20–31); the user should perform any desired interpolation before calling this function.
  - `depths`: 1D array — T-chain depth levels (ordered shallow-to-deep)
  - `adcp_depths`: 1D array — ADCP depth bins
  - `config`: `Config` object
  - `sensor_indices`: optional list of depth indices to process (default: all). For single-sensor mode, pass e.g. `[5]` — the function still uses neighboring sensors from `temp_cal` for gradient computation.
- **Returns:** `xarray.Dataset` with:
  - **Coordinates:** `depth`, `time` (chunk center times)
  - **Variables:** `chi`, `U`, `mean_u`, `mean_v`, `mean_w`, `dtdz`, `dtdx`, `alpha`, `gamma`, `unstab_prop`, `unstab_count`, `unstab_length`, `spectral_slope`, `mean_t`, `mean_t_uncal`, `avrg_lim_actual` (the possibly-adjusted low-frequency limit used for each chunk)
  - **Coordinate:** `time_bnds` — chunk boundary times stored as a coordinate variable with dims `(time, 2)` for provenance
  - **Per-chunk spectra:** `Pt` and `f` arrays from `calc_chi` are stored in the Dataset as variables with dims `(depth, time, frequency)`. This enables downstream spectral QC (binned averaging, slope analysis) without re-running the computation.
- **Processing per chunk:**
  1. Find time indices for the chunk window
  2. Extract uncalibrated and calibrated temperature for the chunk
  3. Find nearest ADCP depth bin, compute `U = sqrt(mean(u)² + mean(v)²)`
  4. Compute `mean_u`, `mean_v`, `mean_w` by interpolating time-averaged ADCP velocity profiles to the T-chain sensor depth (using `np.interp` or `xr.interp`), matching Matlab lines 238–242. Note: this differs from `U` which uses nearest-bin.
  5. Compute vertical gradient via `vertical_gradient()` using calibrated temperature
  6. Compute horizontal gradient via `horizontal_gradient()` using calibrated temperature
  7. Compute instability proportion (fraction of timesteps where dT/dz > 0)
  8. Compute thermal expansion coefficient via `gsw.alpha()`
  9. Adjust low-frequency limit: `avrg_lim[0] = max(avrg_lim[0], U_ref / hab)` where `hab = bottom_depth - depth`
  10. **NaN handling:** Skip chunk if the uncalibrated temperature contains any NaN values; set chi and gamma to NaN. Diagnostic variables (`mean_u`, `mean_v`, `mean_w`, `dtdz`, `dtdx`, `alpha`, `unstab_prop`, `mean_t`) are still computed and saved. For NaN-skipped chunks, `mean_u/v/w` use nearest ADCP depth bin (not interpolation) — this matches the Matlab behavior (lines 262–293) and is preserved intentionally so Python results can be compared directly against Matlab output. This matches Matlab line 166 which checks `length(temp_in) == length(temp_in(~isnan(temp_in)))` — i.e., the entire chunk is rejected if any sample is NaN, rather than removing NaN values and computing spectra on a non-contiguous series.
  11. Infer `sample_freq` from time coordinate spacing (computed once, not per chunk)
  12. Call `calc_chi()` for valid chunks, passing `grad_T_mag = sqrt(dtdz² + dtdx²)`. Store returned `Pt`, `f`, and `spectral_slope`.
  13. After all chunks are processed, compute binned average spectra by grouping per-chunk spectra into `log10(chi)` bins (bin edges `−12:1:−3`) and averaging per depth. Matches Matlab lines 301–307.
  14. Assemble results into output Dataset (primary) and binned spectra Dataset (secondary)

**Spectral QC — binned average spectra:**
The Matlab code bins spectra by `log10(chi)` (bin edges `−12:1:−3`) and averages them per depth for QC plotting. The orchestrator computes this after the main loop: for each depth, each valid-chi chunk's spectrum `Pt` is assigned to a `log10(chi)` bin, and the mean spectrum per bin is stored. Output as a separate `xarray.Dataset` (or as a secondary return value) with:
- `binned_spectra`: mean spectrum per bin, dims `(depth, chi_bin, frequency)`
- `binned_counts`: number of spectra per bin, dims `(depth, chi_bin)`
- `chi_bin_edges`: the log10(chi) bin boundaries
- `frequency`: the frequency vector

## Testing Strategy

### Test Fixtures

A Matlab script (`scripts/export_matlab_fixtures.m`) generates reference data:

1. **Spectral fixtures** — For 3–5 representative chunks: raw temperature input, `csd_odas` output `(Pt, f)`, the Hanning window used, and all scalar inputs to `Calc_Chi_TChain_2`
2. **Gradient fixtures** — For several depth/time combinations: calibrated temperatures at sensor and neighbors, depth values, computed `dtdz` and `dtdx`
3. **Chi fixtures** — For the same chunks: final chi values, all intermediate scalars (alpha, U, gamma, grad_T_mag, dtdz, dtdx, unstab_prop)
4. **End-to-end fixtures** — A small subset (e.g. 3 depths × 10 time chunks): full input arrays and the complete output chi matrix

Fixtures saved as `.mat` files, loaded in Python via `scipy.io.loadmat`.

### Test Modules

- **`test_spectra.py`** — `csd_odas` Python vs Matlab spectra (relative tolerance ~1e-10). Fixtures include the window array to verify normalization. Also compares `csd_odas` vs `scipy.signal.welch` to quantify differences.
- **`test_gradients.py`** — `vertical_gradient` and `horizontal_gradient` vs Matlab values (exact match expected)
- **`test_chi.py`** — `calc_chi` output vs Matlab chi values (relative tolerance ~1e-6, accounting for floating-point path differences). Must include at least one fixture where `U_ref / hab > avrg_lim[0]` to exercise the low-frequency cutoff adjustment.
- **`test_process.py`** — Full pipeline: same inputs as Matlab, compare output Dataset against Matlab results. Includes a test with NaN-containing chunks to verify they produce NaN output.

### Tolerance Rationale

- Spectral estimation: should be near machine precision (~1e-10) since the algorithm is identical
- Chi values: the formula involves medians and power operations, so ~1e-6 relative tolerance is reasonable
- Gradients: simple arithmetic, exact match expected

## Matlab Fixture Export Script

The script `scripts/export_matlab_fixtures.m` will:
1. Run the existing pipeline on a small data subset
2. At each stage, save intermediate arrays to `.mat` files:
   - `fixtures/spectra_chunk_N.mat`: temperature input, Pt, f, window, spectra_size, sample_freq, avrg_lim
   - `fixtures/gradient_chunk_N.mat`: temp_cal (3 sensors), depths, dtdz, dtdx, U
   - `fixtures/chi_chunk_N.mat`: chi, alpha, gamma, U, grad_T_mag, dtdz, dtdx, unstab_prop, mean_t
   - `fixtures/pipeline_subset.mat`: full input/output arrays for end-to-end test
3. Select chunks that cover edge cases: top sensor, bottom sensor, middle sensor; high/low chi; chunks with adjusted avrg_lim (where `U_ref/hab > avrg_lim[0]`); chunks with NaN values
4. Note on data variable names: L1 (uncalibrated) files use variable `t`, L2 (calibrated) files use `__xarray_dataarray_variable__`

## Open Questions

None — all design decisions have been resolved through discussion.

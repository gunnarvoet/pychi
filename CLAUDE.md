# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pychi** — a Python package to estimate the turbulent temperature variance dissipation rate (chi, χ) from oceanographic moored time series. The reference Matlab implementation is under `matlab_version/`.

## Commands

```bash
uv sync                  # Install dependencies (including dev group)
uv run pytest            # Run all tests
uv run pytest -x         # Stop on first failure
uv run ruff check .      # Lint
uv run ruff format .     # Format
```

## Running the Matlab Code

The Matlab code requires paths to BLT mooring data (NetCDF files). The entry point is `T_Chain_Chi_Horiz_Grads_hu_Spectra_Test.m`, which calls `Calc_Chi_TChain_2.m` for each depth/time chunk.

Required data (not in repo):
- ADCP data: `MAVS2_24606.nc` (velocities u, v, w on a vertical grid)
- T-chain uncalibrated: `mavs2_l1/mavs2_YYYYMMDD.nc` (variables: time, depth, `__xarray_dataarray_variable__`)
- T-chain calibrated: `mavs2_l2/mavs2_YYYYMMDD.nc` (variables: time, depth, `__xarray_dataarray_variable__`)

Run in Matlab from the `matlab_version/` directory after pointing paths to data.

## Matlab Code Architecture

### Processing Pipeline

1. **Data loading** — ADCP velocities from NetCDF, interpolated to 5× the native time resolution. T-chain temperatures loaded from 2-day NetCDF files (both uncalibrated L1 and calibrated L2).

2. **Chunking** — Time is divided into 10-minute windows (`chi_time_step=10/60/24` days). Processing loops over all depths × all time windows.

3. **Per-chunk computation** (`Calc_Chi_TChain_2.m`):
   - Computes temperature power spectrum via `csd_odas` (Welch's method with Hanning window, 50% overlap, segment length 2⁷=128 samples)
   - Multiplies spectrum by f^(5/3) to flatten the inertial subrange
   - Takes median spectral level in a frequency band (`avrg_lim`) and applies the Osborn-Cox scaling to estimate χ
   - The χ formula uses: horizontal velocity magnitude U, thermal expansion coefficient α (from `sw_alpha`), vertical temperature gradient dT/dz, horizontal gradient dT/dx (frozen-field hypothesis), mixing efficiency γ=0.2, and gravity g=9.81

4. **Diagnostics** — Spectral slopes are fitted in the inertial subrange and compared to the theoretical −5/3. Spectra are binned by log₁₀(χ) and averaged for QC plots.

5. **Output** — Saved as `.mat` files containing χ, mean velocities, temperature gradients, stability metrics, and averaged spectra.

### Key Physical Parameters
- `spectra_size`: 2⁷ = 128 (FFT segment length at 1 Hz sampling)
- `avrg_lim`: [0.8e-2, 1e-1] Hz (inertial subrange frequency bounds)
- Low-frequency limit adjusted by U/h (velocity/height-above-bottom) to exclude boundary-suppressed eddies
- `gamma`: 0.2 (assumed mixing efficiency)
- Vertical gradient uses centered differences (forward/backward at boundaries)

### `csd_odas.m`
A standalone spectral estimation function (replaces deprecated Matlab `csd`). Supports detrending modes: none, constant, linear, parabolic, cubic. Handles real and complex inputs. Normalizes so that integral from 0 to Nyquist equals signal variance.

## Python Package Development

### Module Structure (`src/pychi/`)
- `config.py` — loads `config/default.yml`; exposes `Config` dataclass with spectral, physics, and QC parameters
- `spectra.py` — `csd_odas()` (Matlab-compatible spectral estimation) and `welch_spectrum()`
- `gradients.py` — `vertical_gradient()` (centered differences) and `horizontal_gradient()` (frozen-field via U)
- `chi.py` — `calc_chi()` (single-chunk χ estimate) and `process_chi()` (full pipeline over depths × time windows)

### Key Libraries (Matlab → Python)
- `csd_odas.m` → `scipy.signal` (via `spectra.py`)
- `sw_alpha`, `sw_pres` → `gsw` (TEOS-10)
- NetCDF I/O → `xarray`

### Configuration
Tunable parameters live in `config/default.yml` (spectral settings, physics constants, QC bin bounds). Override by passing a custom YAML path to `Config`.

## Code Style

- Linter/formatter: `ruff` (line-length 88)
- No type annotations convention enforced yet

## Testing Strategy

The test suite validates Python output against Matlab-processed reference data. Matlab results (intermediate and final) are saved as test fixtures, and Python tests compare against these to ensure numerical equivalence of the port.

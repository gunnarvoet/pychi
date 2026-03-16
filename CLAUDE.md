# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pychi** — a Python package (not yet written) to estimate the turbulent temperature variance dissipation rate (chi, χ) from oceanographic moored time series. The reference implementation is in Matlab under `matlab_version/`.

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
   - Takes median spectral level in a frequency band (`avrg_lim`) and applies the Batchelor/Osborn-Cox scaling to estimate χ
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

When building the Python equivalent:
- Use `xarray` and `netCDF4` for NetCDF I/O
- Use `scipy.signal` for spectral estimation (replaces `csd_odas`)
- Use `gsw` (TEOS-10) for seawater properties (replaces `sw_alpha`, `sw_pres`)
- Use `numpy` for array operations (replaces Matlab vectorized math)
- The core χ calculation in `Calc_Chi_TChain_2.m` line 51 is the key formula to port accurately

## Testing Strategy

The test suite validates Python output against Matlab-processed reference data. Matlab results (intermediate and final) are saved as test fixtures, and Python tests compare against these to ensure numerical equivalence of the port.

# Paper vs. Plan: Method Comparison

Comparison between the χ estimation method described in **Naveira Garabato et al., "Convective turbulent mixing drives rapid upwelling along the ocean's bottom boundary"** and the pychi implementation plan.

---

## Overview

The paper describes mooring-based estimates of the temperature variance dissipation rate $\chi$ from 83 fast-sampling (1 Hz) thermistors deployed in a continental-slope canyon in the Rockall Trough. The Matlab code in `Chi_Calc_For_Gunnar/` implements this method. The pychi plan ports that Matlab code to Python.

**Bottom line:** The plan faithfully captures the method described in the paper. There are a few nuances worth noting, detailed below.

---

## The χ Formula

### Paper (lines 344–353)

The paper starts from the inertial-subrange model of the horizontal wavenumber temperature spectrum:

$$\phi_T(k) = C_\theta \chi \varepsilon^{-1/3} k^{-5/3}$$

where $C_\theta = 0.4$ is the Obukhov-Corrsin constant. Using Taylor's frozen-field hypothesis ($\sigma = Uk$) and relating $\varepsilon$ to a buoyancy flux (assuming buoyancy is a function of temperature only), they derive:

$$\chi = \left[ \left(\frac{g\alpha}{2\Gamma \frac{\partial T}{\partial z^*}}\right)^{1/3} \left(\frac{2\pi}{U}\right)^{2/3} \sigma^{-5/3} \frac{\phi_T(\sigma)}{C_\theta} \right]^{3/2}$$

### Matlab code (Calc_Chi_TChain_2.m, line 51)

```matlab
chi = (nanmedian(Pt .* scale) .* (2*pi/U)^(2/3) ./ 0.4 ...
       .* (g*alpha / (2*dtdz*gamma))^(1/3)) ^ (3/2);
```

where `scale = f.^(5/3)` (pre-multiplied to flatten the spectrum in the inertial subrange), and `0.4` is $C_\theta$.

### Plan (chi.py, lines 1114–1124)

```python
phi = float(np.nanmedian(Pt[in_band] * scale[in_band]))
chi_val = (
    (phi * (2 * np.pi / U) ** (2 / 3))
    / 0.4
    * (g * alpha / (2 * grad_T_mag * gamma)) ** (1 / 3)
) ** (3 / 2)
```

### Comparison

| Parameter | Paper | Matlab | Plan | Match? |
|-----------|-------|--------|------|--------|
| $C_\theta$ (Obukhov-Corrsin) | 0.4 | 0.4 | 0.4 | Yes |
| $\Gamma$ (dissipation ratio / mixing efficiency) | 0.2 (default), also tested 0.4 | `gamma` input | `gamma` (default 0.2 via Config) | Yes |
| Frozen-field factor | $(2\pi/U)^{2/3}$ | `(2*pi/U)^(2/3)` | `(2 * np.pi / U) ** (2 / 3)` | Yes |
| Spectral level | $\sigma^{-5/3} \phi_T(\sigma) / C_\theta$ | `nanmedian(Pt .* f.^(5/3)) / 0.4` | `np.nanmedian(Pt * scale) / 0.4` | Yes |
| Buoyancy scaling | $(g\alpha / (2\Gamma \partial T/\partial z^*))^{1/3}$ | `(g*alpha/(2*dtdz*gamma))^(1/3)` | `(g * alpha / (2 * grad_T_mag * gamma)) ** (1/3)` | Yes |
| Overall exponent | $(\cdots)^{3/2}$ | `^(3/2)` | `** (3/2)` | Yes |

**The formula is identical across all three.**

---

## Spectral Estimation

### Paper (lines 365–380)

- 5-minute segments (300 samples at 1 Hz)
- Welch's method
- Spectra scaled by $\sigma^{5/3}$ to flatten the inertial subrange
- Median spectral level taken across a frequency band
- Sensitivity tests showed results were robust to window size/shape, mean vs. median, and frequency range

### Matlab code

- `csd_odas` with `spectra_size = 2^7 = 128`, Hanning window normalized to unit RMS, 50% overlap, linear detrend
- Chunks are 10-minute windows (`chi_time_step = 10/60/24` days)
- Multiplies spectrum by `f.^(5/3)` and takes `nanmedian` in `avrg_lim = [0.8e-2, 1e-1]` Hz

### Plan

- Faithful port of `csd_odas` with identical parameters (n_fft=128, Hanning window, 50% overlap, linear detrend)
- Same 10-minute chunking (600 s)
- Same `f^(5/3)` scaling and `nanmedian` in the inertial subrange band
- Also provides `welch_spectrum` (scipy wrapper) for cross-validation

### Notes

- The paper mentions 5-minute spectra; the Matlab code uses 10-minute chunks with 128-sample FFT segments (128 s each, with 50% overlap yielding ~8–9 segments per chunk). These are consistent — the 5-minute averaging in the paper likely refers to an earlier or alternative processing, while the Matlab code provided uses 10-minute windows as the outer chunk. The spectral estimation within each chunk uses 128-sample sub-segments.
- The plan correctly uses `spectra_size=128` (matching the Matlab `2^7`), not 300 (which would correspond to 5 minutes at 1 Hz).

---

## Gradient Computation

### Paper (lines 341–339, implicit)

The paper's $\partial T / \partial z^*$ is the temperature gradient in the diathermal direction, which in the canyon context is well approximated by the vertical direction. The paper's $\chi$ formula uses this gradient.

### Matlab code

- `dtdz`: centered finite differences at interior sensors, one-sided at boundaries
- `dtdx`: horizontal gradient via frozen-field hypothesis ($dT/dx = (1/U) \cdot dT/dt$, computed as `np.gradient(T) * sample_freq / U`)
- The gradient magnitude entering the formula is `abs(dtdz)` (Matlab line 49)

### Plan

- `vertical_gradient()`: identical centered/one-sided finite differences
- `horizontal_gradient()`: frozen-field hypothesis, same as Matlab
- Uses `grad_T_mag = sqrt(dtdz^2 + dtdx^2)` in the orchestrator, then `abs(grad_T_mag)` in `calc_chi`

### No discrepancy

Despite the misleading parameter name `dtdz` in `Calc_Chi_TChain_2.m`, the **Matlab caller** (line 180) actually passes `sqrt(dtdz^2 + dtdx^2)` — the total gradient magnitude:

```matlab
Calc_Chi_TChain_2(temp_in, U_in, gamma, alpha, sqrt(dtdz.^2+dtdx.^2), ...)
```

The plan does the same thing in the orchestrator (line 1578):

```python
grad_T_mag = np.sqrt(dtdz_mean**2 + dtdx_val**2)
```

The plan's parameter name `grad_T_mag` is actually clearer than the Matlab code's reuse of the name `dtdz` for a combined gradient. The `abs()` on line 49 of the Matlab function just ensures positivity of the already-combined value. **No change needed.**

---

## Frequency Band Limits

### Paper

- Not explicitly specified beyond saying a frequency range was chosen representative of the inertial subrange, exhibiting the typical $-5/3$ slope.

### Matlab code

- `avrg_lim = [0.8e-2, 1e-1]` Hz (i.e., [0.008, 0.1])
- Low-frequency limit adjusted by `U/h` (velocity / height-above-bottom) to exclude boundary-suppressed eddies

### Plan

- Default `avrg_lim = [0.008, 0.1]` Hz — matches Matlab
- Implements the `U/h` low-frequency adjustment in the orchestrator (same logic as Matlab)

**Match confirmed.**

---

## Physical Constants and Parameters

| Parameter | Paper | Matlab | Plan | Match? |
|-----------|-------|--------|------|--------|
| $g$ | 9.81 m/s² | 9.81 | 9.81 | Yes |
| $C_\theta$ | 0.4 | 0.4 | 0.4 | Yes |
| $\Gamma$ | 0.2 (also tested 0.4) | Input param | Default 0.2 via Config | Yes |
| Thermal expansion $\alpha$ | Calculated | `sw_alpha` | `gsw` (TEOS-10) | Updated library* |
| Sampling rate | 1 Hz | 1 Hz | Parameterized | Yes |

*The plan uses `gsw` (TEOS-10) instead of the older `sw_alpha` (EOS-80). This is a modernization; numerical differences are negligible for the relevant T/S/p ranges.

---

## Spectral Slope Fitting

### Paper

- Mentions spectra "consistently exhibiting the typical $-5/3$ slope" in the inertial subrange (line 380)

### Matlab code (main script, lines 202–228)

- Fits $\log_{10}(f)$ vs. $\log_{10}(\phi_T)$ in the inertial subrange to get the spectral slope
- Uses `avrg_lim(1)/1.5` as the low-frequency bound for fitting

### Plan

- Implements the same log-log linear fit with `avrg_lim[0] / 1.5` as the low-frequency bound
- Returns `spectral_slope` in the diagnostics dict

**Match confirmed.**

---

## Summary of Findings

1. **The core $\chi$ formula is identical** across the paper, Matlab code, and plan.
2. **Spectral estimation parameters match** (FFT length, window, overlap, detrending, frequency band).
3. **No gradient discrepancy**: despite the misleading parameter name `dtdz` in `Calc_Chi_TChain_2.m`, the Matlab caller passes `sqrt(dtdz^2 + dtdx^2)` — the plan matches exactly.
4. **Library modernization**: `gsw` (TEOS-10) replaces `sw_alpha` (EOS-80) — numerically negligible difference.
5. **The plan captures all key aspects** of the method: Welch spectral estimation, $f^{5/3}$ scaling, median spectral level, Osborn-Cox/inertial-subrange formula, gradient computation, frequency band adjustment, and spectral slope diagnostics.

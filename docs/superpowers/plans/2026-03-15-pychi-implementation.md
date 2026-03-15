# pychi Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Matlab chi (χ) calculation pipeline to a Python package (`pychi`) with layered composable functions, YAML configuration, xarray I/O, and a Matlab-validated test suite.

**Architecture:** Three computation layers — `spectra.py` (spectral estimation), `gradients.py` (temperature gradients), `chi.py` (chi formula + orchestrator) — with a `config.py` module for YAML-based configuration. Each layer is independently testable against Matlab reference data. The orchestrator (`process_chi`) composes the layers and returns xarray Datasets.

**Tech Stack:** Python 3.10+, numpy, scipy, xarray, gsw, pyyaml, pytest, ruff, uv

**Spec:** `docs/superpowers/specs/2026-03-15-pychi-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, build config |
| `README.md` | Detailed usage documentation with examples |
| `config/default.yml` | Default parameters (Matlab values) |
| `src/pychi/__init__.py` | Public API exports |
| `src/pychi/config.py` | `Config` dataclass, YAML loading, defaults |
| `src/pychi/spectra.py` | `csd_odas()` port, `welch_spectrum()` wrapper |
| `src/pychi/gradients.py` | `vertical_gradient()`, `horizontal_gradient()` |
| `src/pychi/chi.py` | `calc_chi()` single-chunk, `process_chi()` orchestrator |
| `tests/conftest.py` | Shared fixtures for Matlab reference data |
| `tests/test_config.py` | Config loading and defaults |
| `tests/test_spectra.py` | csd_odas and welch against known signals |
| `tests/test_gradients.py` | Gradient computation |
| `tests/test_chi.py` | calc_chi against known values |
| `tests/test_process.py` | End-to-end orchestrator |
| `scripts/export_matlab_fixtures.m` | Matlab script to generate test fixtures |

---

## Chunk 1: Project Scaffolding and Configuration

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config/default.yml`
- Create: `src/pychi/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "pychi"
version = "0.1.0"
description = "Estimate turbulent temperature variance dissipation rate (chi) from oceanographic moored time series"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "scipy",
    "xarray",
    "gsw",
    "pyyaml",
]

[build-system]
requires = ["uv_build>=0.10.9,<0.11.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-root = "src"

[dependency-groups]
dev = [
    "pytest",
    "ruff",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
src = ["src"]
line-length = 88
```

- [ ] **Step 2: Create default config**

Create `config/default.yml`:

```yaml
spectral:
  spectra_size: 128        # FFT segment length (2^7)
  avrg_lim: [0.008, 0.1]  # Inertial subrange frequency bounds [Hz]
  chi_time_step: 600       # Chunk duration [seconds] (10 minutes)

qc:
  chi_spectra_bin_bounds: [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3]

physics:
  gamma: 0.2               # Mixing efficiency
  salinity: 35.0            # PSU (for thermal expansion coefficient)
  latitude: 54.25           # Degrees N (for pressure calculation)
  bottom_depth: 1466.0      # Meters (for height-above-bottom)
  U_ref: 0.1               # Representative velocity [m/s] (for u/h low-freq cutoff)
```

- [ ] **Step 3: Create __init__.py**

Create `src/pychi/__init__.py`:

```python
from pychi.config import Config
from pychi.spectra import csd_odas, welch_spectrum
from pychi.gradients import vertical_gradient, horizontal_gradient
from pychi.chi import calc_chi, process_chi

__all__ = [
    "Config",
    "csd_odas",
    "welch_spectrum",
    "vertical_gradient",
    "horizontal_gradient",
    "calc_chi",
    "process_chi",
]
```

- [ ] **Step 4: Create virtual environment and install**

```bash
cd /Users/gunnar/Projects/claude/pychi
uv sync
```

- [ ] **Step 5: Verify import fails gracefully**

```bash
uv run python -c "import pychi"
```

Expected: ImportError (modules don't exist yet). This confirms the package structure is wired up.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config/default.yml src/pychi/__init__.py
git commit -m "scaffold: project structure, pyproject.toml, default config"
```

---

### Task 2: Config module

**Files:**
- Create: `src/pychi/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for Config**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_config.py -v
```

Expected: FAIL (ModuleNotFoundError or ImportError)

- [ ] **Step 3: Implement Config**

Create `src/pychi/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "default.yml"


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
            for key in ("gamma", "salinity", "latitude", "bottom_depth", "U_ref"):
                if key in raw["physics"]:
                    kwargs[key] = raw["physics"][key]

        return cls(**kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pychi/config.py tests/test_config.py
git commit -m "feat: add Config dataclass with YAML loading and Matlab defaults"
```

---

## Chunk 2: Spectral Estimation (csd_odas + welch)

### Task 3: csd_odas — faithful port

**Files:**
- Create: `src/pychi/spectra.py`
- Create: `tests/test_spectra.py`

**Reference:** `Chi_Calc_For_Gunnar/csd_odas.m` — the Matlab function we are porting. Only the auto-spectrum path is needed (the `y` parameter is dropped). The function segments the input, optionally detrends each segment with a polynomial, applies a window, computes FFT, averages `|FFT|^2` across segments, and normalizes so the integral from 0 to Nyquist equals signal variance.

- [ ] **Step 1: Write failing test — known variance signal**

The most important property of `csd_odas` is that its integral from 0 to Nyquist equals the signal variance. We test this with a signal of known variance.

Create `tests/test_spectra.py`:

```python
import numpy as np
import pytest

from pychi.spectra import csd_odas


def test_csd_odas_variance_preservation():
    """Integral of auto-spectrum from 0 to Nyquist equals signal variance.

    This is the fundamental normalization property documented in csd_odas.m.
    We generate white noise with known variance and verify the spectral
    integral matches.
    """
    rng = np.random.default_rng(42)
    n = 1024
    rate = 1.0
    x = rng.standard_normal(n)
    n_fft = 128

    Pxx, f = csd_odas(x, n_fft, rate)

    # Integrate using trapezoidal rule: integral from 0 to Nyquist
    spectral_variance = np.trapz(Pxx, f)
    signal_variance = np.var(x, ddof=0)

    # Allow some tolerance — spectral estimation has finite resolution
    assert spectral_variance == pytest.approx(signal_variance, rel=0.15)


def test_csd_odas_frequency_vector():
    """Frequency vector ranges from 0 to Nyquist, with correct spacing."""
    n = 512
    rate = 2.0
    n_fft = 64
    x = np.random.default_rng(0).standard_normal(n)

    Pxx, f = csd_odas(x, n_fft, rate)

    assert f[0] == 0.0
    assert f[-1] == pytest.approx(rate / 2)  # Nyquist
    assert len(f) == n_fft // 2 + 1
    assert len(Pxx) == len(f)


def test_csd_odas_detrend_linear():
    """Linear detrending removes a linear trend from each segment."""
    n = 512
    rate = 1.0
    n_fft = 128
    # Signal is purely a linear ramp — after detrending, spectrum ~ 0
    x = np.linspace(0, 10, n)

    Pxx_none, _ = csd_odas(x, n_fft, rate, detrend="none")
    Pxx_lin, _ = csd_odas(x, n_fft, rate, detrend="linear")

    # With linear detrending, spectral power should be much smaller
    assert np.sum(Pxx_lin) < np.sum(Pxx_none) * 0.01


def test_csd_odas_input_too_short():
    """Raises ValueError if input is shorter than 2 * n_fft."""
    x = np.ones(100)
    with pytest.raises(ValueError, match="twice"):
        csd_odas(x, n_fft=128, rate=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_spectra.py -v
```

Expected: FAIL (ImportError — `csd_odas` doesn't exist)

- [ ] **Step 3: Implement csd_odas**

Create `src/pychi/spectra.py`:

```python
"""Spectral estimation functions.

Provides a faithful port of csd_odas.m (auto-spectrum only) and a
scipy.signal.welch wrapper for cross-validation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def csd_odas(
    x: NDArray[np.floating],
    n_fft: int,
    rate: float,
    window: NDArray[np.floating] | None = None,
    overlap: int | None = None,
    detrend: str = "none",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the auto-spectrum of a real signal using Welch's method.

    Faithful port of csd_odas.m (auto-spectrum path only). The Matlab function
    supports cross-spectra via a ``y`` parameter, but the chi pipeline always
    calls it with x == y, so this port omits ``y`` as a deliberate
    simplification.

    Parameters
    ----------
    x : 1D array
        Input signal. Must be NaN-free and real-valued.
    n_fft : int
        FFT segment length.
    rate : float
        Sampling frequency in Hz.
    window : 1D array of length n_fft, or None
        Window applied to each segment. If None, a Hanning window normalized
        to unit RMS is generated internally.

        Note: the Matlab csd_odas.m calls this a "cosine window" and generates
        it as ``1 + cos(pi * (-1 + 2*k/N))``. This is mathematically identical
        to the Hann (Hanning) window. In numpy, ``np.hanning(n)`` produces the
        same window.
    overlap : int or None
        Number of overlapping points between segments. Default: n_fft // 2.
    detrend : str
        Detrending mode applied to each segment before FFT. One of:
        "none", "constant" (order 0), "linear" (order 1),
        "parabolic" (order 2), "cubic" (order 3).
        Default "none" matches the Matlab function's default. The chi pipeline
        caller passes "linear", matching Calc_Chi_TChain_2.m line 10.

    Returns
    -------
    Pxx : 1D array
        Auto-spectral density. Length is n_fft // 2 + 1 (for even n_fft).
        Normalized so that the integral from 0 to Nyquist equals the signal
        variance.
    f : 1D array
        Frequency vector in Hz, from 0 to Nyquist (inclusive).
    """
    x = np.asarray(x, dtype=float).ravel()

    if len(x) < 2 * n_fft:
        raise ValueError(
            f"Input length ({len(x)}) must be more than twice n_fft ({n_fft})"
        )

    if overlap is None:
        overlap = n_fft // 2

    # --- Window ---
    # If no window provided, generate Hanning (= cosine) window normalized to
    # unit RMS. The Matlab csd_odas.m (lines 131-134) generates this as:
    #   Window = 1 + cos(pi*(-1 + 2*(0:n_fft-1)'/n_fft))
    #   Window = Window / sqrt(mean(Window.^2))
    # This is mathematically identical to numpy.hanning(n_fft) normalized to
    # unit RMS: both produce the Hann window w(k) = 0.5 - 0.5*cos(2*pi*k/N).
    if window is None:
        # Generate window using the exact Matlab csd_odas.m formula (lines 131-132):
        #   Window = 1 + cos(pi*(-1 + 2*(0:n_fft-1)'/n_fft))
        # This is mathematically equivalent to the Hann (Hanning) window but
        # differs from np.hanning() at the endpoints: Matlab's version is
        # periodic-like (endpoint not exactly zero), while np.hanning() is
        # symmetric (both endpoints zero). We use the Matlab formula exactly
        # to achieve ~1e-10 spectral agreement.
        window = 1.0 + np.cos(np.pi * (-1.0 + 2.0 * np.arange(n_fft) / n_fft))
        window = window / np.sqrt(np.mean(window**2))
    else:
        window = np.asarray(window, dtype=float).ravel()
        if len(window) != n_fft:
            raise ValueError(f"Window length ({len(window)}) must equal n_fft ({n_fft})")

    # --- Detrend order ---
    detrend_orders = {
        "none": None,
        "constant": 0,
        "linear": 1,
        "parabolic": 2,
        "cubic": 3,
    }
    if detrend not in detrend_orders:
        raise ValueError(
            f"detrend must be one of {list(detrend_orders.keys())}, got '{detrend}'"
        )
    order = detrend_orders[detrend]

    # --- Segment and accumulate ---
    increment = n_fft - overlap
    num_segments = (len(x) - overlap) // increment
    ramp = np.arange(n_fft, dtype=float)

    Cxy = np.zeros(n_fft)

    for i in range(num_segments):
        start = i * increment
        segment = x[start : start + n_fft]

        if order is not None:
            coeffs = np.polyfit(ramp, segment, order)
            segment = segment - np.polyval(coeffs, ramp)

        xw = window * segment
        X = np.fft.fft(xw, n_fft)
        Cxy += np.abs(X) ** 2

    # --- Select positive frequencies (real input) ---
    if n_fft % 2:  # odd
        freq_range = slice(0, (n_fft + 1) // 2)
    else:  # even — includes DC and Nyquist
        freq_range = slice(0, n_fft // 2 + 1)

    Pxx = Cxy[freq_range]
    f = np.arange(Pxx.shape[0]) * rate / n_fft

    # --- Normalize ---
    # Ensemble mean, then scale so integral 0..Nyquist = variance.
    # Matches Matlab: Cxy / num_segments / (n_fft * rate / 2)
    Pxx = Pxx / num_segments / (n_fft * rate / 2)

    return Pxx, f
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_spectra.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/pychi/spectra.py tests/test_spectra.py
git commit -m "feat: add csd_odas auto-spectrum port with detrending and variance normalization"
```

---

### Task 4: welch_spectrum — scipy wrapper

**Files:**
- Modify: `src/pychi/spectra.py`
- Modify: `tests/test_spectra.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_spectra.py`:

```python
from pychi.spectra import welch_spectrum


def test_welch_spectrum_variance_preservation():
    """welch_spectrum integral from 0 to Nyquist approximates signal variance."""
    rng = np.random.default_rng(42)
    n = 1024
    rate = 1.0
    x = rng.standard_normal(n)
    n_fft = 128

    Pxx, f = welch_spectrum(x, n_fft, rate)

    spectral_variance = np.trapz(Pxx, f)
    signal_variance = np.var(x, ddof=0)
    assert spectral_variance == pytest.approx(signal_variance, rel=0.15)


def test_welch_vs_csd_odas_comparison():
    """welch_spectrum and csd_odas produce similar results for same input.

    They won't be identical due to normalization and detrending differences,
    but spectral shapes should be close. This test documents the comparison.
    """
    rng = np.random.default_rng(99)
    n = 2048
    rate = 1.0
    x = rng.standard_normal(n)
    n_fft = 128

    Pxx_odas, f_odas = csd_odas(x, n_fft, rate, detrend="linear")
    Pxx_welch, f_welch = welch_spectrum(x, n_fft, rate)

    # Frequency vectors should match
    np.testing.assert_allclose(f_odas, f_welch, atol=1e-12)

    # Spectral levels should be in the same ballpark (within 50%)
    # Differences arise from detrending implementation details
    ratio = np.mean(Pxx_odas[1:]) / np.mean(Pxx_welch[1:])  # skip DC
    assert 0.5 < ratio < 2.0
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
uv run pytest tests/test_spectra.py::test_welch_spectrum_variance_preservation tests/test_spectra.py::test_welch_vs_csd_odas_comparison -v
```

Expected: FAIL (ImportError — `welch_spectrum` doesn't exist)

- [ ] **Step 3: Implement welch_spectrum**

Add to `src/pychi/spectra.py`:

```python
from scipy.signal import welch as _scipy_welch
from scipy.signal.windows import hann


def welch_spectrum(
    x: NDArray[np.floating],
    n_fft: int,
    rate: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute auto-spectrum using scipy.signal.welch.

    Configured to match csd_odas as closely as possible: Hanning window,
    50% overlap, linear detrend. Useful for cross-validation against the
    csd_odas port.

    Parameters
    ----------
    x : 1D array
        Input signal.
    n_fft : int
        FFT segment length.
    rate : float
        Sampling frequency in Hz.

    Returns
    -------
    Pxx : 1D array
        Auto-spectral density.
    f : 1D array
        Frequency vector in Hz.
    """
    f, Pxx = _scipy_welch(
        x,
        fs=rate,
        window=hann(n_fft, sym=False),
        nperseg=n_fft,
        noverlap=n_fft // 2,
        detrend="linear",
        scaling="density",
    )
    return Pxx, f
```

- [ ] **Step 4: Run all spectra tests**

```bash
uv run pytest tests/test_spectra.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/pychi/spectra.py tests/test_spectra.py
git commit -m "feat: add welch_spectrum wrapper and csd_odas comparison test"
```

---

## Chunk 3: Gradients

### Task 5: Vertical gradient

**Files:**
- Create: `src/pychi/gradients.py`
- Create: `tests/test_gradients.py`

**Reference:** Matlab `T_Chain_Chi_Horiz_Grads_hu_Spectra_Test.m` lines 137–152. Centered differences at interior sensors, one-sided at boundaries.

- [ ] **Step 1: Write failing tests**

Create `tests/test_gradients.py`:

```python
import numpy as np
import pytest

from pychi.gradients import vertical_gradient


def test_vertical_gradient_interior_sensor():
    """Interior sensor uses centered difference: (T[i-1] - T[i+1]) / (z[i-1] - z[i+1])."""
    # 3 depths, 4 time steps — uniform gradient of 1 °C/m
    depths = np.array([100.0, 110.0, 120.0])
    temp_cal = np.array([
        [20.0, 20.0, 20.0, 20.0],  # depth 100
        [19.0, 19.0, 19.0, 19.0],  # depth 110
        [18.0, 18.0, 18.0, 18.0],  # depth 120
    ])

    dtdz_mean, dtdz_ts = vertical_gradient(temp_cal, depths, sensor_index=1)

    # (T[0] - T[2]) / (z[0] - z[2]) = (20 - 18) / (100 - 120) = -0.1
    assert dtdz_mean == pytest.approx(-0.1)
    np.testing.assert_allclose(dtdz_ts, np.full(4, -0.1))


def test_vertical_gradient_top_sensor():
    """Top sensor (index 0) uses forward difference: (T[0] - T[1]) / (z[0] - z[1])."""
    depths = np.array([100.0, 110.0, 120.0])
    temp_cal = np.array([
        [20.0, 20.0],
        [19.0, 19.0],
        [18.0, 18.0],
    ])

    dtdz_mean, dtdz_ts = vertical_gradient(temp_cal, depths, sensor_index=0)

    # (T[0] - T[1]) / (z[0] - z[1]) = (20 - 19) / (100 - 110) = -0.1
    assert dtdz_mean == pytest.approx(-0.1)


def test_vertical_gradient_bottom_sensor():
    """Bottom sensor (last index) uses backward difference: (T[N-2] - T[N-1]) / (z[N-2] - z[N-1])."""
    depths = np.array([100.0, 110.0, 120.0])
    temp_cal = np.array([
        [20.0, 20.0],
        [19.0, 19.0],
        [18.0, 18.0],
    ])

    dtdz_mean, dtdz_ts = vertical_gradient(temp_cal, depths, sensor_index=2)

    # (T[1] - T[2]) / (z[1] - z[2]) = (19 - 18) / (110 - 120) = -0.1
    assert dtdz_mean == pytest.approx(-0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_gradients.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement vertical_gradient**

Create `src/pychi/gradients.py`:

```python
"""Temperature gradient computation.

Provides vertical (dT/dz) and horizontal (dT/dx) gradient functions
matching the Matlab reference implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def vertical_gradient(
    temp_cal: NDArray[np.floating],
    depths: NDArray[np.floating],
    sensor_index: int,
) -> tuple[float, NDArray[np.floating]]:
    """Compute vertical temperature gradient dT/dz for a given sensor.

    Uses finite differences matching Matlab lines 137-152:
    - Top sensor (index 0): (T[0] - T[1]) / (depth[0] - depth[1])
    - Bottom sensor (index N-1): (T[N-2] - T[N-1]) / (depth[N-2] - depth[N-1])
    - Interior sensors: centered (T[i-1] - T[i+1]) / (depth[i-1] - depth[i+1])

    Parameters
    ----------
    temp_cal : 2D array, shape (n_depths, n_times)
        Calibrated temperature.
    depths : 1D array
        Depth values, ordered shallow-to-deep (increasing depth).
    sensor_index : int
        Index of the sensor being processed.

    Returns
    -------
    dtdz_mean : float
        Chunk-mean vertical gradient.
    dtdz_ts : 1D array
        Per-timestep vertical gradient.
    """
    n_depths = temp_cal.shape[0]

    if sensor_index == 0:
        # Top sensor — forward difference in index
        i_upper, i_lower = 0, 1
    elif sensor_index == n_depths - 1:
        # Bottom sensor — backward difference in index
        i_upper, i_lower = n_depths - 2, n_depths - 1
    else:
        # Interior — centered difference
        i_upper, i_lower = sensor_index - 1, sensor_index + 1

    dz = depths[i_upper] - depths[i_lower]
    dtdz_ts = (temp_cal[i_upper, :] - temp_cal[i_lower, :]) / dz
    # Compute dtdz_mean as gradient of the mean temperatures, matching Matlab
    # lines 139/144/149: (nanmean(T_upper) - nanmean(T_lower)) / dz.
    # This differs from np.nanmean(dtdz_ts) when NaNs are at different
    # positions in the two sensors.
    dtdz_mean = float(
        (np.nanmean(temp_cal[i_upper, :]) - np.nanmean(temp_cal[i_lower, :])) / dz
    )

    return dtdz_mean, dtdz_ts
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_gradients.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pychi/gradients.py tests/test_gradients.py
git commit -m "feat: add vertical_gradient with boundary handling matching Matlab"
```

---

### Task 6: Horizontal gradient

**Files:**
- Modify: `src/pychi/gradients.py`
- Modify: `tests/test_gradients.py`

**Reference:** Matlab line 154: `dtdx = ((temp_in_cal(end)-temp_in_cal(1)) / (chi_time_step*24*60*60)) / U_in`

- [ ] **Step 1: Write failing test**

Append to `tests/test_gradients.py`:

```python
from pychi.gradients import horizontal_gradient


def test_horizontal_gradient_known_values():
    """Frozen-field dT/dx: (T_end - T_start) / (dt * U)."""
    temp_chunk = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    chunk_duration_s = 600.0  # 10 minutes
    U = 0.5  # m/s

    dtdx = horizontal_gradient(temp_chunk, chunk_duration_s, U)

    # (12.0 - 10.0) / (600 * 0.5) = 2.0 / 300.0 = 0.006667
    assert dtdx == pytest.approx(2.0 / 300.0)


def test_horizontal_gradient_negative():
    """Cooling trend produces negative dT/dx."""
    temp_chunk = np.array([15.0, 14.0, 13.0])
    dtdx = horizontal_gradient(temp_chunk, 600.0, 1.0)
    assert dtdx == pytest.approx(-2.0 / 600.0)
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
uv run pytest tests/test_gradients.py::test_horizontal_gradient_known_values tests/test_gradients.py::test_horizontal_gradient_negative -v
```

Expected: FAIL (ImportError)

- [ ] **Step 3: Implement horizontal_gradient**

Add to `src/pychi/gradients.py`:

```python
def horizontal_gradient(
    temp_cal_chunk: NDArray[np.floating],
    chunk_duration_s: float,
    U: float,
) -> float:
    """Compute horizontal temperature gradient dT/dx via frozen-field hypothesis.

    Matches Matlab line 154:
    dtdx = ((temp_in_cal(end) - temp_in_cal(1)) / (chi_time_step * 24*60*60)) / U_in

    Parameters
    ----------
    temp_cal_chunk : 1D array
        Calibrated temperatures for the time chunk.
    chunk_duration_s : float
        Chunk duration in seconds (equals config.chi_time_step).
    U : float
        Mean horizontal velocity magnitude [m/s].

    Returns
    -------
    dtdx : float
        Horizontal temperature gradient estimate.
    """
    return float((temp_cal_chunk[-1] - temp_cal_chunk[0]) / (chunk_duration_s * U))
```

- [ ] **Step 4: Run all gradient tests**

```bash
uv run pytest tests/test_gradients.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/pychi/gradients.py tests/test_gradients.py
git commit -m "feat: add horizontal_gradient (frozen-field dT/dx)"
```

---

## Chunk 4: Chi Calculation

### Task 7: calc_chi — single-chunk chi computation

**Files:**
- Create: `src/pychi/chi.py`
- Create: `tests/test_chi.py`

**Reference:** `Chi_Calc_For_Gunnar/Calc_Chi_TChain_2.m` — the core chi formula on line 51, and spectral slope fitting from lines 202–228 of the main script.

- [ ] **Step 1: Write failing tests**

Create `tests/test_chi.py`:

```python
import numpy as np
import pytest

from pychi.chi import calc_chi
from pychi.config import Config


def test_calc_chi_returns_positive_value():
    """calc_chi returns a positive chi value for a realistic synthetic signal."""
    rng = np.random.default_rng(42)
    # Simulate 600 samples at 1 Hz (10-minute chunk)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    U = 0.3
    gamma = 0.2
    alpha = 2.5e-4
    grad_T_mag = 0.01
    sample_freq = 1.0
    config = Config()

    chi_val, diag = calc_chi(temperature, U, gamma, alpha, grad_T_mag, sample_freq, config)

    assert chi_val > 0
    assert np.isfinite(chi_val)


def test_calc_chi_diagnostics_keys():
    """calc_chi returns diagnostics with expected keys."""
    rng = np.random.default_rng(42)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert "Pt" in diag
    assert "f" in diag
    assert "U" in diag
    assert "mean_t" in diag
    assert "spectral_slope" in diag


def test_calc_chi_diagnostics_spectrum_shape():
    """Pt and f have consistent lengths (n_fft // 2 + 1)."""
    rng = np.random.default_rng(42)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    config = Config(spectra_size=128)

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert len(diag["Pt"]) == 128 // 2 + 1
    assert len(diag["f"]) == len(diag["Pt"])


def test_calc_chi_mean_t():
    """mean_t in diagnostics equals the mean of the input temperature."""
    temperature = np.array([10.0, 11.0, 12.0] * 200)  # 600 samples
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert diag["mean_t"] == pytest.approx(np.mean(temperature))


def test_calc_chi_spectral_slope_is_finite():
    """spectral_slope is a finite float for a typical signal."""
    rng = np.random.default_rng(42)
    temperature = 15.0 + 0.1 * rng.standard_normal(600)
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert np.isfinite(diag["spectral_slope"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_chi.py -v
```

Expected: FAIL (ImportError)

- [ ] **Step 3: Implement calc_chi**

Create `src/pychi/chi.py`:

```python
"""Chi (χ) dissipation rate calculation.

Provides calc_chi (single-chunk computation, port of Calc_Chi_TChain_2.m)
and process_chi (orchestrator over depths and time chunks).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pychi.config import Config
from pychi.spectra import csd_odas


def calc_chi(
    temperature: NDArray[np.floating],
    U: float,
    gamma: float,
    alpha: float,
    grad_T_mag: float,
    sample_freq: float,
    config: Config,
) -> tuple[float, dict]:
    """Compute chi for a single time chunk at a single depth.

    Port of Calc_Chi_TChain_2.m. Computes the temperature power spectrum,
    takes the median spectral level in the inertial subrange (scaled by
    f^(5/3)), and applies the Batchelor/Osborn-Cox scaling.

    Parameters
    ----------
    temperature : 1D array
        NaN-free uncalibrated temperature for the chunk.
    U : float
        Horizontal velocity magnitude [m/s].
    gamma : float
        Mixing efficiency (typically 0.2).
    alpha : float
        Thermal expansion coefficient [1/°C].
    grad_T_mag : float
        Total temperature gradient magnitude sqrt(dtdz^2 + dtdx^2).
        Named grad_T_mag (not dtdz) to avoid confusion with the vertical
        gradient. The Matlab code passes sqrt(dtdz^2 + dtdx^2) as the
        dtdz argument (line 180) and takes abs() inside (line 49).
    sample_freq : float
        Sampling frequency in Hz.
    config : Config
        Provides spectra_size and avrg_lim.

    Returns
    -------
    chi_val : float
        Dissipation rate estimate.
    diagnostics : dict
        Keys: Pt (spectrum), f (frequencies), U, mean_t, spectral_slope.
    """
    n_fft = config.spectra_size

    # Compute auto-spectrum — let csd_odas generate its default window
    # (exact Matlab formula: 1 + cos(pi*(-1 + 2*k/N)), normalized to unit
    # RMS). Use linear detrend and 50% overlap, matching Calc_Chi_TChain_2.m
    # line 10.
    Pt, f = csd_odas(
        temperature, n_fft, sample_freq,
        window=None, overlap=n_fft // 2, detrend="linear",
    )

    # Scale spectrum by f^(5/3) to flatten inertial subrange
    scale = f ** (5 / 3)

    # Select inertial subrange frequencies
    avrg_lim = config.avrg_lim
    in_band = (f > avrg_lim[0]) & (f < avrg_lim[1])

    # Chi formula — Matlab line 51, with explicit operator precedence:
    # chi = ((phi * (2*pi/U)^(2/3)) / 0.4
    #        * (g*alpha / (2*|grad_T_mag|*gamma))^(1/3)) ^ (3/2)
    g = 9.81
    grad_T_mag = abs(grad_T_mag)
    phi = float(np.nanmedian(Pt[in_band] * scale[in_band]))
    chi_val = (
        (phi * (2 * np.pi / U) ** (2 / 3))
        / 0.4
        * (g * alpha / (2 * grad_T_mag * gamma)) ** (1 / 3)
    ) ** (3 / 2)

    # Spectral slope — linear fit in log-log space over the inertial subrange.
    # Matlab uses avrg_lim(1)/1.5 as the low-frequency bound for the fit
    # (line 202) and fits log10(f) vs log10(Pt).
    slope_band = (f > avrg_lim[0] / 1.5) & (f < avrg_lim[1])
    if np.any(slope_band) and np.any(Pt[slope_band] > 0):
        log_f = np.log10(f[slope_band])
        log_Pt = np.log10(Pt[slope_band])
        # Linear fit: log10(Pt) = a + b * log10(f), slope is b
        coeffs = np.polyfit(log_f, log_Pt, 1)
        spectral_slope = float(coeffs[0])
    else:
        spectral_slope = float("nan")

    diagnostics = {
        "Pt": Pt,
        "f": f,
        "U": U,
        "mean_t": float(np.nanmean(temperature)),
        "spectral_slope": spectral_slope,
    }

    return chi_val, diagnostics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_chi.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/pychi/chi.py tests/test_chi.py
git commit -m "feat: add calc_chi with chi formula port and spectral slope fitting"
```

---

## Chunk 5: Orchestrator (process_chi)

### Task 8: process_chi — main pipeline

**Files:**
- Modify: `src/pychi/chi.py`
- Create: `tests/test_process.py`

This is the largest task. The orchestrator loops over depths and time chunks, calls the gradient and chi functions, handles NaN chunks, computes binned average spectra, and assembles an xarray Dataset.

- [ ] **Step 1: Write failing tests**

Create `tests/test_process.py`:

```python
import numpy as np
import pytest
import xarray as xr

from pychi.chi import process_chi
from pychi.config import Config


def _make_synthetic_data(n_depths=3, n_times=1200, rate=1.0):
    """Create synthetic xarray inputs for process_chi.

    Returns data mimicking a T-chain + ADCP setup:
    - 3 depth levels at 100m, 110m, 120m
    - 1200 seconds of data at 1 Hz (covers 2 × 600s chunks)
    - ADCP at 5 depth bins spanning the T-chain range
    """
    rng = np.random.default_rng(42)
    depths = np.array([100.0, 110.0, 120.0])
    times = np.arange(n_times, dtype=float)  # seconds as float

    # Temperature: decreasing with depth, small random fluctuations
    temp_base = np.array([15.0, 14.0, 13.0])[:, None]
    temp_uncal = temp_base + 0.1 * rng.standard_normal((n_depths, n_times))
    temp_cal = temp_uncal + 0.5  # calibration offset

    temp_uncal_da = xr.DataArray(
        temp_uncal, dims=["depth", "time"],
        coords={"depth": depths, "time": times},
    )
    temp_cal_da = xr.DataArray(
        temp_cal, dims=["depth", "time"],
        coords={"depth": depths, "time": times},
    )

    # ADCP velocities
    adcp_depths = np.array([95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    adcp_times = np.arange(n_times, dtype=float)
    n_adcp_z = len(adcp_depths)

    u_vel = 0.2 + 0.02 * rng.standard_normal((n_times, n_adcp_z))
    v_vel = 0.1 + 0.02 * rng.standard_normal((n_times, n_adcp_z))
    w_vel = 0.01 * rng.standard_normal((n_times, n_adcp_z))

    u_da = xr.DataArray(
        u_vel, dims=["time", "z"],
        coords={"time": adcp_times, "z": adcp_depths},
    )
    v_da = xr.DataArray(
        v_vel, dims=["time", "z"],
        coords={"time": adcp_times, "z": adcp_depths},
    )
    w_da = xr.DataArray(
        w_vel, dims=["time", "z"],
        coords={"time": adcp_times, "z": adcp_depths},
    )

    return temp_uncal_da, temp_cal_da, u_da, v_da, w_da, depths, adcp_depths


def test_process_chi_returns_dataset():
    """process_chi returns an xarray Dataset with expected variables."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, binned = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config,
    )

    assert isinstance(result, xr.Dataset)
    expected_vars = {
        "chi", "U", "mean_u", "mean_v", "mean_w",
        "dtdz", "dtdx", "alpha", "gamma",
        "unstab_prop", "unstab_count", "unstab_length",
        "spectral_slope", "mean_t", "mean_t_uncal",
        "avrg_lim_actual", "Pt",
    }
    assert expected_vars.issubset(set(result.data_vars))
    assert "depth" in result.coords
    assert "time" in result.coords
    assert "time_bnds" in result.coords or "time_bnds" in result.data_vars


def test_process_chi_dimensions():
    """Output has correct dimensions: (n_depths, n_chunks)."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, _ = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config,
    )

    # 1200 seconds / 600s per chunk = 2 chunks, 3 depths
    assert result.dims["depth"] == 3
    assert result.dims["time"] == 2


def test_process_chi_single_sensor():
    """Single sensor mode processes only the requested depth."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    result, _ = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config,
        sensor_indices=[1],
    )

    assert result.dims["depth"] == 1
    assert float(result.depth.values[0]) == 110.0


def test_process_chi_nan_chunk_handling():
    """Chunks with NaN in temperature produce NaN chi but valid diagnostics."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    # Inject NaN into the first chunk for depth index 0
    temp_uncal.values[0, 10] = np.nan

    result, _ = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config,
    )

    # Chi should be NaN for depth 0, chunk 0
    assert np.isnan(result["chi"].values[0, 0])
    # But mean_t (calibrated) should still be finite
    assert np.isfinite(result["mean_t"].values[0, 0])
    # Gamma should be NaN for skipped chunks
    assert np.isnan(result["gamma"].values[0, 0])


def test_process_chi_binned_spectra():
    """Binned spectra Dataset has expected structure."""
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    config = Config(chi_time_step=600, bottom_depth=200.0)

    _, binned = process_chi(
        temp_uncal, temp_cal, u, v, w, depths, adcp_depths, config,
    )

    assert isinstance(binned, xr.Dataset)
    assert "binned_spectra" in binned.data_vars
    assert "binned_counts" in binned.data_vars
    assert "frequency" in binned.coords
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_process.py -v
```

Expected: FAIL (ImportError — `process_chi` doesn't exist)

- [ ] **Step 3: Implement process_chi**

Add to `src/pychi/chi.py`:

```python
import xarray as xr

from pychi.gradients import horizontal_gradient, vertical_gradient


def process_chi(
    temp_uncal: xr.DataArray,
    temp_cal: xr.DataArray,
    u_velocity: xr.DataArray,
    v_velocity: xr.DataArray,
    w_velocity: xr.DataArray,
    depths: NDArray[np.floating],
    adcp_depths: NDArray[np.floating],
    config: Config,
    sensor_indices: list[int] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute chi over all depths and time chunks.

    Orchestrator that loops over requested depth levels and time windows,
    calling vertical_gradient, horizontal_gradient, and calc_chi for each
    chunk. Returns primary results and binned spectral QC data.

    Parameters
    ----------
    temp_uncal : DataArray (depth, time)
        Uncalibrated T-chain temperatures.
    temp_cal : DataArray (depth, time)
        Calibrated T-chain temperatures.
    u_velocity, v_velocity, w_velocity : DataArray (time, z)
        ADCP velocity components. Pass at desired time resolution — no
        internal upsampling is performed. The Matlab code upsamples ADCP
        to 5x native resolution (lines 20-31); do this before calling.
    depths : 1D array
        T-chain depth levels, ordered shallow-to-deep.
    adcp_depths : 1D array
        ADCP depth bins.
    config : Config
        Pipeline configuration.
    sensor_indices : list of int, optional
        Depth indices to process. Default: all. For single-sensor mode,
        pass e.g. [5] — neighbors from temp_cal are still used for gradients.

    Returns
    -------
    result : xr.Dataset
        Primary output with chi, diagnostics, and per-chunk spectra.
    binned : xr.Dataset
        Binned average spectra for QC (grouped by log10(chi) bins).
    """
    import gsw

    depths = np.asarray(depths, dtype=float)
    adcp_depths = np.asarray(adcp_depths, dtype=float)
    times = temp_uncal.time.values
    adcp_times = u_velocity.time.values

    # Infer sample frequency from time coordinate.
    # Handle both numeric (seconds) and datetime64 time coordinates.
    dt_raw = times[1] - times[0]
    if np.issubdtype(type(dt_raw), np.timedelta64):
        dt = dt_raw / np.timedelta64(1, "s")
    else:
        dt = float(dt_raw)
    sample_freq = 1.0 / dt if dt != 0 else 1.0

    # Time chunk boundaries.
    # Convert to numeric seconds if datetime64, then chunk.
    if np.issubdtype(times.dtype, np.datetime64):
        t0 = times[0]
        times_numeric = (times - t0) / np.timedelta64(1, "s")
    else:
        t0 = None
        times_numeric = times.astype(float)

    t_start = float(times_numeric[0])
    t_end = float(times_numeric[-1])
    chunk_step = config.chi_time_step
    time_bnds_list = np.arange(t_start, t_end, chunk_step)
    n_chunks = len(time_bnds_list) - 1
    if n_chunks < 1:
        time_bnds_list = np.array([t_start, t_end])
        n_chunks = 1

    # Chunk center times and boundary pairs (numeric seconds)
    chunk_centers_numeric = (time_bnds_list[:-1] + time_bnds_list[1:]) / 2
    time_bnds_pairs = np.column_stack([time_bnds_list[:-1], time_bnds_list[1:]])

    # Convert back to original time type for output coordinates
    if t0 is not None:
        chunk_centers = t0 + (chunk_centers_numeric * 1e9).astype("timedelta64[ns]")
    else:
        chunk_centers = chunk_centers_numeric

    if sensor_indices is None:
        sensor_indices = list(range(len(depths)))

    proc_depths = depths[sensor_indices]
    n_depths = len(sensor_indices)
    n_freq = config.spectra_size // 2 + 1

    # Pre-allocate output arrays
    chi_arr = np.full((n_depths, n_chunks), np.nan)
    U_arr = np.full((n_depths, n_chunks), np.nan)
    mean_u_arr = np.full((n_depths, n_chunks), np.nan)
    mean_v_arr = np.full((n_depths, n_chunks), np.nan)
    mean_w_arr = np.full((n_depths, n_chunks), np.nan)
    dtdz_arr = np.full((n_depths, n_chunks), np.nan)
    dtdx_arr = np.full((n_depths, n_chunks), np.nan)
    alpha_arr = np.full((n_depths, n_chunks), np.nan)
    gamma_arr = np.full((n_depths, n_chunks), np.nan)
    unstab_prop_arr = np.full((n_depths, n_chunks), np.nan)
    unstab_count_arr = np.full((n_depths, n_chunks), np.nan)
    unstab_length_arr = np.full((n_depths, n_chunks), np.nan)
    slope_arr = np.full((n_depths, n_chunks), np.nan)
    mean_t_arr = np.full((n_depths, n_chunks), np.nan)
    mean_t_uncal_arr = np.full((n_depths, n_chunks), np.nan)
    avrg_lim_actual_arr = np.full((n_depths, n_chunks), np.nan)
    Pt_arr = np.full((n_depths, n_chunks, n_freq), np.nan)
    f_arr = None  # Set on first valid computation

    for di, si in enumerate(sensor_indices):
        depth_val = depths[si]

        # Find nearest ADCP depth bin
        z_idx = int(np.argmin(np.abs(adcp_depths - depth_val)))

        for jj in range(n_chunks):
            t_lo = time_bnds_list[jj]
            t_hi = time_bnds_list[jj + 1]

            # --- Time masks (using numeric seconds) ---
            t_mask = (times_numeric >= t_lo) & (times_numeric < t_hi)
            if np.issubdtype(adcp_times.dtype, np.datetime64):
                adcp_numeric = (adcp_times - (t0 if t0 is not None else adcp_times[0])) / np.timedelta64(1, "s")
            else:
                adcp_numeric = adcp_times.astype(float)
            adcp_t_mask = (adcp_numeric >= t_lo) & (adcp_numeric < t_hi)

            # --- Velocities ---
            u_chunk = u_velocity.values[adcp_t_mask, :]
            v_chunk = v_velocity.values[adcp_t_mask, :]
            w_chunk = w_velocity.values[adcp_t_mask, :]

            mean_u_at_z = float(np.nanmean(u_chunk[:, z_idx]))
            mean_v_at_z = float(np.nanmean(v_chunk[:, z_idx]))

            U_val = np.sqrt(mean_u_at_z**2 + mean_v_at_z**2)
            U_arr[di, jj] = U_val

            # --- Temperature ---
            temp_uncal_chunk = temp_uncal.values[si, t_mask]
            temp_cal_chunk = temp_cal.values[:, t_mask]  # all depths for gradient
            temp_cal_sensor = temp_cal.values[si, t_mask]

            mean_t_arr[di, jj] = float(np.nanmean(temp_cal_sensor))

            # --- Vertical gradient ---
            dtdz_mean, dtdz_ts = vertical_gradient(temp_cal_chunk, depths, si)
            dtdz_arr[di, jj] = dtdz_mean

            # --- Horizontal gradient ---
            dtdx_val = horizontal_gradient(temp_cal_sensor, config.chi_time_step, U_val)
            dtdx_arr[di, jj] = dtdx_val

            # --- Instability ---
            valid_mask = ~np.isnan(dtdz_ts)
            unstab_length = int(np.sum(valid_mask))
            unstab_count = int(np.sum(dtdz_ts[valid_mask] > 0))
            unstab_prop = unstab_count / unstab_length if unstab_length > 0 else np.nan
            unstab_prop_arr[di, jj] = unstab_prop
            unstab_count_arr[di, jj] = unstab_count
            unstab_length_arr[di, jj] = unstab_length

            # --- Thermal expansion coefficient ---
            p = gsw.p_from_z(-depth_val, config.latitude)
            SA = gsw.SA_from_SP(config.salinity, p, 0, config.latitude)
            CT = gsw.CT_from_t(SA, mean_t_arr[di, jj], p)
            alpha_val = float(gsw.alpha(SA, CT, p))
            alpha_arr[di, jj] = alpha_val

            # --- Adjust low-freq limit ---
            hab = config.bottom_depth - depth_val
            avrg_lim_lo = max(config.avrg_lim[0], config.U_ref / hab)
            avrg_lim_actual_arr[di, jj] = avrg_lim_lo

            # --- NaN check ---
            has_nan = np.any(np.isnan(temp_uncal_chunk))

            if has_nan:
                # NaN-skipped chunk: chi and gamma are NaN.
                # mean_u/v/w use nearest ADCP bin (not interpolation) to match
                # Matlab behavior (lines 262-293), preserved intentionally so
                # Python results compare directly against Matlab output.
                chi_arr[di, jj] = np.nan
                gamma_arr[di, jj] = np.nan
                mean_t_uncal_arr[di, jj] = np.nan
                mean_u_arr[di, jj] = float(
                    np.nanmean(u_chunk[:, z_idx])
                )
                mean_v_arr[di, jj] = float(
                    np.nanmean(v_chunk[:, z_idx])
                )
                mean_w_arr[di, jj] = float(
                    np.nanmean(w_chunk[:, z_idx])
                )
                slope_arr[di, jj] = np.nan
            else:
                # Valid chunk — interpolate mean velocities to sensor depth
                mean_u_profile = np.nanmean(u_chunk, axis=0)
                mean_v_profile = np.nanmean(v_chunk, axis=0)
                mean_w_profile = np.nanmean(w_chunk, axis=0)
                mean_u_arr[di, jj] = float(
                    np.interp(depth_val, adcp_depths, mean_u_profile)
                )
                mean_v_arr[di, jj] = float(
                    np.interp(depth_val, adcp_depths, mean_v_profile)
                )
                mean_w_arr[di, jj] = float(
                    np.interp(depth_val, adcp_depths, mean_w_profile)
                )

                # Adjusted config for this chunk
                chunk_config = Config(
                    spectra_size=config.spectra_size,
                    avrg_lim=[avrg_lim_lo, config.avrg_lim[1]],
                    chi_time_step=config.chi_time_step,
                    gamma=config.gamma,
                    salinity=config.salinity,
                    latitude=config.latitude,
                    bottom_depth=config.bottom_depth,
                    U_ref=config.U_ref,
                    chi_spectra_bin_bounds=config.chi_spectra_bin_bounds,
                )

                grad_T_mag = np.sqrt(dtdz_mean**2 + dtdx_val**2)

                chi_val, diag = calc_chi(
                    temp_uncal_chunk, U_val, config.gamma, alpha_val,
                    grad_T_mag, sample_freq, chunk_config,
                )

                chi_arr[di, jj] = chi_val
                gamma_arr[di, jj] = config.gamma
                mean_t_uncal_arr[di, jj] = diag["mean_t"]
                slope_arr[di, jj] = diag["spectral_slope"]
                Pt_arr[di, jj, :] = diag["Pt"]
                if f_arr is None:
                    f_arr = diag["f"]

    # If no valid chunks were computed, create a dummy frequency array
    if f_arr is None:
        f_arr = np.arange(n_freq) * sample_freq / config.spectra_size

    # --- Assemble primary Dataset ---
    result = xr.Dataset(
        {
            "chi": (["depth", "time"], chi_arr),
            "U": (["depth", "time"], U_arr),
            "mean_u": (["depth", "time"], mean_u_arr),
            "mean_v": (["depth", "time"], mean_v_arr),
            "mean_w": (["depth", "time"], mean_w_arr),
            "dtdz": (["depth", "time"], dtdz_arr),
            "dtdx": (["depth", "time"], dtdx_arr),
            "alpha": (["depth", "time"], alpha_arr),
            "gamma": (["depth", "time"], gamma_arr),
            "unstab_prop": (["depth", "time"], unstab_prop_arr),
            "unstab_count": (["depth", "time"], unstab_count_arr),
            "unstab_length": (["depth", "time"], unstab_length_arr),
            "spectral_slope": (["depth", "time"], slope_arr),
            "mean_t": (["depth", "time"], mean_t_arr),
            "mean_t_uncal": (["depth", "time"], mean_t_uncal_arr),
            "avrg_lim_actual": (["depth", "time"], avrg_lim_actual_arr),
            "Pt": (["depth", "time", "frequency"], Pt_arr),
            "time_bnds": (["time", "bnds"], time_bnds_pairs),
        },
        coords={
            "depth": proc_depths,
            "time": chunk_centers,
            "frequency": f_arr,
        },
    )

    # --- Binned average spectra ---
    bin_edges = np.array(config.chi_spectra_bin_bounds, dtype=float)
    n_bins = len(bin_edges) - 1
    binned_sum = np.zeros((n_depths, n_bins, n_freq))
    binned_count = np.zeros((n_depths, n_bins), dtype=int)

    for di in range(n_depths):
        for jj in range(n_chunks):
            chi_val = chi_arr[di, jj]
            if np.isnan(chi_val) or chi_val <= 0:
                continue
            log_chi = np.log10(chi_val)
            # Find which bin this chi falls into
            bin_idx = np.searchsorted(bin_edges, log_chi, side="right") - 1
            if 0 <= bin_idx < n_bins:
                binned_sum[di, bin_idx, :] += Pt_arr[di, jj, :]
                binned_count[di, bin_idx] += 1

    # Compute means (avoid division by zero)
    with np.errstate(invalid="ignore", divide="ignore"):
        binned_mean = np.where(
            binned_count[:, :, np.newaxis] > 0,
            binned_sum / binned_count[:, :, np.newaxis],
            np.nan,
        )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned = xr.Dataset(
        {
            "binned_spectra": (["depth", "chi_bin", "frequency"], binned_mean),
            "binned_counts": (["depth", "chi_bin"], binned_count),
        },
        coords={
            "depth": proc_depths,
            "chi_bin": bin_centers,
            "frequency": f_arr,
            "chi_bin_edges": ("chi_bin_edge", bin_edges),
        },
    )

    return result, binned
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_process.py -v
```

Expected: 5 passed

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests pass (config: 3, spectra: 6, gradients: 5, chi: 5, process: 5 = 24 tests)

- [ ] **Step 6: Commit**

```bash
git add src/pychi/chi.py tests/test_process.py
git commit -m "feat: add process_chi orchestrator with NaN handling and binned spectra"
```

### Task 8b: Additional process_chi edge case tests

**Files:**
- Modify: `tests/test_process.py`

- [ ] **Step 1: Add test for avrg_lim low-frequency adjustment**

Append to `tests/test_process.py`:

```python
def test_process_chi_avrg_lim_adjustment():
    """When U_ref / hab > avrg_lim[0], the low-frequency limit is adjusted upward.

    With bottom_depth=150 and sensor at depth=140, hab=10, so
    U_ref/hab = 0.1/10 = 0.01, which exceeds avrg_lim[0] = 0.008.
    """
    temp_uncal, temp_cal, u, v, w, depths, adcp_depths = _make_synthetic_data()
    # Override depths to put a sensor near the bottom
    depths_shallow = np.array([140.0, 145.0, 148.0])
    adcp_depths_shallow = np.array([135.0, 140.0, 145.0, 148.0, 150.0])

    temp_uncal = temp_uncal.assign_coords(depth=depths_shallow)
    temp_cal = temp_cal.assign_coords(depth=depths_shallow)
    u = u.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(z=adcp_depths_shallow)
    v = v.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(z=adcp_depths_shallow)
    w = w.isel(z=slice(0, len(adcp_depths_shallow))).assign_coords(z=adcp_depths_shallow)

    config = Config(chi_time_step=600, bottom_depth=150.0, U_ref=0.1)

    result, _ = process_chi(
        temp_uncal, temp_cal, u, v, w,
        depths_shallow, adcp_depths_shallow, config,
        sensor_indices=[0],  # depth=140, hab=10, U_ref/hab=0.01
    )

    # avrg_lim_actual should be 0.01, not the default 0.008
    assert result["avrg_lim_actual"].values[0, 0] == pytest.approx(0.01)


def test_calc_chi_zero_signal_slope_is_nan():
    """spectral_slope is NaN when all spectral values in the band are zero."""
    # A constant signal produces zero variance after detrending
    temperature = np.full(600, 15.0)
    config = Config()

    _, diag = calc_chi(temperature, 0.3, 0.2, 2.5e-4, 0.01, 1.0, config)

    assert np.isnan(diag["spectral_slope"])
```

- [ ] **Step 2: Add import for calc_chi at top of test_process.py**

Add `from pychi.chi import calc_chi` to the imports in `tests/test_process.py`.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_process.py -v
```

Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_process.py
git commit -m "test: add edge case tests for avrg_lim adjustment and zero-signal slope"
```

---

## Chunk 6: Matlab Fixture Export, conftest, README, and Polish

### Task 9: Matlab fixture export script

**Files:**
- Create: `scripts/export_matlab_fixtures.m`

- [ ] **Step 1: Write the Matlab fixture export script**

Create `scripts/export_matlab_fixtures.m`:

```matlab
%% export_matlab_fixtures.m
% Generates test fixtures for pychi by running the Matlab chi pipeline
% on a small data subset and saving intermediate results at each stage.
%
% Run from the Chi_Calc_For_Gunnar/ directory after pointing paths to data.
% Saves .mat files to ../tests/fixtures/

output_dir = '../tests/fixtures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load ADCP
file_in = 'MAVS2_24606.nc';
ADCP.time = (double(ncread(file_in,'time'))./1e6./60./60./24) + datenum(2021,7,7,6,15,0);
ADCP.z = double(ncread(file_in,'z'));
ADCP.u = double(ncread(file_in,'u'));
ADCP.v = double(ncread(file_in,'v'));
ADCP.w = double(ncread(file_in,'w'));

ADCP.z = ADCP.z(2:17);
ADCP.u = ADCP.u(:,2:17);
ADCP.v = ADCP.v(:,2:17);
ADCP.w = ADCP.w(:,2:17);

ADCP.time_tmp = ADCP.time;
ADCP.time = interp1(1:length(ADCP.time), ADCP.time, 1:1/5:length(ADCP.time));
for ii = 1:length(ADCP.z)
    ADCP.u_tmp(:,ii) = interp1(ADCP.time_tmp, ADCP.u(:,ii), ADCP.time);
    ADCP.v_tmp(:,ii) = interp1(ADCP.time_tmp, ADCP.v(:,ii), ADCP.time);
    ADCP.w_tmp(:,ii) = interp1(ADCP.time_tmp, ADCP.w(:,ii), ADCP.time);
end
ADCP.u = ADCP.u_tmp;
ADCP.v = ADCP.v_tmp;
ADCP.w = ADCP.w_tmp;

%% Load T-chain (uncalibrated L1)
path_l1 = './mavs2_l1';
file_dates = datenum(2021,7,6):2:datenum(2021,7,8);  % Small subset: 2 files
TChain.temp = []; TChain.time = [];
for ii = 1:length(file_dates)
    filename = [path_l1, '/mavs2_', datestr(file_dates(ii),'yyyymmdd'), '.nc'];
    time = double(ncread(filename,'time'))./60./60./24 + file_dates(ii);
    depth = ncread(filename,'depth');
    temp = ncread(filename,'__xarray_dataarray_variable__');
    TChain.time = [TChain.time time'];
    TChain.temp = [TChain.temp temp'];
    TChain.depth = depth;
end

%% Load T-chain (calibrated L2)
path_l2 = './mavs2_l2';
TChain_cal.temp = []; TChain_cal.time = [];
for ii = 1:length(file_dates)
    filename = [path_l2, '/mavs2_', datestr(file_dates(ii),'yyyymmdd'), '.nc'];
    time = double(ncread(filename,'time'))./60./60./24 + file_dates(ii);
    depth = ncread(filename,'depth');
    temp = ncread(filename,'__xarray_dataarray_variable__');
    TChain_cal.time = [TChain_cal.time time'];
    TChain_cal.temp = [TChain_cal.temp temp'];
    TChain_cal.depth = depth;
end

%% Parameters
chi_time_step = 10/60/24;
spectra_size = 2^7;
sample_freq = 1;
gamma = 0.2;

chi.time_bnds = TChain.time(1):chi_time_step:TChain.time(end);
chi.depth = TChain.depth;

%% Select fixture chunks
% Pick 5 chunks that cover edge cases:
%   chunk 1: top sensor (ii=1), early time
%   chunk 2: bottom sensor (ii=length(depth)), early time
%   chunk 3: middle sensor, middle time
%   chunk 4: middle sensor, with adjusted avrg_lim (shallow depth = large hab)
%   chunk 5: chunk with NaN if available

fixture_cases = [1, 1;                       % depth_idx, time_idx
                 length(chi.depth), 1;
                 round(length(chi.depth)/2), round(length(chi.time_bnds)/4);
                 round(length(chi.depth)/2), round(length(chi.time_bnds)/2);
                 1, 2];

for cc = 1:size(fixture_cases, 1)
    ii = fixture_cases(cc, 1);
    jj = fixture_cases(cc, 2);

    if jj >= length(chi.time_bnds), continue; end

    tmp_indx = find(TChain_cal.time >= chi.time_bnds(jj) & TChain_cal.time < chi.time_bnds(jj+1));
    if isempty(tmp_indx), continue; end

    temp_in = TChain.temp(ii, tmp_indx);
    temp_in_cal = TChain_cal.temp(ii, tmp_indx);

    z_indx = find(abs(ADCP.z - chi.depth(ii)) == min(abs(ADCP.z - chi.depth(ii))));
    U_in = sqrt(nanmean(ADCP.u(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2 + ...
                nanmean(ADCP.v(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2);

    % Vertical gradient
    if ii == 1
        dtdz = (nanmean(TChain_cal.temp(ii,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii) - chi.depth(ii+1));
        dtdz_ts = (TChain_cal.temp(ii,tmp_indx) - TChain_cal.temp(ii+1,tmp_indx)) ./ (chi.depth(ii) - chi.depth(ii+1));
    elseif ii == length(chi.depth)
        dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii));
        dtdz_ts = (TChain_cal.temp(ii-1,tmp_indx) - TChain_cal.temp(ii,tmp_indx)) ./ (chi.depth(ii-1) - chi.depth(ii));
    else
        dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii+1));
        dtdz_ts = (TChain_cal.temp(ii-1,tmp_indx) - TChain_cal.temp(ii+1,tmp_indx)) ./ (chi.depth(ii-1) - chi.depth(ii+1));
    end

    dtdx = ((temp_in_cal(end) - temp_in_cal(1)) ./ (chi_time_step.*24.*60.*60)) ./ U_in;

    alpha_val = sw_alpha(35, nanmean(TChain_cal.temp(ii,tmp_indx)), sw_pres(chi.depth(ii), 54+(15/60)));

    avrg_lim = [0.8e-2, 1e-1];
    U_tmp = 0.1;
    hab_tmp = 1466 - chi.depth(ii);
    u_h = U_tmp / hab_tmp;
    if u_h > avrg_lim(1); avrg_lim(1) = u_h; end

    unstab_count = length(find(dtdz_ts > 0));
    unstab_length = length(find(~isnan(dtdz_ts)));

    has_nan = length(temp_in) ~= length(temp_in(~isnan(temp_in)));

    if ~has_nan
        % Compute spectrum
        win = hanning(spectra_size) ./ sqrt(mean(hanning(spectra_size).^2));
        [Pt, f] = csd_odas(temp_in, temp_in, spectra_size, sample_freq, win, spectra_size/2, 'linear');

        grad_T_mag = sqrt(dtdz.^2 + dtdx.^2);
        [chi_val, diag_out] = Calc_Chi_TChain_2(temp_in, U_in, gamma, alpha_val, grad_T_mag, 0, spectra_size, avrg_lim);

        % Save spectral fixture
        save(fullfile(output_dir, ['spectra_chunk_', num2str(cc), '.mat']), ...
            'temp_in', 'Pt', 'f', 'win', 'spectra_size', 'sample_freq', 'avrg_lim');

        % Save chi fixture
        save(fullfile(output_dir, ['chi_chunk_', num2str(cc), '.mat']), ...
            'chi_val', 'alpha_val', 'gamma', 'U_in', 'grad_T_mag', ...
            'dtdz', 'dtdx', 'unstab_count', 'unstab_length', ...
            'temp_in', 'temp_in_cal', 'avrg_lim');
    end

    % Save gradient fixture (always, even if NaN)
    if ii == 1
        temp_cal_neighbors = TChain_cal.temp(1:2, tmp_indx);
        depths_neighbors = chi.depth(1:2);
    elseif ii == length(chi.depth)
        temp_cal_neighbors = TChain_cal.temp(end-1:end, tmp_indx);
        depths_neighbors = chi.depth(end-1:end);
    else
        temp_cal_neighbors = TChain_cal.temp(ii-1:ii+1, tmp_indx);
        depths_neighbors = chi.depth(ii-1:ii+1);
    end

    save(fullfile(output_dir, ['gradient_chunk_', num2str(cc), '.mat']), ...
        'temp_cal_neighbors', 'depths_neighbors', 'dtdz', 'dtdx', 'U_in', ...
        'ii', 'has_nan');
end

%% Save small end-to-end fixture (3 depths x 10 chunks)
% Run the pipeline for a small subset and save full input/output
n_test_depths = min(3, length(chi.depth));
n_test_chunks = min(10, length(chi.time_bnds) - 1);

e2e_chi = nan(n_test_depths, n_test_chunks);
e2e_dtdz = nan(n_test_depths, n_test_chunks);
e2e_dtdx = nan(n_test_depths, n_test_chunks);
e2e_U = nan(n_test_depths, n_test_chunks);
e2e_alpha = nan(n_test_depths, n_test_chunks);
e2e_slope = nan(n_test_depths, n_test_chunks);

ft = fittype('a+b*x');

for ii = 1:n_test_depths
    for jj = 1:n_test_chunks
        tmp_indx = find(TChain_cal.time >= chi.time_bnds(jj) & TChain_cal.time < chi.time_bnds(jj+1));
        if isempty(tmp_indx), continue; end

        temp_in = TChain.temp(ii, tmp_indx);
        temp_in_cal = TChain_cal.temp(ii, tmp_indx);

        z_indx = find(abs(ADCP.z - chi.depth(ii)) == min(abs(ADCP.z - chi.depth(ii))));
        U_in = sqrt(nanmean(ADCP.u(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2 + ...
                    nanmean(ADCP.v(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2);

        if ii == 1
            dtdz = (nanmean(TChain_cal.temp(ii,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii) - chi.depth(ii+1));
        elseif ii == n_test_depths
            dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii));
        else
            dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii+1));
        end

        dtdx = ((temp_in_cal(end) - temp_in_cal(1)) ./ (chi_time_step.*24.*60.*60)) ./ U_in;
        alpha_val = sw_alpha(35, nanmean(TChain_cal.temp(ii,tmp_indx)), sw_pres(chi.depth(ii), 54+(15/60)));

        avrg_lim = [0.8e-2, 1e-1];
        u_h = 0.1 / (1466 - chi.depth(ii));
        if u_h > avrg_lim(1); avrg_lim(1) = u_h; end

        e2e_dtdz(ii,jj) = dtdz;
        e2e_dtdx(ii,jj) = dtdx;
        e2e_U(ii,jj) = U_in;
        e2e_alpha(ii,jj) = alpha_val;

        if length(temp_in) == length(temp_in(~isnan(temp_in)))
            grad_T_mag = sqrt(dtdz.^2 + dtdx.^2);
            [chi_val, diag_out] = Calc_Chi_TChain_2(temp_in, U_in, gamma, alpha_val, grad_T_mag, 0, spectra_size, avrg_lim);
            e2e_chi(ii,jj) = chi_val;

            % Slope
            indx = find(diag_out.f > avrg_lim(1)/1.5 & diag_out.f < avrg_lim(2));
            if ~isempty(find(diag_out.Pt(indx) ~= 0))
                fitobj = fit(log10(diag_out.f(indx)), log10(diag_out.Pt(indx)), ft, 'StartPoint', [1, -5/3]);
                e2e_slope(ii,jj) = fitobj.b;
            end
        end
    end
end

e2e_depths = chi.depth(1:n_test_depths);
e2e_time_bnds = chi.time_bnds(1:n_test_chunks+1);

save(fullfile(output_dir, 'pipeline_subset.mat'), ...
    'e2e_chi', 'e2e_dtdz', 'e2e_dtdx', 'e2e_U', 'e2e_alpha', 'e2e_slope', ...
    'e2e_depths', 'e2e_time_bnds', 'gamma', 'spectra_size', 'sample_freq');

disp('Fixture export complete.');
```

- [ ] **Step 2: Add dependency note at top of script**

Add this comment to line 2 of the export script:

```matlab
% Requires: seawater toolbox (sw_alpha, sw_pres) on Matlab path.
```

- [ ] **Step 3: Commit**

```bash
mkdir -p scripts tests/fixtures
git add scripts/export_matlab_fixtures.m
git commit -m "feat: add Matlab fixture export script for test validation"
```

---

### Task 9b: conftest.py and Matlab-validated test placeholders

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_spectra.py`
- Modify: `tests/test_gradients.py`
- Modify: `tests/test_chi.py`
- Modify: `tests/test_process.py`

- [ ] **Step 1: Create conftest.py with fixture loaders**

Create `tests/conftest.py`:

```python
"""Shared fixtures for loading Matlab reference data.

Matlab fixtures are generated by running scripts/export_matlab_fixtures.m
and saved to tests/fixtures/ as .mat files.
"""

from pathlib import Path

import pytest
import numpy as np

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _has_fixtures():
    """Check if Matlab fixtures have been generated."""
    return FIXTURES_DIR.exists() and any(FIXTURES_DIR.glob("*.mat"))


requires_matlab_fixtures = pytest.mark.skipif(
    not _has_fixtures(),
    reason="Matlab fixtures not generated (run scripts/export_matlab_fixtures.m)",
)


@pytest.fixture
def spectra_fixture():
    """Load a spectral fixture (temperature input, Pt, f, window)."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "spectra_chunk_1.mat"
    data = loadmat(path, squeeze_me=True)
    return data


@pytest.fixture
def gradient_fixture():
    """Load a gradient fixture (temp_cal_neighbors, depths, dtdz, dtdx)."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "gradient_chunk_3.mat"
    data = loadmat(path, squeeze_me=True)
    return data


@pytest.fixture
def chi_fixture():
    """Load a chi fixture (chi_val, all intermediates)."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "chi_chunk_1.mat"
    data = loadmat(path, squeeze_me=True)
    return data


@pytest.fixture
def pipeline_fixture():
    """Load end-to-end pipeline fixture."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "pipeline_subset.mat"
    data = loadmat(path, squeeze_me=True)
    return data
```

- [ ] **Step 2: Add Matlab-validated placeholder tests to test_spectra.py**

Append to `tests/test_spectra.py`:

```python
from conftest import requires_matlab_fixtures


@requires_matlab_fixtures
def test_csd_odas_vs_matlab(spectra_fixture):
    """csd_odas output matches Matlab csd_odas to ~1e-10 relative tolerance."""
    data = spectra_fixture
    temp_in = data["temp_in"]
    Pt_matlab = data["Pt"]
    f_matlab = data["f"]
    n_fft = int(data["spectra_size"])
    rate = float(data["sample_freq"])

    # Use exact Matlab window from fixture
    win = data["win"]

    Pxx, f = csd_odas(temp_in, n_fft, rate, window=win, detrend="linear")

    np.testing.assert_allclose(f, f_matlab, rtol=1e-12)
    np.testing.assert_allclose(Pxx, Pt_matlab, rtol=1e-10)
```

- [ ] **Step 3: Add Matlab-validated placeholder tests to test_gradients.py**

Append to `tests/test_gradients.py`:

```python
from conftest import requires_matlab_fixtures


@requires_matlab_fixtures
def test_vertical_gradient_vs_matlab(gradient_fixture):
    """vertical_gradient matches Matlab dtdz exactly."""
    data = gradient_fixture
    temp_cal = data["temp_cal_neighbors"]
    depths = data["depths_neighbors"]
    dtdz_matlab = float(data["dtdz"])
    ii = int(data["ii"]) - 1  # Matlab 1-indexed to Python 0-indexed

    # Determine sensor_index relative to the neighbor slice
    if temp_cal.shape[0] == 2:
        sensor_index = 0 if ii == 0 else 1
    else:
        sensor_index = 1  # middle of 3

    dtdz_mean, _ = vertical_gradient(temp_cal, depths, sensor_index)
    assert dtdz_mean == pytest.approx(dtdz_matlab, rel=1e-12)
```

- [ ] **Step 4: Add Matlab-validated placeholder tests to test_chi.py**

Append to `tests/test_chi.py`:

```python
from conftest import requires_matlab_fixtures


@requires_matlab_fixtures
def test_calc_chi_vs_matlab(chi_fixture):
    """calc_chi output matches Matlab chi value to ~1e-6 relative tolerance."""
    data = chi_fixture
    temp_in = data["temp_in"]
    U = float(data["U_in"])
    gamma = float(data["gamma"])
    alpha = float(data["alpha_val"])
    grad_T_mag = float(data["grad_T_mag"])
    avrg_lim = data["avrg_lim"].tolist()
    chi_matlab = float(data["chi_val"])

    config = Config(
        spectra_size=128,
        avrg_lim=avrg_lim,
    )

    chi_val, _ = calc_chi(temp_in, U, gamma, alpha, grad_T_mag, 1.0, config)

    np.testing.assert_allclose(chi_val, chi_matlab, rtol=1e-6)
```

- [ ] **Step 5: Run tests (Matlab tests should skip, others pass)**

```bash
uv run pytest -v
```

Expected: Matlab-validated tests SKIP (no fixtures), all other tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_spectra.py tests/test_gradients.py tests/test_chi.py
git commit -m "test: add conftest fixture loaders and Matlab-validated test placeholders"
```

---

### Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `README.md`:

```markdown
# pychi

Estimate the turbulent temperature variance dissipation rate (χ, chi) from oceanographic moored time series.

pychi is a Python port of a Matlab reference implementation. It computes χ from moored T-chain and ADCP data using spectral analysis in the inertial subrange with the Batchelor/Osborn-Cox scaling.

## Installation

```bash
# Clone and install with uv
git clone <repo-url>
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

3. **`chi.calc_chi()`** — Single-chunk chi computation: spectrum → inertial subrange median → Batchelor/Osborn-Cox formula.

4. **`chi.process_chi()`** — Orchestrator that loops over depths and time chunks, calls the above functions, handles NaN chunks, and returns xarray Datasets.

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

To generate Matlab test fixtures, run `scripts/export_matlab_fixtures.m` in Matlab from the `Chi_Calc_For_Gunnar/` directory. This saves intermediate results (spectra, gradients, chi values) to `tests/fixtures/`.

## Dependencies

- numpy, scipy, xarray, gsw, pyyaml
- Dev: pytest, ruff
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add detailed README with usage examples and architecture overview"
```

---

### Task 11: Update __init__.py and lint

**Files:**
- Modify: `src/pychi/__init__.py`

- [ ] **Step 1: Verify __init__.py exports match implemented API**

Read `src/pychi/__init__.py` and confirm all imports resolve now that all modules exist.

```bash
uv run python -c "from pychi import Config, csd_odas, welch_spectrum, vertical_gradient, horizontal_gradient, calc_chi, process_chi; print('All imports OK')"
```

Expected: "All imports OK"

- [ ] **Step 2: Run ruff**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

Fix any issues found.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -u
git commit -m "style: apply ruff formatting and fix lint issues"
```

---

### Task 12: Add .gitignore and final commit

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
# Matlab output files (not test fixtures)
Chi_Calc_For_Gunnar/*.mat
# Keep test fixtures tracked
!tests/fixtures/*.mat
.ruff_cache/
```

- [ ] **Step 2: Final commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

- [ ] **Step 3: Verify clean state**

```bash
git status
uv run pytest -v
```

Expected: clean working tree, all tests pass.

"""Visual comparison plots: Python vs Matlab reference outputs.

Run with:
    uv run pytest tests/test_matlab_plots.py -v --no-header

Plots are saved to tests/plots/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pychi.chi import calc_chi
from pychi.config import Config
from pychi.gradients import vertical_gradient
from pychi.spectra import csd_odas

from conftest import requires_matlab_fixtures, FIXTURES_DIR

PLOT_DIR = Path(__file__).parent / "plots"


@pytest.fixture(autouse=True)
def _ensure_plot_dir():
    PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Spectra comparison
# ---------------------------------------------------------------------------


def _load_spectra_fixture(idx):
    from scipy.io import loadmat

    return loadmat(FIXTURES_DIR / f"spectra_chunk_{idx}.mat", squeeze_me=True)


@requires_matlab_fixtures
@pytest.mark.parametrize("chunk_idx", [1, 2, 3, 4, 5])
def test_plot_spectra_vs_matlab(chunk_idx):
    """Plot Python vs Matlab power spectra for each fixture chunk."""
    data = _load_spectra_fixture(chunk_idx)
    temp_in = data["temp_in"]
    Pt_matlab = data["Pt"]
    f_matlab = data["f"]
    n_fft = int(data["spectra_size"])
    rate = float(data["sample_freq"])
    win = data["win"]

    Pxx_py, f_py = csd_odas(temp_in, n_fft, rate, window=win, detrend="linear")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overlay spectra
    ax = axes[0]
    ax.loglog(f_matlab[1:], Pt_matlab[1:], "k-", lw=1.5, label="Matlab")
    ax.loglog(f_py[1:], Pxx_py[1:], "r--", lw=1.2, label="Python")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [°C²/Hz]")
    ax.set_title(f"Spectra — chunk {chunk_idx}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Right: relative error
    ax = axes[1]
    rel_err = np.abs(Pxx_py[1:] - Pt_matlab[1:]) / np.abs(Pt_matlab[1:])
    ax.semilogy(f_py[1:], rel_err, "b-", lw=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative error")
    ax.set_title(f"Relative error — chunk {chunk_idx}")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"spectra_chunk_{chunk_idx}.png", dpi=150)
    plt.close(fig)

    # Still assert correctness
    np.testing.assert_allclose(Pxx_py, Pt_matlab, rtol=1e-10)


# ---------------------------------------------------------------------------
# Chi comparison
# ---------------------------------------------------------------------------


def _load_chi_fixture(idx):
    from scipy.io import loadmat

    return loadmat(FIXTURES_DIR / f"chi_chunk_{idx}.mat", squeeze_me=True)


@requires_matlab_fixtures
def test_plot_chi_spectra_all_chunks():
    """Plot the f^(5/3)-compensated spectra for all chi fixture chunks,
    annotating the Matlab chi value and the Python chi value."""
    n_chunks = 5
    fig, axes = plt.subplots(1, n_chunks, figsize=(4 * n_chunks, 5), sharey=True)

    for i, ax in enumerate(axes, start=1):
        data = _load_chi_fixture(i)
        temp_in = data["temp_in"]
        U = float(data["U_in"])
        gamma = float(data["gamma"])
        alpha = float(data["alpha_val"])
        grad_T_mag = float(data["grad_T_mag"])
        avrg_lim = data["avrg_lim"].tolist()
        chi_matlab = float(data["chi_val"])

        config = Config(spectra_size=128, avrg_lim=avrg_lim)
        chi_py, diag = calc_chi(temp_in, U, gamma, alpha, grad_T_mag, 1.0, config)

        f = diag["f"]
        Pt = diag["Pt"]
        compensated = Pt * f ** (5 / 3)

        ax.loglog(f[1:], Pt[1:], "0.6", lw=0.8, label="PSD")
        ax.loglog(f[1:], compensated[1:], "k-", lw=1.2, label="f^{5/3} × PSD")

        # Shade inertial subrange
        ax.axvspan(avrg_lim[0], avrg_lim[1], alpha=0.15, color="green",
                   label="inertial subrange")

        # -5/3 reference slope
        f_ref = f[(f > avrg_lim[0] / 2) & (f < avrg_lim[1] * 2)]
        if len(f_ref) > 0:
            ref_level = np.median(Pt[(f > avrg_lim[0]) & (f < avrg_lim[1])])
            f_mid = np.sqrt(f_ref[0] * f_ref[-1])
            slope_ref = ref_level * (f_ref / f_mid) ** (-5 / 3)
            ax.loglog(f_ref, slope_ref, "b:", lw=1, label="-5/3 slope")

        ax.set_xlabel("Frequency [Hz]")
        if i == 1:
            ax.set_ylabel("PSD / Compensated PSD")

        ax.set_title(f"Chunk {i}")
        ax.text(
            0.03, 0.03,
            f"χ_mat = {chi_matlab:.2e}\nχ_py  = {chi_py:.2e}",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        if i == 1:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Chi estimation: Python vs Matlab", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "chi_all_chunks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Verify all match
    for i in range(1, n_chunks + 1):
        data = _load_chi_fixture(i)
        config = Config(spectra_size=128, avrg_lim=data["avrg_lim"].tolist())
        chi_py, _ = calc_chi(
            data["temp_in"], float(data["U_in"]), float(data["gamma"]),
            float(data["alpha_val"]), float(data["grad_T_mag"]), 1.0, config,
        )
        np.testing.assert_allclose(chi_py, float(data["chi_val"]), rtol=1e-6)


# ---------------------------------------------------------------------------
# Gradient comparison
# ---------------------------------------------------------------------------


def _load_gradient_fixture(idx):
    from scipy.io import loadmat

    return loadmat(FIXTURES_DIR / f"gradient_chunk_{idx}.mat", squeeze_me=True)


@requires_matlab_fixtures
def test_plot_gradient_vs_matlab():
    """Bar chart comparing Python vs Matlab vertical gradients across chunks."""
    n_chunks = 5
    dtdz_matlab = []
    dtdz_python = []
    labels = []

    for i in range(1, n_chunks + 1):
        data = _load_gradient_fixture(i)
        temp_cal = data["temp_cal_neighbors"]
        depths = data["depths_neighbors"]
        matlab_val = float(data["dtdz"])
        ii = int(data["ii"]) - 1

        if temp_cal.shape[0] == 2:
            sensor_index = 0 if ii == 0 else 1
        else:
            sensor_index = 1

        py_val, _ = vertical_gradient(temp_cal, depths, sensor_index)

        dtdz_matlab.append(matlab_val)
        dtdz_python.append(py_val)
        labels.append(f"Chunk {i}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(n_chunks)
    w = 0.35

    ax = axes[0]
    ax.bar(x - w / 2, dtdz_matlab, w, label="Matlab", color="steelblue")
    ax.bar(x + w / 2, dtdz_python, w, label="Python", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("dT/dz [°C/m]")
    ax.set_title("Vertical gradient comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    abs_err = np.abs(np.array(dtdz_python) - np.array(dtdz_matlab))
    ax.bar(x, abs_err, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute error [°C/m]")
    ax.set_title("Gradient absolute error")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gradient_comparison.png", dpi=150)
    plt.close(fig)

    for m, p in zip(dtdz_matlab, dtdz_python):
        assert p == pytest.approx(m, rel=1e-12)

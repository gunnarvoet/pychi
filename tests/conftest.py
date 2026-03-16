"""Shared fixtures for loading Matlab reference data."""

from pathlib import Path

import pytest

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
    """Load a spectral fixture."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "spectra_chunk_1.mat"
    return loadmat(path, squeeze_me=True)


@pytest.fixture
def gradient_fixture():
    """Load a gradient fixture."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "gradient_chunk_3.mat"
    return loadmat(path, squeeze_me=True)


@pytest.fixture
def chi_fixture():
    """Load a chi fixture."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "chi_chunk_1.mat"
    return loadmat(path, squeeze_me=True)


@pytest.fixture
def pipeline_fixture():
    """Load end-to-end pipeline fixture."""
    from scipy.io import loadmat

    path = FIXTURES_DIR / "pipeline_subset.mat"
    return loadmat(path, squeeze_me=True)

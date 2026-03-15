import numpy as np
import pytest

from pychi.gradients import vertical_gradient


def test_vertical_gradient_interior_sensor():
    """Interior sensor uses centered difference: (T[i-1] - T[i+1]) / (z[i-1] - z[i+1])."""
    depths = np.array([100.0, 110.0, 120.0])
    temp_cal = np.array([
        [20.0, 20.0, 20.0, 20.0],
        [19.0, 19.0, 19.0, 19.0],
        [18.0, 18.0, 18.0, 18.0],
    ])

    dtdz_mean, dtdz_ts = vertical_gradient(temp_cal, depths, sensor_index=1)

    assert dtdz_mean == pytest.approx(-0.1)
    np.testing.assert_allclose(dtdz_ts, np.full(4, -0.1))


def test_vertical_gradient_top_sensor():
    """Top sensor (index 0) uses forward difference."""
    depths = np.array([100.0, 110.0, 120.0])
    temp_cal = np.array([
        [20.0, 20.0],
        [19.0, 19.0],
        [18.0, 18.0],
    ])

    dtdz_mean, dtdz_ts = vertical_gradient(temp_cal, depths, sensor_index=0)
    assert dtdz_mean == pytest.approx(-0.1)


def test_vertical_gradient_bottom_sensor():
    """Bottom sensor (last index) uses backward difference."""
    depths = np.array([100.0, 110.0, 120.0])
    temp_cal = np.array([
        [20.0, 20.0],
        [19.0, 19.0],
        [18.0, 18.0],
    ])

    dtdz_mean, dtdz_ts = vertical_gradient(temp_cal, depths, sensor_index=2)
    assert dtdz_mean == pytest.approx(-0.1)

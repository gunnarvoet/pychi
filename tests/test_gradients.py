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


from pychi.gradients import horizontal_gradient


def test_horizontal_gradient_known_values():
    """Frozen-field dT/dx: (T_end - T_start) / (dt * U)."""
    temp_chunk = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    chunk_duration_s = 600.0
    U = 0.5

    dtdx = horizontal_gradient(temp_chunk, chunk_duration_s, U)
    assert dtdx == pytest.approx(2.0 / 300.0)


def test_horizontal_gradient_negative():
    """Cooling trend produces negative dT/dx."""
    temp_chunk = np.array([15.0, 14.0, 13.0])
    dtdx = horizontal_gradient(temp_chunk, 600.0, 1.0)
    assert dtdx == pytest.approx(-2.0 / 600.0)

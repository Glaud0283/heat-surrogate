"""Unit tests for the finite-difference solver: physics we can check by hand."""
import numpy as np
import pytest

from src.solver import solve

pytestmark = pytest.mark.unit


def test_shape():
    assert solve(2, 0.5, 100, 20, 80, 50, 50, n=16).shape == (16, 16)


def test_uniform_edges_no_source_gives_uniform_field():
    # If every edge is at 50 and nothing heats the plate, the whole plate is at 50,
    # whatever the conductivity does.
    field = solve(3, 1.2, 0, 50, 50, 50, 50, n=16)
    np.testing.assert_allclose(field, 50, atol=1e-8)


def test_heat_source_warms_the_plate():
    cold = solve(2, 0, 0, 20, 20, 20, 20, n=16)
    heated = solve(2, 0, 300, 20, 20, 20, 20, n=16)
    assert (heated > cold).all()


def test_left_right_symmetry_without_grading():
    # g = 0 and equal left/right temperatures: the field is a mirror image.
    field = solve(2, 0, 150, 10, 90, 40, 40, n=16)
    np.testing.assert_allclose(field, field[:, ::-1], atol=1e-8)

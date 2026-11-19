"""Unit tests for the network definition."""
import pytest
import torch

from src.model import N_PARAMS, make_mlp

pytestmark = pytest.mark.unit


def test_output_shape():
    model = make_mlp(64 * 64, width=16)
    assert model(torch.zeros(5, N_PARAMS)).shape == (5, 64 * 64)


def test_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    model = make_mlp(10, width=16).eval()
    x = torch.randn(3, N_PARAMS)
    assert torch.equal(model(x), model(x))

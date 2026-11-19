"""Runs only where a CUDA GPU exists (your RTX 5060), never on GitHub-hosted runners."""
import pytest
import torch

from src.model import N_PARAMS, make_mlp

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_one_training_step_on_gpu():
    torch.manual_seed(0)
    model = make_mlp(64 * 64, width=64).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = torch.randn(32, N_PARAMS, device="cuda"), torch.randn(32, 64 * 64, device="cuda")
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)

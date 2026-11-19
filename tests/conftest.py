"""Shared test fixtures. Imports live inside fixtures, so the GPU test can run
in the training image, which has no FastAPI."""
import pytest
import torch

from src.model import make_mlp


@pytest.fixture
def tiny_model_path(tmp_path):
    """A small, untrained checkpoint with the real file format. Fast, and needs no download."""
    torch.manual_seed(0)
    model = make_mlp(64 * 64, width=8)
    path = tmp_path / "tiny.pt"
    torch.save({"state_dict": model.state_dict(), "width": 8,
                "x_mu": torch.zeros(7), "x_sd": torch.ones(7),
                "y_mu": torch.zeros(64 * 64), "y_sd": torch.ones(64 * 64)}, path)
    return path


@pytest.fixture
def client(tiny_model_path, monkeypatch):
    """The real app, started (its lifespan runs) with the tiny model."""
    from fastapi.testclient import TestClient

    import serve

    monkeypatch.setenv("MODEL_PATH", str(tiny_model_path))
    with TestClient(serve.app) as c:
        yield c

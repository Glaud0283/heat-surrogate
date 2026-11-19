"""HTTP inference API for the heat surrogate."""
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.model import make_mlp

# Set at image build time by CI (Dockerfile.serve ARG -> ENV); "unknown" when run locally.
GIT_SHA = os.environ.get("GIT_SHA", "unknown")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

state = {}


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = make_mlp(ckpt["y_mu"].numel(), ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@asynccontextmanager
async def lifespan(app):
    # Load once at startup, not at import: tests can choose the model, and a bad
    # model file stops the server from starting instead of failing on first request.
    state["model"], state["ckpt"] = load_model(os.environ.get("MODEL_PATH", "outputs/model.pt"))
    yield
    state.clear()


app = FastAPI(title="Heat surrogate", lifespan=lifespan)


class Params(BaseModel):
    k0: float = Field(gt=0, description="conductivity at the left edge")
    g: float = Field(description="conductivity grading: k = k0 * exp(g * x)")
    q: float = Field(description="volumetric heat source")
    t_top: float
    t_bottom: float
    t_left: float
    t_right: float


@app.get("/health")
def health():
    return {"status": "ok", "git_sha": GIT_SHA, "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(p: Params):
    model, ckpt = state["model"], state["ckpt"]
    x = torch.tensor([[p.k0, p.g, p.q, p.t_top, p.t_bottom, p.t_left, p.t_right]])
    with torch.no_grad():
        field = (model((x - ckpt["x_mu"]) / ckpt["x_sd"]) * ckpt["y_sd"] + ckpt["y_mu"]).reshape(64, 64)
    return {"t_max": field.max().item(), "t_mean": field.mean().item(), "field": field.tolist()}

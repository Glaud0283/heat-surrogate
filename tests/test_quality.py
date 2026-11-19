"""Model-quality gate: the released model, served by the real API, must match the solver."""
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import serve
from src.solver import solve

pytestmark = pytest.mark.quality

MEDIAN_LIMIT = 0.01   # 1 % median relative L2 error (the release measured ~0.4 %)
P95_LIMIT = 0.03


def test_released_model_matches_solver():
    path = Path(os.environ.get("MODEL_PATH", "models/model.pt"))
    if not path.exists():
        pytest.fail(f"{path} not found: run scripts/fetch_model.sh first")

    rng = np.random.default_rng(123)  # not the training seed: unseen cases
    names = ["k0", "g", "q", "t_top", "t_bottom", "t_left", "t_right"]
    cases = np.column_stack([rng.uniform(1, 10, 30), rng.uniform(-1.5, 1.5, 30),
                             rng.uniform(0, 500, 30), rng.uniform(0, 100, (30, 4))])

    errors = []
    with TestClient(serve.app) as client:  # through the API: tests the serving code too
        for p in cases:
            field = np.array(client.post("/predict", json=dict(zip(names, p.tolist()))).json()["field"])
            truth = solve(*p)
            errors.append(np.linalg.norm(field - truth) / np.linalg.norm(truth))

    median, p95 = np.median(errors), np.percentile(errors, 95)
    print(f"median rel L2 {median:.4f}, p95 {p95:.4f}")
    assert median < MEDIAN_LIMIT, f"median error {median:.2%} exceeds {MEDIAN_LIMIT:.0%}"
    assert p95 < P95_LIMIT, f"p95 error {p95:.2%} exceeds {P95_LIMIT:.0%}"

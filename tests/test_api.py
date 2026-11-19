"""Integration tests: the HTTP API end to end, with a tiny model."""
import pytest

pytestmark = pytest.mark.integration

CASE = {"k0": 5, "g": 0.8, "q": 300, "t_top": 20, "t_bottom": 80, "t_left": 50, "t_right": 50}


def test_health_reports_version_fields(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert {"git_sha", "model_version"} <= r.json().keys()


def test_predict_returns_a_64_by_64_field(client):
    r = client.post("/predict", json=CASE)
    assert r.status_code == 200
    field = r.json()["field"]
    assert len(field) == 64 and all(len(row) == 64 for row in field)


@pytest.mark.parametrize("bad", [{**CASE, "k0": 0}, {k: v for k, v in CASE.items() if k != "q"}])
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422

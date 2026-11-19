"""Make training data with the solver, train the surrogate, save model + metrics.

    python train.py                      # defaults: 2000 simulations, 150 epochs
    python train.py --n-samples 300 --epochs 20 --out /tmp/quick   # a fast try-out
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import make_mlp
from src.solver import solve

ap = argparse.ArgumentParser()
ap.add_argument("--n-samples", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--width", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs")
args = ap.parse_args()
torch.manual_seed(args.seed)

# 1. Data: random physical parameters -> solver -> 64 x 64 temperature fields
t0 = time.time()
rng = np.random.default_rng(args.seed)
n = args.n_samples
params = np.column_stack([
    rng.uniform(1, 10, n),          # k0: conductivity at the left edge
    rng.uniform(-1.5, 1.5, n),      # g: grading, k = k0 * exp(g * x)
    rng.uniform(0, 500, n),         # q: heat source
    rng.uniform(0, 100, (n, 4)),    # top, bottom, left, right edge temperatures
])
fields = np.stack([solve(*p) for p in params])
print(f"{n} simulations in {time.time() - t0:.0f} s", flush=True)

# 2. Normalise with statistics from the training split only
X = torch.tensor(params, dtype=torch.float32)
Y = torch.tensor(fields, dtype=torch.float32).reshape(n, -1)
n_train = int(0.8 * n)
x_mu, x_sd = X[:n_train].mean(0), X[:n_train].std(0)
y_mu, y_sd = Y[:n_train].mean(0), Y[:n_train].std(0) + 1e-6
Xn, Yn = (X - x_mu) / x_sd, (Y - y_mu) / y_sd

# 3. Train
model = make_mlp(Y.shape[1], args.width)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(Xn[:n_train], Yn[:n_train]), batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(args.seed))
for epoch in range(args.epochs):
    for xb, yb in loader:
        loss = nn.functional.mse_loss(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 25 == 0:
        print(f"epoch {epoch + 1}: train loss {loss.item():.3e}", flush=True)

# 4. Evaluate on the held-out 20 % and save
model.eval()
with torch.no_grad():
    pred = model(Xn[n_train:]) * y_sd + y_mu
rel_l2 = (torch.linalg.norm(pred - Y[n_train:], dim=1) / torch.linalg.norm(Y[n_train:], dim=1)).mean().item()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "width": args.width,
            "x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd}, out / "model.pt")
metrics = {"test_rel_l2": round(rel_l2, 5), "train_seconds": round(time.time() - t0), **vars(args)}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

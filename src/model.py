"""The surrogate network, shared by training and inference."""
from torch import nn

N_PARAMS = 7  # k0, g, q, t_top, t_bottom, t_left, t_right


def make_mlp(n_out, width=256):
    return nn.Sequential(
        nn.Linear(N_PARAMS, width), nn.GELU(),
        nn.Linear(width, width), nn.GELU(),
        nn.Linear(width, n_out),
    )

from __future__ import annotations

import torch
from torch import nn

from src.data import INPUT_DIM

torch.set_default_device("cpu")


def build_model() -> nn.Module:
    """Build Opacus-compatible binary classifier."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from flamby.datasets.fed_heart_disease import FedHeartDisease, NUM_CLIENTS as FLAMBY_NUM_CLIENTS

torch.set_default_device("cpu")

NUM_CLIENTS: int = int(FLAMBY_NUM_CLIENTS)

_sample_dataset = FedHeartDisease(center=0, train=True, pooled=False)
_x0, _ = _sample_dataset[0]
INPUT_DIM: int = int(_x0.shape[-1])


def get_client_dataloader(client_id: int, batch_size: int = 32, train: bool = True) -> DataLoader:
    """Return per-site dataloader for Fed-Heart-Disease on CPU."""
    dataset = FedHeartDisease(center=int(client_id), train=train, pooled=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_site_sizes() -> Dict[int, int]:
    """Return training set size for each site."""
    return {cid: len(FedHeartDisease(center=cid, train=True, pooled=False)) for cid in range(NUM_CLIENTS)}

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from torch import nn

from src.metrics import bce_loss, compute_auc, compute_brier

torch.set_default_device("cpu")


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader,
        test_loader,
        local_epochs: int,
        lr: float,
        run_id: Optional[str] = None,
    ) -> None:
        self.client_id = int(client_id)
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = int(local_epochs)
        self.lr = float(lr)
        self.run_id = run_id if run_id is not None else os.environ.get("RUN_ID", "default")
        self._keys = list(self.model.state_dict().keys())
        self._init_logged = False

        try:
            params = self.get_parameters({})
            flat = np.concatenate([p.flatten() for p in params]).tobytes()
            self._init_hash = hashlib.sha1(flat).hexdigest()[:12]
        except Exception:
            self._init_hash = "unknown"

    def get_parameters(self, config) -> List[np.ndarray]:
        state = self.model.state_dict()
        return [state[k].detach().cpu().numpy() for k in self._keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state = self.model.state_dict()
        mapped = {k: torch.tensor(v, dtype=state[k].dtype) for k, v in zip(self._keys, parameters)}
        self.model.load_state_dict(mapped, strict=True)

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        if int(config.get("server_round", 0)) == 1 and not getattr(self, "_init_logged", False):
            try:
                log_path = Path("results") / "diagnostics_init.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[{self.run_id}] client_id={self.client_id} init_param_hash={self._init_hash}\n"
                    )
                self._init_logged = True
            except Exception:
                pass

        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.BCELoss()

        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x = x.float()
                y = y.float().view(-1, 1)
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        y_true_all, y_prob_all = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.float()
                y_hat = self.model(x).view(-1).detach().cpu().numpy()
                y_true = y.view(-1).detach().cpu().numpy()
                y_prob_all.append(y_hat)
                y_true_all.append(y_true)

        y_true_arr = np.concatenate(y_true_all).astype(float)
        y_prob_arr = np.concatenate(y_prob_all).astype(float)
        loss = bce_loss(y_true_arr, y_prob_arr)
        auc = compute_auc(y_true_arr, y_prob_arr)
        brier = compute_brier(y_true_arr, y_prob_arr)

        run_id = str(config.get("run_id", self.run_id))
        server_round = int(config.get("server_round", 0))
        out_dir = Path("results") / "predictions" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"round{server_round}_site{self.client_id}.npz"
        np.savez(out_path, y_true=y_true_arr, y_prob=y_prob_arr)

        return float(loss), len(self.test_loader.dataset), {"client_id": self.client_id, "auc": float(auc), "brier": float(brier)}

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.data import get_client_dataloader
from src.metrics import bce_loss, compute_auc, compute_brier, compute_calibration_intercept_slope
from src.model import build_model

torch.set_default_device("cpu")


def evaluate_global_model_per_site(
    model_path: str,
    num_clients: int,
    save_predictions: bool = False,
) -> Dict[int, Dict]:
    model_name = model_path if str(model_path).endswith(".pt") else f"{model_path}.pt"
    path = Path("results") / "models" / model_name
    preds_dir_name = str(model_path)[:-3] if str(model_path).endswith(".pt") else str(model_path)

    model = build_model()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if save_predictions:
        preds_dir = Path("results") / "predictions" / preds_dir_name
        preds_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[int, Dict] = {}
    for client_id in range(num_clients):
        loader = get_client_dataloader(client_id=client_id, train=False)
        y_true_all, y_prob_all = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.float()
                prob = model(x).view(-1).detach().cpu().numpy()
                y_true = y.view(-1).detach().cpu().numpy()
                y_prob_all.append(prob)
                y_true_all.append(y_true)

        y_true_arr = np.concatenate(y_true_all).astype(float)
        y_prob_arr = np.concatenate(y_prob_all).astype(float)
        calib_intercept, calib_slope = compute_calibration_intercept_slope(y_true_arr, y_prob_arr)
        out[client_id] = {
            "n_test": int(len(y_true_arr)),
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
            "auc": float(compute_auc(y_true_arr, y_prob_arr)),
            "brier": float(compute_brier(y_true_arr, y_prob_arr)),
            "calib_intercept": float(calib_intercept),
            "calib_slope": float(calib_slope),
            "loss": float(bce_loss(y_true_arr, y_prob_arr)),
        }

        if save_predictions:
            np.savez(
                Path("results") / "predictions" / preds_dir_name / f"site{client_id}.npz",
                y_true=y_true_arr,
                y_prob=y_prob_arr,
            )
    return out

"""Experiment 1 — FedAvg aggregation exponent sweep (α ∈ {0, 0.5, 1.0}).

Trains one global model per (α, seed), evaluates per-site test loss/AUC, and
writes ``results/exp1_results.{json,csv}``. Checkpoints use run IDs
``exp1_alpha{α}_seed{s}`` (used by Experiment 3’s default checkpoint).

Next step: ``python -m experiments.exp1_figure`` for bootstrap CI table + plot.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

torch.set_default_device("cpu")

from src.data import NUM_CLIENTS, get_site_sizes
from src.eval_per_site import evaluate_global_model_per_site
from src.strategies import AlphaWeightedFedAvg
from src.train import run_federation

ALPHAS: List[float] = [0.0, 0.5, 1.0]
SEEDS: List[int] = [0, 1, 2, 3, 4]
NUM_ROUNDS: int = 20
LOCAL_EPOCHS: int = 2
BATCH_SIZE: int = 32
LR: float = 0.01


def _to_scalar(value: Any) -> float:
    if value is None:
        return float(np.nan)
    try:
        return float(value)
    except Exception:
        return float(np.nan)


def _per_site_scalars_only(per_site: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for cid, m in per_site.items():
        out[int(cid)] = {
            "n_test": int(m["n_test"]),
            "auc": _to_scalar(m.get("auc")),
            "brier": _to_scalar(m.get("brier")),
            "calib_intercept": _to_scalar(m.get("calib_intercept")),
            "calib_slope": _to_scalar(m.get("calib_slope")),
            "loss": _to_scalar(m.get("loss")),
        }
    return out


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(v)
    if not mask.any() or float(np.sum(w[mask])) <= 0.0:
        return float(np.nan)
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def main() -> None:
    try:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        _ = get_site_sizes()
        results: List[Dict[str, Any]] = []

        for alpha in ALPHAS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                strategy = AlphaWeightedFedAvg(
                    alpha=alpha,
                    run_id=None,
                    fraction_fit=1.0,
                    fraction_evaluate=1.0,
                    min_fit_clients=NUM_CLIENTS,
                    min_evaluate_clients=NUM_CLIENTS,
                    min_available_clients=NUM_CLIENTS,
                )
                run_id = f"exp1_alpha{alpha}_seed{seed}"

                run_federation(
                    strategy=strategy,
                    num_clients=NUM_CLIENTS,
                    num_rounds=NUM_ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    run_id=run_id,
                    seed=seed,
                )

                per_site = evaluate_global_model_per_site(
                    model_path=run_id,
                    num_clients=NUM_CLIENTS,
                    save_predictions=True,
                )
                per_site_scalars = _per_site_scalars_only(per_site)

                ks = sorted(per_site.keys())
                n_tests = [int(per_site[k]["n_test"]) for k in ks]
                losses = [_to_scalar(per_site[k]["loss"]) for k in ks]
                aucs = [_to_scalar(per_site[k]["auc"]) for k in ks]

                patient_loss = _weighted_mean(losses, n_tests)
                worst_site_loss = float(np.nanmax(np.asarray(losses, dtype=float)))
                patient_auc = _weighted_mean(aucs, n_tests)
                worst_site_auc = float(np.nanmin(np.asarray(aucs, dtype=float)))

                results.append(
                    {
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "patient_loss": patient_loss,
                        "worst_site_loss": worst_site_loss,
                        "patient_auc": patient_auc,
                        "worst_site_auc": worst_site_auc,
                        "per_site": per_site_scalars,
                    }
                )

        json_path = results_dir / "exp1_results.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        flat_rows = []
        for row in results:
            flat_rows.append(
                {
                    "alpha": row["alpha"],
                    "seed": row["seed"],
                    "patient_loss": row["patient_loss"],
                    "worst_site_loss": row["worst_site_loss"],
                    "patient_auc": row["patient_auc"],
                    "worst_site_auc": row["worst_site_auc"],
                }
            )
        df = pd.DataFrame(flat_rows).sort_values(by=["alpha", "seed"]).reset_index(drop=True)
        csv_path = results_dir / "exp1_results.csv"
        df.to_csv(csv_path, index=False)

        print(f"Wrote {csv_path} ({len(df)} rows) and {json_path}")
        print(df.to_string(index=False))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

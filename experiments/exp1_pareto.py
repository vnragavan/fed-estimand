from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_default_device("cpu")

from src.data import NUM_CLIENTS, get_site_sizes
from src.eval_per_site import evaluate_global_model_per_site
from src.strategies import AlphaWeightedFedAvg
from src.train import run_federation

ALPHAS: List[float] = [0.0, 0.5, 1.0]
SEEDS: List[int] = [0, 1, 2]
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
    serializable: Dict[int, Dict[str, float]] = {}
    for cid, metrics in per_site.items():
        serializable[int(cid)] = {
            "n_test": int(metrics["n_test"]),
            "auc": _to_scalar(metrics.get("auc")),
            "brier": _to_scalar(metrics.get("brier")),
            "calib_intercept": _to_scalar(metrics.get("calib_intercept")),
            "calib_slope": _to_scalar(metrics.get("calib_slope")),
            "loss": _to_scalar(metrics.get("loss")),
        }
    return serializable


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if np.all(np.isnan(v)) or np.sum(w[~np.isnan(v)]) <= 0:
        return float(np.nan)
    mask = ~np.isnan(v)
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def main() -> None:
    try:
        results_dir = Path("results")
        figures_dir = Path("figures")
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        _ = get_site_sizes()
        results: List[Dict[str, Any]] = []

        for alpha in ALPHAS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                strategy = AlphaWeightedFedAvg(
                    alpha=alpha,
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

                per_site = evaluate_global_model_per_site(model_path=run_id, num_clients=NUM_CLIENTS)
                per_site_serializable = _per_site_scalars_only(per_site)

                n_tests = [int(per_site[k]["n_test"]) for k in sorted(per_site.keys())]
                losses = [_to_scalar(per_site[k]["loss"]) for k in sorted(per_site.keys())]
                aucs = [_to_scalar(per_site[k]["auc"]) for k in sorted(per_site.keys())]

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
                        "per_site": per_site_serializable,
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

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        colors = plt.get_cmap("tab10").colors
        color_map = {alpha: colors[i] for i, alpha in enumerate(ALPHAS)}

        for alpha in ALPHAS:
            sub = df[df["alpha"] == alpha]
            ax.scatter(
                sub["patient_loss"].values,
                sub["worst_site_loss"].values,
                s=70,
                color=color_map[alpha],
                alpha=0.4,
                edgecolors="none",
            )

        mean_points = []
        for alpha in ALPHAS:
            sub = df[df["alpha"] == alpha]
            mx = float(sub["patient_loss"].mean())
            my = float(sub["worst_site_loss"].mean())
            mean_points.append((alpha, mx, my))
            ax.scatter([mx], [my], s=200, color=color_map[alpha], alpha=1.0)

        xs = [p[1] for p in mean_points]
        ys = [p[2] for p in mean_points]
        ax.plot(xs, ys, linestyle="--", color="gray", linewidth=1.5)

        labels = {
            0.0: "alpha = 0.0  (uniform site weighting)",
            0.5: "alpha = 0.5  (sqrt sample-size weighting)",
            1.0: "alpha = 1.0  (sample-size weighting; FedAvg default)",
        }
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[a], markersize=10, label=labels[a])
            for a in ALPHAS
        ]
        ax.legend(handles=handles, loc="best")

        ax.set_title(
            "Patient-weighted vs. worst-site loss across FedAvg weighting exponents\n"
            "(Fed-Heart-Disease, 4 sites, 20 rounds)"
        )
        ax.set_xlabel("Patient-weighted test loss (lower is better)")
        ax.set_ylabel("Worst-site test loss (lower is better)")

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.annotate(
            "better",
            xy=(x_min + 0.08 * (x_max - x_min), y_min + 0.08 * (y_max - y_min)),
            xytext=(x_min + 0.22 * (x_max - x_min), y_min + 0.22 * (y_max - y_min)),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=9,
            color="black",
        )

        plt.tight_layout()
        fig_png = figures_dir / "exp1_pareto.png"
        fig_pdf = figures_dir / "exp1_pareto.pdf"
        fig.savefig(fig_png, dpi=300)
        fig.savefig(fig_pdf, dpi=300)
        plt.close(fig)

        mean_by_alpha = df.groupby("alpha", as_index=False)[["patient_loss", "worst_site_loss"]].mean()
        alpha_to_mean = {float(r["alpha"]): r for r in mean_by_alpha.to_dict(orient="records")}

        overall_patient_min = float(df["patient_loss"].min())
        overall_patient_max = float(df["patient_loss"].max())
        overall_worst_min = float(df["worst_site_loss"].min())
        overall_worst_max = float(df["worst_site_loss"].max())

        frontier_visible = (
            alpha_to_mean[0.0]["worst_site_loss"] <= alpha_to_mean[0.5]["worst_site_loss"]
            and alpha_to_mean[0.5]["worst_site_loss"] <= alpha_to_mean[1.0]["worst_site_loss"]
        )

        print("Experiment 1 — Patient-vs-Site Pareto Frontier")
        print(f"Patient-weighted loss range across alphas: [{overall_patient_min:.6f}, {overall_patient_max:.6f}]")
        print(f"Worst-site loss range across alphas: [{overall_worst_min:.6f}, {overall_worst_max:.6f}]")
        print(
            "Mean patient-weighted loss at alpha=1.0 (FedAvg default): "
            f"{alpha_to_mean[1.0]['patient_loss']:.6f}"
        )
        print(f"Mean worst-site loss at alpha=1.0: {alpha_to_mean[1.0]['worst_site_loss']:.6f}")
        print(f"Mean patient-weighted loss at alpha=0.0 (uniform): {alpha_to_mean[0.0]['patient_loss']:.6f}")
        print(f"Mean worst-site loss at alpha=0.0: {alpha_to_mean[0.0]['worst_site_loss']:.6f}")
        print(f"Frontier visible: {'YES' if frontier_visible else 'NO'}")

        print(fig_png.as_posix())
        print(df.head(10).to_string(index=False))

    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

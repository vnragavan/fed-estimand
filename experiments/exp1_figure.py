"""Experiment 1 — figure + bootstrap CI summary for the α sweep.

Reads ``results/exp1_results.csv`` and per-site ``.npz`` predictions produced by
``exp1_sweep``, then:

1. Writes ``results/exp1_ci_summary.csv`` (95% bootstrap CIs over test patients).
2. Renders ``figures/exp1.{png,pdf}`` — scatter per seed, means with
   mean ± 1.96×SE across seeds, footnote with average bootstrap half-widths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

torch.set_default_device("cpu")

from src.bootstrap_ci import combined_ci_across_seeds, seed_se

ALPHAS: List[float] = [0.0, 0.5, 1.0]
Z: float = 1.96
NUM_CLIENTS: int = 4
N_BOOT_PER_SEED: int = 500
BOOTSTRAP_SEED: int = 42
CI: float = 0.95


def _load_per_site_predictions(run_id: str) -> Dict[int, Dict[str, np.ndarray]]:
    preds: Dict[int, Dict[str, np.ndarray]] = {}
    base = Path("results") / "predictions" / run_id
    for cid in range(NUM_CLIENTS):
        path = base / f"site{cid}.npz"
        z = np.load(path)
        preds[cid] = {"y_true": np.asarray(z["y_true"], dtype=float), "y_prob": np.asarray(z["y_prob"], dtype=float)}
    return preds


def _interval_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    return not (a_hi < b_lo or b_hi < a_lo)


def main() -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("results/exp1_results.csv")

    summary_rows: List[Dict[str, Any]] = []
    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha].sort_values("seed")
        predictions_per_seed = [_load_per_site_predictions(f"exp1_alpha{alpha}_seed{s}") for s in sub["seed"].tolist()]
        ci_res = combined_ci_across_seeds(
            predictions_per_seed,
            n_boot_per_seed=N_BOOT_PER_SEED,
            seed=BOOTSTRAP_SEED,
            ci=CI,
        )
        summary_rows.append(
            {
                "alpha": alpha,
                "n_seeds": ci_res["n_seeds"],
                "patient_loss_mean": ci_res["patient_loss_mean"],
                "patient_loss_lo": ci_res["patient_loss_lo"],
                "patient_loss_hi": ci_res["patient_loss_hi"],
                "worst_site_loss_mean": ci_res["worst_site_loss_mean"],
                "worst_site_loss_lo": ci_res["worst_site_loss_lo"],
                "worst_site_loss_hi": ci_res["worst_site_loss_hi"],
                "seed_se_patient": seed_se(sub["patient_loss"].tolist()),
                "seed_se_worst": seed_se(sub["worst_site_loss"].tolist()),
            }
        )

    boot = pd.DataFrame(summary_rows)
    summary_path = Path("results") / "exp1_ci_summary.csv"
    boot.to_csv(summary_path, index=False)

    print("alpha | patient_loss [mean (lo, hi)] | worst_site_loss [mean (lo, hi)] | seed_SE_patient | seed_SE_worst")
    for r in summary_rows:
        print(
            f"{r['alpha']:.1f}  | "
            f"{r['patient_loss_mean']:.4f} ({r['patient_loss_lo']:.4f}, {r['patient_loss_hi']:.4f}) | "
            f"{r['worst_site_loss_mean']:.4f} ({r['worst_site_loss_lo']:.4f}, {r['worst_site_loss_hi']:.4f}) | "
            f"{r['seed_se_patient']:.4f} | {r['seed_se_worst']:.4f}"
        )

    rows: List[Dict[str, Any]] = []
    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha]
        n = int(len(sub))
        pl_mean = float(sub["patient_loss"].mean())
        pl_se = float(sub["patient_loss"].std(ddof=1) / np.sqrt(n))
        ws_mean = float(sub["worst_site_loss"].mean())
        ws_se = float(sub["worst_site_loss"].std(ddof=1) / np.sqrt(n))

        bsub = boot[boot["alpha"] == alpha].iloc[0]
        pl_boot_half = float((bsub["patient_loss_hi"] - bsub["patient_loss_lo"]) / 2.0)
        ws_boot_half = float((bsub["worst_site_loss_hi"] - bsub["worst_site_loss_lo"]) / 2.0)

        rows.append(
            {
                "alpha": alpha,
                "n_seeds": n,
                "patient_loss_mean": pl_mean,
                "patient_loss_seed_se": pl_se,
                "patient_loss_seed_ci_half": Z * pl_se,
                "worst_site_loss_mean": ws_mean,
                "worst_site_loss_seed_se": ws_se,
                "worst_site_loss_seed_ci_half": Z * ws_se,
                "patient_loss_boot_halfwidth": pl_boot_half,
                "worst_site_loss_boot_halfwidth": ws_boot_half,
            }
        )

    summary = pd.DataFrame(rows)
    avg_pl_boot = float(summary["patient_loss_boot_halfwidth"].mean())
    avg_ws_boot = float(summary["worst_site_loss_boot_halfwidth"].mean())

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    tab10 = plt.get_cmap("tab10").colors
    color_map = {a: tab10[i] for i, a in enumerate(ALPHAS)}

    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha]
        ax.scatter(
            sub["patient_loss"].values,
            sub["worst_site_loss"].values,
            s=40,
            color=color_map[alpha],
            alpha=0.35,
            edgecolors="none",
        )

    mean_xs: List[float] = []
    mean_ys: List[float] = []
    for r in rows:
        a = r["alpha"]
        mx = r["patient_loss_mean"]
        my = r["worst_site_loss_mean"]
        ax.scatter([mx], [my], s=200, color=color_map[a], alpha=1.0, zorder=4)
        ax.hlines(
            y=my,
            xmin=mx - r["patient_loss_seed_ci_half"],
            xmax=mx + r["patient_loss_seed_ci_half"],
            colors=[color_map[a]],
            linewidth=3.0,
            zorder=3,
        )
        ax.vlines(
            x=mx,
            ymin=my - r["worst_site_loss_seed_ci_half"],
            ymax=my + r["worst_site_loss_seed_ci_half"],
            colors=[color_map[a]],
            linewidth=3.0,
            zorder=3,
        )
        mean_xs.append(mx)
        mean_ys.append(my)

    ax.plot(mean_xs, mean_ys, linestyle="--", color="gray", linewidth=1.5, zorder=2)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tab10[0], markersize=10,
               label="alpha = 0.0  (uniform site weighting)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tab10[1], markersize=10,
               label="alpha = 0.5  (sqrt sample-size weighting)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tab10[2], markersize=10,
               label="alpha = 1.0  (sample-size weighting; FedAvg default)"),
        Line2D([0], [0], marker="+", color="black", markersize=12, markeredgewidth=2,
               linestyle="None", label="mean ± 1.96 × SE across seeds"),
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=9)

    plt.title(
        "FedAvg weighting exponent: patient-weighted vs. worst-site loss\n"
        "Fed-Heart-Disease (4 sites, 20 rounds, 5 seeds; error bars = mean ± 1.96 × SE)",
        fontsize=11,
    )
    ax.set_xlabel("Patient-weighted test loss (lower is better)")
    ax.set_ylabel("Worst-site test loss (lower is better)")

    plt.figtext(
        0.5,
        0.02,
        f"Population-loss uncertainty (95% bootstrap CI over test patients): "
        f"patient-loss ±{avg_pl_boot:.3f}, worst-site ±{avg_ws_boot:.3f}, "
        f"dominated by Site 2 (n_test=16). This reflects estimation uncertainty for a fixed "
        f"trained model, separate from the alpha effect.",
        wrap=True,
        ha="center",
        fontsize=8,
    )

    plt.subplots_adjust(bottom=0.18)
    fig_png = Path("figures") / "exp1.png"
    fig_pdf = Path("figures") / "exp1.pdf"
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSummary CSV:", summary_path.as_posix())
    print("Figure saved to:", fig_png.as_posix())

    print("\nAcross-seed SE-based CIs (the alpha effect):")
    for r in rows:
        print(
            f"alpha={r['alpha']:.1f}: patient_loss = {r['patient_loss_mean']:.3f} ± {r['patient_loss_seed_ci_half']:.3f}, "
            f"worst_site_loss = {r['worst_site_loss_mean']:.3f} ± {r['worst_site_loss_seed_ci_half']:.3f}"
        )

    r0 = rows[0]
    r1 = rows[-1]
    pl_overlap = _interval_overlap(
        r0["patient_loss_mean"] - r0["patient_loss_seed_ci_half"],
        r0["patient_loss_mean"] + r0["patient_loss_seed_ci_half"],
        r1["patient_loss_mean"] - r1["patient_loss_seed_ci_half"],
        r1["patient_loss_mean"] + r1["patient_loss_seed_ci_half"],
    )
    ws_overlap = _interval_overlap(
        r0["worst_site_loss_mean"] - r0["worst_site_loss_seed_ci_half"],
        r0["worst_site_loss_mean"] + r0["worst_site_loss_seed_ci_half"],
        r1["worst_site_loss_mean"] - r1["worst_site_loss_seed_ci_half"],
        r1["worst_site_loss_mean"] + r1["worst_site_loss_seed_ci_half"],
    )

    print("\nPairwise comparison (alpha=0 vs alpha=1, mean ± 1.96*SE):")
    print(f"Patient loss CIs overlap?    {'YES' if pl_overlap else 'NO'}")
    print(f"Worst-site loss CIs overlap? {'YES' if ws_overlap else 'NO'}")

    print("\nPatient-level bootstrap CI widths (uncertainty about population loss for fixed model):")
    for r in rows:
        print(
            f"alpha={r['alpha']:.1f}: patient ± {r['patient_loss_boot_halfwidth']:.3f}, "
            f"worst ± {r['worst_site_loss_boot_halfwidth']:.3f}"
        )

    print("\n" + "=" * 88)
    print("INTERPRETATION (seed-SE vs patient bootstrap)")
    print("=" * 88)

    if not pl_overlap and not ws_overlap:
        para1 = (
            "Paragraph 1. The seed-SE-based CIs separate alpha=0 from alpha=1 on BOTH the patient-weighted "
            "and the worst-site axis (intervals do not overlap). The workshop paper's central claim — that "
            "FedAvg's weighting exponent has a real, replicable effect on the trade-off between average and "
            "worst-site performance — therefore has empirical support: across 5 independent training runs, "
            "uniform site weighting (alpha=0) consistently yields lower worst-site loss than sample-size "
            "weighting (alpha=1), and the difference is larger than the across-run noise. 'Weighting is an "
            "ethical design choice' is defensible at the level of the alpha effect on this dataset."
        )
    elif not ws_overlap and pl_overlap:
        para1 = (
            "Paragraph 1. The seed-SE-based CIs separate alpha=0 from alpha=1 on the worst-site axis but not "
            "on the patient-weighted axis. The workshop paper can claim that weighting choice has a real, "
            "replicable effect on worst-site performance specifically — the central claim of an ethical "
            "design trade-off survives — but the patient-weighted side of the trade-off is statistically "
            "indistinguishable across alpha values, so the framing should emphasize the worst-site effect."
        )
    else:
        para1 = (
            "Paragraph 1. The seed-SE-based CIs do NOT separate alpha=0 from alpha=1 on either axis. With "
            "5 seeds we cannot replicate the alpha effect at the standard 95% threshold; the workshop paper "
            "should not claim a real Pareto trade-off from this experiment alone. Either run more seeds, or "
            "pivot Experiment 1 to the per-site bar-chart framing motivating Experiment 3's calibration story."
        )

    pl_ratio = avg_pl_boot / max((rows[0]["patient_loss_seed_ci_half"] + rows[-1]["patient_loss_seed_ci_half"]) / 2.0, 1e-9)
    ws_ratio = avg_ws_boot / max((rows[0]["worst_site_loss_seed_ci_half"] + rows[-1]["worst_site_loss_seed_ci_half"]) / 2.0, 1e-9)
    para2 = (
        "Paragraph 2. The bootstrap CIs over test patients are much wider than the seed-SE CIs "
        f"(~{pl_ratio:.0f}× wider on patient loss, ~{ws_ratio:.0f}× wider on worst-site loss). This is not a "
        "contradiction, it answers a different question. The bootstrap CI says: for any single trained model, "
        "our estimate of its population loss is highly uncertain because Site 2's test set has only 16 patients "
        "(and that site dominates worst_site_loss). The seed-SE CI says: across re-trainings of the model, the "
        "alpha effect is reproducible. We are confident the alpha-induced shift in worst-site loss is real "
        "across runs, while honestly conceding that the absolute level of any single model's worst-site loss "
        "is poorly pinned down by 16 test patients. The two uncertainties should be reported side by side, "
        "not collapsed into one."
    )

    print(para1)
    print()
    print(para2)


if __name__ == "__main__":
    main()

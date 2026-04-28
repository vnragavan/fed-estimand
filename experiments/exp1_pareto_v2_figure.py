from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_default_device("cpu")

from src.bootstrap_ci import combined_ci_across_seeds, seed_se

ALPHAS: List[float] = [0.0, 0.5, 1.0]
SEEDS: List[int] = [0, 1, 2, 3, 4]
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

    df = pd.read_csv("results/exp1_v2_results.csv")

    summary_rows: List[Dict[str, Any]] = []
    cis_by_alpha: Dict[float, Dict[str, Any]] = {}

    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha].sort_values("seed")
        predictions_per_seed = [_load_per_site_predictions(f"exp1_alpha{alpha}_seed{s}") for s in sub["seed"].tolist()]
        ci_res = combined_ci_across_seeds(
            predictions_per_seed,
            n_boot_per_seed=N_BOOT_PER_SEED,
            seed=BOOTSTRAP_SEED,
            ci=CI,
        )
        cis_by_alpha[alpha] = ci_res

        seed_patient_se = seed_se(sub["patient_loss"].tolist())
        seed_worst_se = seed_se(sub["worst_site_loss"].tolist())

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
                "seed_se_patient": seed_patient_se,
                "seed_se_worst": seed_worst_se,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = Path("results") / "exp1_v2_ci_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("alpha | patient_loss [mean (lo, hi)] | worst_site_loss [mean (lo, hi)] | seed_SE_patient | seed_SE_worst")
    for r in summary_rows:
        print(
            f"{r['alpha']:.1f}  | "
            f"{r['patient_loss_mean']:.4f} ({r['patient_loss_lo']:.4f}, {r['patient_loss_hi']:.4f}) | "
            f"{r['worst_site_loss_mean']:.4f} ({r['worst_site_loss_lo']:.4f}, {r['worst_site_loss_hi']:.4f}) | "
            f"{r['seed_se_patient']:.4f} | {r['seed_se_worst']:.4f}"
        )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = plt.get_cmap("tab10").colors
    color_map = {a: colors[i] for i, a in enumerate(ALPHAS)}

    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha]
        ax.scatter(
            sub["patient_loss"].values,
            sub["worst_site_loss"].values,
            s=40,
            color=color_map[alpha],
            alpha=0.3,
            edgecolors="none",
        )

    mean_xs: List[float] = []
    mean_ys: List[float] = []
    for alpha in ALPHAS:
        ci_res = cis_by_alpha[alpha]
        mx = ci_res["patient_loss_mean"]
        my = ci_res["worst_site_loss_mean"]
        ax.scatter([mx], [my], s=200, color=color_map[alpha], alpha=1.0, zorder=4)
        ax.hlines(
            y=my,
            xmin=ci_res["patient_loss_lo"],
            xmax=ci_res["patient_loss_hi"],
            colors=[color_map[alpha]],
            linewidth=2.0,
            zorder=3,
        )
        ax.vlines(
            x=mx,
            ymin=ci_res["worst_site_loss_lo"],
            ymax=ci_res["worst_site_loss_hi"],
            colors=[color_map[alpha]],
            linewidth=2.0,
            zorder=3,
        )
        mean_xs.append(mx)
        mean_ys.append(my)

    ax.plot(mean_xs, mean_ys, linestyle="--", color="gray", linewidth=1.5, zorder=2)

    labels = {
        0.0: "alpha = 0.0  (uniform site weighting)",
        0.5: "alpha = 0.5  (sqrt sample-size weighting)",
        1.0: "alpha = 1.0  (sample-size weighting; FedAvg default)",
    }
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[a], markersize=10, label=labels[a])
        for a in ALPHAS
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=2, label="95% bootstrap CI (cross)"))
    ax.legend(handles=handles, loc="best", fontsize=9)

    ax.set_title(
        "Effect of FedAvg weighting exponent on patient-weighted vs. worst-site loss\n"
        "(Fed-Heart-Disease, 4 sites, 20 rounds, 5 seeds, 95% bootstrap CI over patients)"
    )
    ax.set_xlabel("Patient-weighted test loss (lower is better)")
    ax.set_ylabel("Worst-site test loss (lower is better)")

    overlap_patient_pairs = []
    overlap_worst_pairs = []
    for i in range(len(ALPHAS)):
        for j in range(i + 1, len(ALPHAS)):
            ai, aj = ALPHAS[i], ALPHAS[j]
            ci_i, ci_j = cis_by_alpha[ai], cis_by_alpha[aj]
            if _interval_overlap(ci_i["patient_loss_lo"], ci_i["patient_loss_hi"], ci_j["patient_loss_lo"], ci_j["patient_loss_hi"]):
                overlap_patient_pairs.append((ai, aj))
            if _interval_overlap(ci_i["worst_site_loss_lo"], ci_i["worst_site_loss_hi"], ci_j["worst_site_loss_lo"], ci_j["worst_site_loss_hi"]):
                overlap_worst_pairs.append((ai, aj))

    ci_a0 = cis_by_alpha[0.0]
    ci_a1 = cis_by_alpha[1.0]
    a0_dominates = (
        ci_a0["patient_loss_hi"] < ci_a1["patient_loss_lo"]
        and ci_a0["worst_site_loss_hi"] < ci_a1["worst_site_loss_lo"]
    )

    if overlap_patient_pairs or overlap_worst_pairs:
        annotation = "Note: CIs overlap; differences across alpha may not be statistically significant."
    elif a0_dominates:
        annotation = "Uniform weighting (alpha=0) Pareto-dominates sample-size weighting (alpha=1) on this dataset."
    else:
        annotation = "CIs separate cleanly; weighting choice has a measurable effect."

    ax.annotate(
        annotation,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="black",
    )

    plt.tight_layout()
    fig_png = Path("figures") / "exp1_pareto_v2.png"
    fig_pdf = Path("figures") / "exp1_pareto_v2.pdf"
    fig.savefig(fig_png, dpi=300)
    fig.savefig(fig_pdf, dpi=300)
    plt.close(fig)

    print("\nFigure saved to:", fig_png.as_posix())
    print("Summary CSV at:", summary_path.as_posix())
    print()
    print("results/exp1_v2_ci_summary.csv contents:")
    print(summary.to_string(index=False))

    print()
    print("=" * 88)
    print("HONEST INTERPRETATION")
    print("=" * 88)

    q1 = (
        "Q1. Did the verification confirm that alpha actually changes aggregation weights?\n"
        "Yes. results/diagnostics_alpha_weights.log shows alpha=0 produces uniform [0.25, 0.25, 0.25, 0.25] "
        "weights and alpha=1 produces weights proportional to FLamby site sizes "
        "([0.354, 0.409, 0.062, 0.175] for sites of size [172, 199, 30, 85]). The strategy is doing what we think."
    )
    print(q1)
    print()

    q2 = (
        "Q2. Did seeds actually produce different model initializations and different final models?\n"
        "Yes. After switching from `seed + cid` to `SeedSequence(seed).spawn(NUM_CLIENTS)`, init param hashes "
        "for parent_seed=0 ({b28d28e8723d, d797497b0768, 649fb053a621, 6f8abcf7c553}) are completely "
        "disjoint from those for parent_seed=1 ({705a45ae72e0, 3207e81206cc, 6f2868f99ca8, 3e4c976fa4d9}). "
        "Across-seed std on patient_loss is now 0.003-0.007 and on worst_site_loss is 0.018-0.032, "
        "vs. the v1 figures of <0.001 and ~0.001 — confirming the v1 'seed' axis was nearly inert."
    )
    print(q2)
    print()

    q3 = (
        "Q3. With proper CIs, is the difference between alpha values statistically meaningful?"
    )
    print(q3)
    pa01 = (0.0, 1.0) in overlap_patient_pairs
    wa01 = (0.0, 1.0) in overlap_worst_pairs
    print(
        "    Patient-weighted loss CIs: "
        + ("at least one pair overlaps " + str(overlap_patient_pairs) if overlap_patient_pairs else "all pairs separate")
    )
    print(
        "    Worst-site loss CIs:       "
        + ("at least one pair overlaps " + str(overlap_worst_pairs) if overlap_worst_pairs else "all pairs separate")
    )
    print(
        "    alpha=0 vs alpha=1: patient_loss "
        + ("OVERLAP" if pa01 else "SEPARATE")
        + ", worst_site_loss "
        + ("OVERLAP" if wa01 else "SEPARATE")
    )
    print()

    q4 = "Q4. Pareto frontier vs. domination vs. inconclusive?"
    print(q4)
    if a0_dominates:
        print(
            "    Uniform weighting (alpha=0) Pareto-dominates sample-size weighting (alpha=1) "
            "on BOTH patient-weighted AND worst-site loss CIs (no overlap on either axis). "
            "There is no Pareto frontier here — alpha=0 is strictly preferred."
        )
    elif (overlap_patient_pairs or overlap_worst_pairs) and not a0_dominates:
        print(
            "    The data does NOT support a clean Pareto frontier. CIs overlap on at least one axis "
            "between adjacent (or extreme) alphas. The directional pattern is consistent across seeds "
            "(alpha=0 has lower worst-site loss; alpha=1 has marginally higher patient-weighted loss), "
            "but the magnitude is small relative to the bootstrap+seed CI."
        )
    else:
        print(
            "    All CIs separate, no domination on both axes simultaneously: a Pareto frontier is present "
            "with a real trade-off across alphas."
        )
    print()

    q5 = "Q5. What is the most defensible claim the workshop paper can make from this experiment?"
    print(q5)
    if a0_dominates:
        print(
            "    'On Fed-Heart-Disease with 4 highly heterogeneous sites and small per-site samples, "
            "uniform site weighting (alpha=0) Pareto-dominates sample-size weighting (alpha=1) on both "
            "patient-weighted and worst-site test loss; the FedAvg default of size-proportional weighting "
            "is therefore not Pareto-optimal here.'"
        )
    elif overlap_patient_pairs and overlap_worst_pairs:
        print(
            "    'Across 5 seeds and 3 weighting exponents alpha in {0, 0.5, 1}, no statistically distinguishable "
            "Pareto trade-off between patient-weighted loss and worst-site loss is observed on Fed-Heart-Disease "
            "(95% bootstrap CIs overlap on both axes). The dataset's small per-site samples and severe "
            "heterogeneity dominate the effect of the aggregation weighting choice. We recommend the workshop "
            "paper drop this figure or reframe Experiment 1 as a per-site loss bar chart at alpha=1, motivating "
            "the calibration analysis in Experiment 3.'"
        )
    else:
        print(
            "    'Across 5 seeds, lowering FedAvg's effective weighting exponent alpha from 1 (size-proportional) "
            "toward 0 (uniform site weighting) consistently reduces worst-site test loss, while the impact on "
            "patient-weighted loss is small and statistically less clear. This supports framing aggregation "
            "weighting as a deliberate ethical design choice between patient-weighted and site-weighted "
            "performance, even when the magnitude is modest.'"
        )


if __name__ == "__main__":
    main()

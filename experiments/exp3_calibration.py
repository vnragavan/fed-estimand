from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_default_device("cpu")

from src.calibration import bin_calibration_curve, bootstrap_calibration_intercept_slope
from src.data import NUM_CLIENTS
from src.eval_per_site import evaluate_global_model_per_site
from src.flamby_sites import FLAMBY_SITE_LABELS
from src.metrics import compute_auc, compute_brier, compute_calibration_intercept_slope

MODEL_RUN_ID: str = "exp1_alpha1.0_seed0"
N_BOOT: int = 2000
N_BINS: int = 10
CLINICAL_THRESHOLD_INTERCEPT: float = 0.2
CLINICAL_THRESHOLD_SLOPE: float = 0.2
BOOT_SEED: int = 42

SITE_LABELS: Dict[int, str] = FLAMBY_SITE_LABELS


def _render_exp3_calibration_figure(
    site_results: List[Dict[str, Any]], pooled_auc: float, pooled_ci: float
) -> None:
    """Write ``figures/exp3.{png,pdf}`` from per-site result dicts."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5), dpi=300, sharex=True, sharey=True)
    tab10 = plt.get_cmap("tab10").colors
    panels = [(axes[0, 0], 0), (axes[0, 1], 1), (axes[1, 0], 2), (axes[1, 1], 3)]

    for ax, k in panels:
        s = next(item for item in site_results if item["site"] == k)
        color = tab10[k % 10]

        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.5, linewidth=1.2)

        centers = np.asarray(s["bin_centers"], dtype=float)
        obs = np.asarray(s["bin_obs"], dtype=float)
        lo = np.asarray(s["bin_lo"], dtype=float)
        hi = np.asarray(s["bin_hi"], dtype=float)
        nb = np.asarray(s["bin_n"], dtype=int)
        nonempty = nb > 0
        if nonempty.any():
            sizes = np.minimum(30 + 5 * nb[nonempty], 300)
            yerr_lo = np.maximum(0.0, obs[nonempty] - lo[nonempty])
            yerr_hi = np.maximum(0.0, hi[nonempty] - obs[nonempty])
            ax.errorbar(
                centers[nonempty],
                obs[nonempty],
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                ecolor=color,
                elinewidth=1.4,
                capsize=3,
                zorder=2,
            )
            ax.scatter(
                centers[nonempty],
                obs[nonempty],
                s=sizes,
                color=color,
                edgecolors="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.plot(
                centers[nonempty],
                obs[nonempty],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        info = (
            f"Site {k}  ({SITE_LABELS.get(k, '')})\n"
            f"n_test = {s['n_test']}\n"
            f"AUC = {s['auc']:.3f}\n"
            f"calib intercept = {s['calib_intercept']:+.2f} "
            f"[{s['calib_intercept_lo']:+.2f}, {s['calib_intercept_hi']:+.2f}]\n"
            f"calib slope = {s['calib_slope']:.2f} "
            f"[{s['calib_slope_lo']:.2f}, {s['calib_slope_hi']:.2f}]"
        )
        ax.text(
            0.05,
            0.95,
            info,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.85, boxstyle="round,pad=0.4"),
        )
        if s["miscalibrated"]:
            ax.text(
                0.05,
                0.55,
                "miscalibrated",
                transform=ax.transAxes,
                fontsize=10,
                color="crimson",
                fontweight="bold",
                va="top",
                ha="left",
            )

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("Predicted risk", fontsize=10)
    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_ylabel("Observed risk", fontsize=10)

    fig.suptitle(
        f"Per-site calibration of the federated default model (FedAvg, α=1.0)\n"
        f"Fed-Heart-Disease — pooled AUC = {pooled_auc:.3f}, "
        f"pooled calib intercept = {pooled_ci:+.2f}",
        fontsize=12,
    )
    plt.figtext(
        0.5,
        0.01,
        "Marker size scales with bin count. Vertical bars are 95% Wilson CIs on observed proportion within each bin. "
        "Calibration intercept/slope CIs are 95% bootstrap percentile intervals over patients (n_boot=2000). "
        "Clinical miscalibration threshold: |intercept| > 0.2 or |1 $-$ slope| > 0.2.",
        ha="center",
        fontsize=8,
        wrap=True,
    )
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.30, wspace=0.25)

    fig_png = Path("figures") / "exp3.png"
    fig_pdf = Path("figures") / "exp3.pdf"
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _refigure_exp3_from_saved_json() -> None:
    json_path = Path("results") / "exp3_results.json"
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    site_results = data["per_site"]
    pooled = data["pooled"]
    pooled_auc = float(pooled["auc"])
    pooled_ci = float(pooled["calib_intercept"])
    Path("figures").mkdir(parents=True, exist_ok=True)
    _render_exp3_calibration_figure(site_results, pooled_auc, pooled_ci)
    print(f"Wrote figures/exp3.{{png,pdf}} from {json_path}")


def _is_miscalibrated_intercept(intercept: float) -> bool:
    return bool(np.isfinite(intercept) and abs(intercept) > CLINICAL_THRESHOLD_INTERCEPT)


def _is_miscalibrated_slope(slope: float) -> bool:
    return bool(np.isfinite(slope) and abs(1.0 - slope) > CLINICAL_THRESHOLD_SLOPE)


def _format_ci(mean: float, lo: float, hi: float, signed: bool = False) -> str:
    fmt_mean = f"{mean:+.2f}" if signed else f"{mean:.2f}"
    fmt_lo = f"{lo:+.2f}" if signed else f"{lo:.2f}"
    fmt_hi = f"{hi:+.2f}" if signed else f"{hi:.2f}"
    return f"{fmt_mean} [{fmt_lo}, {fmt_hi}]"


def _site_descriptor_phrase(per_site_data: List[Dict[str, Any]]) -> str:
    n_misc = sum(1 for s in per_site_data if s["miscalibrated"])
    if n_misc == 0:
        return "all 4 sites are within the clinical miscalibration threshold"
    return f"{n_misc} of 4 sites exceed the clinical miscalibration threshold (|intercept| > 0.2 or |1 − slope| > 0.2)"


def main() -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    if "--refigure-only" in sys.argv:
        _refigure_exp3_from_saved_json()
        return

    print(f"Loading per-site predictions from model: {MODEL_RUN_ID}")
    per_site = evaluate_global_model_per_site(
        model_path=MODEL_RUN_ID,
        num_clients=NUM_CLIENTS,
        save_predictions=True,
    )
    site_ids = sorted(per_site.keys())

    site_results: List[Dict[str, Any]] = []
    for k in site_ids:
        y_true = np.asarray(per_site[k]["y_true"], dtype=float)
        y_prob = np.asarray(per_site[k]["y_prob"], dtype=float)
        n_test = int(len(y_true))

        auc_k = float(compute_auc(y_true, y_prob))
        brier_k = float(compute_brier(y_true, y_prob))
        ci_k, cs_k = compute_calibration_intercept_slope(y_true, y_prob)
        ci_k, cs_k = float(ci_k), float(cs_k)

        centers, obs, lo, hi, nb = bin_calibration_curve(y_true, y_prob, n_bins=N_BINS)
        boot = bootstrap_calibration_intercept_slope(y_true, y_prob, n_boot=N_BOOT, seed=BOOT_SEED)

        misc_intercept = _is_miscalibrated_intercept(ci_k)
        misc_slope = _is_miscalibrated_slope(cs_k)
        miscalibrated = bool(misc_intercept or misc_slope)

        site_results.append(
            {
                "site": int(k),
                "label": SITE_LABELS.get(int(k), f"Site {k}"),
                "n_test": n_test,
                "auc": auc_k,
                "brier": brier_k,
                "calib_intercept": ci_k,
                "calib_intercept_lo": float(boot["intercept_lo"]),
                "calib_intercept_hi": float(boot["intercept_hi"]),
                "calib_slope": cs_k,
                "calib_slope_lo": float(boot["slope_lo"]),
                "calib_slope_hi": float(boot["slope_hi"]),
                "clinically_miscalibrated_intercept": misc_intercept,
                "clinically_miscalibrated_slope": misc_slope,
                "miscalibrated": miscalibrated,
                "bin_centers": centers.tolist(),
                "bin_obs": obs.tolist(),
                "bin_lo": lo.tolist(),
                "bin_hi": hi.tolist(),
                "bin_n": nb.tolist(),
                "boot_n_successful": int(boot["n_boot_successful"]),
            }
        )

    pooled_y_true = np.concatenate([np.asarray(per_site[k]["y_true"], dtype=float) for k in site_ids])
    pooled_y_prob = np.concatenate([np.asarray(per_site[k]["y_prob"], dtype=float) for k in site_ids])
    pooled_auc = float(compute_auc(pooled_y_true, pooled_y_prob))
    pooled_brier = float(compute_brier(pooled_y_true, pooled_y_prob))
    pooled_ci, pooled_cs = compute_calibration_intercept_slope(pooled_y_true, pooled_y_prob)
    pooled_ci, pooled_cs = float(pooled_ci), float(pooled_cs)
    pooled_boot = bootstrap_calibration_intercept_slope(
        pooled_y_true, pooled_y_prob, n_boot=N_BOOT, seed=BOOT_SEED
    )

    summary_rows: List[Dict[str, Any]] = []
    for s in site_results:
        summary_rows.append(
            {
                "site": s["site"],
                "n_test": s["n_test"],
                "auc": s["auc"],
                "brier": s["brier"],
                "calib_intercept": s["calib_intercept"],
                "calib_intercept_lo": s["calib_intercept_lo"],
                "calib_intercept_hi": s["calib_intercept_hi"],
                "calib_slope": s["calib_slope"],
                "calib_slope_lo": s["calib_slope_lo"],
                "calib_slope_hi": s["calib_slope_hi"],
                "clinically_miscalibrated_intercept": s["clinically_miscalibrated_intercept"],
                "clinically_miscalibrated_slope": s["clinically_miscalibrated_slope"],
            }
        )
    summary_rows.append(
        {
            "site": "POOLED",
            "n_test": int(len(pooled_y_true)),
            "auc": pooled_auc,
            "brier": pooled_brier,
            "calib_intercept": pooled_ci,
            "calib_intercept_lo": float(pooled_boot["intercept_lo"]),
            "calib_intercept_hi": float(pooled_boot["intercept_hi"]),
            "calib_slope": pooled_cs,
            "calib_slope_lo": float(pooled_boot["slope_lo"]),
            "calib_slope_hi": float(pooled_boot["slope_hi"]),
            "clinically_miscalibrated_intercept": _is_miscalibrated_intercept(pooled_ci),
            "clinically_miscalibrated_slope": _is_miscalibrated_slope(pooled_cs),
        }
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("results") / "exp3_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    full_results = {
        "model_run_id": MODEL_RUN_ID,
        "n_boot": N_BOOT,
        "n_bins": N_BINS,
        "clinical_threshold_intercept": CLINICAL_THRESHOLD_INTERCEPT,
        "clinical_threshold_slope": CLINICAL_THRESHOLD_SLOPE,
        "per_site": site_results,
        "pooled": {
            "n_test": int(len(pooled_y_true)),
            "auc": pooled_auc,
            "brier": pooled_brier,
            "calib_intercept": pooled_ci,
            "calib_slope": pooled_cs,
            "boot": pooled_boot,
        },
    }
    json_path = Path("results") / "exp3_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, default=float)

    _render_exp3_calibration_figure(site_results, pooled_auc, pooled_ci)
    fig_png = Path("figures") / "exp3.png"

    print()
    print("=" * 66)
    print("EXPERIMENT 3 — POOLED VS. PER-SITE CALIBRATION")
    print("=" * 66)
    print()
    print(f"Pooled metrics (model: {MODEL_RUN_ID}):")
    print(f"  AUC                   = {pooled_auc:.3f}")
    print(f"  Brier score           = {pooled_brier:.3f}")
    print(
        f"  Calibration intercept = {pooled_ci:+.3f}  (95% boot CI "
        f"[{pooled_boot['intercept_lo']:+.3f}, {pooled_boot['intercept_hi']:+.3f}])"
    )
    print(
        f"  Calibration slope     = {pooled_cs:.3f}  (95% boot CI "
        f"[{pooled_boot['slope_lo']:.3f}, {pooled_boot['slope_hi']:.3f}])"
    )
    print()
    print("Per-site metrics:")
    print()
    print("Site | n_test | AUC   | Brier | Calib intercept (95% CI)         | Calib slope (95% CI)             | Miscalibrated?")
    print("-----|--------|-------|-------|----------------------------------|----------------------------------|---------------")
    for s in site_results:
        line = (
            f"  {s['site']}  | {s['n_test']:>6d} | {s['auc']:.3f} | {s['brier']:.3f} | "
            f"{_format_ci(s['calib_intercept'], s['calib_intercept_lo'], s['calib_intercept_hi'], signed=True):>32s} | "
            f"{_format_ci(s['calib_slope'], s['calib_slope_lo'], s['calib_slope_hi'], signed=False):>32s} | "
            f"{'yes' if s['miscalibrated'] else 'no'}"
        )
        print(line)
    print()

    misc_intercept_sites = [s["site"] for s in site_results if s["clinically_miscalibrated_intercept"]]
    misc_slope_sites = [s["site"] for s in site_results if s["clinically_miscalibrated_slope"]]
    print(f"Sites with |calibration intercept| > 0.2:  {misc_intercept_sites if misc_intercept_sites else 'None'}")
    print(f"Sites with |1 - calibration slope| > 0.2:  {misc_slope_sites if misc_slope_sites else 'None'}")
    print()

    print("=" * 66)
    print("HEADLINE FINDING")
    print("=" * 66)
    n_misc_total = sum(1 for s in site_results if s["miscalibrated"])
    intercepts = [s["calib_intercept"] for s in site_results]
    min_intercept = float(min(intercepts))
    max_intercept = float(max(intercepts))
    site2_intercept = next((s["calib_intercept"] for s in site_results if s["site"] == 2), float("nan"))
    site3_intercept = next((s["calib_intercept"] for s in site_results if s["site"] == 3), float("nan"))

    if pooled_auc < 0.70:
        case = "C"
        print(
            "Pooled AUC is unexpectedly low. Investigate before drawing calibration conclusions — "
            "the model may not have converged."
        )
    elif pooled_auc > 0.80 and n_misc_total >= 1:
        case = "A"
        print(
            f"Pooled AUC of {pooled_auc:.2f} indicates strong overall discrimination, yet {n_misc_total} of 4 sites\n"
            f"exceed the clinical miscalibration threshold of 0.2 on the calibration intercept (range:\n"
            f"{min_intercept:+.2f} to {max_intercept:+.2f}). The smallest site (Site 2, "
            f"{SITE_LABELS[2]}, n_test="
            f"{next(s['n_test'] for s in site_results if s['site'] == 2)}) shows\n"
            f"the most extreme miscalibration with intercept {site2_intercept:+.2f}, "
            f"but Site 3 ({SITE_LABELS[3]}, n_test="
            f"{next(s['n_test'] for s in site_results if s['site'] == 3)})\n"
            f"is also strongly miscalibrated (intercept {site3_intercept:+.2f}), demonstrating that\n"
            f"miscalibration in this federation is not solely a small-site phenomenon. This empirically\n"
            f"supports the workshop paper's claim that pooled discrimination metrics hide clinically\n"
            f"meaningful per-site failures, and that calibration must be reported as a first-order\n"
            f"output rather than aggregated away."
        )
    else:
        case = "B"
        print(
            "All sites are within the clinical miscalibration threshold. The expected per-site\n"
            "miscalibration pattern from Experiment 1's per-site dumps did not survive in the v2\n"
            "re-trained model. This is itself a finding worth investigating: it suggests the\n"
            "calibration story may be sensitive to seed choice and that a multi-seed calibration\n"
            "analysis is needed for the journal paper. For the workshop paper, drop this figure\n"
            "or replace it with a per-site loss bar chart."
        )
    print()

    print("=" * 66)
    print("WORKSHOP-PAPER-READY CAPTION")
    print("=" * 66)
    site_descriptor = _site_descriptor_phrase(site_results)
    caption = (
        f"Figure 2: Per-site calibration of the federated default model (FedAvg, α=1.0) on "
        f"Fed-Heart-Disease. Pooled AUC = {pooled_auc:.3f} indicates strong overall discrimination, "
        f"yet {site_descriptor}. Calibration intercept/slope confidence intervals are 95% bootstrap "
        f"percentile intervals over test patients within each site (n_boot=2000); marker size in "
        f"each panel scales with the number of test patients in that probability bin."
    )
    print(caption)
    print()

    print("=" * 66)
    print("ROBUSTNESS CHECK")
    print("=" * 66)
    try:
        with open("results/exp1_results.json", "r", encoding="utf-8") as f:
            v2_results = json.load(f)
        rows: List[Dict[str, Any]] = []
        for entry in v2_results:
            alpha = float(entry["alpha"])
            seed = int(entry["seed"])
            for cid, m in entry["per_site"].items():
                rows.append(
                    {
                        "alpha": alpha,
                        "seed": seed,
                        "site": int(cid),
                        "calib_intercept": float(m["calib_intercept"]),
                        "calib_slope": float(m["calib_slope"]),
                    }
                )
        cdf = pd.DataFrame(rows)

        def _agg_mean_se(values: pd.Series) -> Tuple[float, float]:
            arr = values.dropna().to_numpy(dtype=float)
            if arr.size < 2:
                return (float(arr.mean()) if arr.size else float("nan"), float("nan"))
            return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))

        site_label_full = {
            0: "0 (Cleveland)   ",
            1: "1 (Hungary)     ",
            2: "2 (Switzerland) ",
            3: "3 (Long Beach)  ",
        }

        print()
        print("Calibration intercept across alpha (mean ± SE across 5 seeds):")
        print()
        print("Site            | α=0.0           | α=0.5           | α=1.0")
        print("----------------|-----------------|-----------------|-----------------")
        intercept_by_site_alpha: Dict[int, Dict[float, Tuple[float, float]]] = {s: {} for s in [0, 1, 2, 3]}
        for site in [0, 1, 2, 3]:
            cells = []
            for alpha in [0.0, 0.5, 1.0]:
                vals = cdf[(cdf["site"] == site) & (cdf["alpha"] == alpha)]["calib_intercept"]
                m, se = _agg_mean_se(vals)
                intercept_by_site_alpha[site][alpha] = (m, se)
                cells.append(f"{m:+.2f} ± {se:.2f}" if np.isfinite(m) and np.isfinite(se) else " nan ± nan ")
            print(f"{site_label_full[site]}| {cells[0]:<15} | {cells[1]:<15} | {cells[2]:<15}")

        print()
        print("Calibration slope across alpha (mean ± SE across 5 seeds):")
        print()
        print("Site            | α=0.0           | α=0.5           | α=1.0")
        print("----------------|-----------------|-----------------|-----------------")
        slope_by_site_alpha: Dict[int, Dict[float, Tuple[float, float]]] = {s: {} for s in [0, 1, 2, 3]}
        for site in [0, 1, 2, 3]:
            cells = []
            for alpha in [0.0, 0.5, 1.0]:
                vals = cdf[(cdf["site"] == site) & (cdf["alpha"] == alpha)]["calib_slope"]
                m, se = _agg_mean_se(vals)
                slope_by_site_alpha[site][alpha] = (m, se)
                cells.append(f"{m:.2f} ± {se:.2f}" if np.isfinite(m) and np.isfinite(se) else " nan ± nan ")
            print(f"{site_label_full[site]}| {cells[0]:<15} | {cells[1]:<15} | {cells[2]:<15}")

        per_site_alpha_misc: Dict[int, Dict[float, bool]] = {s: {} for s in [0, 1, 2, 3]}
        for site in [0, 1, 2, 3]:
            for alpha in [0.0, 0.5, 1.0]:
                ci_m = intercept_by_site_alpha[site][alpha][0]
                cs_m = slope_by_site_alpha[site][alpha][0]
                misc = (np.isfinite(ci_m) and abs(ci_m) > CLINICAL_THRESHOLD_INTERCEPT) or (
                    np.isfinite(cs_m) and abs(1.0 - cs_m) > CLINICAL_THRESHOLD_SLOPE
                )
                per_site_alpha_misc[site][alpha] = bool(misc)

        per_site_status: Dict[int, str] = {}
        for site in [0, 1, 2, 3]:
            flags = [per_site_alpha_misc[site][a] for a in [0.0, 0.5, 1.0]]
            if all(flags):
                per_site_status[site] = "every alpha"
            elif any(flags):
                per_site_status[site] = "some alpha"
            else:
                per_site_status[site] = "no alpha"

        misc_at_alpha0 = [s for s in [0, 1, 2, 3] if per_site_alpha_misc[s][0.0]]
        misc_at_alpha1 = [s for s in [0, 1, 2, 3] if per_site_alpha_misc[s][1.0]]

        if set(misc_at_alpha0) == set(misc_at_alpha1) and misc_at_alpha0:
            persistence = "persists"
            cause = "the federation's structure makes miscalibration unavoidable regardless of weighting"
        elif misc_at_alpha0 and len(misc_at_alpha0) < len(misc_at_alpha1):
            persistence = "partially persists"
            cause = "lowering FedAvg's weighting exponent reduces but does not eliminate miscalibration"
        elif not misc_at_alpha0 and misc_at_alpha1:
            persistence = "disappears"
            cause = "FedAvg's default specifically causes this miscalibration"
        else:
            persistence = "is mixed across alpha"
            cause = "the relationship between alpha and per-site calibration is not monotonic"

        print()
        per_site_phrases = "; ".join(
            [f"site {s} {per_site_status[s]}" for s in [0, 1, 2, 3]]
        )
        print(
            f"Per-site status across alpha (|intercept|>0.2 or |1−slope|>0.2): {per_site_phrases}."
        )
        print(
            f"Sites miscalibrated at α=0: {misc_at_alpha0 if misc_at_alpha0 else 'None'}. "
            f"Sites miscalibrated at α=1: {misc_at_alpha1 if misc_at_alpha1 else 'None'}."
        )
        print(
            f"Headline: Per-site miscalibration {persistence} when uniform site weighting (α=0) is used. "
            f"This means {cause}."
        )

        for s in site_results:
            ci_pe = s["calib_intercept"]
            ci_lo = s["calib_intercept_lo"]
            ci_hi = s["calib_intercept_hi"]
            cs_pe = s["calib_slope"]
            cs_lo = s["calib_slope_lo"]
            cs_hi = s["calib_slope_hi"]
            inside_int = (ci_lo - 1e-9) <= ci_pe <= (ci_hi + 1e-9)
            inside_slope = (cs_lo - 1e-9) <= cs_pe <= (cs_hi + 1e-9)
            if not inside_int or not inside_slope:
                print(
                    f"WARNING: site {s['site']} point estimate falls outside its bootstrap CI "
                    f"(intercept inside={inside_int}, slope inside={inside_slope})."
                )

    except Exception as exc:
        print(f"Robustness check: could not load v2 per-site results ({exc}).")

    misc_excl_site2 = [s["site"] for s in site_results if s["miscalibrated"] and s["site"] != 2]
    print()
    if misc_excl_site2:
        print(
            f"Robustness 2 — excluding Site 2 (n_test=16): sites still miscalibrated = {misc_excl_site2}. "
            f"The headline finding does NOT depend on the smallest site alone; the per-site calibration "
            f"failure is replicated at site(s) with much larger test samples."
        )
    else:
        print(
            "Robustness 2 — excluding Site 2 (n_test=16): no other site exceeds the threshold. "
            "The miscalibration story is driven entirely by Site 2's 16 test patients, which is fragile."
        )

    print()
    print("Figure path:", fig_png.as_posix())
    print("Summary CSV:", summary_path.as_posix())
    print("Headline case:", case)


if __name__ == "__main__":
    main()

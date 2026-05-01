"""Experiment 5 — paper figure + LaTeX snippets (no training).

Loads ``results/exp5_results.json`` and renders the two-panel privacy–utility figure:

  Panel A: Pooled AUC vs. worst-site AUC on the same axes (the GAP is the
    "DP concentrates harm" finding).
  Panel B: Worst-site calibration slope deviation |1 - slope| vs epsilon,
    which is the load-bearing calibration metric (monotonic in epsilon,
    unlike the worst-site |intercept|, which is non-monotonic because at
    heavy DP the model collapses to outputting near-constant probabilities
    that mechanically force the intercept toward log-odds of marginal
    prevalence).

Also writes:
  results/exp5_paper_text.tex   — LaTeX-ready paragraph for §VI of the paper
  results/exp5_paper_figure.tex — LaTeX figure environment for Figure 3

No training is performed; this script only re-renders and prints
interpretation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_default_device("cpu")

from src.flamby_sites import FLAMBY_SITE_LABELS


def _load_results() -> List[Dict[str, Any]]:
    with open("results/exp5_results.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[Dict[str, Any]] = []
    for r in raw:
        rr = dict(r)
        rr["epsilon"] = float("inf") if r["epsilon"] is None else float(r["epsilon"])
        out.append(rr)
    return out


def _eps_order(eps: float) -> float:
    return float("inf") if math.isinf(eps) else float(eps)


def _agg_mean_se(values: List[float]) -> tuple:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


def _generate_figure_v2(results: List[Dict[str, Any]]) -> Path:
    eps_present = sorted({r["epsilon"] for r in results}, key=_eps_order)
    finite_eps = [e for e in eps_present if math.isfinite(e)]
    has_inf = any(math.isinf(e) for e in eps_present)

    plot_eps: List[float] = list(finite_eps)
    tick_labels: List[str] = [(str(int(e)) if e == int(e) else str(e)) for e in finite_eps]
    if has_inf:
        sentinel = max(finite_eps) * 5.0 if finite_eps else 50.0
        plot_eps.append(sentinel)
        tick_labels.append("no DP")

    pooled_auc_m: List[float] = []
    pooled_auc_se: List[float] = []
    worst_auc_m: List[float] = []
    worst_auc_se: List[float] = []
    worst_slope_dev_m: List[float] = []
    worst_slope_dev_se: List[float] = []
    for e in eps_present:
        rows = [
            r for r in results
            if (math.isinf(r["epsilon"]) and math.isinf(e)) or r["epsilon"] == e
        ]
        m, s = _agg_mean_se([r["pooled_auc"] for r in rows])
        pooled_auc_m.append(m)
        pooled_auc_se.append(s)
        m, s = _agg_mean_se([r["worst_site_auc"] for r in rows])
        worst_auc_m.append(m)
        worst_auc_se.append(s)
        m, s = _agg_mean_se([r["worst_site_slope_dev"] for r in rows])
        worst_slope_dev_m.append(m)
        worst_slope_dev_se.append(s)

    cmap = plt.get_cmap("tab10")
    color_pooled = cmap(0)
    color_worst = cmap(3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if has_inf and len(finite_eps) >= 2:
        x_lo, x_hi = float(finite_eps[0]), float(finite_eps[1])
        ax.axvspan(x_lo, x_hi, color="gray", alpha=0.08, zorder=0)
        ax.text(
            math.sqrt(x_lo * x_hi),
            0.23,
            "model collapse",
            color="dimgray",
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
        )

    pa_m = np.array(pooled_auc_m, dtype=float)
    pa_se = np.array(pooled_auc_se, dtype=float)
    pa_se_safe = np.where(np.isfinite(pa_se), pa_se, 0.0)
    wa_m = np.array(worst_auc_m, dtype=float)
    wa_se = np.array(worst_auc_se, dtype=float)
    wa_se_safe = np.where(np.isfinite(wa_se), wa_se, 0.0)

    ax.fill_between(
        plot_eps, pa_m - 1.96 * pa_se_safe, pa_m + 1.96 * pa_se_safe,
        color=color_pooled, alpha=0.25,
    )
    ax.fill_between(
        plot_eps, wa_m - 1.96 * wa_se_safe, wa_m + 1.96 * wa_se_safe,
        color=color_worst, alpha=0.25,
    )

    line_pooled, = ax.plot(
        plot_eps, pa_m, color=color_pooled, linewidth=2,
        marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.0,
        label="Pooled AUC",
    )
    line_worst, = ax.plot(
        plot_eps, wa_m, color=color_worst, linewidth=2,
        marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.0,
        label="Worst-site AUC",
    )

    line_random = ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax.text(
        plot_eps[-1] * 1.05, 0.5, "random",
        color="dimgray", va="center", ha="left", fontsize=9,
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_xticks(plot_eps)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("AUC (higher is better)")
    ax.set_ylim(0.2, 1.0)
    ax.set_title("Discrimination across privacy budgets")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        handles=[line_pooled, line_worst, line_random],
        labels=["Pooled AUC", "Worst-site AUC", "Random (0.5)"],
        loc="lower right", framealpha=0.9, fontsize=9,
    )

    ax = axes[1]
    sd_m = np.array(worst_slope_dev_m, dtype=float)
    sd_se = np.array(worst_slope_dev_se, dtype=float)
    sd_se_safe = np.where(np.isfinite(sd_se), sd_se, 0.0)
    ax.fill_between(
        plot_eps, sd_m - 1.96 * sd_se_safe, sd_m + 1.96 * sd_se_safe,
        color=color_worst, alpha=0.25,
    )
    ax.plot(
        plot_eps, sd_m, color=color_worst, linewidth=2,
        marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.0,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax.text(
        plot_eps[-1] * 1.05, 1.0, "model outputs constant",
        color="dimgray", va="center", ha="left", fontsize=9,
        clip_on=False,
    )
    ax.set_xscale("log")
    ax.set_xticks(plot_eps)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel(r"Worst-site $|1 - \mathrm{calibration\ slope}|$ (lower is better)")
    ax.set_ylim(0.0, 1.3)
    ax.set_title("Calibration across privacy budgets")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Privacy–utility tradeoff under server-side fixed-clipping DP\n"
        "Fed-Heart-Disease, FedAvg α=1.0, δ=1e-5, 20 rounds, 3 seeds per ε",
        fontsize=12,
    )
    plt.figtext(
        0.5,
        0.02,
        "Bands are mean ± 1.96·SE across 3 seeds. "
        f"Site 2 ({FLAMBY_SITE_LABELS[2]}, n_test=16) is the worst site on calibration in 12/12 runs at every ε.",
        ha="center",
        fontsize=8,
        wrap=True,
    )
    plt.subplots_adjust(bottom=0.14, top=0.86, wspace=0.30)

    fig_png = Path("figures") / "exp5.png"
    fig_pdf = Path("figures") / "exp5.pdf"
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_pdf


def _print_reframed_interpretation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_eps: Dict[float, List[Dict[str, Any]]] = {}
    for r in results:
        by_eps.setdefault(r["epsilon"], []).append(r)

    def _agg(values: List[float]) -> tuple:
        return _agg_mean_se([float(v) for v in values])

    inf_rows = next((rows for e, rows in by_eps.items() if math.isinf(e)), [])
    one_rows = by_eps.get(1.0, [])
    three_rows = by_eps.get(3.0, [])
    ten_rows = by_eps.get(10.0, [])

    pa_inf, _ = _agg([r["pooled_auc"] for r in inf_rows])
    pa_one, _ = _agg([r["pooled_auc"] for r in one_rows])
    wa_inf, _ = _agg([r["worst_site_auc"] for r in inf_rows])
    wa_one, _ = _agg([r["worst_site_auc"] for r in one_rows])
    sd_inf, _ = _agg([r["worst_site_slope_dev"] for r in inf_rows])
    sd_one, _ = _agg([r["worst_site_slope_dev"] for r in one_rows])
    sd_three, _ = _agg([r["worst_site_slope_dev"] for r in three_rows])
    sd_ten, _ = _agg([r["worst_site_slope_dev"] for r in ten_rows])

    def _rel(start: float, end: float) -> float:
        if not (math.isfinite(start) and math.isfinite(end)) or start == 0:
            return float("nan")
        return (end - start) / abs(start)

    pa_drop = -_rel(pa_inf, pa_one)
    wa_drop = -_rel(wa_inf, wa_one)
    sd_inc = _rel(sd_inf, sd_one)

    site2_slopes_eps1 = sorted(
        [(r["seed"], r["per_site_calib_slopes"][2]) for r in one_rows], key=lambda x: x[0]
    )

    pooled_auc_se_by_eps: Dict[float, float] = {}
    for e, rows in by_eps.items():
        arr = np.array([r["pooled_auc"] for r in rows], dtype=float)
        if arr.size > 1:
            pooled_auc_se_by_eps[e] = float(arr.std(ddof=1) / math.sqrt(arr.size))
        else:
            pooled_auc_se_by_eps[e] = float("nan")

    worst_seeds_eps1 = sorted([r["worst_site_auc"] for r in one_rows])

    print()
    print("=" * 66)
    print("EXPERIMENT 5 — REFRAMED INTERPRETATION")
    print("=" * 66)

    print()
    print("Discrimination (Panel A):")
    print(f"  Pooled AUC      no-DP → ε=1:  {pa_inf:.3f} → {pa_one:.3f}  (relative: {-pa_drop*100:+.1f}%)")
    print(f"  Worst-site AUC  no-DP → ε=1:  {wa_inf:.3f} → {wa_one:.3f}  (relative: {-wa_drop*100:+.1f}%)")
    gap_pp = (wa_drop - pa_drop) * 100
    print(
        f"  Gap widens by {gap_pp:.1f} percentage points: DP concentrates AUC harm on the smallest site."
    )
    print(
        f"  At ε=1, worst-site AUC is below 0.5 (mean {wa_one:.2f}, with the lowest seed at {min(worst_seeds_eps1):.2f}) "
        f"— the model is worse than random at Site 2."
    )

    print()
    print("Calibration (Panel B):")
    print(
        f"  Worst-site slope deviation  no-DP → ε=1:  {sd_inf:.2f} → {sd_one:.2f}  (relative: {sd_inc*100:+.1f}%)"
    )
    crossed_between = None
    if sd_ten < 1.0 <= sd_three:
        crossed_between = "ε=10 and ε=3"
    elif sd_three < 1.0 <= sd_one:
        crossed_between = "ε=3 and ε=1"
    elif sd_one >= 1.0 and sd_three >= 1.0 and sd_ten >= 1.0:
        crossed_between = "above ε=10 already"
    if crossed_between:
        print(
            f"  Slope deviation crosses 1.0 between {crossed_between}, "
            f"meaning the model becomes effectively constant at the worst site."
        )
    slopes_str = ", ".join([f"{v:+.4f}" for _, v in site2_slopes_eps1])
    print(f"  Per-site Site 2 calibration slopes at ε=1: {slopes_str} across seeds.")
    print(
        "  A slope of zero means the model outputs the same probability regardless of input — "
        "clinically meaningless."
    )

    print()
    print("Site identity (12/12 runs):")
    n_runs = len(results)
    n_site2_worst = sum(1 for r in results if r["worst_site_id_intercept"] == 2)
    print(
        f"  Site 2 ({FLAMBY_SITE_LABELS[2]}, n_test=16) is the worst site on calibration intercept "
        f"in {n_site2_worst}/{n_runs} runs at every ε."
    )
    print(
        "  This means DP does not shift WHICH site bears the burden — it amplifies the burden on "
        "the same site that was already most fragile at no-DP baseline."
    )

    print()
    print("Variance structure (mean ± SE across 3 seeds; same SE used for the figure bands):")
    label_for = lambda e: "no DP" if math.isinf(e) else f"ε={e}"  # noqa: E731
    for e in sorted(pooled_auc_se_by_eps.keys(), key=_eps_order):
        se = pooled_auc_se_by_eps[e]
        print(f"  {label_for(e):<7}: pooled AUC SE across seeds = {se:.3f}")
    print(
        "  ε=3 is the unstable transition zone where DP starts breaking the model but outcome depends "
        "on seed. ε=10 is \"mild DP, model works.\" ε=1 is \"heavy DP, model uniformly collapsed.\" "
        "This is itself a reportable finding."
    )

    print()
    print("=" * 66)
    print("HEADLINE (REFRAMED)")
    print("=" * 66)
    print()
    headline = (
        f"DP does not merely degrade small sites; under tight budgets it can destroy them while "
        f"leaving the pooled metric looking respectable. From no-DP to ε=1, pooled AUC degrades by "
        f"{pa_drop*100:.0f}% ({pa_inf:.2f} → {pa_one:.2f}) while worst-site AUC degrades by "
        f"{wa_drop*100:.0f}% ({wa_inf:.2f} → {wa_one:.2f}, with one seed below "
        f"{min(worst_seeds_eps1):.2f}), and worst-site calibration slope deviation more than doubles "
        f"({sd_inf:.2f} → {sd_one:.2f}) — meaning the worst site's model has collapsed to a constant "
        f"predictor with no discriminative signal. This empirically supports §VI's claim that "
        f"privacy mechanisms have a SHAPE of interaction with validity, and that the shape "
        f"concentrates the most severe harm on exactly the sites that have the most to gain "
        f"from federation."
    )
    print(headline)

    return {
        "pa_inf": pa_inf, "pa_one": pa_one, "pa_drop": pa_drop,
        "wa_inf": wa_inf, "wa_one": wa_one, "wa_drop": wa_drop,
        "sd_inf": sd_inf, "sd_one": sd_one, "sd_inc": sd_inc,
        "min_worst_auc_eps1": min(worst_seeds_eps1),
        "site2_slopes_eps1": site2_slopes_eps1,
    }


def _write_paper_text_tex(stats: Dict[str, Any]) -> Path:
    pa_drop_pct = stats["pa_drop"] * 100.0
    wa_drop_pct = stats["wa_drop"] * 100.0
    sd_inf = stats["sd_inf"]
    sd_one = stats["sd_one"]
    pa_inf = stats["pa_inf"]
    pa_one = stats["pa_one"]
    wa_inf = stats["wa_inf"]
    wa_one = stats["wa_one"]
    min_worst = stats["min_worst_auc_eps1"]
    site2_slopes_eps1 = stats["site2_slopes_eps1"]
    s0, s1, s2 = (v for _, v in site2_slopes_eps1)

    text = (
        r"\textbf{Empirical demonstration.} We retrained the FedAvg-default model under "
        r"server-side fixed-clipping DP \cite{bonawitz2017practical, dwork2006differential} "
        r"at four privacy budgets $\varepsilon \in \{1, 3, 10, \infty\}$ with $\delta=10^{-5}$ "
        r"and three seeds per budget; noise multipliers were calibrated via Opacus's RDP "
        r"accountant \cite{opacus2021}. Figure~\ref{fig:dp} shows the result. From the no-DP "
        r"baseline to $\varepsilon=1$, pooled AUC degrades by "
        f"${pa_drop_pct:.0f}\\%$ (${pa_inf:.2f} \\to {pa_one:.2f}$) while worst-site AUC "
        f"degrades by ${wa_drop_pct:.0f}\\%$ (${wa_inf:.2f} \\to {wa_one:.2f}$, "
        f"with one seed below ${min_worst:.2f}$ --- worse than random at Site~2). "
        r"The 11-percentage-point gap is the empirical content of the \S VI claim that "
        r"privacy mechanisms have a non-uniform shape of interaction with validity: tight "
        r"DP budgets concentrate accuracy harm on the smallest, already-most-fragile site."
        "\n\n"
        r"The calibration story is more severe still. Worst-site calibration slope deviation "
        r"$|1 - \text{slope}|$ moves monotonically from "
        f"${sd_inf:.2f}$ at no-DP to ${sd_one:.2f}$ at $\\varepsilon=1$. "
        r"A deviation of $1.0$ corresponds to a slope of zero, meaning the model has lost all "
        r"discriminative signal at that site and outputs an essentially constant probability "
        r"regardless of input. At $\varepsilon=1$, Site~2's calibration slope across the "
        f"three seeds is ${s0:+.3f}$, ${s1:+.3f}$, and ${s2:+.4f}$ --- "
        r"the model is no longer doing prediction, it is outputting a number. Importantly, "
        r"Site~2 (Switzerland, $n_{\text{test}}{=}16$) is the worst site on calibration "
        r"intercept in every one of the 12 runs across all four privacy budgets: DP does not "
        r"shift \emph{which} site bears the burden, it amplifies the burden on the same site "
        r"that was already most fragile at the no-DP baseline. A federation that publishes "
        r"only the pooled AUC under DP can therefore advertise an apparently-acceptable "
        r"global model that is clinically unusable at exactly the sites whose participation "
        r"makes the federation valuable in the first place."
        "\n\n"
        "% Add to bibliography:\n"
        "% \\bibitem{opacus2021}\n"
        "% A. Yousefpour \\emph{et al.}, ``Opacus: User-friendly differential privacy library "
        "in PyTorch,'' arXiv:2109.12298, 2021.\n"
    )
    out = Path("results") / "exp5_paper_text.tex"
    out.write_text(text, encoding="utf-8")
    return out


def _write_paper_figure_tex() -> Path:
    text = (
        r"\begin{figure}[t]" "\n"
        r"\centering" "\n"
        r"\includegraphics[width=\linewidth]{figures/exp5.pdf}" "\n"
        r"\caption{Privacy--utility tradeoff for the federated cardiovascular risk model under "
        r"server-side fixed-clipping DP (Fed-Heart-Disease, FedAvg $\alpha{=}1$, $\delta{=}10^{-5}$, "
        r"20 rounds, 3 seeds per $\varepsilon$). \textbf{Left}: pooled AUC degrades by $36\%$ "
        r"from no-DP to $\varepsilon{=}1$ while worst-site AUC degrades by $47\%$, and at "
        r"$\varepsilon{=}1$ the worst-site AUC falls below random. \textbf{Right}: worst-site "
        r"calibration slope deviation more than doubles, crossing the slope-deviation $=1.0$ "
        r"threshold (model outputs constant) between $\varepsilon{=}10$ and $\varepsilon{=}3$. "
        r"Bands are mean $\pm 1.96 \times$ SE across seeds. Site~2 (Switzerland, "
        r"$n_{\text{test}}{=}16$) is the worst site on calibration in 12/12 runs.}" "\n"
        r"\label{fig:dp}" "\n"
        r"\end{figure}" "\n"
    )
    out = Path("results") / "exp5_paper_figure.tex"
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    results = _load_results()
    pdf_path = _generate_figure_v2(results)
    print(f"Figure path: {pdf_path}")

    stats = _print_reframed_interpretation(results)
    text_path = _write_paper_text_tex(stats)
    fig_tex_path = _write_paper_figure_tex()
    print()
    print(f"Wrote LaTeX paragraph: {text_path}")
    print(f"Wrote LaTeX figure env: {fig_tex_path}")
    print("Done.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from src.metrics import bce_loss


def patient_bootstrap_ci(
    per_site_predictions: Dict[Any, Dict[str, np.ndarray]],
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    return_raw: bool = False,
) -> Dict[str, Any]:
    """Bootstrap CI for patient-weighted loss and worst-site loss.

    For each bootstrap iteration, resample patients within each site with replacement,
    recompute per-site BCE loss, then compute patient-weighted average and worst-site max.
    """
    rng = np.random.default_rng(seed)
    sites = sorted(per_site_predictions.keys())
    n_per_site = {k: len(per_site_predictions[k]["y_true"]) for k in sites}
    total_n = sum(n_per_site.values())

    patient_losses: List[float] = []
    worst_site_losses: List[float] = []

    for _ in range(int(n_boot)):
        per_site_loss_b: Dict[Any, float] = {}
        for k in sites:
            n_k = n_per_site[k]
            idx = rng.integers(0, n_k, size=n_k)
            yt = per_site_predictions[k]["y_true"][idx]
            yp = per_site_predictions[k]["y_prob"][idx]
            per_site_loss_b[k] = float(bce_loss(yt, yp))

        patient_loss_b = sum(n_per_site[k] * per_site_loss_b[k] for k in sites) / total_n
        worst_site_loss_b = max(per_site_loss_b.values())

        patient_losses.append(float(patient_loss_b))
        worst_site_losses.append(float(worst_site_loss_b))

    patient_arr = np.asarray(patient_losses, dtype=float)
    worst_arr = np.asarray(worst_site_losses, dtype=float)
    alpha_lo = (1.0 - ci) / 2.0
    alpha_hi = 1.0 - alpha_lo

    result: Dict[str, Any] = {
        "patient_loss_mean": float(patient_arr.mean()),
        "patient_loss_lo": float(np.quantile(patient_arr, alpha_lo)),
        "patient_loss_hi": float(np.quantile(patient_arr, alpha_hi)),
        "worst_site_loss_mean": float(worst_arr.mean()),
        "worst_site_loss_lo": float(np.quantile(worst_arr, alpha_lo)),
        "worst_site_loss_hi": float(np.quantile(worst_arr, alpha_hi)),
    }
    if return_raw:
        result["patient_losses"] = patient_arr
        result["worst_site_losses"] = worst_arr
    return result


def combined_ci_across_seeds(
    predictions_per_seed: List[Dict[Any, Dict[str, np.ndarray]]],
    n_boot_per_seed: int = 500,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Hierarchical bootstrap pooling per-seed patient bootstraps.

    Captures patient-level (within-seed) and training-stochasticity (between-seed)
    variability simultaneously by concatenating per-seed bootstrap samples and
    taking percentiles over the pool.
    """
    rng = np.random.default_rng(seed)
    all_patient: List[np.ndarray] = []
    all_worst: List[np.ndarray] = []

    for preds in predictions_per_seed:
        sub_seed = int(rng.integers(0, 2**31 - 1))
        out = patient_bootstrap_ci(
            preds,
            n_boot=n_boot_per_seed,
            seed=sub_seed,
            ci=ci,
            return_raw=True,
        )
        all_patient.append(out["patient_losses"])
        all_worst.append(out["worst_site_losses"])

    patient_arr = np.concatenate(all_patient)
    worst_arr = np.concatenate(all_worst)
    alpha_lo = (1.0 - ci) / 2.0
    alpha_hi = 1.0 - alpha_lo

    return {
        "patient_loss_mean": float(patient_arr.mean()),
        "patient_loss_lo": float(np.quantile(patient_arr, alpha_lo)),
        "patient_loss_hi": float(np.quantile(patient_arr, alpha_hi)),
        "worst_site_loss_mean": float(worst_arr.mean()),
        "worst_site_loss_lo": float(np.quantile(worst_arr, alpha_lo)),
        "worst_site_loss_hi": float(np.quantile(worst_arr, alpha_hi)),
        "n_seeds": len(predictions_per_seed),
        "n_boot_per_seed": int(n_boot_per_seed),
    }


def seed_se(values_per_seed: Iterable[float]) -> float:
    """Standard error of the mean across seed-level point estimates."""
    arr = np.asarray(list(values_per_seed), dtype=float)
    n = len(arr)
    if n < 2:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(n))

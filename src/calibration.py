"""
Calibration utilities for Experiment 3.

Provides:
  - bin_calibration_curve: 10-bin equal-width calibration curve with
    per-bin Wilson 95% CIs.
  - bootstrap_calibration_intercept_slope: bootstrap distribution of
    calibration intercept and slope under patient-level resampling.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    halfwidth = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return max(0.0, centre - halfwidth), min(1.0, centre + halfwidth)


def bin_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (centers, observed_rate, lo, hi, n_per_bin) for an equal-width binning."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    obs = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    n = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (y_prob >= edges[b]) & (y_prob <= edges[b + 1])
        else:
            mask = (y_prob >= edges[b]) & (y_prob < edges[b + 1])
        n[b] = int(mask.sum())
        if n[b] > 0:
            k = int(y_true[mask].sum())
            obs[b] = k / n[b]
            lo[b], hi[b] = wilson_interval(k, n[b])
    return centers, obs, lo, hi, n


def bootstrap_calibration_intercept_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    pathological_threshold: float = 10.0,
) -> dict:
    """Bootstrap calibration intercept/slope with patient-level resampling.

    Uses regularized logistic recalibration (C=1.0) to avoid perfect-separation
    artifacts on small bootstrap resamples. Resamples whose absolute intercept
    or slope exceeds `pathological_threshold` after fitting are dropped.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    intercepts: list = []
    slopes: list = []
    n_failed_fit = 0
    n_pathological = 0
    eps = 1e-6
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            n_failed_fit += 1
            continue
        yp_clip = np.clip(yp, eps, 1 - eps)
        logit = np.log(yp_clip / (1 - yp_clip))
        try:
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            model.fit(logit.reshape(-1, 1), yt)
            b = float(model.intercept_[0])
            w = float(model.coef_[0, 0])
        except Exception:
            n_failed_fit += 1
            continue
        if abs(b) > pathological_threshold or abs(w) > pathological_threshold:
            n_pathological += 1
            continue
        intercepts.append(b)
        slopes.append(w)
    if not intercepts:
        return {
            "intercept_mean": float("nan"),
            "intercept_lo": float("nan"),
            "intercept_hi": float("nan"),
            "slope_mean": float("nan"),
            "slope_lo": float("nan"),
            "slope_hi": float("nan"),
            "n_boot_successful": 0,
            "n_boot_failed_fit": int(n_failed_fit),
            "n_boot_pathological_dropped": int(n_pathological),
        }
    intercepts_arr = np.array(intercepts)
    slopes_arr = np.array(slopes)
    return {
        "intercept_mean": float(intercepts_arr.mean()),
        "intercept_lo": float(np.quantile(intercepts_arr, 0.025)),
        "intercept_hi": float(np.quantile(intercepts_arr, 0.975)),
        "slope_mean": float(slopes_arr.mean()),
        "slope_lo": float(np.quantile(slopes_arr, 0.025)),
        "slope_hi": float(np.quantile(slopes_arr, 0.975)),
        "n_boot_successful": int(len(intercepts_arr)),
        "n_boot_failed_fit": int(n_failed_fit),
        "n_boot_pathological_dropped": int(n_pathological),
    }

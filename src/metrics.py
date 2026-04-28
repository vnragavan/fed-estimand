from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _invalid_small_sample(y_true: np.ndarray, y_prob: np.ndarray) -> bool:
    return len(y_true) < 10 or len(y_prob) < 10


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if _invalid_small_sample(y_true, y_prob):
        return float(np.nan)
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if _invalid_small_sample(y_true, y_prob):
        return float(np.nan)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    if _invalid_small_sample(y_true, y_prob):
        return float(np.nan), float(np.nan)
    if np.unique(y_true).size < 2:
        return float(np.nan), float(np.nan)
    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    logit = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    clf.fit(logit, y_true.astype(int))
    intercept = float(clf.intercept_[0])
    slope = float(clf.coef_[0][0])
    return intercept, slope


def bce_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if _invalid_small_sample(y_true, y_prob):
        return float(np.nan)
    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss)

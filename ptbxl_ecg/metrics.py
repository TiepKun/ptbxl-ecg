"""Chỉ số đánh giá đa nhãn: AUC theo lớp, macro-AUC, F1 với ngưỡng chọn trên tập val."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .data import SUPERCLASSES


def per_class_auc(y: np.ndarray, p: np.ndarray) -> list[float]:
    out = []
    for k in range(y.shape[1]):
        if len(np.unique(y[:, k])) < 2:  # lớp không đủ cả 0 và 1 -> không tính được AUC
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(y[:, k], p[:, k])))
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    aucs = per_class_auc(y, p)
    return float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")


def best_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> np.ndarray:
    """Với mỗi lớp chọn ngưỡng cho F1 cao nhất trên tập validation."""
    grid = np.arange(0.05, 0.96, 0.01)
    thr = np.full(y_val.shape[1], 0.5)
    for k in range(y_val.shape[1]):
        scores = [f1_score(y_val[:, k], p_val[:, k] >= t, zero_division=0) for t in grid]
        thr[k] = grid[int(np.argmax(scores))]
    return thr


def f1_per_class(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> list[float]:
    return [float(f1_score(y[:, k], p[:, k] >= thr[k], zero_division=0)) for k in range(y.shape[1])]


def summarize(y_val, p_val, y_test, p_test) -> dict:
    """Gói toàn bộ chỉ số val/test vào một dict phẳng (dùng cho log CSV)."""
    thr = best_thresholds(y_val, p_val)
    auc_t = per_class_auc(y_test, p_test)
    f1_t = f1_per_class(y_test, p_test, thr)
    res = {
        "val_auc": macro_auc(y_val, p_val),
        "test_auc": macro_auc(y_test, p_test),
        "test_f1_macro": float(np.mean(f1_t)),
    }
    for c, a, f in zip(SUPERCLASSES, auc_t, f1_t):
        res[f"auc_{c}"] = a
        res[f"f1_{c}"] = f
    res["thresholds"] = [float(t) for t in thr]
    return res

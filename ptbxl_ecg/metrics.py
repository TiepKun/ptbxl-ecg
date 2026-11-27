"""Chỉ số đánh giá đa nhãn: AUC theo lớp, macro-AUC, F1 với ngưỡng chọn trên tập val."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .data import SUPERCLASSES


def per_class_auc(y: np.ndarray, p: np.ndarray) -> list[float]:
    """AUC của từng lớp (đa nhãn). y, p: (N, K), y là 0/1, p là xác suất.

    Trả về list K số float. Lớp mà y chỉ có một giá trị (toàn 0 hoặc toàn 1) thì không tính được AUC:
    trả về NaN cho lớp đó. Gợi ý: sklearn.metrics.roc_auc_score.
    """
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for k in range(y.shape[1]):
        if len(np.unique(y[:, k])) < 2:
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(y[:, k], p[:, k])))
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Trung bình AUC các lớp, bỏ qua lớp NaN. Nếu mọi lớp đều NaN thì trả về NaN (float).
    """
    a = np.array(per_class_auc(y, p), dtype=float)
    if np.isnan(a).all():  # tránh RuntimeWarning "Mean of empty slice" của nanmean
        return float("nan")
    return float(np.nanmean(a))


THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def best_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> np.ndarray:
    """Với mỗi lớp chọn ngưỡng cho F1 cao nhất trên tập validation."""
    y = np.asarray(y_val).astype(bool)
    p = np.asarray(p_val)
    thr = np.empty(y.shape[1])
    for k in range(y.shape[1]):
        # Tính F1 cho mọi ngưỡng trên lưới cùng lúc: pred có shape (N, số ngưỡng)
        pred = p[:, k, None] >= THRESHOLD_GRID[None, :]
        tp = (pred & y[:, k, None]).sum(0)
        fp = (pred & ~y[:, k, None]).sum(0)
        fn = (~pred & y[:, k, None]).sum(0)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        thr[k] = THRESHOLD_GRID[int(np.argmax(f1))]  # hòa thì lấy ngưỡng nhỏ nhất
    return thr


def f1_per_class(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> list[float]:
    """F1 của từng lớp với ngưỡng riêng thr[k]: dự đoán dương khi p[:, k] >= thr[k].

    Trả về list K số float; dùng zero_division=0 để lớp không có mẫu dương không gây lỗi.
    """
    y, p = np.asarray(y), np.asarray(p)
    return [float(f1_score(y[:, k], (p[:, k] >= thr[k]).astype(int), zero_division=0))
            for k in range(y.shape[1])]


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

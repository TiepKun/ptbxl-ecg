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
    raise NotImplementedError("TODO [A]: AUC theo lớp, NaN nếu lớp không đủ cả hai nhãn")


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Trung bình AUC các lớp, bỏ qua lớp NaN. Nếu mọi lớp đều NaN thì trả về NaN (float).
    """
    raise NotImplementedError("TODO [A]: macro-AUC (nanmean của per_class_auc)")


def best_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> np.ndarray:
    """Với mỗi lớp chọn ngưỡng cho F1 cao nhất trên tập validation."""
    raise NotImplementedError("TODO [A]: chọn ngưỡng tối ưu F1 theo từng lớp trên tập validation")


def f1_per_class(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> list[float]:
    """F1 của từng lớp với ngưỡng riêng thr[k]: dự đoán dương khi p[:, k] >= thr[k].

    Trả về list K số float; dùng zero_division=0 để lớp không có mẫu dương không gây lỗi.
    """
    raise NotImplementedError("TODO [A]: F1 theo lớp với ngưỡng cho trước")


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

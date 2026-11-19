"""Đặc trưng thủ công cho baseline cổ điển (Logistic Regression / Random Forest).

Mỗi chuyển đạo cho 11 đặc trưng: trung bình, độ lệch chuẩn, min, max, độ lệch (skewness),
độ nhọn (kurtosis) và log-năng lượng trong 5 dải tần (0.5-4, 4-8, 8-15, 15-30, 30-50 Hz).
12 chuyển đạo -> 132 đặc trưng.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from .data import FS

BANDS = [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 50)]


def extract_features(X: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """X: (N, C, T) tín hiệu GỐC (chưa chuẩn hóa) -> (N, C * 11)."""
    feats = []
    freqs = np.fft.rfftfreq(X.shape[-1], d=1.0 / FS)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        stats = [x.mean(-1), x.std(-1), x.min(-1), x.max(-1), skew(x, axis=-1), kurtosis(x, axis=-1)]
        power = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2 / x.shape[-1]
        bands = [np.log1p(power[..., (freqs >= lo) & (freqs < hi)].sum(-1)) for lo, hi in BANDS]
        f = np.stack(stats + bands, axis=-1)  # (n, C, 11)
        feats.append(f.reshape(len(x), -1))
    return np.nan_to_num(np.concatenate(feats).astype(np.float32))

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
    X = np.asarray(X, dtype=np.float32)
    n, c, t = X.shape
    freqs = np.fft.rfftfreq(t, d=1.0 / FS)
    nyq = FS / 2
    # Dải nửa mở [lo, hi) để các dải không chồng lên nhau; riêng dải chạm Nyquist lấy cả hi.
    masks = [(freqs >= lo) & ((freqs < hi) if hi < nyq else (freqs <= hi)) for lo, hi in BANDS]

    out = np.empty((n, c, 6 + len(BANDS)), dtype=np.float32)
    for s in range(0, n, chunk):  # xử lý theo khối để FFT không chiếm quá nhiều RAM
        x = X[s:s + chunk].astype(np.float64)
        out[s:s + chunk, :, 0] = x.mean(-1)
        out[s:s + chunk, :, 1] = x.std(-1)
        out[s:s + chunk, :, 2] = x.min(-1)
        out[s:s + chunk, :, 3] = x.max(-1)
        out[s:s + chunk, :, 4] = skew(x, axis=-1)
        out[s:s + chunk, :, 5] = kurtosis(x, axis=-1)
        power = np.abs(np.fft.rfft(x, axis=-1)) ** 2 / t
        for b, m in enumerate(masks):
            out[s:s + chunk, :, 6 + b] = np.log(power[..., m].sum(-1) + 1e-8)

    # Chuyển đạo phẳng (toàn 0) cho skew/kurtosis = NaN -> thay bằng 0.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(n, c * (6 + len(BANDS)))  # thứ tự: [chuyển đạo 0: 11 đặc trưng, chuyển đạo 1: ...]

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
    raise NotImplementedError("TODO [A]: trích 11 đặc trưng cho mỗi chuyển đạo (xem docstring của module)")

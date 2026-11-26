"""Đăng ký mô hình: ánh xạ tên -> lớp. File chung của cả nhóm, hạn chế sửa.

Mọi mô hình nhận x có shape (B, C, T) và trả về logit (B, n_classes).
Số tham số mặc định của 4 mô hình chính xấp xỉ nhau (khoảng 0,5 triệu) để
so sánh công bằng theo yêu cầu ablation (c). Dùng `count_params` để kiểm tra.
"""
from __future__ import annotations

import torch.nn as nn

from .models_cnn import CNNAttention, ResNet1D, SimpleCNN  # B
from .models_common import PoolHead, conv_stem, count_params  # noqa: F401  (dùng chung)
from .models_rnn import CRNN  # C

# --------------------------------------------------------------------------- #
# Đăng ký mô hình
# --------------------------------------------------------------------------- #
MODEL_NAMES = ["cnn_simple", "resnet1d", "gru", "lstm", "cnn_attn"]


def build_model(name: str, in_ch: int, n_classes: int = 5, seq_len: int = 1000,
                **kwargs) -> nn.Module:
    if name == "cnn_simple":
        return SimpleCNN(in_ch, n_classes, **kwargs)
    if name == "resnet1d":
        return ResNet1D(in_ch, n_classes, **kwargs)
    if name in ("gru", "lstm"):
        return CRNN(in_ch, n_classes, rnn=name, **kwargs)
    if name == "cnn_attn":
        return CNNAttention(in_ch, n_classes, seq_len=seq_len, **kwargs)
    raise ValueError(f"Mô hình không hợp lệ: {name}. Chọn trong {MODEL_NAMES}")

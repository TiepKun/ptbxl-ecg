"""Mô hình CNN của vai trò B: SimpleCNN, ResNet 1D và CNN + self-attention.

Mọi mô hình nhận x có shape (B, C, T) và trả về logit (B, n_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models_common import PoolHead, conv_stem


# --------------------------------------------------------------------------- #
# 1. CNN đơn giản: baseline học sâu (3 tầng conv)
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    """Baseline học sâu, cỡ nhỏ (khoảng 55 nghìn tham số với mặc định).

    Gợi ý kiến trúc: 3 khối [Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(2)] với số kênh width, 2*width, 4*width
    (kernel 7, 5, 5, padding giữ độ dài), sau đó PoolHead(4*width, n_classes).
    Nhận x (B, in_ch, T), trả về logit (B, n_classes).
    """

    def __init__(self, in_ch: int, n_classes: int, width: int = 32):
        super().__init__()
        raise NotImplementedError("TODO [B]: cài đặt SimpleCNN theo docstring của lớp")

    def forward(self, x):
        raise NotImplementedError("TODO [B]: forward của SimpleCNN")


# --------------------------------------------------------------------------- #
# 2. ResNet 1D
# --------------------------------------------------------------------------- #
class BasicBlock1D(nn.Module):
    """Khối residual 1D.

    Gợi ý: conv1 (có stride) -> BN -> ReLU -> conv2 -> BN; cộng với đường tắt rồi ReLU.
    Đường tắt là Conv1d kernel 1 (stride cùng stride) + BN khi stride != 1 hoặc cin != cout, ngược lại là
    chính x. Kernel `kernel` với padding kernel//2 để độ dài chỉ đổi theo stride.
    """

    def __init__(self, cin: int, cout: int, stride: int, kernel: int = 7):
        super().__init__()
        raise NotImplementedError("TODO [B]: cài đặt BasicBlock1D theo docstring của lớp")

    def forward(self, x):
        raise NotImplementedError("TODO [B]: forward của BasicBlock1D")


class ResNet1D(nn.Module):
    """ResNet 1D nhỏ (khoảng 0,5 triệu tham số với mặc định).

    Gợi ý: stem Conv1d(in_ch, width, kernel 15, stride 2, padding 7) + BN + ReLU; sau đó các BasicBlock1D
    với số kênh [w, 2w, 4w, 4w] và số khối mỗi giai đoạn lấy từ `blocks` (stride 2 ở khối đầu của mọi giai
    đoạn trừ giai đoạn đầu); cuối cùng PoolHead.
    """

    def __init__(self, in_ch: int, n_classes: int, width: int = 32, blocks=(1, 1, 1, 1),
                 kernel: int = 7):
        super().__init__()
        raise NotImplementedError("TODO [B]: cài đặt ResNet1D theo docstring của lớp")

    def forward(self, x):
        raise NotImplementedError("TODO [B]: forward của ResNet1D")


# --------------------------------------------------------------------------- #
# 3. CNN + self-attention (Transformer encoder nhỏ)
# --------------------------------------------------------------------------- #
class CNNAttention(nn.Module):
    """CNN + self-attention (Transformer encoder nhỏ, khoảng 0,5 triệu tham số với mặc định).

    Gợi ý: stem = conv_stem(in_ch, d_model) giảm 1000 -> 250 token; cộng positional embedding học được
    (nn.Parameter shape (1, n_tokens, d_model), khởi tạo trunc_normal std 0.02; n_tokens lấy bằng cách chạy
    thử stem với tensor 0 độ dài seq_len); nn.TransformerEncoder (batch_first=True, norm_first=True,
    dim_feedforward = d_model * ff_mult, enable_nested_tensor=False); LayerNorm; PoolHead(d_model, n_classes).
    Lưu ý: đảo trục (B, D, T') <-> (B, T', D) cho đúng với batch_first.
    """

    def __init__(self, in_ch: int, n_classes: int, d_model: int = 128, heads: int = 4,
                 layers: int = 2, ff_mult: int = 4, dropout: float = 0.1, seq_len: int = 1000):
        super().__init__()
        raise NotImplementedError("TODO [B]: cài đặt CNNAttention theo docstring của lớp")

    def forward(self, x):
        raise NotImplementedError("TODO [B]: forward của CNNAttention")

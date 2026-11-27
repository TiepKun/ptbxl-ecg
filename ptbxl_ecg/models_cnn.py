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

        def block(cin: int, cout: int, k: int) -> nn.Sequential:
            # bias=False vì BatchNorm ngay sau đã có tham số dịch (beta)
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
                                 nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.MaxPool1d(2))

        self.features = nn.Sequential(
            block(in_ch, width, 7),              # 1000 -> 500
            block(width, 2 * width, 5),          # 500  -> 250
            block(2 * width, 4 * width, 5),      # 250  -> 125
        )
        self.head = PoolHead(4 * width, n_classes)

    def forward(self, x):
        return self.head(self.features(x))


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
        pad = kernel // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            # Đường tắt phải khớp cả số kênh lẫn độ dài với nhánh chính
            self.shortcut = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                          nn.BatchNorm1d(cout))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """ResNet 1D nhỏ (khoảng 0,5 triệu tham số với mặc định).

    Gợi ý: stem Conv1d(in_ch, width, kernel 15, stride 2, padding 7) + BN + ReLU; sau đó các BasicBlock1D
    với số kênh [w, 2w, 4w, 4w] và số khối mỗi giai đoạn lấy từ `blocks` (stride 2 ở khối đầu của mọi giai
    đoạn trừ giai đoạn đầu); cuối cùng PoolHead.
    """

    def __init__(self, in_ch: int, n_classes: int, width: int = 32, blocks=(1, 1, 1, 1),
                 kernel: int = 7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 15, stride=2, padding=7, bias=False),   # 1000 -> 500
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
        )
        chans = [width, 2 * width, 4 * width, 4 * width]
        layers, cin = [], width
        for stage, (cout, n_blocks) in enumerate(zip(chans, blocks)):
            for b in range(n_blocks):
                stride = 2 if (b == 0 and stage > 0) else 1                  # 500 -> 250 -> 125 -> 63
                layers.append(BasicBlock1D(cin, cout, stride, kernel))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.head = PoolHead(cin, n_classes)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


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
        self.stem = conv_stem(in_ch, d_model)                                # 1000 -> 250 token
        # Chạy thử stem để biết số token. Tạm để eval() để lượt chạy thử với tensor 0
        # không làm lệch running_mean/running_var của BatchNorm.
        self.stem.eval()
        with torch.no_grad():
            n_tokens = self.stem(torch.zeros(1, in_ch, seq_len)).shape[-1]
        self.stem.train()

        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model * ff_mult,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # norm_first=True: đầu ra encoder chưa qua LayerNorm cuối -> thêm một LayerNorm
        self.norm = nn.LayerNorm(d_model)
        self.head = PoolHead(d_model, n_classes)

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)                                     # (B, D, T') -> (B, T', D)
        h = h + self.pos[:, :h.size(1)]
        h = self.norm(self.encoder(h))
        return self.head(h.transpose(1, 2))                                  # (B, T', D) -> (B, D, T')

"""Các mô hình cho ECG: CNN đơn giản (baseline DL), ResNet 1D, BiGRU/BiLSTM, CNN + attention.

Mọi mô hình nhận x có shape (B, C, T) và trả về logit (B, n_classes).
Số tham số mặc định của 4 mô hình chính xấp xỉ nhau (khoảng 0,5 triệu) để
so sánh công bằng theo yêu cầu ablation (c). Dùng `count_params` để kiểm tra.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PoolHead(nn.Module):
    """Gộp theo thời gian: nối trung bình và max, rồi Linear."""

    def __init__(self, dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * dim, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (B, D, T)
        z = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.fc(self.drop(z))


# --------------------------------------------------------------------------- #
# 1. CNN đơn giản: baseline học sâu (3 tầng conv)
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, w, 7, padding=3), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(w, 2 * w, 5, padding=2), nn.BatchNorm1d(2 * w), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(2 * w, 4 * w, 5, padding=2), nn.BatchNorm1d(4 * w), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = PoolHead(4 * w, n_classes)

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------- #
# 2. ResNet 1D
# --------------------------------------------------------------------------- #
class BasicBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.ReLU()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.short is None else self.short(x)
        return self.act(out + res)


class ResNet1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 32, blocks=(1, 1, 1, 1),
                 kernel: int = 7):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, w, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(w), nn.ReLU(),
        )
        chans = [w, 2 * w, 4 * w, 4 * w]
        layers, cin = [], w
        for i, (c, n) in enumerate(zip(chans, blocks)):
            for j in range(n):
                stride = 2 if (j == 0 and i > 0) else 1
                layers.append(BasicBlock1D(cin, c, stride, kernel))
                cin = c
        self.layers = nn.Sequential(*layers)
        self.head = PoolHead(cin, n_classes)

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


# --------------------------------------------------------------------------- #
# 3. CRNN: conv giảm mẫu + BiGRU / BiLSTM
# --------------------------------------------------------------------------- #
def conv_stem(in_ch: int, dim: int) -> nn.Sequential:
    """Giảm độ dài chuỗi 4 lần (1000 -> 250 bước) để RNN/Transformer chạy nhanh."""
    return nn.Sequential(
        nn.Conv1d(in_ch, dim, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(dim), nn.ReLU(),
        nn.Conv1d(dim, dim, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(dim), nn.ReLU(),
    )


class CRNN(nn.Module):
    # hidden mặc định chọn để GRU và LSTM đều có ~0,5 triệu tham số (LSTM có 4 cổng, GRU có 3)
    DEFAULT_HIDDEN = {"gru": 130, "lstm": 112}

    def __init__(self, in_ch: int, n_classes: int, rnn: str = "gru", stem_dim: int = 64,
                 hidden: int | None = None, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        hidden = hidden or self.DEFAULT_HIDDEN[rnn]
        self.stem = conv_stem(in_ch, stem_dim)
        cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn]
        self.rnn = cls(stem_dim, hidden, num_layers=layers, batch_first=True,
                       bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = PoolHead(2 * hidden, n_classes)

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)  # (B, T', D)
        out, _ = self.rnn(h)  # (B, T', 2H)
        return self.head(out.transpose(1, 2))


# --------------------------------------------------------------------------- #
# 4. CNN + self-attention (Transformer encoder nhỏ)
# --------------------------------------------------------------------------- #
class CNNAttention(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, d_model: int = 128, heads: int = 4,
                 layers: int = 2, ff_mult: int = 4, dropout: float = 0.1, seq_len: int = 1000):
        super().__init__()
        self.stem = conv_stem(in_ch, d_model)
        with torch.no_grad():  # xác định số token sau stem để tạo positional embedding
            n_tokens = self.stem.eval()(torch.zeros(1, in_ch, seq_len)).shape[-1]
        self.stem.train()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * ff_mult, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = PoolHead(d_model, n_classes)

    def forward(self, x):
        h = self.stem(x).transpose(1, 2) + self.pos  # (B, T', D)
        h = self.norm(self.encoder(h))
        return self.head(h.transpose(1, 2))


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

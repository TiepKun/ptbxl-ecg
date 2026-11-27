"""Mô hình RNN của vai trò C: CRNN (conv giảm mẫu + BiGRU / BiLSTM).

Nhận x có shape (B, C, T) và trả về logit (B, n_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models_common import PoolHead, conv_stem


# --------------------------------------------------------------------------- #
class CRNN(nn.Module):
    # hidden mặc định chọn để GRU và LSTM đều có ~0,5 triệu tham số (LSTM có 4 cổng, GRU có 3)
    """CRNN: tích chập giảm mẫu rồi RNN hai chiều (GRU hoặc LSTM). Khoảng 0,5 triệu tham số.

    `rnn` là "gru" hoặc "lstm"; hidden mặc định lấy từ DEFAULT_HIDDEN (đã chọn để hai loại có số tham số
    xấp xỉ nhau vì LSTM có 4 cổng, GRU có 3).
    Gợi ý: stem = conv_stem(in_ch, stem_dim) (1000 -> 250 bước); RNN batch_first, bidirectional=True,
    `layers` lớp, dropout giữa các lớp (chỉ khi layers > 1); PoolHead(2 * hidden, n_classes).
    Lưu ý: đảo trục (B, D, T') <-> (B, T', D) trước/sau RNN.
    """

    DEFAULT_HIDDEN = {"gru": 130, "lstm": 112}

    def __init__(self, in_ch: int, n_classes: int, rnn: str = "gru", stem_dim: int = 64,
                 hidden: int | None = None, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        rnn = rnn.lower()
        if rnn not in self.DEFAULT_HIDDEN:
            raise ValueError(f"rnn phải là 'gru' hoặc 'lstm', nhận được {rnn!r}")
        hidden = hidden or self.DEFAULT_HIDDEN[rnn]
        self.stem = conv_stem(in_ch, stem_dim)                               # 1000 -> 250 bước
        rnn_cls = nn.GRU if rnn == "gru" else nn.LSTM
        # PyTorch chỉ áp dropout GIỮA các lớp RNN; với 1 lớp phải để 0 (nếu không sẽ có cảnh báo)
        self.rnn = rnn_cls(stem_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = PoolHead(2 * hidden, n_classes)                          # 2 chiều -> 2 * hidden

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)                                     # (B, D, T') -> (B, T', D)
        h, _ = self.rnn(h)                                                   # (B, T', 2H)
        return self.head(h.transpose(1, 2))                                  # (B, 2H, T')

"""Thành phần dùng chung cho các mô hình (cả nhóm dùng, KHÔNG thuộc riêng ai).

Nếu cần sửa file này, báo cả nhóm trước vì sẽ ảnh hưởng đến mô hình của người khác.
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


def conv_stem(in_ch: int, dim: int) -> nn.Sequential:
    """Giảm độ dài chuỗi 4 lần (1000 -> 250 bước) để RNN/Transformer chạy nhanh."""
    return nn.Sequential(
        nn.Conv1d(in_ch, dim, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(dim), nn.ReLU(),
        nn.Conv1d(dim, dim, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(dim), nn.ReLU(),
    )

"""Test cho vai trò B: mô hình CNN (SimpleCNN, ResNet1D, CNN+attention) và vòng huấn luyện.

Chạy: python -m pytest -q tests/test_B_cnn_train.py
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from ptbxl_ecg.data import LEAD_SETS
from ptbxl_ecg.models import build_model, count_params
from ptbxl_ecg.train import EarlyStopper, train_one_epoch

CNN_MODELS = ["cnn_simple", "resnet1d", "cnn_attn"]


@pytest.mark.parametrize("name", CNN_MODELS)
@pytest.mark.parametrize("leads", ["12", "II"])
def test_forward_shape(name, leads):
    ch = len(LEAD_SETS[leads])
    m = build_model(name, ch).eval()
    out = m(torch.randn(3, ch, 1000))
    assert out.shape == (3, 5) and torch.isfinite(out).all()


def test_param_budgets():
    """Hai mô hình chính của B phải cùng ngân sách ~0,5 triệu tham số; baseline CNN thì nhỏ hơn nhiều."""
    assert count_params(build_model("cnn_simple", 12)) < 200_000
    for name in ("resnet1d", "cnn_attn"):
        n = count_params(build_model(name, 12))
        assert 300_000 <= n <= 800_000, f"{name}: {n:,} tham số"


@pytest.mark.parametrize("name", CNN_MODELS)
def test_eval_is_deterministic_and_input_dependent(name):
    m = build_model(name, 12).eval()
    x = torch.randn(2, 12, 1000)
    assert torch.allclose(m(x), m(x))
    assert not torch.allclose(m(x), m(torch.randn(2, 12, 1000)))


# ------------------------------ vòng huấn luyện ------------------------------ #
def test_early_stopper_logic():
    model = nn.Linear(2, 1)
    es = EarlyStopper(patience=2)
    assert es.update(0.5, 1, model) is False
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    assert es.update(0.6, 2, model) is False and es.best_epoch == 2
    with torch.no_grad():
        model.weight.add_(1.0)                                      # đổi trọng số sau khi lưu
    assert es.update(0.55, 3, model) is False and es.bad == 1
    assert es.update(0.54, 4, model) is True                        # 2 epoch liên tiếp không cải thiện
    assert es.best_epoch == 2 and es.best_score == pytest.approx(0.6)
    # best_state là BẢN SAO tại epoch 2, không bị đổi theo trọng số hiện tại
    assert not torch.equal(es.best_state["weight"], model.weight)


def test_early_stopper_nan_counts_as_no_improvement():
    es = EarlyStopper(patience=1)
    assert es.update(float("nan"), 1, nn.Linear(2, 1)) is True


def test_train_one_epoch_updates_weights_and_scheduler():
    torch.manual_seed(0)
    X = torch.randn(64, 12, 1000)
    Y = (torch.rand(64, 5) > 0.5).float()
    model = build_model("cnn_simple", 12)
    before = [p.detach().clone() for p in model.parameters()]
    bs, steps = 16, 4
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=3 * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_one_epoch(model, X, Y, opt, sched, scaler, nn.BCEWithLogitsLoss(), bs=bs, steps=steps,
                           device=torch.device("cpu"), use_amp=False, aug=True)
    assert isinstance(loss, float) and np.isfinite(loss) and loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert sched.last_epoch == steps                                # sched.step() sau mỗi bước


def test_train_one_epoch_reduces_loss_on_easy_task():
    torch.manual_seed(0)
    X = torch.randn(128, 12, 1000)
    Y = torch.zeros(128, 5); Y[:, 0] = (X[:, 0].mean(-1) > 0).float()
    X[:, 0] += (Y[:, 0:1] * 2 - 1) * 1.0                            # gắn tín hiệu rõ ràng vào chuyển đạo 0
    model = build_model("cnn_simple", 12)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=5 * 8)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    losses = [train_one_epoch(model, X, Y, opt, sched, scaler, nn.BCEWithLogitsLoss(), bs=16, steps=8,
                              device=torch.device("cpu"), use_amp=False, aug=False) for _ in range(5)]
    assert losses[-1] < losses[0]

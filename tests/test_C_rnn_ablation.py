"""Test cho vai trò C: mô hình RNN (GRU/LSTM), lưới ablation và saliency.

Chạy: python -m pytest -q tests/test_C_rnn_ablation.py
"""
import pytest
import torch
import torch.nn as nn

from ptbxl_ecg.ablation import build_jobs, job_name
from ptbxl_ecg.data import LEAD_SETS
from ptbxl_ecg.models import MODEL_NAMES, build_model, count_params
from ptbxl_ecg.saliency import lead_importance


# ------------------------------- mô hình RNN ------------------------------- #
@pytest.mark.parametrize("name", ["gru", "lstm"])
@pytest.mark.parametrize("leads", ["12", "II"])
def test_forward_shape(name, leads):
    ch = len(LEAD_SETS[leads])
    m = build_model(name, ch).eval()
    out = m(torch.randn(3, ch, 1000))
    assert out.shape == (3, 5) and torch.isfinite(out).all()


def test_param_budget_and_fairness():
    n_gru = count_params(build_model("gru", 12))
    n_lstm = count_params(build_model("lstm", 12))
    for n in (n_gru, n_lstm):
        assert 300_000 <= n <= 800_000
    assert abs(n_gru - n_lstm) / max(n_gru, n_lstm) < 0.15          # GRU và LSTM cùng ngân sách


def test_rnn_eval_is_deterministic():
    m = build_model("gru", 12).eval()
    x = torch.randn(2, 12, 1000)
    assert torch.allclose(m(x), m(x))


# --------------------------------- ablation --------------------------------- #
def test_build_jobs_counts():
    seeds = [0, 1, 2]
    assert len(build_jobs("main", "resnet1d", seeds)) == 5 * 3      # 5 mô hình x 3 seed
    assert len(build_jobs("leads", "resnet1d", seeds)) == 2 * 3     # limb6 và II
    assert len(build_jobs("aug", "resnet1d", seeds)) == 1 * 3       # tắt augmentation
    assert len(build_jobs("all", "resnet1d", seeds)) == 8 * 3


def test_build_jobs_content():
    jobs = build_jobs("all", "gru", [0, 1])
    main = [j for j in jobs if j["group"] == "main"]
    assert {j["model"] for j in main} == set(MODEL_NAMES)
    assert all(j["leads"] == "12" and j["aug"] is True for j in main)
    leads = [j for j in jobs if j["group"] == "leads"]
    assert {j["leads"] for j in leads} == {"limb6", "II"} and {j["model"] for j in leads} == {"gru"}
    aug = [j for j in jobs if j["group"] == "aug"]
    assert all(j["aug"] is False and j["leads"] == "12" and j["model"] == "gru" for j in aug)
    assert {j["seed"] for j in jobs} == {0, 1}
    names = [job_name(j) for j in jobs]
    assert len(set(names)) == len(names)                             # không trùng lượt chạy


def test_ablation_cli_models_filter(capsys):
    from ptbxl_ecg.ablation import main
    main(["--group", "main", "--models", "gru", "lstm", "--seeds", "0", "1", "--dry-run"])
    out = capsys.readouterr().out
    assert "Tổng số lượt chạy: 4" in out                              # 2 mô hình x 2 seed
    assert "main_gru_12_aug1_s0" in out and "main_resnet1d" not in out


# --------------------------------- saliency --------------------------------- #
def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv1d(3, 4, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(4, 5)).eval()


def test_lead_importance_shape_nonnegative():
    imp = lead_importance(_tiny(), torch.randn(8, 3, 50), cls=1)
    assert imp.shape == (3,) and (imp >= 0).all()


def test_lead_importance_zero_for_zero_lead():
    x = torch.randn(8, 3, 50)
    x[:, 2] = 0.0                                                    # gradient x input = 0 ở chuyển đạo này
    imp = lead_importance(_tiny(), x, cls=0)
    assert imp[2] == pytest.approx(0.0) and imp[0] > 0


def test_lead_importance_works_with_gru():
    m = build_model("gru", 12).eval()
    assert lead_importance(m, torch.randn(4, 12, 1000), cls=0).shape == (12,)

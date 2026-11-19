"""Kiểm thử nhanh trên CPU bằng dữ liệu giả lập: chạy được toàn bộ pipeline."""
import json

import numpy as np
import pytest
import torch

from ptbxl_ecg import synthetic
from ptbxl_ecg.data import LEAD_SETS, SUPERCLASSES, augment, load_processed, prepare
from ptbxl_ecg.models import MODEL_NAMES, build_model, count_params
from ptbxl_ecg.train import get_args, run


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    X, Y, folds = synthetic.make(n=200, seed=0)
    np.save(d / "X.npy", X); np.save(d / "Y.npy", Y); np.save(d / "folds.npy", folds)
    return d


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("leads", ["12", "II"])
def test_forward_shape(name, leads):
    ch = len(LEAD_SETS[leads])
    m = build_model(name, ch)
    assert m(torch.zeros(2, ch, 1000)).shape == (2, 5)
    assert count_params(m) > 0


def test_augment_keeps_shape():
    x = torch.randn(4, 12, 1000)
    assert augment(x).shape == x.shape


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_one_epoch(name, synth_dir, tmp_path):
    args = get_args(["--data-dir", str(synth_dir), "--model", name, "--epochs", "1",
                     "--batch-size", "32", "--out-dir", str(tmp_path / "runs"),
                     "--log-csv", str(tmp_path / "log.csv"), "--group", "test"])
    row = run(args)
    metrics = json.loads((tmp_path / "runs" / f"test_{name}_12_aug1_s0" / "test_metrics.json").read_text())
    assert 0.0 <= metrics["test_auc"] <= 1.0
    assert (tmp_path / "log.csv").exists() and row["params"] > 0


def test_prepare_from_fake_ptbxl(tmp_path):
    """Tạo thư mục PTB-XL giả (định dạng WFDB thật) để kiểm tra bước đọc nhãn + tín hiệu."""
    import pandas as pd
    import wfdb

    raw = tmp_path / "raw"
    (raw / "records100" / "00000").mkdir(parents=True)
    rows = []
    codes = [{"NORM": 100.0}, {"IMI": 100.0}, {"STACH": 100.0}, {"LAFB": 80.0, "NORM": 0.0}]
    for i in range(1, 5):
        sig = np.random.randn(1000, 12) * 0.5
        wfdb.wrsamp(f"{i:05d}_lr", fs=100, units=["mV"] * 12,
                    sig_name=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                    p_signal=sig, fmt=["16"] * 12, write_dir=str(raw / "records100" / "00000"))
        rows.append({"ecg_id": i, "scp_codes": str(codes[i - 1]), "strat_fold": 9 + (i % 2),
                     "filename_lr": f"records100/00000/{i:05d}_lr"})
    pd.DataFrame(rows).to_csv(raw / "ptbxl_database.csv", index=False)
    pd.DataFrame({"": ["NORM", "IMI", "STACH", "LAFB"], "diagnostic": [1, 1, float("nan"), 1],
                  "diagnostic_class": ["NORM", "MI", None, "CD"]}).to_csv(
        raw / "scp_statements.csv", index=False)

    prepare(raw, tmp_path / "out")
    d = load_processed(tmp_path / "out")
    # bản ghi 3 chỉ có mã STACH (không phải chẩn đoán) -> bị loại; còn 3 bản ghi
    assert d["X"].shape == (3, 12, 1000)
    assert d["Y"].shape == (3, len(SUPERCLASSES))
    assert d["Y"][0, 0] == 1 and d["Y"][1, 1] == 1  # NORM, MI
    assert d["Y"][2, 3] == 1 and d["Y"][2, 0] == 1  # CD (+ NORM vì mã NORM có trong dict)

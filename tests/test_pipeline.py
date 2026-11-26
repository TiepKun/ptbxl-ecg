"""Kiểm thử toàn pipeline trên CPU bằng dữ liệu giả lập. Chỉ chạy được khi A, B, C đã xong phần của mình.

Chạy file này SAU CÙNG (khi các bộ test_A, test_B, test_C đã xanh).
"""
import json

import pytest

from ptbxl_ecg.data import SUPERCLASSES, load_processed, prepare
from ptbxl_ecg.models import MODEL_NAMES
from ptbxl_ecg.train import get_args, run


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_one_epoch(name, synth_dir, tmp_path):
    args = get_args(["--data-dir", str(synth_dir), "--model", name, "--epochs", "1",
                     "--batch-size", "32", "--out-dir", str(tmp_path / "runs"),
                     "--log-csv", str(tmp_path / "log.csv"), "--group", "test"])
    row = run(args)
    metrics = json.loads((tmp_path / "runs" / f"test_{name}_12_aug1_s0" / "test_metrics.json").read_text())
    assert 0.0 <= metrics["test_auc"] <= 1.0
    assert (tmp_path / "log.csv").exists() and row["params"] > 0


def test_prepare_from_fake_ptbxl(fake_ptbxl, tmp_path):
    prepare(fake_ptbxl, tmp_path / "out")
    d = load_processed(tmp_path / "out")
    # bản ghi 3 chỉ có mã STACH (không phải chẩn đoán) -> bị loại; còn 3 bản ghi
    assert d["X"].shape == (3, 12, 1000)
    assert d["Y"].shape == (3, len(SUPERCLASSES))
    assert d["Y"][0, 0] == 1 and d["Y"][1, 1] == 1  # NORM, MI
    assert d["Y"][2, 3] == 1 and d["Y"][2, 0] == 1  # CD (+ NORM vì mã NORM có trong dict)

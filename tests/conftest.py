"""Fixture dùng chung cho các bộ test."""
import numpy as np
import pandas as pd
import pytest

from ptbxl_ecg import synthetic


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    """Dữ liệu giả lập (X, Y, folds) để chạy pipeline trên CPU."""
    d = tmp_path_factory.mktemp("synth")
    X, Y, folds = synthetic.make(n=200, seed=0)
    np.save(d / "X.npy", X); np.save(d / "Y.npy", Y); np.save(d / "folds.npy", folds)
    return d


@pytest.fixture()
def fake_ptbxl(tmp_path):
    """Thư mục PTB-XL giả (định dạng WFDB thật, 4 bản ghi) để thử đọc nhãn + tín hiệu."""
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
    return raw

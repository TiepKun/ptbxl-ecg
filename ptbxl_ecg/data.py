"""Dữ liệu PTB-XL: ánh xạ nhãn, cache ra .npy, chia tập, chuẩn hóa, augmentation.

Quy ước (đúng khuyến nghị của tác giả PTB-XL):
    - fold 1-8  -> train
    - fold 9    -> validation
    - fold 10   -> test
    - 5 siêu lớp chẩn đoán: NORM, MI, STTC, CD, HYP (đa nhãn)
    - dùng bản 100 Hz, mỗi bản ghi 10 giây = 1000 mẫu, 12 chuyển đạo
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Tập chuyển đạo dùng cho ablation (a): 12 / 6 chuyển đạo chi / chỉ chuyển đạo II
LEAD_SETS = {"12": list(range(12)), "limb6": [0, 1, 2, 3, 4, 5], "II": [1]}
FS = 100
SIG_LEN = 1000


# --------------------------------------------------------------------------- #
# Bước 1: đọc metadata + nhãn
# --------------------------------------------------------------------------- #
def build_labels(raw_dir: str | Path) -> pd.DataFrame:
    """Đọc ptbxl_database.csv, ánh xạ scp_codes sang 5 siêu lớp.

    Trả về DataFrame (index = ecg_id) có các cột: filename_lr, strat_fold,
    và 5 cột 0/1 tương ứng 5 siêu lớp. Bản ghi không có nhãn chẩn đoán nào bị bỏ.
    """
    raise NotImplementedError("TODO [A]: đọc metadata + ánh xạ scp_codes sang 5 siêu lớp (xem docstring)")


# --------------------------------------------------------------------------- #
# Bước 2: đọc tín hiệu và cache
# --------------------------------------------------------------------------- #
def prepare(raw_dir: str | Path, out_dir: str | Path) -> None:
    """Đọc toàn bộ tín hiệu 100 Hz và lưu thành X.npy, Y.npy, folds.npy, meta.json."""
    import wfdb  # import muộn để các phần khác chạy được khi chưa cài wfdb
    from tqdm import tqdm

    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_labels(raw)
    print(f"Số bản ghi có nhãn chẩn đoán: {len(df)}")

    X = np.zeros((len(df), 12, SIG_LEN), dtype=np.float32)
    for i, fname in enumerate(tqdm(df["filename_lr"].tolist(), desc="Đọc ECG")):
        sig, _ = wfdb.rdsamp(str(raw / fname))  # (1000, 12)
        X[i] = sig.T.astype(np.float32)

    X = np.nan_to_num(X)  # phòng trường hợp có NaN
    np.save(out / "X.npy", X)
    np.save(out / "Y.npy", df[SUPERCLASSES].to_numpy(dtype=np.uint8))
    np.save(out / "folds.npy", df["strat_fold"].to_numpy(dtype=np.int16))
    np.save(out / "ecg_ids.npy", df.index.to_numpy(dtype=np.int64))
    meta = {"classes": SUPERCLASSES, "leads": LEAD_NAMES, "fs": FS, "n": int(len(df))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Đã lưu vào {out}")


# --------------------------------------------------------------------------- #
# Bước 3: nạp dữ liệu đã cache, chia tập, chuẩn hóa
# --------------------------------------------------------------------------- #
def load_processed(data_dir: str | Path) -> dict:
    d = Path(data_dir)
    if not (d / "X.npy").exists():
        raise FileNotFoundError(
            f"Không thấy {d}/X.npy. Hãy chạy `python -m ptbxl_ecg.prepare_data` "
            "(hoặc `python -m ptbxl_ecg.synthetic` để tạo dữ liệu giả lập thử code)."
        )
    return {
        "X": np.load(d / "X.npy"),
        "Y": np.load(d / "Y.npy"),
        "folds": np.load(d / "folds.npy"),
    }


def split_indices(folds: np.ndarray) -> dict[str, np.ndarray]:
    """Chia chỉ số theo strat_fold: train = fold 1-8, val = fold 9, test = fold 10.

    folds: mảng (N,) chứa số fold của từng bản ghi. Trả về dict {"train", "val", "test"} -> mảng chỉ số
    (tăng dần). Ba tập không giao nhau.
    """
    raise NotImplementedError("TODO [A]: chia train/val/test theo strat_fold")


def zscore_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tính (mean, std) theo từng chuyển đạo từ tập TRAIN.

    X_train: (N, C, T). Trả về hai mảng shape (1, C, 1) để broadcast khi chuẩn hóa.
    std cộng thêm 1e-6 để tránh chia cho 0.
    """
    raise NotImplementedError("TODO [A]: tính mean, std theo từng chuyển đạo trên tập train")


def get_splits(data_dir: str | Path, leads: str = "12", max_train: int | None = None,
               seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Trả về {'train': (X, Y), 'val': ..., 'test': ...} đã chọn chuyển đạo và chuẩn hóa.

    Chuẩn hóa z-score theo từng chuyển đạo, thống kê chỉ tính trên tập train
    (tránh rò rỉ thông tin từ val/test).
    """
    data = load_processed(data_dir)
    idx = split_indices(data["folds"])
    if max_train is not None and len(idx["train"]) > max_train:
        rng = np.random.default_rng(seed)
        idx["train"] = np.sort(rng.choice(idx["train"], max_train, replace=False))

    lead_idx = LEAD_SETS[leads]
    X = data["X"][:, lead_idx, :]
    mean, std = zscore_stats(X[idx["train"]])
    out = {}
    for name, ii in idx.items():
        out[name] = (((X[ii] - mean) / std).astype(np.float32), data["Y"][ii].astype(np.float32))
    return out


# --------------------------------------------------------------------------- #
# Augmentation (chạy trên batch, trên GPU/CPU)
# --------------------------------------------------------------------------- #
def augment(x, noise_std: float = 0.05, scale_range: float = 0.2, max_shift: int = 50):
    """Nhiễu Gauss + co giãn biên độ + dịch thời gian (vòng tròn) cho batch (B, C, T)."""
    import torch

    b, c, t = x.shape
    dev = x.device
    scale = 1.0 + (torch.rand(b, 1, 1, device=dev) - 0.5) * 2 * scale_range
    x = x * scale
    x = x + noise_std * torch.randn_like(x)
    shifts = torch.randint(-max_shift, max_shift + 1, (b,), device=dev)
    idx = (torch.arange(t, device=dev)[None, :] - shifts[:, None]) % t  # (B, T)
    return torch.gather(x, 2, idx[:, None, :].expand(b, c, t))

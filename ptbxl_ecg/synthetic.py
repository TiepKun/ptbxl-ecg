"""Sinh dữ liệu ECG GIẢ LẬP cùng định dạng với PTB-XL đã cache, chỉ để thử code chạy được.

KHÔNG dùng cho báo cáo kết quả thật.
    python -m ptbxl_ecg.synthetic --out-dir data/synthetic --n 1000
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import FS, LEAD_NAMES, SIG_LEN, SUPERCLASSES


def make(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(SIG_LEN) / FS
    Y = (rng.random((n, len(SUPERCLASSES))) < 0.3).astype(np.uint8)
    Y[Y.sum(1) == 0, 0] = 1  # mỗi mẫu có ít nhất 1 nhãn
    X = np.zeros((n, 12, SIG_LEN), dtype=np.float32)
    for i in range(n):
        hr = rng.uniform(1.0, 1.6)  # nhịp tim giả (Hz)
        base = np.sin(2 * np.pi * hr * t)[None, :] ** 15  # chuỗi xung giả QRS
        sig = base * rng.uniform(0.8, 1.2, (12, 1))
        for k in range(1, len(SUPERCLASSES)):  # mỗi lớp bệnh thêm một dấu hiệu riêng
            if Y[i, k]:
                sig[(2 * k) % 12:(2 * k) % 12 + 3] += 0.4 * np.sin(2 * np.pi * (3 + 2 * k) * t)
        X[i] = sig + 0.15 * rng.standard_normal((12, SIG_LEN))
    folds = rng.integers(1, 11, n).astype(np.int16)
    return X, Y, folds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    X, Y, folds = make(a.n, a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "Y.npy", Y)
    np.save(out / "folds.npy", folds)
    np.save(out / "ecg_ids.npy", np.arange(a.n))
    (out / "meta.json").write_text(json.dumps({"classes": SUPERCLASSES, "leads": LEAD_NAMES,
                                               "fs": FS, "n": a.n, "synthetic": True}))
    print(f"Đã tạo dữ liệu giả lập {a.n} mẫu tại {out}")


if __name__ == "__main__":
    main()

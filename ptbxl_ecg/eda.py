"""Thống kê mô tả: phân bố nhãn theo tập train/val/test và hình minh họa tín hiệu ECG.

    python -m ptbxl_ecg.eda --data-dir data/processed
Kết quả: results/class_distribution.csv, results/class_distribution.png, results/ecg_examples.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FS, LEAD_NAMES, SUPERCLASSES, load_processed, split_indices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = load_processed(a.data_dir)
    idx = split_indices(d["folds"])
    rows = []
    for split, ii in idx.items():
        y = d["Y"][ii]
        rows.append({"split": split, "n": len(ii),
                     **{c: int(y[:, k].sum()) for k, c in enumerate(SUPERCLASSES)},
                     "mean_labels_per_record": float(y.sum(1).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "class_distribution.csv", index=False)
    print(df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frac = df.set_index("split")[SUPERCLASSES].div(df.set_index("split")["n"], axis=0)
    ax = frac.T.plot.bar(figsize=(6, 3.5))
    ax.set_ylabel("Tỉ lệ bản ghi có nhãn")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(len(SUPERCLASSES), 1, figsize=(10, 2 * len(SUPERCLASSES)), sharex=True)
    t = np.arange(d["X"].shape[-1]) / FS
    for k, c in enumerate(SUPERCLASSES):
        i = np.where((d["Y"][:, k] == 1) & (d["Y"].sum(1) == 1))[0]
        i = i[0] if len(i) else np.where(d["Y"][:, k] == 1)[0][0]
        axes[k].plot(t, d["X"][i, 1], lw=0.8)  # chuyển đạo II
        axes[k].set_ylabel(c)
    axes[-1].set_xlabel(f"Thời gian (s) - chuyển đạo {LEAD_NAMES[1]}")
    plt.tight_layout()
    plt.savefig(out / "ecg_examples.png", dpi=150)
    print("Đã lưu hình vào", out)


if __name__ == "__main__":
    main()

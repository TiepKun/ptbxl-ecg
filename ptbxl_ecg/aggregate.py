"""Tổng hợp logs/experiments.csv thành bảng trung bình ± độ lệch chuẩn theo cấu hình + biểu đồ.

    python -m ptbxl_ecg.aggregate
Kết quả: results/summary.csv, results/summary.md, results/test_auc.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SUPERCLASSES


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp các lượt chạy thành bảng trung bình ± độ lệch chuẩn theo (model, leads, aug).

    - Bỏ các dòng có group == "dev".
    - Nếu cùng (model, leads, aug, seed) chạy nhiều lần thì giữ lần cuối.
    - Kết quả có các cột: model, leads, aug, n_seeds, params, và với mỗi chỉ số trong
      ["test_auc", "val_auc", "test_f1_macro", "auc_<lớp>" ...] có cặp <chỉ số>_mean, <chỉ số>_std
      (std dùng ddof=1; chỉ có 1 seed thì std = 0).
    - Sắp xếp giảm dần theo test_auc_mean.
    """
    keys = ["model", "leads", "aug"]
    df = df[df["group"] != "dev"]
    df = df.drop_duplicates(subset=keys + ["seed"], keep="last")  # CSV ghi nối tiếp -> dòng sau là lần chạy mới hơn

    metrics = ["test_auc", "val_auc", "test_f1_macro"] + [f"auc_{c}" for c in SUPERCLASSES]
    g = df.groupby(keys, sort=False)
    out = g.agg(n_seeds=("seed", "size"), params=("params", "first"))
    for m in metrics:
        out[f"{m}_mean"] = g[m].mean()
        out[f"{m}_std"] = g[m].std(ddof=1).fillna(0.0)  # 1 seed -> std NaN -> 0
    return out.reset_index().sort_values("test_auc_mean", ascending=False).reset_index(drop=True)


def aggregate(log_csv: str, out_dir: str) -> pd.DataFrame:
    out = summarize_runs(pd.read_csv(log_csv))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(out_dir) / "summary.csv", index=False)

    lines = ["| Mô hình | Chuyển đạo | Aug | Số seed | Tham số | Test macro-AUC | Test macro-F1 |",
             "|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        lines.append(f"| {r.model} | {r.leads} | {r.aug} | {r.n_seeds} | {int(r.params):,} | "
                     f"{r.test_auc_mean:.4f} ± {r.test_auc_std:.4f} | "
                     f"{r.test_f1_macro_mean:.4f} ± {r.test_f1_macro_std:.4f} |")
    lines += ["", "AUC theo từng lớp (trung bình các seed):", "",
              "| Mô hình | Chuyển đạo | Aug | " + " | ".join(SUPERCLASSES) + " |",
              "|---|---|---|" + "---|" * len(SUPERCLASSES)]
    for r in out.itertuples():
        vals = " | ".join(f"{getattr(r, f'auc_{c}_mean'):.3f}" for c in SUPERCLASSES)
        lines.append(f"| {r.model} | {r.leads} | {r.aug} | {vals} |")
    (Path(out_dir) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r.model}\n{r.leads}/aug{r.aug}" for r in out.itertuples()]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(out)), 4))
    ax.bar(range(len(out)), out["test_auc_mean"], yerr=out["test_auc_std"], capsize=3)
    ax.set_xticks(range(len(out)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    lo = float(np.nanmin(out["test_auc_mean"] - out["test_auc_std"]))
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.set_ylabel("Test macro-AUC (trung bình ± std)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "test_auc.png", dpi=150)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-csv", default="logs/experiments.csv")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    out = aggregate(a.log_csv, a.out_dir)
    print(out[["model", "leads", "aug", "n_seeds", "test_auc_mean", "test_auc_std"]].to_string(index=False))
    print(f"Đã ghi {a.out_dir}/summary.csv, summary.md, test_auc.png")


if __name__ == "__main__":
    main()

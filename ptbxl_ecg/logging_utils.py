"""Nhật ký thí nghiệm: mọi lượt chạy được ghi thêm một dòng vào logs/experiments.csv."""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path

from .data import SUPERCLASSES

FIELDS = (
    ["timestamp", "run_name", "group", "model", "leads", "aug", "seed", "params",
     "epochs_run", "best_epoch", "val_auc", "test_auc", "test_f1_macro"]
    + [f"auc_{c}" for c in SUPERCLASSES]
    + [f"f1_{c}" for c in SUPERCLASSES]
    + ["train_time_s", "device", "git_commit", "config"]
)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - không có git thì bỏ qua
        return "n/a"


def append_row(csv_path: str | Path, row: dict) -> None:
    """Ghi thêm một dòng; tự tạo file + header nếu chưa có."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "git_commit": git_commit(), **row}
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)

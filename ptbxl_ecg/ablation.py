"""Chạy lưới thí nghiệm ablation theo kế hoạch của đề xuất.

Các nhóm (group):
    main   : so sánh kiến trúc (cnn_simple, resnet1d, gru, lstm, cnn_attn), 12 chuyển đạo, có augmentation
    leads  : ablation (a) số chuyển đạo: limb6 và II (12 chuyển đạo đã có ở nhóm main)
    aug    : ablation (b) augmentation tắt (bật đã có ở nhóm main)
    all    : cả ba nhóm trên

Ví dụ:
    python -m ptbxl_ecg.ablation --group all --seeds 0 1 2 --epochs 30 --model resnet1d
    python -m ptbxl_ecg.ablation --group all --dry-run        # chỉ in danh sách lượt chạy

Lượt chạy nào đã có test_metrics.json sẽ được bỏ qua, nên có thể chạy lại sau khi Colab bị ngắt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .models import MODEL_NAMES
from .train import get_args, run

MAIN_MODELS = ["cnn_simple", "resnet1d", "gru", "lstm", "cnn_attn"]


def build_jobs(group: str, model: str, seeds: list[int]) -> list[dict]:
    jobs = []
    for s in seeds:
        if group in ("main", "all"):
            jobs += [dict(group="main", model=m, leads="12", aug=True, seed=s) for m in MAIN_MODELS]
        if group in ("leads", "all"):
            jobs += [dict(group="leads", model=model, leads=l, aug=True, seed=s)
                     for l in ("limb6", "II")]
        if group in ("aug", "all"):
            jobs.append(dict(group="aug", model=model, leads="12", aug=False, seed=s))
    return jobs


def job_name(j: dict) -> str:
    return f"{j['group']}_{j['model']}_{j['leads']}_aug{int(j['aug'])}_s{j['seed']}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--group", choices=["main", "leads", "aug", "all"], default="all")
    ap.add_argument("--model", choices=MODEL_NAMES, default="resnet1d",
                    help="mô hình dùng cho ablation leads/aug (chọn mô hình tốt nhất ở nhóm main)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--log-csv", default="logs/experiments.csv")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    jobs = build_jobs(a.group, a.model, a.seeds)
    print(f"Tổng số lượt chạy: {len(jobs)}")
    for i, j in enumerate(jobs, 1):
        name = job_name(j)
        done = (Path(a.out_dir) / name / "test_metrics.json").exists()
        print(f"[{i}/{len(jobs)}] {name} {'(đã xong, bỏ qua)' if done else ''}")
        if a.dry_run or done:
            continue
        argv_i = ["--data-dir", a.data_dir, "--out-dir", a.out_dir, "--log-csv", a.log_csv,
                  "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                  "--group", j["group"], "--model", j["model"], "--leads", j["leads"],
                  "--seed", str(j["seed"]), "--aug" if j["aug"] else "--no-aug"]
        if a.max_train:
            argv_i += ["--max-train", str(a.max_train)]
        run(get_args(argv_i))


if __name__ == "__main__":
    main()

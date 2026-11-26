"""Huấn luyện một mô hình trên PTB-XL và ghi log.

Ví dụ:
    python -m ptbxl_ecg.train --model resnet1d --seed 0 --group main
    python -m ptbxl_ecg.train --model gru --leads limb6 --no-aug --group leads
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import SUPERCLASSES, LEAD_SETS, augment, get_splits
from .logging_utils import append_row
from .metrics import macro_auc, summarize
from .models import MODEL_NAMES, build_model, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--model", choices=MODEL_NAMES, default="resnet1d")
    p.add_argument("--model-kwargs", default="{}", help='JSON, ví dụ \'{"hidden": 128}\'')
    p.add_argument("--leads", choices=list(LEAD_SETS), default="12")
    p.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10, help="dừng sớm nếu val AUC không tăng")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-train", type=int, default=None, help="giới hạn số mẫu train (debug)")
    p.add_argument("--group", default="dev", help="nhãn nhóm thí nghiệm: main / leads / aug / dev")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--log-csv", default="logs/experiments.csv")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, device: torch.device, bs: int = 512):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model(X[i:i + bs].to(device)).float().cpu())
    return torch.cat(outs)


class EarlyStopper:
    """Theo dõi điểm validation, giữ trọng số tốt nhất và báo khi nên dừng.

    update(score, epoch, model) -> True nếu nên dừng.
    - score là NaN thì coi như -1.0 (không cải thiện).
    - Cải thiện khi score > best_score (nghiêm ngặt): lưu bản sao state_dict, đặt best_epoch,
      đưa bộ đếm 'không cải thiện' về 0.
    - Ngược lại tăng bộ đếm; dừng khi bộ đếm >= patience.
    Thuộc tính: best_score (khởi tạo -1.0), best_epoch (0), best_state (None), bad (0).
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score, self.best_epoch, self.best_state, self.bad = -1.0, 0, None, 0

    def update(self, score: float, epoch: int, model: nn.Module) -> bool:
        raise NotImplementedError("TODO [B]: cập nhật điểm tốt nhất, lưu bản sao trọng số, quyết định dừng sớm")


def train_one_epoch(model, X, Y, opt, sched, scaler, loss_fn, *, bs: int, steps: int,
                    device: torch.device, use_amp: bool, aug: bool) -> float:
    """Huấn luyện một epoch, trả về loss trung bình trên `steps` bước.

    Mỗi bước: lấy batch ngẫu nhiên không lặp (hoán vị mới mỗi epoch), đưa lên `device`,
    augment nếu `aug`, tính loss trong autocast (chuyển logit về float trước khi tính loss),
    backward qua `scaler`, unscale rồi cắt gradient (max-norm 1.0), bước optimizer, scaler.update,
    bước scheduler (sched.step() sau MỖI bước, vì OneCycleLR đếm theo bước).
    """
    raise NotImplementedError("TODO [B]: cài đặt một epoch huấn luyện (xem docstring)")


def run(args: argparse.Namespace) -> dict:
    """Huấn luyện + đánh giá một cấu hình. Trả về dict kết quả (cũng được ghi vào CSV)."""
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    use_amp = bool(args.amp and device.type == "cuda")
    run_name = args.run_name or f"{args.group}_{args.model}_{args.leads}_aug{int(args.aug)}_s{args.seed}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- dữ liệu ----
    splits = get_splits(args.data_dir, args.leads, args.max_train, args.seed)
    Xtr, Ytr = (torch.from_numpy(a) for a in splits["train"])
    Xva, Yva = (torch.from_numpy(a) for a in splits["val"])
    Xte, Yte = (torch.from_numpy(a) for a in splits["test"])
    if device.type == "cuda":  # PTB-XL 100 Hz chỉ ~1 GB nên để hẳn trên GPU cho nhanh
        Xtr, Ytr = Xtr.to(device), Ytr.to(device)

    # ---- mô hình ----
    kw = json.loads(args.model_kwargs)
    model = build_model(args.model, in_ch=Xtr.shape[1], n_classes=len(SUPERCLASSES),
                        seq_len=Xtr.shape[2], **kw).to(device)
    n_params = count_params(model)
    print(f"[{run_name}] {args.model}: {n_params:,} tham số | device={device} | "
          f"train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    bs = min(args.batch_size, len(Xtr))
    steps = max(1, len(Xtr) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps,
                                                pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()

    # ---- vòng huấn luyện ----
    history, stopper = [], EarlyStopper(args.patience)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, Xtr, Ytr, opt, sched, scaler, loss_fn, bs=bs, steps=steps,
                                     device=device, use_amp=use_amp, aug=args.aug)
        val_logits = predict_logits(model, Xva, device)
        val_loss = loss_fn(val_logits, Yva).item()
        val_auc = macro_auc(Yva.numpy(), torch.sigmoid(val_logits).numpy())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_auc": val_auc, "lr": sched.get_last_lr()[0],
                        "elapsed_s": time.time() - t0})
        print(f"  epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val AUC {val_auc:.4f}")
        if stopper.update(val_auc, epoch, model):
            print(f"  dừng sớm ở epoch {epoch} (tốt nhất: epoch {stopper.best_epoch})")
            break
    train_time = time.time() - t0

    # ---- đánh giá với checkpoint tốt nhất ----
    model.load_state_dict(stopper.best_state)
    p_val = torch.sigmoid(predict_logits(model, Xva, device)).numpy()
    p_test = torch.sigmoid(predict_logits(model, Xte, device)).numpy()
    metrics = summarize(Yva.numpy(), p_val, Yte.numpy(), p_test)

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    torch.save({"state_dict": stopper.best_state, "model": args.model, "model_kwargs": kw,
                "leads": args.leads, "n_classes": len(SUPERCLASSES)}, run_dir / "best.pt")
    np.save(run_dir / "preds_val.npy", p_val)
    np.save(run_dir / "preds_test.npy", p_test)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    row = {"run_name": run_name, "group": args.group, "model": args.model, "leads": args.leads,
           "aug": int(args.aug), "seed": args.seed, "params": n_params,
           "epochs_run": len(history), "best_epoch": stopper.best_epoch, "train_time_s": round(train_time, 1),
           "device": str(device), "config": json.dumps({"lr": args.lr, "bs": bs,
                                                         "wd": args.weight_decay, "kw": kw}),
           **metrics}
    append_row(args.log_csv, row)
    print(f"[{run_name}] val AUC {metrics['val_auc']:.4f} | TEST AUC {metrics['test_auc']:.4f} "
          f"| test F1 {metrics['test_f1_macro']:.4f}")
    return row


def main() -> None:
    run(get_args())


if __name__ == "__main__":
    main()

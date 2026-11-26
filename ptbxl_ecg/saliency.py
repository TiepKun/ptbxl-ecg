"""Saliency theo chuyển đạo: chuyển đạo nào đóng góp nhiều nhất cho từng lớp chẩn đoán.

    python -m ptbxl_ecg.saliency --run-dir runs/main_resnet1d_12_aug1_s0 --n 100
Kết quả: results/saliency_<run>.png (độ quan trọng trung bình theo chuyển đạo và theo lớp).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import LEAD_NAMES, LEAD_SETS, SUPERCLASSES, get_splits
from .models import build_model


def lead_importance(model, X: torch.Tensor, cls: int) -> np.ndarray:
    """|gradient x input| của logit lớp `cls`, trung bình theo mẫu và thời gian -> (C,)."""
    raise NotImplementedError("TODO [C]: tính |gradient x input| theo chuyển đạo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--n", type=int, default=100, help="số mẫu test dương tính mỗi lớp")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    ckpt = torch.load(Path(a.run_dir) / "best.pt", map_location="cpu", weights_only=False)
    leads = ckpt["leads"]
    X, Y = get_splits(a.data_dir, leads)["test"]
    model = build_model(ckpt["model"], in_ch=X.shape[1], n_classes=ckpt["n_classes"],
                        seq_len=X.shape[2], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    imp = np.zeros((len(SUPERCLASSES), X.shape[1]))
    for k in range(len(SUPERCLASSES)):
        pos = np.where(Y[:, k] == 1)[0][: a.n]
        if len(pos):
            v = lead_importance(model, torch.from_numpy(X[pos]), k)
            imp[k] = v / (v.sum() + 1e-12)  # chuẩn hóa để tổng = 1 theo từng lớp

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [LEAD_NAMES[i] for i in LEAD_SETS[leads]]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(imp, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(SUPERCLASSES)), SUPERCLASSES)
    ax.set_title(f"Độ quan trọng theo chuyển đạo ({Path(a.run_dir).name})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"saliency_{Path(a.run_dir).name}.png", dpi=150)
    (out / f"saliency_{Path(a.run_dir).name}.json").write_text(
        json.dumps({"leads": names, "classes": SUPERCLASSES, "importance": imp.tolist()}, indent=2))
    print("Đã lưu hình saliency vào", out)


if __name__ == "__main__":
    main()

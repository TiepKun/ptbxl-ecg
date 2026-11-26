"""Baseline cổ điển: đặc trưng thủ công + Logistic Regression / Random Forest.

    python -m ptbxl_ecg.baseline --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import LEAD_SETS, SUPERCLASSES, load_processed, split_indices
from .features import extract_features
from .logging_utils import append_row
from .metrics import summarize


def _proba(clf, X, n_classes):
    """Chuẩn hóa đầu ra predict_proba của LR (ma trận) và RF (list) về (N, K)."""
    p = clf.predict_proba(X)
    if isinstance(p, list):
        out = np.zeros((len(X), n_classes))
        for k, pk in enumerate(p):
            out[:, k] = pk[:, 1] if pk.shape[1] > 1 else 0.0
        return out
    return p


def build_baselines(seed: int) -> dict:
    """Trả về {tên: bộ phân loại} cho baseline cổ điển (đa nhãn, có fit và predict_proba).

    Cần có tên "baseline_lr" (Logistic Regression) và "baseline_rf" (Random Forest).
    """
    raise NotImplementedError("TODO [A]: tạo Logistic Regression (chuẩn hóa + OneVsRest) và Random Forest")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--leads", choices=list(LEAD_SETS), default="12")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--log-csv", default="logs/experiments.csv")
    ap.add_argument("--max-train", type=int, default=None)
    a = ap.parse_args(argv)

    data = load_processed(a.data_dir)
    idx = split_indices(data["folds"])
    if a.max_train:
        idx["train"] = idx["train"][: a.max_train]
    X = data["X"][:, LEAD_SETS[a.leads], :]
    print("Trích đặc trưng thủ công ...")
    F = {k: extract_features(X[ii]) for k, ii in idx.items()}
    Y = {k: data["Y"][ii] for k, ii in idx.items()}
    K = len(SUPERCLASSES)

    for seed in a.seeds:
        models = build_baselines(seed)
        for name, clf in models.items():
            t0 = time.time()
            clf.fit(F["train"], Y["train"])
            m = summarize(Y["val"], _proba(clf, F["val"], K), Y["test"], _proba(clf, F["test"], K))
            append_row(a.log_csv, {
                "run_name": f"baseline_{name}_{a.leads}_s{seed}", "group": "baseline",
                "model": name, "leads": a.leads, "aug": 0, "seed": seed, "params": 0,
                "epochs_run": 0, "best_epoch": 0, "train_time_s": round(time.time() - t0, 1),
                "device": "cpu", "config": json.dumps({}), **m})
            print(f"[{name} seed={seed}] val AUC {m['val_auc']:.4f} | TEST AUC {m['test_auc']:.4f}")


if __name__ == "__main__":
    main()

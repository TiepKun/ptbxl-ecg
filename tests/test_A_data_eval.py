"""Test cho vai trò A: dữ liệu, đặc trưng, chỉ số đánh giá, baseline, tổng hợp kết quả.

Chạy: python -m pytest -q tests/test_A_data_eval.py
"""
import numpy as np
import pandas as pd
import pytest

from ptbxl_ecg import aggregate, baseline, data, features, metrics
from ptbxl_ecg.data import SUPERCLASSES


# ----------------------------- data.py ----------------------------- #
def test_build_labels(fake_ptbxl):
    df = data.build_labels(fake_ptbxl)
    assert list(df.columns) == ["filename_lr", "strat_fold", *SUPERCLASSES]
    assert len(df) == 3                      # bản ghi 3 chỉ có mã không phải chẩn đoán -> bị loại
    assert 3 not in df.index
    assert df.loc[1, "NORM"] == 1 and df.loc[2, "MI"] == 1
    assert df.loc[4, "CD"] == 1              # LAFB -> CD
    assert df[SUPERCLASSES].isin([0, 1]).all().all()


def test_split_indices():
    folds = np.array([1, 5, 8, 9, 10, 3, 9, 10])
    idx = data.split_indices(folds)
    assert set(idx) == {"train", "val", "test"}
    assert idx["train"].tolist() == [0, 1, 2, 5]
    assert idx["val"].tolist() == [3, 6]
    assert idx["test"].tolist() == [4, 7]


def test_zscore_stats_shape_and_values():
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, size=(50, 4, 100)).astype(np.float32)
    mean, std = data.zscore_stats(X)
    assert mean.shape == (1, 4, 1) and std.shape == (1, 4, 1)
    z = (X - mean) / std
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-4)
    assert np.allclose(z.std(axis=(0, 2)), 1, atol=1e-3)


def test_get_splits_no_leakage(synth_dir, tmp_path):
    """Đổi dữ liệu ở fold test không được làm thay đổi dữ liệu train đã chuẩn hóa."""
    before = data.get_splits(synth_dir, "12")
    d = data.load_processed(synth_dir)
    X2 = d["X"].copy()
    X2[d["folds"] == 10] += 1000.0
    out = tmp_path / "mod"; out.mkdir()
    np.save(out / "X.npy", X2); np.save(out / "Y.npy", d["Y"]); np.save(out / "folds.npy", d["folds"])
    after = data.get_splits(out, "12")
    assert np.allclose(before["train"][0], after["train"][0])
    assert not np.allclose(before["test"][0], after["test"][0])


# --------------------------- features.py --------------------------- #
def test_extract_features_shape_and_finite():
    X = np.random.randn(5, 12, 1000).astype(np.float32)
    F = features.extract_features(X)
    assert F.shape == (5, 132) and F.dtype == np.float32 and np.isfinite(F).all()


def test_extract_features_band_power_peaks_in_right_band():
    t = np.arange(1000) / 100.0
    X = np.sin(2 * np.pi * 10 * t)[None, None, :].astype(np.float32)   # sóng 10 Hz
    f = features.extract_features(X)[0]                                # 11 đặc trưng: 6 thống kê + 5 dải
    band_logpower = f[6:]
    assert int(np.argmax(band_logpower)) == 2                          # dải (8, 15) Hz


# ---------------------------- metrics.py --------------------------- #
def test_per_class_auc_perfect_reversed_and_single_class():
    y = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1]])
    p = np.array([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5], [0.2, 0.8, 0.5], [0.8, 0.2, 0.5]])
    auc = metrics.per_class_auc(y, p)
    assert auc[0] == pytest.approx(1.0) and auc[1] == pytest.approx(1.0)
    assert np.isnan(auc[2])                                            # cột chỉ có nhãn 1
    assert metrics.per_class_auc(y, 1 - p)[0] == pytest.approx(0.0)


def test_macro_auc_ignores_nan_and_handles_all_nan():
    y = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    p = np.array([[0.1, 0.3], [0.9, 0.3], [0.2, 0.3], [0.8, 0.3]])
    assert metrics.macro_auc(y, p) == pytest.approx(1.0)
    assert np.isnan(metrics.macro_auc(np.ones((4, 2)), p))


def test_best_thresholds_and_f1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(200, 3))
    p = np.clip(y * 0.6 + rng.uniform(0, 0.4, size=y.shape), 0, 1)     # tách được ở ngưỡng ~0.5-0.6
    thr = metrics.best_thresholds(y, p)
    assert thr.shape == (3,) and ((thr >= 0.05) & (thr <= 0.95)).all()
    f1 = metrics.f1_per_class(y, p, thr)
    assert len(f1) == 3 and all(f > 0.95 for f in f1)
    assert metrics.f1_per_class(y, p, np.full(3, 0.99)) != f1          # ngưỡng khác -> F1 khác


# ----------------------------- baseline ---------------------------- #
def test_build_baselines_fit_predict():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(80, 10))
    Y = rng.integers(0, 2, size=(80, 5))
    models = baseline.build_baselines(seed=0)
    assert {"baseline_lr", "baseline_rf"} <= set(models)
    for clf in models.values():
        clf.fit(F, Y)
        assert baseline._proba(clf, F, 5).shape == (80, 5)


# ----------------------------- aggregate --------------------------- #
def _runs(rows):
    base = {"group": "main", "leads": "12", "aug": 1, "params": 1000, "val_auc": 0.8,
            "test_f1_macro": 0.5, **{f"auc_{c}": 0.8 for c in SUPERCLASSES}}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_summarize_runs_mean_std_and_order():
    df = _runs([
        dict(model="a", seed=0, test_auc=0.80), dict(model="a", seed=1, test_auc=0.90),
        dict(model="a", seed=2, test_auc=0.85), dict(model="b", seed=0, test_auc=0.95),
    ])
    out = aggregate.summarize_runs(df)
    a = out[out["model"] == "a"].iloc[0]
    assert a["n_seeds"] == 3
    assert a["test_auc_mean"] == pytest.approx(0.85)
    assert a["test_auc_std"] == pytest.approx(0.05)                    # ddof=1
    b = out[out["model"] == "b"].iloc[0]
    assert b["n_seeds"] == 1 and b["test_auc_std"] == 0.0              # 1 seed -> std = 0
    assert out.iloc[0]["model"] == "b"                                 # giảm dần theo test_auc_mean


def test_summarize_runs_drops_dev_and_keeps_last_duplicate():
    df = _runs([
        dict(model="a", seed=0, test_auc=0.10),
        dict(model="a", seed=0, test_auc=0.90),                        # chạy lại cùng seed -> giữ lần cuối
        dict(model="a", seed=1, test_auc=0.50, group="dev"),           # nhóm dev bị bỏ
    ])
    out = aggregate.summarize_runs(df)
    assert len(out) == 1 and out.iloc[0]["n_seeds"] == 1
    assert out.iloc[0]["test_auc_mean"] == pytest.approx(0.90)

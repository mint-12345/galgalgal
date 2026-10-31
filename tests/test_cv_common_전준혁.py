"""`src/cv_common.py`(공통 fold·보정 t-검정)를 확인한다.

1. 보정 t-검정: 비율 0이면 일반 paired t-검정과 같고, 비율이 있으면 손으로 계산한 값과 같다
2. fold 누수 점검: 같은 키·같은 WAFER_ID가 학습·평가 양쪽에 있으면 멈춘다
3. 실제 데이터(있을 때만): 25개 fold, 반복마다 모든 행이 한 번씩 평가, 누수 0, 조각 크기 479~481

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv_common as cv  # noqa: E402

HAS_DATA = cv.FEATURES.exists() and cv.FOLDS.exists() and cv.TABLE.exists()

DIFF = np.array([0.8, -0.2, 1.1, 0.4, 0.9, 0.3, -0.5, 1.4, 0.6, 0.2])


def test_ttest_ratio_zero_equals_plain_paired_t():
    out = cv.corrected_ttest(DIFF, n_train=1, n_eval=1e-12)   # 비율 ≈ 0
    ref = stats.ttest_1samp(DIFF, 0.0)
    assert out["t"] == pytest.approx(ref.statistic, rel=1e-9)
    assert out["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert out["df"] == len(DIFF) - 1


def test_ttest_matches_hand_formula():
    n_tr = np.array([400, 410, 390, 400, 405, 395, 400, 400, 402, 398])
    n_ev = np.array([100, 95, 105, 100, 98, 102, 100, 100, 99, 101])
    out = cv.corrected_ttest(DIFF, n_tr, n_ev)
    ratio = (n_ev / n_tr).mean()
    se = np.sqrt((1 / len(DIFF) + ratio) * DIFF.var(ddof=1))
    t = DIFF.mean() / se
    assert out["ratio"] == pytest.approx(ratio)
    assert out["t"] == pytest.approx(t)
    assert out["p"] == pytest.approx(2 * stats.t.sf(abs(t), len(DIFF) - 1))
    # 보정 t는 일반 t보다 작다 (보수적)
    assert abs(out["t"]) < abs(stats.ttest_1samp(DIFF, 0).statistic)


def test_ttest_sign_and_degenerate_cases():
    assert cv.corrected_ttest(-DIFF, 400, 100)["t"] < 0
    flat = cv.corrected_ttest(np.zeros(5), 400, 100)
    assert flat["p"] == 1.0 and flat["t"] == 0.0
    with pytest.raises(ValueError):
        cv.corrected_ttest([1.0], 400, 100)
    with pytest.raises(ValueError):
        cv.corrected_ttest([1.0, np.nan], 400, 100)


def test_compare_pairs_by_fold():
    rows = []
    for r in range(4):
        rows.append({"model": "a", "r_id": r, "subset": "all", "n_train": 300, "n_eval": 100, "mse": 1.0 + r})
        rows.append({"model": "b", "r_id": 3 - r, "subset": "all", "n_train": 300, "n_eval": 100,
                     "mse": 2.0 + (3 - r) * 1.1})
    out = cv.compare(pd.DataFrame(rows), "a", "b")
    d = np.array([1.0 + r - (2.0 + r * 1.1) for r in range(4)])   # r_id로 짝지은 차이
    assert out["mean_diff"] == pytest.approx(d.mean())
    assert out["ratio"] == pytest.approx(1 / 3)


def _toy():
    d = pd.DataFrame({"WAFER_ID": [1, 1, 2, 3], "STAGE": ["A", "B", "A", "A"]})
    return d


def test_leak_check_catches_shared_wafer_id():
    d = _toy()
    cv.check_no_leak(d, np.array([0, 1]), np.array([2, 3]))       # 정상
    with pytest.raises(ValueError, match="WAFER_ID"):
        cv.check_no_leak(d, np.array([0, 2]), np.array([1, 3]))  # WAFER_ID 1의 A·B가 나뉨
    with pytest.raises(ValueError):
        cv.check_no_leak(d, np.array([0, 2]), np.array([2, 3]))  # 같은 행


@pytest.mark.skipif(not HAS_DATA, reason="data/processed 파일 없음")
def test_real_folds_are_complete_and_leak_free():
    data = cv.load_dataset()
    assert len(data) == cv.N_ROWS
    assert data["split"].isin(["train", "val"]).all()
    folds = list(cv.iter_folds(data))     # iter_folds 안에서 누수 점검
    assert [f.r_id for f in folds] == list(range(25))
    for rep in range(5):
        ev = np.concatenate([f.eval for f in folds if f.repeat == rep])
        assert np.array_equal(np.sort(ev), np.arange(cv.N_ROWS))   # 반복마다 정확히 한 번
    assert all(479 <= len(f.eval) <= 481 for f in folds)          # folds_v1.md 검증 3
    assert all(len(f.train) + len(f.eval) == cv.N_ROWS for f in folds)


@pytest.mark.skipif(not HAS_DATA, reason="data/processed 파일 없음")
def test_cv_evaluate_filters_and_counts():
    data = cv.load_dataset()
    a123 = lambda df: df["group"] == "A-123"   # noqa: E731
    scores, preds = cv.cv_evaluate(data, lambda tr, ev, fd: np.full(len(ev), tr["MRR"].mean()),
                                   train_filter=a123, eval_filter=a123)
    s = scores[scores["subset"] == "all"]
    assert len(s) == 25
    assert (s.groupby("repeat")["n_eval"].sum() == 431).all()      # A-123 평가 = 431장씩
    assert (s["n_train"] + s["n_eval"] == 431).all()
    assert set(preds["group"]) == {"A-123"}
    assert preds.groupby("r_id").size().sum() == 431 * 5

"""`src/model_zoo.py`(예측 모델 정비 비교군)를 확인한다.

1. one-hot은 고정 수준 목록으로 만들어 학습·평가 쪽 열이 항상 같고, 알 수 없는 수준이면 멈춘다
2. 선형회귀의 중앙값 대체·표준화는 학습 쪽 값만 쓴다 (평가 쪽 값을 바꿔도 예측이 평가 행별로만 바뀐다)
3. 고정 하이퍼파라미터가 3.2절 값과 같다
4. 같은 입력이면 모든 모델의 예측이 재현된다

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv_common as cv  # noqa: E402
import model_zoo as mz  # noqa: E402


def toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array(cv.GROUPS)[rng.integers(0, 3, n)]
    d = pd.DataFrame({"group": g, "STAGE": np.where(g == "B-456", "B", "A"),
                      "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    d.loc[d.index[:5], "x2"] = np.nan
    d["MRR"] = 3 * d["x1"] + (d["group"] == "A-123") * 70 + rng.normal(scale=0.1, size=n)
    return d


def test_one_hot_fixed_columns():
    d = toy()
    oh = mz.one_hot(d[d["group"] == "A-456"])          # 수준 하나만 있어도 열은 5개
    assert list(oh.columns) == ["is_group_A-456", "is_group_B-456", "is_group_A-123", "is_STAGE_A", "is_STAGE_B"]
    assert (oh["is_group_A-456"] == 1).all() and (oh["is_group_A-123"] == 0).all()
    bad = d.head(2).assign(group="C-789")
    with pytest.raises(ValueError):
        mz.one_hot(bad)


def test_lr_preprocessing_uses_train_only():
    d = toy()
    tr, ev = d.iloc[:40], d.iloc[40:].copy()
    fp = mz.make_fit_predict("lr", ["x1", "x2"], "onehot")
    p1 = fp(tr, ev)
    ev2 = ev.copy()
    ev2.iloc[0, ev2.columns.get_loc("x2")] = 1e6      # 평가 행 하나를 극단값으로
    p2 = fp(tr, ev2)
    assert np.allclose(p1[1:], p2[1:])                 # 다른 평가 행의 예측은 그대로 (평가 쪽 통계 미사용)


def test_fixed_hyperparameters():
    assert {k: cv.XGB_PARAMS[k] for k in ("n_estimators", "max_depth", "learning_rate", "subsample",
                                          "colsample_bytree", "min_child_weight")} == \
        dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=3)
    assert {k: cv.CAT_PARAMS[k] for k in ("iterations", "depth", "learning_rate")} == \
        dict(iterations=600, depth=4, learning_rate=0.05)
    assert mz.estimator("xgb", depth=6).get_params()["max_depth"] == 6
    assert mz.RF_PARAMS["max_features"] == 1.0


@pytest.mark.parametrize("model,cat", [("xgb", "onehot"), ("xgb", None), ("rf", "onehot"), ("lr", "onehot"),
                                       ("catboost", "native"), ("catboost", "onehot")])
def test_models_run_and_reproduce(model, cat, monkeypatch):
    if model == "catboost":
        pytest.importorskip("catboost")
        # CatBoost 1.2.10 범주형(CTR)은 이 환경에서 반복당 약 0.3초라 테스트에서만 반복 수를 줄인다
        monkeypatch.setitem(cv.CAT_PARAMS, "iterations", 20)
    d = toy()
    tr, ev = d.iloc[:40], d.iloc[40:]
    fp = mz.make_fit_predict(model, ["x1", "x2"], cat)
    a, b = fp(tr, ev), fp(tr, ev)
    assert a.shape == (len(ev),) and np.isfinite(a).all()
    # RF는 n_jobs 병렬 예측의 합산 순서 때문에 마지막 자리(~1e-15)가 달라질 수 있다
    assert np.allclose(a, b, rtol=0, atol=1e-9)


def test_native_only_for_catboost():
    with pytest.raises(ValueError):
        mz.make_fit_predict("xgb", ["x1"], "native")

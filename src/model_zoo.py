"""예측 모델 정비(preregistration.md 3.2절)의 비교군을 `cv_common.cv_evaluate`에 넣을 수 있는 형태로 만든다.

- 모델 (#15): XGBoost(기본), CatBoost, RF, 선형회귀. 하이퍼파라미터는 3.2절 고정값(#16), 튜닝 없음.
  RF = `RandomForestRegressor(max_features=1.0)` (CLAUDE.md 9절 tree bagging, prereg 4.5절과 같은 설정).
- 범주형(그룹·스테이지): 0/1 단일 변수로 넣지 않는다.
  `onehot` = 고정된 수준 목록으로 만든 one-hot(데이터에 따라 열이 바뀌지 않음), `native` = CatBoost 범주형.
  범주형을 쓰지 않는 모델(그룹 하나만 학습하는 분리 모델)은 `cat=None`.
- 결측(NaN): XGBoost·CatBoost·RF는 자체 처리. 선형회귀만 학습 쪽 중앙값으로 채우고 학습 쪽 기준으로 표준화한다
  (모든 정규화는 fold 학습 쪽 기준, prereg 1절). 학습 쪽에서 전부 NaN인 열은 선형회귀에서 빠진다.
- 새 피처를 만들지 않는다 (CLAUDE.md 3절 5번). one-hot은 기존 범주형 열의 표현 방식이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import cv_common as cv

CAT_COLS = ["group", "STAGE"]
LEVELS = {"group": cv.GROUPS, "STAGE": ["A", "B"]}
RF_PARAMS = dict(max_features=1.0, random_state=cv.SEED, n_jobs=4)
MODELS = ["xgb", "catboost", "rf", "lr"]


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in CAT_COLS:
        unknown = set(df[c]) - set(LEVELS[c])
        if unknown:
            raise ValueError(f"{c}에 알 수 없는 수준 {unknown}")
        for lv in LEVELS[c]:
            out[f"is_{c}_{lv}"] = (df[c] == lv).astype(float).to_numpy()
    return pd.DataFrame(out, index=df.index)


def design(df: pd.DataFrame, cols: list[str], cat: str | None) -> pd.DataFrame:
    X = df[cols].astype(float)
    if cat == "onehot":
        X = pd.concat([X, one_hot(df)], axis=1)
    elif cat == "native":
        X = pd.concat([X, df[CAT_COLS].astype(str)], axis=1)
    elif cat is not None:
        raise ValueError(f"알 수 없는 범주형 방식: {cat}")
    return X


def estimator(model: str, depth: int | None = None, cat: str | None = None):
    if model == "xgb":
        return XGBRegressor(**{**cv.XGB_PARAMS, **({} if depth is None else {"max_depth": depth})})
    if model == "catboost":
        from catboost import CatBoostRegressor
        p = {**cv.CAT_PARAMS, **({} if depth is None else {"depth": depth})}
        if cat == "native":
            p["cat_features"] = CAT_COLS
        return CatBoostRegressor(**p)
    if model == "rf":
        return RandomForestRegressor(**RF_PARAMS)
    if model == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    raise ValueError(f"알 수 없는 모델: {model}")


def make_fit_predict(model: str, cols: list[str], cat: str | None, depth: int | None = None):
    """`cv_common.cv_evaluate`용 `fit_predict(train_df, eval_df, fold)`."""
    if cat == "native" and model != "catboost":
        raise ValueError("native 범주형은 CatBoost만")

    def fit_predict(tr: pd.DataFrame, ev: pd.DataFrame, fd=None) -> np.ndarray:
        est = estimator(model, depth, cat)
        est.fit(design(tr, cols, cat), tr["MRR"].to_numpy(float))
        return np.asarray(est.predict(design(ev, cols, cat)), float)

    return fit_predict


def params_text(model: str, depth: int | None = None, cat: str | None = None) -> str:
    if model == "xgb":
        p = {**cv.XGB_PARAMS, **({} if depth is None else {"max_depth": depth})}
        return "XGBRegressor(" + ", ".join(f"{k}={v!r}" for k, v in p.items()) + ")"
    if model == "catboost":
        p = {**cv.CAT_PARAMS, **({} if depth is None else {"depth": depth})}
        if cat == "native":
            p["cat_features"] = CAT_COLS
        return "CatBoostRegressor(" + ", ".join(f"{k}={v!r}" for k, v in p.items()) + ")"
    if model == "rf":
        return "RandomForestRegressor(" + ", ".join(f"{k}={v!r}" for k, v in RF_PARAMS.items()) + ")  # n_estimators=100 기본값"
    return "SimpleImputer(median) → StandardScaler → LinearRegression() (학습 쪽 기준)"

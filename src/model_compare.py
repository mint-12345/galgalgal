"""예측 모델 정비: 모델군·범주형 처리 비교 (preregistration.md 3.2절 #15·#16).

- 데이터: train+val 2,401행(이상치 제외), F-free 113개 + 범주형(group, STAGE). 세 그룹을 한 모델로 학습(통합).
- fold: folds_v1.csv 25개 (`cv_common`). 평가는 전체와 그룹별로 RMSE·MAE·MSE.
- 비교군: XGBoost + one-hot(기본) / CatBoost + 범주형 그대로 / CatBoost + one-hot / RF + one-hot / 선형회귀 + one-hot.
  하이퍼파라미터는 3.2절 고정값. 튜닝·피처 선택 없음.
- 검정(참고): XGBoost 기준 대비 각 모델, CatBoost 범주형 대 one-hot을 보정 t-검정. 3.2절에 모델 선택 규칙이 없으므로
  결과로 기본 모델을 바꾸지 않는다.

실행: python src/model_compare.py   (출력이 있으면 멈춘다)
출력: results/{날짜}_전준혁_model_compare.csv (+ .yaml, 검정표 _tests.csv)
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cv  # noqa: E402
import model_zoo as mz  # noqa: E402

AUTHOR = "전준혁"
CONFIGS = [("xgb", "onehot"), ("catboost", "native"), ("catboost", "onehot"), ("rf", "onehot"), ("lr", "onehot")]
BASE = "xgb+onehot"


def label(model: str, cat: str) -> str:
    return f"{model}+{cat}"


def main() -> None:
    today = date.today().isoformat()
    out = cv.ROOT / "results" / f"{today}_{AUTHOR}_model_compare.csv"
    tests_out = out.with_name(out.stem + "_tests.csv")
    for p in (out, out.with_suffix(".yaml"), tests_out):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (덮어쓰지 않음)")
    data = cv.load_dataset()
    cols = cv.feature_columns("full")
    rows, cost = [], {}
    for model, cat in CONFIGS:
        t0 = time.time()
        s, _ = cv.cv_evaluate(data, mz.make_fit_predict(model, cols, cat))
        cost[label(model, cat)] = round(time.time() - t0, 1)
        rows.append(s.assign(model=label(model, cat)))
        print(f"  {label(model, cat):18s} {cost[label(model, cat)]:6.1f}s  RMSE "
              + ", ".join(f"{g} {x:.3f}" for g, x in s.groupby("subset")["rmse"].mean().items()))
    res = pd.concat(rows, ignore_index=True)

    tests = []
    pairs = [(label(m, c), BASE) for m, c in CONFIGS if label(m, c) != BASE] + [("catboost+native", "catboost+onehot")]
    for a, b in pairs:
        for sub in ["all"] + cv.GROUPS:
            tests.append(cv.compare(res, a, b, subset=sub, metric="mse"))
    tests = pd.DataFrame(tests)

    res = res.rename(columns={"subset": "group", "r_id": "fold"}).assign(featureset="F-free+cat")
    res = res[["group", "model", "featureset", "fold", "repeat", "mse", "mae", "rmse", "n_train", "n_eval"]]
    res.round(6).to_csv(out, index=False, lineterminator="\n")
    tests.round(6).to_csv(tests_out, index=False, lineterminator="\n")

    summ = res.groupby(["model", "group"])[["rmse", "mae", "mse"]].agg(["mean", "std"]).round(3)
    print(summ.to_string())
    print(tests[["model_a", "model_b", "subset", "mean_a", "mean_b", "t", "p"]].round(4).to_string())
    lines = [
        f"date: {today}", f"author: {AUTHOR}", "script: src/model_compare.py",
        "preregistration: v0.2.1 3.2절 (#15 비교군, #16 고정 하이퍼파라미터)",
        "input: data/processed/features_v1.csv (F-free 113) + wafer_table.csv (MRR, train+val, is_outlier=0)",
        "folds: data/processed/folds_v1.csv 25 fold (repeat 0~4, 시드 42~46), cv_common.iter_folds 누수 점검",
        "featureset: F-free+cat = F-free 113개 + 범주형 group·STAGE (one-hot 또는 CatBoost 범주형)",
        "design: 세 그룹 통합 학습 1개 모델, 평가는 전체(all)와 그룹별",
        "models:",
    ] + [f"  {label(m, c)}: {mz.params_text(m, None, c)}" for m, c in CONFIGS] + [
        "categorical_levels: group [A-456, B-456, A-123], STAGE [A, B] (고정 목록 one-hot)",
        "missing: xgb·catboost·rf 자체 처리, lr만 학습 쪽 중앙값 대체 + 학습 쪽 표준화",
        "tuning: 없음", f"seed: {cv.SEED}",
        "tests: 보정 t-검정(Nadeau & Bengio), 지표 MSE, 차이 = model_a − model_b, 비율 = fold별 n_eval/n_train 평균 (참고용, 선택 규칙 없음)",
        "labels: train+val만. test·data/locked/ 사용 안 함",
        f"runtime_sec: {cost}",
        "summary_rmse_mean:",
    ]
    for (m, g), r in res.groupby(["model", "group"])["rmse"].mean().round(3).items():
        lines.append(f"  - {{model: {m}, group: {g}, rmse: {r}}}")
    out.with_suffix(".yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out.relative_to(cv.ROOT)}, {tests_out.name}")


if __name__ == "__main__":
    main()

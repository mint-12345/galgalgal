"""Step 9: Model 1 (Pure ML) 첫 기준 모델 - 1차 다중선형회귀, 5-fold 평가.

- 입력 표: data/processed/step7_wafer_stage_table_v1.csv (training, 1,977행)
- fold: data/processed/step8_cv_folds_v1.csv (WAFER_ID + STAGE 로 연결, 그대로 사용)
- 입력에서 제외: WAFER_ID, AVG_REMOVAL_RATE(정답)
- 문자 열(STAGE, CHAMBER_PATH)은 one-hot 으로, 숫자 열은 표준화로 바꾼다.
  두 변환과 회귀를 하나의 Pipeline 으로 묶어, 매 fold 마다 학습용 4개 fold 로만 fit 한다.
  평가용 fold 에는 transform / predict 만 적용하므로 평가용 정보가 변환에 쓰이지 않는다.
- 2차식, 트리 모델, feature 변경, 성능 개선은 하지 않는다. test / validation 은 사용하지 않는다.
출력: reports/step9_model1_mlr/ (fold_metrics.csv, summary.csv, oof_predictions.csv)
실행: python -B scripts/step9_model1_mlr_baseline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.width", 200)

ROOT = Path(__file__).resolve().parents[1]
TABLE_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
OUT_DIR = ROOT / "reports" / "step9_model1_mlr"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
CATEGORICAL = ["STAGE", "CHAMBER_PATH"]


def load():
    table = pd.read_csv(TABLE_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    data = table.merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(data) == 1977 and data["FOLD"].notna().all() and data[TARGET].notna().all()
    for c in CATEGORICAL:
        data[c] = data[c].astype(str)
    features = [c for c in table.columns if c not in ("WAFER_ID", TARGET)]
    numeric = [c for c in features if c not in CATEGORICAL]
    return data, features, numeric


def make_model(numeric):
    prep = ColumnTransformer([
        # drop="first": 절편과 겹치지 않도록 각 문자 열의 첫 범주를 기준으로 둔다
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", prep), ("mlr", LinearRegression())])


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def main():
    data, features, numeric = load()
    print(f"행 {len(data):,} / 입력 열 {len(features)}개 (문자 {len(CATEGORICAL)} + 숫자 {len(numeric)}) / 정답 {TARGET}")

    rows, preds = [], []
    for k in sorted(data["FOLD"].unique()):
        train, test = data[data["FOLD"] != k], data[data["FOLD"] == k]
        assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])
        model = make_model(numeric).fit(train[features], train[TARGET])
        y_pred = model.predict(test[features])

        cat_names = model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
        unseen = {c: sorted(set(test[c]) - set(train[c])) for c in CATEGORICAL}
        rows.append(dict(FOLD=int(k), n_train=len(train), n_test=len(test),
                         **metrics(test[TARGET], y_pred),
                         n_model_inputs=len(cat_names) + len(numeric),
                         onehot_columns=";".join(cat_names),
                         test_categories_not_in_train=str({c: v for c, v in unseen.items() if v} or "")))
        preds.append(test[KEY + ["CHAMBER_PATH", "FOLD"]].assign(
            MRR_TRUE=test[TARGET].to_numpy(), MRR_PRED=y_pred, ERROR=y_pred - test[TARGET].to_numpy()))

    fold_metrics = pd.DataFrame(rows)
    oof = pd.concat(preds).loc[data.index]  # Step 7 표와 같은 행 순서
    stats = fold_metrics[["R2", "MAE", "RMSE"]].agg(["mean", "std", "min", "max"]).T
    stats = stats.rename(columns={"std": "std_ddof1"})
    stats["pooled_out_of_fold"] = pd.Series(metrics(oof["MRR_TRUE"], oof["MRR_PRED"]))

    # 참고용: 모든 fold 의 예측을 모아 STAGE x CHAMBER_PATH 별로 같은 지표를 계산 (모델은 바꾸지 않음)
    group_rows = []
    for (stage, path), g in oof.groupby(["STAGE", "CHAMBER_PATH"]):
        m = metrics(g["MRR_TRUE"], g["MRR_PRED"]) if len(g) > 1 else dict(
            R2=np.nan, MAE=mean_absolute_error(g["MRR_TRUE"], g["MRR_PRED"]),
            RMSE=float(np.sqrt(mean_squared_error(g["MRR_TRUE"], g["MRR_PRED"]))))
        group_rows.append(dict(STAGE=stage, CHAMBER_PATH=path, n=len(g), **m,
                               MRR_TRUE_mean=g["MRR_TRUE"].mean(), MRR_TRUE_std=g["MRR_TRUE"].std()))
    for path, g in oof.groupby("CHAMBER_PATH"):
        group_rows.append(dict(STAGE="all", CHAMBER_PATH=path, n=len(g), **metrics(g["MRR_TRUE"], g["MRR_PRED"]),
                               MRR_TRUE_mean=g["MRR_TRUE"].mean(), MRR_TRUE_std=g["MRR_TRUE"].std()))
    group_metrics = pd.DataFrame(group_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    stats.to_csv(OUT_DIR / "summary.csv", index_label="metric")
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    group_metrics.to_csv(OUT_DIR / "group_metrics_reference.csv", index=False)

    print("\n[fold 별 평가]")
    print(fold_metrics[["FOLD", "n_train", "n_test", "R2", "MAE", "RMSE", "n_model_inputs"]].round(4).to_string(index=False))
    print("one-hot 열 (fold 1 기준):", fold_metrics.loc[0, "onehot_columns"])
    print("평가용에만 있는 범주:", fold_metrics["test_categories_not_in_train"].replace("", "없음").tolist())
    print("\n[5-fold 요약] mean, 표준편차(ddof=1), min, max, 참고: 5개 fold 예측을 모두 모아 한 번에 계산한 값")
    print(stats.round(4).to_string())
    print(f"\n예측 파일: {(OUT_DIR / 'oof_predictions.csv').relative_to(ROOT).as_posix()} ({len(oof):,}행)")
    print("예측값 범위:", round(float(oof["MRR_PRED"].min()), 3), "~", round(float(oof["MRR_PRED"].max()), 3),
          "/ 실제값 범위:", round(float(oof["MRR_TRUE"].min()), 3), "~", round(float(oof["MRR_TRUE"].max()), 3))
    print("\n[참고] STAGE x CHAMBER_PATH 별 지표 (5개 fold 예측을 모아 계산)")
    print(group_metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

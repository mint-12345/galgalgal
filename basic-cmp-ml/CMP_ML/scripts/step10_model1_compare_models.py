"""Step 10: Model 1 (Pure ML) - 같은 feature, 같은 5-fold 에서 모델 종류만 바꿔 비교.

- 입력 표: data/processed/step7_wafer_stage_table_v1.csv (training, 1,977행, feature 32개 그대로)
- fold: data/processed/step8_cv_folds_v1.csv 그대로
- 입력에서 제외: WAFER_ID, AVG_REMOVAL_RATE(정답)
- 모델 (세부 튜닝 없음, 라이브러리 기본값. 무작위성이 있는 모델만 random_state=42 고정)
    MLR   : Step 9 와 같은 1차 다중선형회귀 (one-hot + 표준화 + LinearRegression)
    POLY2 : one-hot + 표준화 → 2차 PolynomialFeatures(제곱항, 교차항) → LinearRegression
    RF    : one-hot + 숫자 그대로 → RandomForestRegressor 기본값
    XGB   : one-hot + 숫자 그대로 → XGBRegressor 기본값
  모든 전처리는 Pipeline 안에서 학습용 4개 fold 로만 fit 한다.
- 지표: fold 마다 전체 / CHAMBER_PATH 1-2-3 안 / 4-5-6 안의 R², MAE, RMSE
- 어떤 모델이 최종적으로 좋은지는 결정하지 않는다. test / validation 은 사용하지 않는다.
출력: reports/step10_model1_compare/
실행: python -B scripts/step10_model1_compare_models.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

ROOT = Path(__file__).resolve().parents[1]
TABLE_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
STEP9_OOF = ROOT / "reports" / "step9_model1_mlr" / "oof_predictions.csv"
OUT_DIR = ROOT / "reports" / "step10_model1_compare"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
CATEGORICAL = ["STAGE", "CHAMBER_PATH"]
SEED = 42
SCOPES = {"all": None, "path_1-2-3": "1-2-3", "path_4-5-6": "4-5-6"}


def load():
    table = pd.read_csv(TABLE_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    data = table.merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(data) == 1977 and data["FOLD"].notna().all()
    for c in CATEGORICAL:
        data[c] = data[c].astype(str)
    features = [c for c in table.columns if c not in ("WAFER_ID", TARGET)]
    assert len(features) == 32 and "WAFER_ID" not in features
    numeric = [c for c in features if c not in CATEGORICAL]
    return data, features, numeric


def onehot():
    return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)


def build_models(numeric):
    scaled = lambda: ColumnTransformer([("cat", onehot(), CATEGORICAL), ("num", StandardScaler(), numeric)])
    raw_num = lambda: ColumnTransformer([("cat", onehot(), CATEGORICAL), ("num", "passthrough", numeric)])
    return {
        "MLR": lambda: Pipeline([("prep", scaled()), ("reg", LinearRegression())]),
        "POLY2": lambda: Pipeline([("prep", scaled()),
                                   ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                                   ("reg", LinearRegression())]),
        "RF": lambda: Pipeline([("prep", raw_num()),
                                # n_jobs=1: 병렬 예측 시 트리 평균 순서 때문에 실행마다 1e-13 수준 차이가 나서
                                # 결과 파일을 완전히 같게 재현하려고 단일 스레드로 둔다 (모델 설정은 기본값 그대로)
                                ("reg", RandomForestRegressor(random_state=SEED, n_jobs=1))]),
        "XGB": lambda: Pipeline([("prep", raw_num()),
                                 ("reg", XGBRegressor(random_state=SEED, n_jobs=-1))]),
    }


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def scoped(frame, pred_col):
    out = {}
    for scope, path in SCOPES.items():
        part = frame if path is None else frame[frame["CHAMBER_PATH"] == path]
        out[scope] = metrics(part["MRR_TRUE"], part[pred_col]) | {"n": len(part)}
    return out


def main():
    data, features, numeric = load()
    models = build_models(numeric)
    print(f"행 {len(data):,} / feature {len(features)}개 / 모델 {list(models)}")

    fold_rows, timing_rows = [], []
    oof = data[KEY + ["CHAMBER_PATH", "FOLD"]].assign(MRR_TRUE=data[TARGET])
    for name, factory in models.items():
        pred = pd.Series(np.nan, index=data.index)
        for k in sorted(data["FOLD"].unique()):
            train, test = data[data["FOLD"] != k], data[data["FOLD"] == k]
            assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])
            t0 = time.perf_counter()
            model = factory().fit(train[features], train[TARGET])
            pred.loc[test.index] = model.predict(test[features])
            seconds = time.perf_counter() - t0
            n_inputs = model[:-1].transform(train[features].iloc[:1]).shape[1]
            frame = oof.loc[test.index].assign(PRED=pred.loc[test.index])
            for scope, m in scoped(frame, "PRED").items():
                fold_rows.append(dict(model=name, FOLD=int(k), scope=scope, **m, n_model_inputs=n_inputs))
            # 실행 시간은 실행마다 달라지므로 결과 지표 파일과 분리해 저장
            timing_rows.append(dict(model=name, FOLD=int(k), fit_predict_seconds=round(seconds, 3)))
        oof[f"PRED_{name}"] = pred
        print(f"  {name}: 완료 (모델 입력 열 {n_inputs}개)")

    # Step 9 와 같은 MLR 인지 확인
    step9 = pd.read_csv(STEP9_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    same = np.allclose(oof["PRED_MLR"].to_numpy(), step9["MRR_PRED"].to_numpy(), rtol=0, atol=1e-8)
    print(f"MLR 예측이 Step 9 저장 예측과 같은가: {same}")
    assert same

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (fold_metrics.melt(id_vars=["model", "FOLD", "scope"], value_vars=["R2", "MAE", "RMSE"],
                                 var_name="metric")
               .groupby(["model", "scope", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index())
    pooled = []
    for name in models:
        for scope, m in scoped(oof, f"PRED_{name}").items():
            pooled += [dict(model=name, scope=scope, metric=k, pooled_out_of_fold=m[k]) for k in ("R2", "MAE", "RMSE")]
    summary = summary.merge(pd.DataFrame(pooled), on=["model", "scope", "metric"])

    # 참고용: 모델별 예측 범위와 절대오차 분포 (모델은 바꾸지 않음, 임의 기준값 없음)
    true_lo, true_hi = oof["MRR_TRUE"].min(), oof["MRR_TRUE"].max()
    spread_rows = []
    for name in models:
        p = oof[f"PRED_{name}"]
        err = (p - oof["MRR_TRUE"]).abs()
        spread_rows.append(dict(model=name, pred_min=p.min(), pred_max=p.max(),
                                n_pred_outside_observed_mrr_range=int(((p < true_lo) | (p > true_hi)).sum()),
                                abs_error_median=err.median(), abs_error_p95=err.quantile(0.95),
                                abs_error_p99=err.quantile(0.99), abs_error_max=err.max()))
    spread = pd.DataFrame(spread_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    spread.to_csv(OUT_DIR / "prediction_error_distribution_reference.csv", index=False)
    for name in models:
        oof[KEY + ["CHAMBER_PATH", "FOLD", "MRR_TRUE"]].assign(
            MRR_PRED=oof[f"PRED_{name}"], ERROR=oof[f"PRED_{name}"] - oof["MRR_TRUE"]
        ).to_csv(OUT_DIR / f"oof_predictions_{name}.csv", index=False)

    order = {m: i for i, m in enumerate(models)}
    view = summary.assign(cell=summary["mean"].map("{:.4f}".format) + " ± " + summary["std_ddof1"].map("{:.4f}".format))
    for metric in ("R2", "MAE", "RMSE"):
        print(f"\n[{metric}] 5-fold 평균 ± 표준편차")
        t = view[view["metric"] == metric].pivot(index="model", columns="scope", values="cell")
        print(t.loc[sorted(t.index, key=order.get), list(SCOPES)].to_string())
    print("\n[fold 별 값]")
    wide = fold_metrics.pivot_table(index=["model", "scope", "FOLD"], values=["R2", "MAE", "RMSE"]).round(4)
    print(wide.reindex(sorted(wide.index, key=lambda i: (order[i[0]], list(SCOPES).index(i[1]), i[2])))
          [["R2", "MAE", "RMSE"]].to_string())
    print("\n[예측값 범위]")
    for name in models:
        p = oof[f"PRED_{name}"]
        print(f"  {name}: {p.min():.3f} ~ {p.max():.3f} (실제 {oof['MRR_TRUE'].min():.3f} ~ {oof['MRR_TRUE'].max():.3f})")
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(OUT_DIR / "run_time_reference.csv", index=False)
    print("\n[모델 입력 열 수 / fold 당 학습+예측 시간(초) 평균, 시간은 실행마다 달라짐]")
    print(fold_metrics[fold_metrics["scope"] == "all"].groupby("model", sort=False)["n_model_inputs"].mean()
          .to_frame().join(timing.groupby("model")["fit_predict_seconds"].mean()).round(3).to_string())
    print("\n[예측 범위와 절대오차 분포]")
    print(spread.round(3).to_string(index=False))
    print(f"\n저장 폴더: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

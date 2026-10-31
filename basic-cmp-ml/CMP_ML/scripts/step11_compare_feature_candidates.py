"""Step 11-2: Model 1 feature 후보 비교 (feature 구성 5개 x 모델 3개, Step 8 의 같은 5-fold).

- feature 구성: Baseline(Step 7 표), C1_chamber, C2_airbag, C3_usage, C4_combined (data/processed/step11/)
- 모델: Step 9 / Step 10 과 같은 설정 (튜닝 없음)
    MLR : one-hot(STAGE, CHAMBER_PATH) + 숫자 [빈 값 처리 → 표준화] → LinearRegression
          빈 값을 직접 다루지 못하므로 학습용 fold 평균으로 채우고, 학습용 fold 에 빈 값이 있던 열마다
          "원래 빈 값이었음" 0/1 열을 추가한다 (SimpleImputer(add_indicator=True)).
          Baseline 처럼 빈 값이 없으면 채우기와 표시 열이 생기지 않아 Step 9 와 같다.
    RF  : one-hot + 숫자 그대로 → RandomForestRegressor(random_state=42, n_jobs=1). 빈 값은 모델이 직접 처리.
    XGB : one-hot + 숫자 그대로 → XGBRegressor(random_state=42, n_jobs=-1). 빈 값은 모델이 직접 처리.
  모든 전처리는 Pipeline 안에서 학습용 4개 fold 로만 fit 한다.
- 지표: fold 마다 전체 / CHAMBER_PATH 1-2-3 / 4-5-6 의 R², MAE, RMSE → 평균, 표준편차(ddof=1)
- CHAMBER_PATH other 4행은 R² 없이 행별 실제값, 예측값, 절대오차만 저장한다.
- 최종 선택은 하지 않는다. test / validation 은 사용하지 않는다.
출력: reports/step11_feature_candidates/
실행: python -B scripts/step11_compare_feature_candidates.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FOLD_FILE = PROC / "step8_cv_folds_v1.csv"
STEP10_DIR = ROOT / "reports" / "step10_model1_compare"
OUT_DIR = ROOT / "reports" / "step11_feature_candidates"
FEATURE_SETS = {
    "Baseline": PROC / "step7_wafer_stage_table_v1.csv",
    "C1_chamber": PROC / "step11" / "step11_candidate1_chamber.csv",
    "C2_airbag": PROC / "step11" / "step11_candidate2_airbag.csv",
    "C3_usage": PROC / "step11" / "step11_candidate3_usage.csv",
    "C4_combined": PROC / "step11" / "step11_candidate4_combined.csv",
}
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
CATEGORICAL = ["STAGE", "CHAMBER_PATH"]
SEED = 42
SCOPES = {"all": None, "path_1-2-3": "1-2-3", "path_4-5-6": "4-5-6"}
METRICS = ["R2", "MAE", "RMSE"]


def onehot():
    return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)


def build_models(numeric):
    return {
        "MLR": lambda: Pipeline([
            ("prep", ColumnTransformer([
                ("cat", onehot(), CATEGORICAL),
                ("num", Pipeline([("impute", SimpleImputer(strategy="mean", add_indicator=True)),
                                  ("scale", StandardScaler())]), numeric)])),
            ("reg", LinearRegression())]),
        "RF": lambda: Pipeline([
            ("prep", ColumnTransformer([("cat", onehot(), CATEGORICAL), ("num", "passthrough", numeric)])),
            ("reg", RandomForestRegressor(random_state=SEED, n_jobs=1))]),
        "XGB": lambda: Pipeline([
            ("prep", ColumnTransformer([("cat", onehot(), CATEGORICAL), ("num", "passthrough", numeric)])),
            ("reg", XGBRegressor(random_state=SEED, n_jobs=-1))]),
    }


def load(path, folds):
    table = pd.read_csv(path, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    data = table.merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(data) == 1977 and data["FOLD"].notna().all()
    for c in CATEGORICAL:
        data[c] = data[c].astype(str)
    features = [c for c in table.columns if c not in ("WAFER_ID", TARGET)]
    numeric = [c for c in features if c not in CATEGORICAL]
    return data, features, numeric


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
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    fold_rows, timing_rows, other_rows, pooled_rows, set_info = [], [], [], [], []
    (OUT_DIR / "oof").mkdir(parents=True, exist_ok=True)

    for set_name, path in FEATURE_SETS.items():
        data, features, numeric = load(path, folds)
        n_missing_cols = int(data[numeric].isna().any().sum())
        set_info.append(dict(feature_set=set_name, n_features=len(features), n_numeric=len(numeric),
                             n_numeric_with_missing=n_missing_cols,
                             n_missing_cells=int(data[numeric].isna().to_numpy().sum())))
        print(f"\n[{set_name}] feature {len(features)}개 (숫자 {len(numeric)}, 빈 값 있는 숫자 열 {n_missing_cols})")
        for model_name, factory in build_models(numeric).items():
            pred = pd.Series(np.nan, index=data.index)
            inputs = []
            for k in sorted(data["FOLD"].unique()):
                train, test = data[data["FOLD"] != k], data[data["FOLD"] == k]
                assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])
                t0 = time.perf_counter()
                model = factory().fit(train[features], train[TARGET])
                pred.loc[test.index] = model.predict(test[features])
                timing_rows.append(dict(feature_set=set_name, model=model_name, FOLD=int(k),
                                        fit_predict_seconds=round(time.perf_counter() - t0, 3)))
                n_in = model[:-1].transform(train[features].iloc[:1]).shape[1]
                inputs.append(n_in)
                frame = data.loc[test.index, KEY + ["CHAMBER_PATH", "FOLD"]].assign(
                    MRR_TRUE=test[TARGET], PRED=pred.loc[test.index])
                for scope, m in scoped(frame, "PRED").items():
                    fold_rows.append(dict(feature_set=set_name, model=model_name, FOLD=int(k), scope=scope,
                                          **m, n_model_inputs=n_in))
            assert pred.notna().all()

            oof = data[KEY + ["CHAMBER_PATH", "FOLD"]].assign(MRR_TRUE=data[TARGET], MRR_PRED=pred,
                                                              ERROR=pred - data[TARGET])
            oof.to_csv(OUT_DIR / "oof" / f"oof_{set_name}_{model_name}.csv", index=False)
            for scope, m in scoped(oof.rename(columns={"MRR_PRED": "PRED"}), "PRED").items():
                pooled_rows += [dict(feature_set=set_name, model=model_name, scope=scope, metric=x,
                                     pooled_out_of_fold=m[x]) for x in METRICS]
            other = oof[oof["CHAMBER_PATH"] == "other"]
            other_rows += [dict(feature_set=set_name, model=model_name, WAFER_ID=r.WAFER_ID, STAGE=r.STAGE,
                                FOLD=r.FOLD, MRR_TRUE=r.MRR_TRUE, MRR_PRED=r.MRR_PRED,
                                ABS_ERROR=abs(r.MRR_PRED - r.MRR_TRUE)) for r in other.itertuples()]

            if set_name == "Baseline":  # Step 9/10 과 같은 모델인지 확인
                ref = pd.read_csv(STEP10_DIR / f"oof_predictions_{model_name}.csv")
                same = np.allclose(pred.to_numpy(), ref["MRR_PRED"].to_numpy(), rtol=0, atol=1e-8)
                print(f"  {model_name}: Baseline 예측이 Step 10 저장 예측과 같은가: {same}")
                assert same
            print(f"  {model_name}: 완료 (모델 입력 열 {min(inputs)}~{max(inputs)}개)")

    fold_metrics = pd.DataFrame(fold_rows)
    long = fold_metrics.melt(id_vars=["feature_set", "model", "FOLD", "scope"], value_vars=METRICS, var_name="metric")
    summary = (long.groupby(["feature_set", "model", "scope", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index()
               .merge(pd.DataFrame(pooled_rows), on=["feature_set", "model", "scope", "metric"]))

    base = long[long["feature_set"] == "Baseline"].drop(columns="feature_set").rename(columns={"value": "baseline"})
    paired = long[long["feature_set"] != "Baseline"].merge(base, on=["model", "FOLD", "scope", "metric"])
    paired["diff"] = paired["value"] - paired["baseline"]
    paired_summary = (paired.groupby(["feature_set", "model", "scope", "metric"], sort=False)["diff"]
                      .agg(mean_diff="mean", std_diff_ddof1="std", min_diff="min", max_diff="max").reset_index())

    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    paired_summary.to_csv(OUT_DIR / "paired_difference_vs_baseline.csv", index=False)
    pd.DataFrame(other_rows).to_csv(OUT_DIR / "other_path_rows.csv", index=False)
    pd.DataFrame(set_info).to_csv(OUT_DIR / "feature_set_info.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUT_DIR / "run_time_reference.csv", index=False)

    sets, models = list(FEATURE_SETS), ["MLR", "RF", "XGB"]
    view = summary.assign(cell=summary["mean"].map("{:.4f}".format) + " ± " + summary["std_ddof1"].map("{:.4f}".format))
    for scope in SCOPES:
        for metric in METRICS:
            t = view[(view["scope"] == scope) & (view["metric"] == metric)].pivot(index="feature_set", columns="model", values="cell")
            print(f"\n[{scope} / {metric}] 5-fold 평균 ± 표준편차")
            print(t.loc[sets, models].to_string())
    pv = paired_summary.assign(cell=paired_summary["mean_diff"].map("{:+.4f}".format) + " ± "
                               + paired_summary["std_diff_ddof1"].map("{:.4f}".format))
    for scope in SCOPES:
        for metric in METRICS:
            t = pv[(pv["scope"] == scope) & (pv["metric"] == metric)].pivot(index="feature_set", columns="model", values="cell")
            print(f"\n[Baseline 대비 같은 fold 차이: {scope} / {metric}] 평균 ± 표준편차 (후보 - Baseline)")
            print(t.loc[sets[1:], models].to_string())
    print("\n[CHAMBER_PATH other 4행]")
    print(pd.DataFrame(other_rows).round(3).to_string(index=False))
    print("\n[feature 구성 정보]")
    print(pd.DataFrame(set_info).to_string(index=False))
    print(f"\n저장 폴더: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

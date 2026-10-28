"""Step 17: Model 3 (Physics-Informed Hybrid) - MRR = k(x) x PV_proxy.

- 대상: Step 16 에서 확정한 Chamber path 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold.
  test / validation 은 사용하지 않는다. raw 파일은 수정하지 않는다.
- PV_proxy 와 k_target 은 Step 16 표를 그대로 사용한다 (P = RETAINER_RING_PRESSURE,
  V = |WAFER_ROTATION - HEAD_ROTATION|, 두 값이 모두 0 이 아닌 row 의 mean(P x V)). P/V 정의는 바꾸지 않는다.
- 학습 정답은 k_target = 실제 MRR / PV_proxy. 평가 sample 의 실제 MRR 과 k_target 은 입력에 쓰지 않는다.
  예측은 k_pred 이고, 최종 MRR 예측 = k_pred x PV_proxy. k_pred 를 보정하거나 clipping 하지 않는다.
- feature 구성
    A State-only        : STAGE, usage 평균 4개, SLURRY A/B/C 평균과 표준편차, DRESSING_WATER_STATUS 평균과 표준편차
    B Physics-separated : Baseline 에서 WAFER_ID, MRR, CHAMBER_PATH 와
                          PV_proxy 를 만드는 데 직접 쓰인 RETAINER_RING_PRESSURE / WAFER_ROTATION / HEAD_ROTATION
                          의 평균과 표준편차 6개를 제외
    C Full              : Baseline 에서 WAFER_ID, MRR, CHAMBER_PATH 만 제외
- 모델 (튜닝 없음)
    LinearRegression : Step 9 방식 (one-hot + 표준화 + LinearRegression)
    RandomForest     : Step 10 기본값 (random_state=42, n_jobs=1)
    XGBoost          : Step 10 기본값 (random_state=42, n_jobs=-1)
  전처리는 fold 마다 학습 sample 로만 fit 한다.
- 평가: k 자체 (k_target vs k_pred) 와 최종 MRR (실제 MRR vs k_pred x PV_proxy) 의 R², MAE, RMSE
- Step 16 의 같은 1,609개 기준 Model 1 (RF, XGB) 과 Model 2 (fixed-k) 결과를 같은 표에 넣는다.
- 최종 조합은 선택하지 않는다.
출력: reports/step17_model3_hybrid/
실행: python -B scripts/step17_model3_hybrid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 250)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
STEP16_DIR = ROOT / "reports" / "step16_model3_prep"
OUT_DIR = ROOT / "reports" / "step17_model3_hybrid"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
SEED = 42
METRICS = ["R2", "MAE", "RMSE"]

PV_SOURCE_FEATURES = ["RETAINER_RING_PRESSURE_MEAN", "RETAINER_RING_PRESSURE_STD",
                      "WAFER_ROTATION_MEAN", "WAFER_ROTATION_STD",
                      "HEAD_ROTATION_MEAN", "HEAD_ROTATION_STD"]
STATE_ONLY = ["STAGE", "USAGE_OF_DRESSER_MEAN", "USAGE_OF_POLISHING_TABLE_MEAN", "USAGE_OF_DRESSER_TABLE_MEAN",
              "USAGE_OF_BACKING_FILM_MEAN",
              "SLURRY_FLOW_LINE_A_MEAN", "SLURRY_FLOW_LINE_A_STD",
              "SLURRY_FLOW_LINE_B_MEAN", "SLURRY_FLOW_LINE_B_STD",
              "SLURRY_FLOW_LINE_C_MEAN", "SLURRY_FLOW_LINE_C_STD",
              "DRESSING_WATER_STATUS_MEAN", "DRESSING_WATER_STATUS_STD"]


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def build_model(name, numeric, categorical):
    onehot = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    if name == "LinearRegression":
        prep = ColumnTransformer([("cat", onehot, categorical), ("num", StandardScaler(), numeric)])
        return Pipeline([("prep", prep), ("reg", LinearRegression())])
    prep = ColumnTransformer([("cat", onehot, categorical), ("num", "passthrough", numeric)])
    reg = (RandomForestRegressor(random_state=SEED, n_jobs=1) if name == "RandomForest"
           else XGBRegressor(random_state=SEED, n_jobs=-1))
    return Pipeline([("prep", prep), ("reg", reg)])


def main():
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    prep = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    assert len(prep) == 1609 and (prep["CHAMBER_PATH"] == "4-5-6").all()
    data = prep.merge(step7.drop(columns=["CHAMBER_PATH", TARGET]), on=KEY, how="left", validate="one_to_one")
    assert len(data) == 1609 and data.notna().all().all()
    assert np.allclose(data["K_TARGET"], data[TARGET] / data["PV_PROXY"], rtol=0, atol=1e-12)
    data["STAGE"] = data["STAGE"].astype(str)

    baseline = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    feature_sets = {
        "A_state_only": STATE_ONLY,
        "B_physics_separated": [c for c in baseline if c not in PV_SOURCE_FEATURES],
        "C_full": baseline,
    }
    for name, cols in feature_sets.items():
        missing = [c for c in cols if c not in data.columns]
        assert not missing, missing
        print(f"[{name}] feature {len(cols)}개 (문자 1개 + 숫자 {len(cols) - 1}개)")
    print("B 에서 제외한 PV 생성 feature:", PV_SOURCE_FEATURES)

    folds = sorted(data["FOLD"].unique())
    rows, preds, k_check_rows, stage_rows = [], [], [], []
    for fs_name, cols in feature_sets.items():
        categorical = [c for c in cols if c == "STAGE"]
        numeric = [c for c in cols if c != "STAGE"]
        for model_name in ["LinearRegression", "RandomForest", "XGBoost"]:
            combo = f"{fs_name}|{model_name}"
            k_pred = pd.Series(np.nan, index=data.index)
            for f in folds:
                train, test = data[data["FOLD"] != f], data[data["FOLD"] == f]
                assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])
                model = build_model(model_name, numeric, categorical).fit(train[cols], train["K_TARGET"])
                k_pred.loc[test.index] = model.predict(test[cols])
                kk = metrics(test["K_TARGET"], k_pred.loc[test.index])
                mm = metrics(test[TARGET], k_pred.loc[test.index] * test["PV_PROXY"])
                rows.append(dict(feature_set=fs_name, model=model_name, combo=combo, FOLD=int(f),
                                 n_train=len(train), n_test=len(test),
                                 **{f"k_{m}": kk[m] for m in METRICS}, **{f"mrr_{m}": mm[m] for m in METRICS}))
            assert k_pred.notna().all()
            mrr_pred = k_pred * data["PV_PROXY"]
            out = data[KEY + ["FOLD", "PV_PROXY", "K_TARGET", TARGET]].assign(
                FEATURE_SET=fs_name, MODEL=model_name, COMBO=combo, K_PRED=k_pred, MRR_PRED=mrr_pred,
                K_ERROR=k_pred - data["K_TARGET"], MRR_ERROR=mrr_pred - data[TARGET])
            preds.append(out)
            lo, hi = data["K_TARGET"].min(), data["K_TARGET"].max()
            k_check_rows.append(dict(combo=combo, feature_set=fs_name, model=model_name,
                                     k_pred_min=k_pred.min(), k_pred_max=k_pred.max(),
                                     n_negative=int((k_pred < 0).sum()),
                                     n_below_k_target_min=int((k_pred < lo).sum()),
                                     n_above_k_target_max=int((k_pred > hi).sum()),
                                     k_target_min=lo, k_target_max=hi))
            for stage, g in out.groupby("STAGE"):
                mm = metrics(g[TARGET], g["MRR_PRED"])
                kk = metrics(g["K_TARGET"], g["K_PRED"])
                stage_rows.append(dict(combo=combo, feature_set=fs_name, model=model_name, STAGE=stage, n=len(g),
                                       **{f"k_{m}": kk[m] for m in METRICS}, **{f"mrr_{m}": mm[m] for m in METRICS}))

    fold_metrics = pd.DataFrame(rows)
    value_cols = [f"{p}_{m}" for p in ("k", "mrr") for m in METRICS]
    summary = (fold_metrics.melt(id_vars=["feature_set", "model", "combo", "FOLD"], value_vars=value_cols,
                                 var_name="metric")
               .groupby(["feature_set", "model", "combo", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index())
    oof = pd.concat(preds, ignore_index=True)
    pooled = pd.DataFrame([dict(combo=c, **{f"k_{m}": metrics(g["K_TARGET"], g["K_PRED"])[m] for m in METRICS},
                                **{f"mrr_{m}": metrics(g[TARGET], g["MRR_PRED"])[m] for m in METRICS})
                           for c, g in oof.groupby("COMBO")])
    k_checks = pd.DataFrame(k_check_rows)
    stage_metrics = pd.DataFrame(stage_rows)

    # Step 16 의 같은 1,609개 기준 결과와 한 표로 묶기 (MRR 기준)
    s16_fold = pd.read_csv(STEP16_DIR / "fold_metrics.csv", float_precision="round_trip")
    s16 = (s16_fold.melt(id_vars=["model", "FOLD"], value_vars=METRICS, var_name="metric")
           .groupby(["model", "metric"], sort=False)["value"].agg(mean="mean", std_ddof1="std").reset_index())
    compare_rows = []
    for m in ["Model1_RF", "Model1_XGB", "Model2_fixed_k"]:
        row = dict(source="Step 16", model_group={"Model1_RF": "Model 1", "Model1_XGB": "Model 1",
                                                  "Model2_fixed_k": "Model 2"}[m], combo=m)
        for metric in METRICS:
            r = s16[(s16["model"] == m) & (s16["metric"] == metric)].iloc[0]
            row[f"mrr_{metric}_mean"] = r["mean"]
            row[f"mrr_{metric}_std"] = r["std_ddof1"]
        compare_rows.append(row)
    for c, g in summary.groupby("combo"):
        row = dict(source="Step 17", model_group="Model 3", combo=c)
        for metric in METRICS:
            r = g[g["metric"] == f"mrr_{metric}"].iloc[0]
            row[f"mrr_{metric}_mean"] = r["mean"]
            row[f"mrr_{metric}_std"] = r["std_ddof1"]
        compare_rows.append(row)
    comparison = pd.DataFrame(compare_rows).sort_values("mrr_RMSE_mean")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pooled.to_csv(OUT_DIR / "pooled_metrics.csv", index=False)
    k_checks.to_csv(OUT_DIR / "k_pred_checks.csv", index=False)
    stage_metrics.to_csv(OUT_DIR / "stage_metrics.csv", index=False)
    comparison.to_csv(OUT_DIR / "comparison_with_step16.csv", index=False)
    oof[["WAFER_ID", "STAGE", "FOLD", "FEATURE_SET", "MODEL", "COMBO", "PV_PROXY", "K_TARGET", "K_PRED",
         TARGET, "MRR_PRED", "K_ERROR", "MRR_ERROR"]].to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    cell = lambda r, col: f"{r[f'{col}_mean']:.4f} ± {r[f'{col}_std']:.4f}"
    piv = summary.assign(cell=summary["mean"].map("{:.6f}".format) + " ± " + summary["std_ddof1"].map("{:.6f}".format))
    print("\n[k 예측 성능] k_target vs k_pred, 5-fold 평균 ± 표준편차")
    print(piv[piv.metric.str.startswith("k_")].pivot(index="combo", columns="metric", values="cell")
          [["k_R2", "k_MAE", "k_RMSE"]].to_string())
    piv2 = summary.assign(cell=summary["mean"].map("{:.4f}".format) + " ± " + summary["std_ddof1"].map("{:.4f}".format))
    print("\n[최종 MRR 예측 성능] 실제 MRR vs k_pred x PV_proxy, 5-fold 평균 ± 표준편차")
    print(piv2[piv2.metric.str.startswith("mrr_")].pivot(index="combo", columns="metric", values="cell")
          [["mrr_R2", "mrr_MAE", "mrr_RMSE"]].to_string())
    print("\n[Step 16 결과와 함께 본 MRR 성능 순위]")
    print(comparison.assign(R2=lambda d: d.apply(lambda r: cell(r, "mrr_R2"), axis=1),
                            MAE=lambda d: d.apply(lambda r: cell(r, "mrr_MAE"), axis=1),
                            RMSE=lambda d: d.apply(lambda r: cell(r, "mrr_RMSE"), axis=1))
          [["source", "model_group", "combo", "R2", "MAE", "RMSE"]].to_string(index=False))
    print("\n[k_pred 확인] 음수, 최소/최대, k_target 범위 밖")
    print(k_checks.round(8).to_string(index=False))
    print("\n[Stage 별 최종 MRR 성능]")
    print(stage_metrics.pivot(index="combo", columns="STAGE", values=["mrr_RMSE", "mrr_R2"]).round(4).to_string())
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

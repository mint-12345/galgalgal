"""Step 18: Model 3 - k_target 표준화로 XGBoost 의 상수 예측 문제 확인.

- Step 17 과 같은 대상: Chamber path 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold.
  test / validation 은 사용하지 않는다. raw 파일은 수정하지 않는다.
- Hybrid 구조 유지: MRR = k(x) x PV_proxy. PV_proxy 와 k_target 은 Step 16 값을 그대로 사용한다.
  feature 구성 A / B / C 도 Step 17 과 같다. feature 를 더하거나 빼지 않고 P/V 정의도 바꾸지 않는다.
- 이번 변경점은 하나뿐이다. XGBoost 학습 때 k_target 을 그대로 쓰지 않고
      k_scaled = (k_target - training_mean) / training_std
  로 바꿔 학습한 뒤, 예측을 k_pred = k_scaled_pred x training_std + training_mean 으로 되돌린다.
  training_mean 과 training_std 는 그 fold 의 학습 sample 로만 계산한다 (평가 fold 통계는 쓰지 않는다).
- XGBoost 설정은 Step 10 기본값 그대로 (random_state=42, n_jobs=-1). 튜닝하지 않는다.
- k_pred 를 clipping 하거나 평가 fold 의 k_target 으로 보정하지 않는다.
출력: reports/step18_model3_scaled_xgb/
실행: python -B scripts/step18_model3_scaled_xgb.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 250)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
STEP16_DIR = ROOT / "reports" / "step16_model3_prep"
STEP17_DIR = ROOT / "reports" / "step17_model3_hybrid"
OUT_DIR = ROOT / "reports" / "step18_model3_scaled_xgb"
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


def build_xgb(numeric, categorical):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep), ("reg", XGBRegressor(random_state=SEED, n_jobs=-1))])


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
    print("feature 수:", {k: len(v) for k, v in feature_sets.items()})

    folds = sorted(data["FOLD"].unique())
    rows, preds, scale_rows, check_rows, stage_rows = [], [], [], [], []
    k_lo, k_hi = data["K_TARGET"].min(), data["K_TARGET"].max()
    for fs_name, cols in feature_sets.items():
        categorical = [c for c in cols if c == "STAGE"]
        numeric = [c for c in cols if c != "STAGE"]
        combo = f"{fs_name}|XGBoost_scaled"
        k_pred = pd.Series(np.nan, index=data.index)
        for f in folds:
            train, test = data[data["FOLD"] != f], data[data["FOLD"] == f]
            assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])
            mean_k = float(train["K_TARGET"].mean())      # 학습 fold 만 사용
            std_k = float(train["K_TARGET"].std(ddof=0))  # 학습 fold 만 사용
            assert std_k > 0
            k_scaled = (train["K_TARGET"] - mean_k) / std_k
            model = build_xgb(numeric, categorical).fit(train[cols], k_scaled)
            pred_scaled = model.predict(test[cols])
            k_pred.loc[test.index] = pred_scaled * std_k + mean_k   # 원래 k 단위로 복원
            scale_rows.append(dict(feature_set=fs_name, FOLD=int(f), training_mean=mean_k, training_std=std_k,
                                   n_train=len(train), n_test=len(test),
                                   k_scaled_pred_min=float(pred_scaled.min()),
                                   k_scaled_pred_max=float(pred_scaled.max()),
                                   n_unique_k_pred=int(pd.Series(k_pred.loc[test.index]).nunique())))
            kk = metrics(test["K_TARGET"], k_pred.loc[test.index])
            mm = metrics(test[TARGET], k_pred.loc[test.index] * test["PV_PROXY"])
            rows.append(dict(feature_set=fs_name, combo=combo, FOLD=int(f), n_train=len(train), n_test=len(test),
                             **{f"k_{m}": kk[m] for m in METRICS}, **{f"mrr_{m}": mm[m] for m in METRICS}))
        assert k_pred.notna().all()
        mrr_pred = k_pred * data["PV_PROXY"]
        out = data[KEY + ["FOLD", "PV_PROXY", "K_TARGET", TARGET]].assign(
            FEATURE_SET=fs_name, MODEL="XGBoost_scaled", COMBO=combo, K_PRED=k_pred, MRR_PRED=mrr_pred,
            K_ERROR=k_pred - data["K_TARGET"], MRR_ERROR=mrr_pred - data[TARGET])
        preds.append(out)
        check_rows.append(dict(combo=combo, feature_set=fs_name, k_pred_min=k_pred.min(), k_pred_max=k_pred.max(),
                               n_unique_k_pred=int(k_pred.nunique()), n_negative=int((k_pred < 0).sum()),
                               n_below_k_target_min=int((k_pred < k_lo).sum()),
                               n_above_k_target_max=int((k_pred > k_hi).sum()),
                               k_target_min=k_lo, k_target_max=k_hi))
        for stage, g in out.groupby("STAGE"):
            stage_rows.append(dict(combo=combo, feature_set=fs_name, STAGE=stage, n=len(g),
                                   **{f"k_{m}": metrics(g["K_TARGET"], g["K_PRED"])[m] for m in METRICS},
                                   **{f"mrr_{m}": metrics(g[TARGET], g["MRR_PRED"])[m] for m in METRICS}))

    fold_metrics = pd.DataFrame(rows)
    value_cols = [f"{p}_{m}" for p in ("k", "mrr") for m in METRICS]
    summary = (fold_metrics.melt(id_vars=["feature_set", "combo", "FOLD"], value_vars=value_cols, var_name="metric")
               .groupby(["feature_set", "combo", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index())
    oof = pd.concat(preds, ignore_index=True)
    pooled = pd.DataFrame([dict(combo=c, **{f"k_{m}": metrics(g["K_TARGET"], g["K_PRED"])[m] for m in METRICS},
                                **{f"mrr_{m}": metrics(g[TARGET], g["MRR_PRED"])[m] for m in METRICS})
                           for c, g in oof.groupby("COMBO")])
    checks = pd.DataFrame(check_rows)
    scales = pd.DataFrame(scale_rows)
    stage_metrics = pd.DataFrame(stage_rows)

    # 비교 표: 이번 결과 + Step 17 (raw XGB, C_full RF) + Step 16 (Model 1, Model 2)
    s17 = pd.read_csv(STEP17_DIR / "summary.csv", float_precision="round_trip")
    s16_fold = pd.read_csv(STEP16_DIR / "fold_metrics.csv", float_precision="round_trip")
    s16 = (s16_fold.melt(id_vars=["model", "FOLD"], value_vars=METRICS, var_name="metric")
           .groupby(["model", "metric"], sort=False)["value"].agg(mean="mean", std_ddof1="std").reset_index())

    def from_long(df, combo, prefix):
        row = {}
        for m in METRICS:
            r = df[(df["combo"] == combo) & (df["metric"] == f"{prefix}_{m}")].iloc[0]
            row[f"mrr_{m}_mean" if prefix == "mrr" else f"k_{m}_mean"] = r["mean"]
            row[f"mrr_{m}_std" if prefix == "mrr" else f"k_{m}_std"] = r["std_ddof1"]
        return row

    compare = []
    for fs in feature_sets:
        compare.append(dict(source="Step 18", group="Model 3 (표준화 XGB)", combo=f"{fs}|XGBoost_scaled",
                            **from_long(summary, f"{fs}|XGBoost_scaled", "mrr"),
                            **from_long(summary, f"{fs}|XGBoost_scaled", "k")))
        compare.append(dict(source="Step 17", group="Model 3 (원래 크기 XGB)", combo=f"{fs}|XGBoost",
                            **from_long(s17, f"{fs}|XGBoost", "mrr"), **from_long(s17, f"{fs}|XGBoost", "k")))
    compare.append(dict(source="Step 17", group="Model 3 (Random Forest)", combo="C_full|RandomForest",
                        **from_long(s17, "C_full|RandomForest", "mrr"), **from_long(s17, "C_full|RandomForest", "k")))
    for m, label in [("Model1_RF", "Model 1"), ("Model1_XGB", "Model 1"), ("Model2_fixed_k", "Model 2")]:
        row = dict(source="Step 16", group=label, combo=m)
        for metric in METRICS:
            r = s16[(s16["model"] == m) & (s16["metric"] == metric)].iloc[0]
            row[f"mrr_{metric}_mean"], row[f"mrr_{metric}_std"] = r["mean"], r["std_ddof1"]
        compare.append(row)
    comparison = pd.DataFrame(compare).sort_values("mrr_RMSE_mean")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pooled.to_csv(OUT_DIR / "pooled_metrics.csv", index=False)
    checks.to_csv(OUT_DIR / "k_pred_checks.csv", index=False)
    scales.to_csv(OUT_DIR / "fold_scaling_values.csv", index=False)
    stage_metrics.to_csv(OUT_DIR / "stage_metrics.csv", index=False)
    comparison.to_csv(OUT_DIR / "comparison_all_models.csv", index=False)
    oof[["WAFER_ID", "STAGE", "FOLD", "FEATURE_SET", "MODEL", "COMBO", "PV_PROXY", "K_TARGET", "K_PRED",
         TARGET, "MRR_PRED", "K_ERROR", "MRR_ERROR"]].to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    piv = summary.assign(cell=summary["mean"].map("{:.6f}".format) + " ± " + summary["std_ddof1"].map("{:.6f}".format))
    print("\n[k 예측 성능] 표준화 XGBoost, 5-fold 평균 ± 표준편차")
    print(piv[piv.metric.str.startswith("k_")].pivot(index="combo", columns="metric", values="cell")
          [["k_R2", "k_MAE", "k_RMSE"]].to_string())
    piv2 = summary.assign(cell=summary["mean"].map("{:.4f}".format) + " ± " + summary["std_ddof1"].map("{:.4f}".format))
    print("\n[최종 MRR 예측 성능] 표준화 XGBoost")
    print(piv2[piv2.metric.str.startswith("mrr_")].pivot(index="combo", columns="metric", values="cell")
          [["mrr_R2", "mrr_MAE", "mrr_RMSE"]].to_string())
    print("\n[전체 비교: MRR RMSE 낮은 순]")
    show = comparison.assign(
        R2=lambda d: d.mrr_R2_mean.map("{:.4f}".format) + " ± " + d.mrr_R2_std.map("{:.4f}".format),
        MAE=lambda d: d.mrr_MAE_mean.map("{:.4f}".format) + " ± " + d.mrr_MAE_std.map("{:.4f}".format),
        RMSE=lambda d: d.mrr_RMSE_mean.map("{:.4f}".format) + " ± " + d.mrr_RMSE_std.map("{:.4f}".format))
    print(show[["source", "group", "combo", "R2", "MAE", "RMSE"]].to_string(index=False))
    print("\n[fold 별 표준화 값과 서로 다른 k_pred 개수]")
    print(scales.round(9).to_string(index=False))
    print("\n[k_pred 확인]")
    print(checks.round(8).to_string(index=False))
    print("\n[Stage 별 최종 MRR 성능]")
    print(stage_metrics.pivot(index="combo", columns="STAGE", values=["mrr_RMSE", "mrr_R2"]).round(4).to_string())
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

"""Step 20: Hybrid (P9 x V3 고정) 에서 k(x) XGBoost 설정만 제한적으로 비교.

- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold.
  test / validation 은 사용하지 않는다. raw 파일은 수정하지 않는다.
  제외 기준은 앞 단계와 같다 (연속 완전 중복 1,819행, MRR > 1,000 인 4개).
- 구조 고정: MRR_pred = k_pred(x) x PV_proxy, k_target = 실제 MRR / PV_proxy
    P9 = pressure 6개 전체의 raw row 별 평균, V3 = |HEAD_ROTATION|
    PV_proxy = P 와 V 가 모두 0 이 아닌 row 만의 mean(P x V)  (Step 19 와 동일)
    feature 는 Step 18 / 19 의 C_full 31개 (WAFER_ID, 실제 MRR, k_target, CHAMBER_PATH 제외)
    k 는 fold 마다 학습 sample 로만 표준화하고 예측 후 원래 단위로 되돌린다 (Step 18 방식)
- 이번에 비교하는 것: XGBoost 설정 54개
    n_estimators {300, 600, 900} x max_depth {2, 3, 4} x learning_rate {0.03, 0.05, 0.1}
    x min_child_weight {1, 3}
  고정: subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
  reg_alpha, reg_lambda, gamma 등 나머지는 Step 18 기본값 그대로 두고 탐색하지 않는다.
- 주 비교 기준은 최종 MRR RMSE 이다. k_pred 를 clipping 하거나 평가 fold 값으로 보정하지 않는다.
- 최종 설정은 선택하지 않는다.
출력: reports/step20_hybrid_xgb_tuning/
실행: python -B scripts/step20_hybrid_xgb_tuning.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
STEP16_DIR = ROOT / "reports" / "step16_model3_prep"
STEP19_DIR = ROOT / "reports" / "step19_hybrid_pv_comparison"
OUT_DIR = ROOT / "reports" / "step20_hybrid_xgb_tuning"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
SEED = 42
METRICS = ["R2", "MAE", "RMSE"]
N_TOP = 10
N_SAVE_OOF = 5

PRESSURES = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
             "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE"]
GRID = dict(n_estimators=[300, 600, 900], max_depth=[2, 3, 4], learning_rate=[0.03, 0.05, 0.1],
            min_child_weight=[1, 3])
FIXED = dict(subsample=0.9, colsample_bytree=0.9)


def load_filtered():
    frames = []
    for f in sorted(TRAIN_DIR.glob("CMP-training-*.csv")):
        df = pd.read_csv(f, dtype={"WAFER_ID": "int64", "STAGE": "string"})
        if df.empty:
            continue
        df["FILE"] = f.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    cols = [c for c in raw.columns if c != "FILE"]
    same_prev = raw[cols].eq(raw[cols].shift()).all(axis=1) & raw["FILE"].eq(raw["FILE"].shift())
    data = raw[~same_prev]
    mrr = pd.read_csv(MRR_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    big = mrr.loc[mrr[TARGET] > MRR_LIMIT, KEY]
    data = data[~data.set_index(KEY).index.isin(big.set_index(KEY).index)]
    assert int(same_prev.sum()) == 1819 and len(big) == 4
    return data


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def config_id(params):
    return f"n{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}_mcw{params['min_child_weight']}"


def build(params, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["STAGE"]),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep),
                     ("reg", XGBRegressor(random_state=SEED, n_jobs=-1, **FIXED, **params))])


def main():
    raw = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    prep = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    assert len(prep) == 1609 and (prep["CHAMBER_PATH"] == "4-5-6").all()
    base = prep[KEY + ["FOLD", TARGET]].merge(step7.drop(columns=["CHAMBER_PATH", TARGET]), on=KEY,
                                              how="left", validate="one_to_one")
    base["STAGE"] = base["STAGE"].astype(str)
    assert len(base) == 1609 and base.notna().all().all()

    # PV_proxy (P9 x V3, nonzero 방식)
    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(base["WAFER_ID"], base["STAGE"]))}
    keys = list(zip(raw["WAFER_ID"].astype(int), raw["STAGE"].astype(str)))
    keep = np.array([k in order for k in keys])
    rows = raw[keep]
    codes = np.array([order[k] for k, k_ok in zip(keys, keep) if k_ok], dtype=np.int64)
    p_row = np.mean([rows[c].to_numpy() for c in PRESSURES], axis=0)
    v_row = np.abs(rows["HEAD_ROTATION"].to_numpy())
    mask = (p_row != 0) & (v_row != 0)
    n_used = np.bincount(codes, weights=mask.astype(float), minlength=len(base))
    sum_pv = np.bincount(codes, weights=p_row * v_row * mask, minlength=len(base))
    assert (n_used > 0).all()
    pv = sum_pv / n_used
    base["PV_PROXY"] = pv
    base["K_TARGET"] = base[TARGET] / base["PV_PROXY"]

    prev = pd.read_csv(STEP19_DIR / "oof_predictions.csv", float_precision="round_trip")
    prev = prev[prev.COMBO == "P9xV3"].set_index(KEY).loc[list(zip(base.WAFER_ID, base.STAGE))]
    same_pv = np.allclose(prev["PV_PROXY"].to_numpy(), pv, rtol=0, atol=0)
    print(f"PV_proxy 가 Step 19 의 P9xV3 값과 같은가: {same_pv}")
    assert same_pv

    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    assert len(features) == 31
    folds = sorted(base["FOLD"].unique())
    y = base[TARGET].to_numpy()
    k_target = base["K_TARGET"].to_numpy()
    sample_fold = base["FOLD"].to_numpy()

    configs = [dict(zip(GRID, vals)) for vals in itertools.product(*GRID.values())]
    print(f"설정 {len(configs)}개 x fold {len(folds)}개 = 학습 {len(configs) * len(folds)}회")

    fold_rows, stage_rows, check_rows, scale_rows, preds = [], [], [], [], {}
    k_lo, k_hi = k_target.min(), k_target.max()
    for params in configs:
        cid = config_id(params)
        k_pred = np.full(len(base), np.nan)
        for f in folds:
            is_test = sample_fold == f
            tr = ~is_test
            mean_k, std_k = float(k_target[tr].mean()), float(k_target[tr].std(ddof=0))
            model = build(params, numeric).fit(base.loc[tr, features], (k_target[tr] - mean_k) / std_k)
            k_pred[is_test] = model.predict(base.loc[is_test, features]) * std_k + mean_k
            scale_rows.append(dict(config_id=cid, FOLD=int(f), training_mean=mean_k, training_std=std_k,
                                   n_train=int(tr.sum()), n_test=int(is_test.sum()),
                                   n_unique_k_pred=int(pd.Series(k_pred[is_test]).nunique())))
            kk = metrics(k_target[is_test], k_pred[is_test])
            mm = metrics(y[is_test], k_pred[is_test] * pv[is_test])
            fold_rows.append(dict(config_id=cid, **params, FOLD=int(f),
                                  **{f"k_{m}": kk[m] for m in METRICS}, **{f"mrr_{m}": mm[m] for m in METRICS}))
        mrr_pred = k_pred * pv
        out = base[KEY + ["FOLD", "PV_PROXY", "K_TARGET", TARGET]].assign(
            CONFIG=cid, K_PRED=k_pred, MRR_PRED=mrr_pred, K_ERROR=k_pred - k_target, MRR_ERROR=mrr_pred - y)
        preds[cid] = out
        check_rows.append(dict(config_id=cid, **params, k_pred_min=k_pred.min(), k_pred_max=k_pred.max(),
                               n_unique_k_pred=int(pd.Series(k_pred).nunique()), n_negative=int((k_pred < 0).sum()),
                               n_below_k_target_min=int((k_pred < k_lo).sum()),
                               n_above_k_target_max=int((k_pred > k_hi).sum()),
                               k_target_min=k_lo, k_target_max=k_hi))
        for stage, g in out.groupby("STAGE"):
            stage_rows.append(dict(config_id=cid, STAGE=stage, n=len(g),
                                   **{f"mrr_{m}": metrics(g[TARGET], g["MRR_PRED"])[m] for m in METRICS},
                                   **{f"k_{m}": metrics(g["K_TARGET"], g["K_PRED"])[m] for m in METRICS}))

    fold_metrics = pd.DataFrame(fold_rows)
    value_cols = [f"{p}_{m}" for p in ("k", "mrr") for m in METRICS]
    summary = (fold_metrics.melt(id_vars=["config_id", "FOLD"], value_vars=value_cols, var_name="metric")
               .groupby(["config_id", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index())

    # Step 19 기본 Hybrid (P9xV3, 기본 XGBoost) 와 같은 fold 비교
    s19_fold = pd.read_csv(STEP19_DIR / "fold_metrics.csv", float_precision="round_trip")
    base_fold = s19_fold[s19_fold.combo == "P9xV3"].set_index("FOLD").mrr_RMSE
    piv = fold_metrics.pivot(index="FOLD", columns="config_id", values="mrr_RMSE")
    paired = pd.DataFrame([dict(config_id=c, mean_diff_vs_step19=float((piv[c] - base_fold).mean()),
                                std_diff=float((piv[c] - base_fold).std()),
                                folds_better_than_step19=int((piv[c] < base_fold).sum())) for c in piv.columns])

    wide = summary.pivot_table(index="config_id", columns="metric", values=["mean", "std_ddof1"])
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    stage_wide = (pd.DataFrame(stage_rows).pivot(index="config_id", columns="STAGE",
                                                 values=["mrr_RMSE", "mrr_R2", "mrr_MAE"]))
    stage_wide.columns = [f"{m}_stage_{s}" for m, s in stage_wide.columns]
    table = (pd.DataFrame(check_rows).set_index("config_id").join(wide).join(paired.set_index("config_id"))
             .join(stage_wide).reset_index().sort_values("mrr_RMSE_mean"))
    table["rank_by_mrr_RMSE"] = range(1, len(table) + 1)

    s16_fold = pd.read_csv(STEP16_DIR / "fold_metrics.csv", float_precision="round_trip")
    s16 = (s16_fold.melt(id_vars=["model", "FOLD"], value_vars=METRICS, var_name="metric")
           .groupby(["model", "metric"], sort=False)["value"].agg(mean="mean", std_ddof1="std").reset_index())
    s19 = pd.read_csv(STEP19_DIR / "combo_results.csv", float_precision="round_trip").set_index("combo")
    ref = [dict(source="Step 19", group="Model 3 Hybrid (기본 XGB)", name="P9xV3",
                mrr_R2_mean=s19.loc["P9xV3", "mrr_R2_mean"], mrr_R2_std=s19.loc["P9xV3", "mrr_R2_std_ddof1"],
                mrr_MAE_mean=s19.loc["P9xV3", "mrr_MAE_mean"], mrr_MAE_std=s19.loc["P9xV3", "mrr_MAE_std_ddof1"],
                mrr_RMSE_mean=s19.loc["P9xV3", "mrr_RMSE_mean"], mrr_RMSE_std=s19.loc["P9xV3", "mrr_RMSE_std_ddof1"])]
    for m, label in [("Model1_XGB", "Model 1"), ("Model1_RF", "Model 1"), ("Model2_fixed_k", "Model 2")]:
        row = dict(source="Step 16", group=label, name=m)
        for metric in METRICS:
            r = s16[(s16["model"] == m) & (s16["metric"] == metric)].iloc[0]
            row[f"mrr_{metric}_mean"], row[f"mrr_{metric}_std"] = r["mean"], r["std_ddof1"]
        ref.append(row)
    for _, r in table.head(N_TOP).iterrows():
        ref.append(dict(source="Step 20", group="Model 3 Hybrid (조정 XGB)", name=r["config_id"],
                        mrr_R2_mean=r["mrr_R2_mean"], mrr_R2_std=r["mrr_R2_std_ddof1"],
                        mrr_MAE_mean=r["mrr_MAE_mean"], mrr_MAE_std=r["mrr_MAE_std_ddof1"],
                        mrr_RMSE_mean=r["mrr_RMSE_mean"], mrr_RMSE_std=r["mrr_RMSE_std_ddof1"]))
    comparison = pd.DataFrame(ref).sort_values("mrr_RMSE_mean")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "oof_top").mkdir(exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_long.csv", index=False)
    table.to_csv(OUT_DIR / "config_results.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT_DIR / "stage_metrics.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUT_DIR / "k_pred_checks.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(OUT_DIR / "fold_scaling_values.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_difference_vs_step19.csv", index=False)
    comparison.to_csv(OUT_DIR / "comparison_with_reference_models.csv", index=False)
    for cid in table.head(N_SAVE_OOF)["config_id"]:
        preds[cid][KEY + ["FOLD", "CONFIG", "PV_PROXY", "K_TARGET", "K_PRED", TARGET, "MRR_PRED",
                          "K_ERROR", "MRR_ERROR"]].to_csv(OUT_DIR / "oof_top" / f"oof_{cid}.csv", index=False)

    top = table.head(N_TOP)
    print(f"\n[상위 {N_TOP}개 설정: 최종 MRR RMSE 낮은 순]")
    show = top.assign(
        RMSE=lambda d: d.mrr_RMSE_mean.map("{:.4f}".format) + " ± " + d.mrr_RMSE_std_ddof1.map("{:.4f}".format),
        MAE=lambda d: d.mrr_MAE_mean.map("{:.4f}".format), R2=lambda d: d.mrr_R2_mean.map("{:.4f}".format),
        kR2=lambda d: d.k_R2_mean.map("{:.4f}".format),
        A=lambda d: d.mrr_RMSE_stage_A.map("{:.4f}".format), B=lambda d: d.mrr_RMSE_stage_B.map("{:.4f}".format),
        vs19=lambda d: d.mean_diff_vs_step19.map("{:+.4f}".format) + " (" +
                       d.folds_better_than_step19.astype(int).astype(str) + "/5)")
    print(show[["rank_by_mrr_RMSE", "config_id", "RMSE", "MAE", "R2", "kR2", "A", "B", "vs19"]].to_string(index=False))
    print("\n[참고 기준과 함께]")
    print(comparison.assign(
        R2=lambda d: d.mrr_R2_mean.map("{:.4f}".format) + " ± " + d.mrr_R2_std.map("{:.4f}".format),
        RMSE=lambda d: d.mrr_RMSE_mean.map("{:.4f}".format) + " ± " + d.mrr_RMSE_std.map("{:.4f}".format))
        [["source", "group", "name", "R2", "RMSE"]].to_string(index=False))
    print("\n[54개 설정의 MRR RMSE 분포]", table.mrr_RMSE_mean.describe().round(4).to_dict())
    print("설정값별 평균 MRR RMSE:")
    for p in GRID:
        print(f"  {p}:", table.groupby(p).mrr_RMSE_mean.mean().round(4).to_dict())
    print("\n[상위 10개 k_pred 확인]")
    print(top[["config_id", "k_pred_min", "k_pred_max", "n_unique_k_pred", "n_negative",
               "n_below_k_target_min", "n_above_k_target_max"]].round(8).to_string(index=False))
    print(f"\nk_target 범위: {k_lo:.6f} ~ {k_hi:.6f}")
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

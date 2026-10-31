"""Step 19: Hybrid (C_full standardized XGBoost) 에서 P/V proxy 12개 조합 비교.

- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold.
  test / validation 은 사용하지 않는다. raw 파일은 수정하지 않는다.
  제외 기준은 앞 단계와 같다 (연속 완전 중복 1,819행, MRR > 1,000 인 4개).
- 구조: MRR_pred = k_pred(x) x PV_proxy, k_target = 실제 MRR / PV_proxy.
  k(x) 는 Step 18 과 같은 standardized-target XGBoost 이다.
    fold 마다 학습 sample 로만 k_scaled = (k_target - training_mean) / training_std 를 만들어 학습하고,
    예측을 k_pred = k_scaled_pred x training_std + training_mean 으로 되돌린다.
    XGBoost 설정은 Step 10 기본값 (random_state=42, n_jobs=-1). 튜닝하지 않는다.
  feature 는 Step 18 의 C_full 구성만 사용한다 (WAFER_ID, 실제 MRR, k_target, CHAMBER_PATH 제외).
- P 후보: P4 retainer ring, P8 air-bag 4개 + retainer 의 행별 평균, P9 pressure 6개 평균,
          P11 정규화한 5개(air-bag 4개 + retainer)의 행별 평균.
          P11 의 정규화 기준은 fold 마다 "학습 sample 의 raw row 중 0 이 아닌 값의 중앙값" 이다.
          이번 분석은 4-5-6 만 다루므로 학습 row 도 4-5-6 학습 sample 의 row 이다.
- V 후보: V3 |HEAD_ROTATION|, V5 |WAFER_ROTATION - HEAD_ROTATION|, V6 |STAGE_ROTATION - HEAD_ROTATION|
          V5, V6 은 실제 선속도가 아니라 rotation 차이로 만든 relative-motion proxy 이다.
- PV_proxy = P 와 V 가 모두 0 이 아닌 row 만의 mean(P x V) (Step 15 nonzero 방식과 동일).
  계산할 수 없는 sample 은 0 으로 채우지 않고 따로 보고한다.
- 최종 후보는 선택하지 않는다. k_pred 를 clipping 하거나 평가 fold 정보로 보정하지 않는다.
출력: reports/step19_hybrid_pv_comparison/
실행: python -B scripts/step19_hybrid_pv_comparison.py
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
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
STEP16_DIR = ROOT / "reports" / "step16_model3_prep"
STEP18_DIR = ROOT / "reports" / "step18_model3_scaled_xgb"
OUT_DIR = ROOT / "reports" / "step19_hybrid_pv_comparison"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
SEED = 42
METRICS = ["R2", "MAE", "RMSE"]
BASELINE_COMBO = "P4xV5"

CHAMBER_P = "PRESSURIZED_CHAMBER_PRESSURE"
RETAINER = "RETAINER_RING_PRESSURE"
AIR_BAGS = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE",
            "EDGE_AIR_BAG_PRESSURE"]
PRESSURES = [CHAMBER_P] + AIR_BAGS + [RETAINER]
P8_COLS = AIR_BAGS + [RETAINER]
P_DEFINITION = {"P4": "RETAINER_RING_PRESSURE",
                "P8": "mean of 4 air-bag + retainer ring (raw)",
                "P9": "mean of all 6 pressures (raw)",
                "P11": "mean of normalized 4 air-bag + retainer ring (train-fold nonzero median)"}
V_DEFINITION = {"V3": "|HEAD_ROTATION|",
                "V5": "|WAFER_ROTATION - HEAD_ROTATION| (relative-motion proxy)",
                "V6": "|STAGE_ROTATION - HEAD_ROTATION| (relative-motion proxy)"}


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
    assert (data[PRESSURES].to_numpy() >= 0).all()
    return data


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def build_xgb(numeric, categorical):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep), ("reg", XGBRegressor(random_state=SEED, n_jobs=-1))])


def main():
    raw = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    prep = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    assert len(prep) == 1609 and (prep["CHAMBER_PATH"] == "4-5-6").all()
    base = prep[KEY + ["FOLD", TARGET]].merge(step7.drop(columns=["CHAMBER_PATH", TARGET]), on=KEY,
                                              how="left", validate="one_to_one")
    base["STAGE"] = base["STAGE"].astype(str)
    assert len(base) == 1609 and base.notna().all().all()

    # 1,609개 sample 의 raw row 만 남기고, sample 인덱스와 fold 를 붙인다
    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(base["WAFER_ID"], base["STAGE"]))}
    keys = list(zip(raw["WAFER_ID"].astype(int), raw["STAGE"].astype(str)))
    in_subset = np.array([k in order for k in keys])
    rows = raw[in_subset]
    codes = np.array([order[k] for k, keep in zip(keys, in_subset) if keep], dtype=np.int64)
    sample_fold = base["FOLD"].to_numpy()
    row_fold = sample_fold[codes]
    y = base[TARGET].to_numpy()
    folds = sorted(base["FOLD"].unique())
    print(f"raw 행 {len(raw):,} 중 4-5-6 sample 의 행 {len(rows):,} 사용 / sample {len(base):,}개")

    press = {c: rows[c].to_numpy() for c in PRESSURES}
    rot = {k: rows[f"{k}_ROTATION"].to_numpy() for k in ["WAFER", "STAGE", "HEAD"]}
    V_ROW = {"V3": np.abs(rot["HEAD"]), "V5": np.abs(rot["WAFER"] - rot["HEAD"]),
             "V6": np.abs(rot["STAGE"] - rot["HEAD"])}
    P_ROW_FIXED = {"P4": press[RETAINER],
                   "P8": np.mean([press[c] for c in P8_COLS], axis=0),
                   "P9": np.mean([press[c] for c in PRESSURES], axis=0)}
    median_rows, P11_BY_FOLD = [], {}
    for f in folds:
        train_rows = row_fold != f
        med = {}
        for c in P8_COLS:
            v = press[c][train_rows]
            nz = v[v != 0]
            assert nz.size > 0
            med[c] = float(np.median(nz))
            median_rows.append(dict(FOLD=int(f), pressure=c, train_nonzero_median=med[c], n_train_nonzero=int(nz.size)))
        P11_BY_FOLD[f] = np.mean([press[c] / med[c] for c in P8_COLS], axis=0)

    n_samples = len(base)

    def pv_proxy(p_row, v_row):
        mask = (p_row != 0) & (v_row != 0)
        n = np.bincount(codes, weights=mask.astype(float), minlength=n_samples)
        s = np.bincount(codes, weights=p_row * v_row * mask, minlength=n_samples)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.where(n > 0, n, 1.0), np.nan), n.astype(int)

    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    categorical = ["STAGE"]
    assert len(features) == 31

    combos = [(p, v) for p in ["P4", "P8", "P9", "P11"] for v in ["V3", "V5", "V6"]]
    fold_rows, preds, avail_rows, check_rows, stage_rows, scale_rows = [], [], [], [], [], []
    for p, v in combos:
        combo = f"{p}x{v}"
        pv_by_fold, rows_used_by_fold = {}, {}
        for f in folds:
            p_row = P11_BY_FOLD[f] if p == "P11" else P_ROW_FIXED[p]
            pv, n_used = pv_proxy(p_row, V_ROW[v])
            pv_by_fold[f], rows_used_by_fold[f] = pv, n_used
        any_pv = pv_by_fold[folds[0]]
        n_missing = int(np.isnan(any_pv).sum())
        avail_rows.append(dict(combo=combo, P=p, V=v, P_definition=P_DEFINITION[p], V_definition=V_DEFINITION[v],
                               n_samples=n_samples, n_pv_computable=int(np.isfinite(any_pv).sum()),
                               n_pv_missing=n_missing, n_pv_zero=int(np.nansum(any_pv == 0)),
                               rows_used_min=int(rows_used_by_fold[folds[0]].min()),
                               rows_used_median=float(np.median(rows_used_by_fold[folds[0]])),
                               rows_used_max=int(rows_used_by_fold[folds[0]].max())))
        assert n_missing == 0, f"{combo}: PV_proxy 를 계산할 수 없는 sample {n_missing}개"

        k_pred = np.full(n_samples, np.nan)
        k_target_oof = np.full(n_samples, np.nan)
        pv_oof = np.full(n_samples, np.nan)
        for f in folds:
            pv = pv_by_fold[f]
            k_target = y / pv
            is_test = sample_fold == f
            tr = ~is_test
            mean_k, std_k = float(k_target[tr].mean()), float(k_target[tr].std(ddof=0))
            assert std_k > 0
            model = build_xgb(numeric, categorical).fit(base.loc[tr, features], (k_target[tr] - mean_k) / std_k)
            k_pred[is_test] = model.predict(base.loc[is_test, features]) * std_k + mean_k
            k_target_oof[is_test] = k_target[is_test]
            pv_oof[is_test] = pv[is_test]
            scale_rows.append(dict(combo=combo, FOLD=int(f), training_mean=mean_k, training_std=std_k,
                                   n_train=int(tr.sum()), n_test=int(is_test.sum()),
                                   n_unique_k_pred=int(pd.Series(k_pred[is_test]).nunique())))
            kk = metrics(k_target[is_test], k_pred[is_test])
            mm = metrics(y[is_test], k_pred[is_test] * pv[is_test])
            fold_rows.append(dict(combo=combo, P=p, V=v, FOLD=int(f),
                                  **{f"k_{m}": kk[m] for m in METRICS}, **{f"mrr_{m}": mm[m] for m in METRICS}))
        mrr_pred = k_pred * pv_oof
        out = base[KEY + ["FOLD"]].assign(COMBO=combo, P=p, V=v, PV_PROXY=pv_oof, K_TARGET=k_target_oof,
                                          K_PRED=k_pred, **{TARGET: y}, MRR_PRED=mrr_pred,
                                          K_ERROR=k_pred - k_target_oof, MRR_ERROR=mrr_pred - y)
        preds.append(out)
        lo, hi = np.nanmin(k_target_oof), np.nanmax(k_target_oof)
        check_rows.append(dict(combo=combo, k_target_min=lo, k_target_max=hi, k_pred_min=k_pred.min(),
                               k_pred_max=k_pred.max(), n_unique_k_pred=int(pd.Series(k_pred).nunique()),
                               n_negative=int((k_pred < 0).sum()), n_below_k_target_min=int((k_pred < lo).sum()),
                               n_above_k_target_max=int((k_pred > hi).sum())))
        for stage, g in out.groupby("STAGE"):
            stage_rows.append(dict(combo=combo, STAGE=stage, n=len(g),
                                   **{f"mrr_{m}": metrics(g[TARGET], g["MRR_PRED"])[m] for m in METRICS},
                                   **{f"k_{m}": metrics(g["K_TARGET"], g["K_PRED"])[m] for m in METRICS}))

    fold_metrics = pd.DataFrame(fold_rows)
    value_cols = [f"{p}_{m}" for p in ("k", "mrr") for m in METRICS]
    summary = (fold_metrics.melt(id_vars=["combo", "P", "V", "FOLD"], value_vars=value_cols, var_name="metric")
               .groupby(["combo", "P", "V", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index())
    oof = pd.concat(preds, ignore_index=True)
    pooled = pd.DataFrame([dict(combo=c, **{f"mrr_{m}": metrics(g[TARGET], g["MRR_PRED"])[m] for m in METRICS},
                                **{f"k_{m}": metrics(g["K_TARGET"], g["K_PRED"])[m] for m in METRICS})
                           for c, g in oof.groupby("COMBO")])

    # 기준 조합(P4xV5) 과 같은 fold 비교
    piv = fold_metrics.pivot(index="FOLD", columns="combo", values="mrr_RMSE")
    paired = []
    for c in piv.columns:
        d = piv[c] - piv[BASELINE_COMBO]
        paired.append(dict(combo=c, mean_diff_vs_P4xV5=float(d.mean()), std_diff=float(d.std()),
                           folds_better_than_P4xV5=int((d < 0).sum())))
    paired = pd.DataFrame(paired)

    wide = summary.pivot_table(index="combo", columns="metric", values=["mean", "std_ddof1"])
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    table = (pd.DataFrame(avail_rows).set_index("combo").join(wide).join(paired.set_index("combo"))
             .reset_index().sort_values("mrr_RMSE_mean"))
    table["rank_by_mrr_RMSE"] = range(1, len(table) + 1)

    # 참고 기준 (Step 16) 과 Step 18 재현 확인
    s16_fold = pd.read_csv(STEP16_DIR / "fold_metrics.csv", float_precision="round_trip")
    s16 = (s16_fold.melt(id_vars=["model", "FOLD"], value_vars=METRICS, var_name="metric")
           .groupby(["model", "metric"], sort=False)["value"].agg(mean="mean", std_ddof1="std").reset_index())
    ref_rows = []
    for _, r in table.iterrows():
        ref_rows.append(dict(source="Step 19", group="Model 3 Hybrid", combo=r["combo"],
                             mrr_R2_mean=r["mrr_R2_mean"], mrr_R2_std=r["mrr_R2_std_ddof1"],
                             mrr_MAE_mean=r["mrr_MAE_mean"], mrr_MAE_std=r["mrr_MAE_std_ddof1"],
                             mrr_RMSE_mean=r["mrr_RMSE_mean"], mrr_RMSE_std=r["mrr_RMSE_std_ddof1"]))
    for m, label in [("Model1_XGB", "Model 1"), ("Model1_RF", "Model 1"), ("Model2_fixed_k", "Model 2")]:
        row = dict(source="Step 16", group=label, combo=m)
        for metric in METRICS:
            r = s16[(s16["model"] == m) & (s16["metric"] == metric)].iloc[0]
            row[f"mrr_{metric}_mean"], row[f"mrr_{metric}_std"] = r["mean"], r["std_ddof1"]
        ref_rows.append(row)
    comparison = pd.DataFrame(ref_rows).sort_values("mrr_RMSE_mean")

    s18 = pd.read_csv(STEP18_DIR / "summary.csv", float_precision="round_trip").set_index(["combo", "metric"])
    base_row = table[table["combo"] == BASELINE_COMBO].iloc[0]
    same_as_s18 = all(np.isclose(base_row[f"{p}_{m}_mean"], s18.loc[(f"C_full|XGBoost_scaled", f"{p}_{m}"), "mean"],
                                 rtol=0, atol=1e-12) for p in ("k", "mrr") for m in METRICS)
    print(f"P4xV5 결과가 Step 18 의 C_full standardized XGBoost 와 같은가: {same_as_s18}")
    assert same_as_s18

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_long.csv", index=False)
    table.to_csv(OUT_DIR / "combo_results.csv", index=False)
    pooled.to_csv(OUT_DIR / "pooled_metrics.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_difference_vs_P4xV5.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUT_DIR / "k_pred_checks.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT_DIR / "stage_metrics.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(OUT_DIR / "fold_scaling_values.csv", index=False)
    pd.DataFrame(median_rows).to_csv(OUT_DIR / "p11_normalization_medians.csv", index=False)
    comparison.to_csv(OUT_DIR / "comparison_with_reference_models.csv", index=False)
    oof[KEY + ["FOLD", "COMBO", "P", "V", "PV_PROXY", "K_TARGET", "K_PRED", TARGET, "MRR_PRED",
               "K_ERROR", "MRR_ERROR"]].to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    print("\n[PV_proxy 계산 가능 여부]")
    print(pd.DataFrame(avail_rows)[["combo", "n_pv_computable", "n_pv_missing", "n_pv_zero",
                                    "rows_used_min", "rows_used_median", "rows_used_max"]].to_string(index=False))
    print("\n[12개 조합: 최종 MRR RMSE 낮은 순]")
    show = table.assign(
        mrr_R2=lambda d: d.mrr_R2_mean.map("{:.4f}".format) + " ± " + d.mrr_R2_std_ddof1.map("{:.4f}".format),
        mrr_MAE=lambda d: d.mrr_MAE_mean.map("{:.4f}".format) + " ± " + d.mrr_MAE_std_ddof1.map("{:.4f}".format),
        mrr_RMSE=lambda d: d.mrr_RMSE_mean.map("{:.4f}".format) + " ± " + d.mrr_RMSE_std_ddof1.map("{:.4f}".format),
        k_R2=lambda d: d.k_R2_mean.map("{:.4f}".format) + " ± " + d.k_R2_std_ddof1.map("{:.4f}".format),
        vs_base=lambda d: d.mean_diff_vs_P4xV5.map("{:+.4f}".format) + " (" +
                          d.folds_better_than_P4xV5.astype(int).astype(str) + "/5)")
    print(show[["rank_by_mrr_RMSE", "combo", "mrr_RMSE", "mrr_MAE", "mrr_R2", "k_R2", "vs_base"]].to_string(index=False))
    print("\n[참고 기준과 함께]")
    print(comparison.assign(
        R2=lambda d: d.mrr_R2_mean.map("{:.4f}".format) + " ± " + d.mrr_R2_std.map("{:.4f}".format),
        RMSE=lambda d: d.mrr_RMSE_mean.map("{:.4f}".format) + " ± " + d.mrr_RMSE_std.map("{:.4f}".format))
        [["source", "group", "combo", "R2", "RMSE"]].to_string(index=False))
    print("\n[k_pred 확인]")
    print(pd.DataFrame(check_rows).round(8).to_string(index=False))
    print("\n[Stage 별 MRR 성능]")
    st = pd.DataFrame(stage_rows).pivot(index="combo", columns="STAGE", values=["mrr_RMSE", "mrr_R2"])
    print(st.round(4).to_string())
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

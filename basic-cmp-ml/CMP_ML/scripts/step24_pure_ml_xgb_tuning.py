"""Step 24: Pure ML XGBoost 에 Hybrid(Step 20) 와 같은 크기의 tuning 기회를 준다.

- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold.
  training 만 사용한다. test / validation 은 사용하지 않는다.
- Pure ML 구조: 공정 feature x -> MRR 직접 예측.
  feature 는 Step 16 Model 1 의 Baseline 구성 그대로 (STAGE + 숫자형 30개 = 31개).
  WAFER_ID 는 입력에서 제외, 실제 MRR 은 target 으로만 사용,
  CHAMBER_PATH 는 전부 4-5-6 이므로 입력하지 않는다. 새 feature 를 만들거나 지우지 않는다.
- 탐색 격자는 Step 20 과 동일한 54개:
    n_estimators {300, 600, 900} x max_depth {2, 3, 4} x learning_rate {0.03, 0.05, 0.1}
    x min_child_weight {1, 3}
  고정: subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
  reg_alpha, reg_lambda, gamma 등 나머지는 기본값을 유지하고 탐색하지 않는다.
- Hybrid 는 다시 튜닝하지 않는다. Step 20 에 저장된 대표 Hybrid 결과를 읽어서 비교만 한다.
- 유일한 최적 설정을 고르지 않는다. 상위 설정 간 차이가 fold 변동보다 작으면 그 사실을 그대로 보고한다.
출력: reports/step24_pure_ml_tuning/
실행: python -B scripts/step24_pure_ml_xgb_tuning.py
      python -B scripts/step24_pure_ml_xgb_tuning.py --reuse-predictions   (저장된 예측으로 재집계만)
"""
import argparse
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
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
STEP16_DIR = ROOT / "reports" / "step16_model3_prep"
STEP20_DIR = ROOT / "reports" / "step20_hybrid_xgb_tuning"
OUT_DIR = ROOT / "reports" / "step24_pure_ml_tuning"
OOF_DIR = OUT_DIR / "oof_top"
OOF_CACHE = OUT_DIR / "oof_all_configs.csv"

KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
SEED = 42
METRICS = ["R2", "MAE", "RMSE"]
N_TOP = 10
N_SAVE_OOF = 5
HYBRID_CONFIG = "n900_d2_lr0.1_mcw1"
GRID = dict(n_estimators=[300, 600, 900], max_depth=[2, 3, 4], learning_rate=[0.03, 0.05, 0.1],
            min_child_weight=[1, 3])
FIXED = dict(subsample=0.9, colsample_bytree=0.9)


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def config_id(p):
    return f"n{p['n_estimators']}_d{p['max_depth']}_lr{p['learning_rate']}_mcw{p['min_child_weight']}"


def build(params, numeric):
    """Step 16 / 20 과 같은 전처리 + XGBoost."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["STAGE"]),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep),
                     ("reg", XGBRegressor(random_state=SEED, n_jobs=-1, **FIXED, **params))])


def load_base():
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                        float_precision="round_trip")
    prep = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                       float_precision="round_trip")
    assert len(prep) == 1609 and (prep["CHAMBER_PATH"] == "4-5-6").all()
    base = prep[KEY + ["FOLD", TARGET]].merge(step7.drop(columns=["CHAMBER_PATH", TARGET]), on=KEY,
                                              how="left", validate="one_to_one")
    base["STAGE"] = base["STAGE"].astype(str)
    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    assert len(base) == 1609 and len(features) == 31 and base.notna().all().all()
    return base, features, numeric


def run_grid(base, features, numeric):
    """54개 설정 x 5 fold 를 학습하고 sample 별 out-of-fold 예측을 모은다."""
    folds = sorted(base["FOLD"].unique())
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    assert len(combos) == 54
    out = []
    for i, params in enumerate(combos, start=1):
        cid = config_id(params)
        pred = np.empty(len(base), dtype=float)
        for fold in folds:
            te = base["FOLD"].to_numpy() == fold
            model = build(params, numeric).fit(base.loc[~te, features], base.loc[~te, TARGET])
            pred[te] = model.predict(base.loc[te, features])
        out.append(pd.DataFrame({"config_id": cid, **{k: params[k] for k in GRID},
                                 "WAFER_ID": base["WAFER_ID"], "STAGE": base["STAGE"],
                                 "FOLD": base["FOLD"], TARGET: base[TARGET], "MRR_PRED": pred}))
        print(f"  [{i:2d}/54] {cid}")
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-predictions", action="store_true",
                    help="저장된 oof_all_configs.csv 로 집계만 다시 한다 (재학습 없음)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    base, features, numeric = load_base()

    if args.reuse_predictions and OOF_CACHE.exists():
        oof = pd.read_csv(OOF_CACHE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                          float_precision="round_trip")
        oof["STAGE"] = oof["STAGE"].astype(str)
        print(f"저장된 예측 재사용: {OOF_CACHE.name}")
    else:
        print("54개 설정 x 5 fold 학습")
        oof = run_grid(base, features, numeric)
        oof.to_csv(OOF_CACHE, index=False)

    # ---- fold 별 지표 --------------------------------------------------
    fold_rows, stage_rows = [], []
    for cid, g in oof.groupby("config_id", sort=False):
        params = {k: g[k].iloc[0] for k in GRID}
        for fold, gf in g.groupby("FOLD", sort=True):
            fold_rows.append(dict(config_id=cid, **params, FOLD=int(fold),
                                  n_train=len(g) - len(gf), n_test=len(gf),
                                  **metrics(gf[TARGET], gf.MRR_PRED)))
        for st, gs in g.groupby("STAGE", sort=True):
            stage_rows.append(dict(config_id=cid, STAGE=st, n=len(gs), **metrics(gs[TARGET], gs.MRR_PRED)))
    fold_metrics = pd.DataFrame(fold_rows)
    stage_metrics = pd.DataFrame(stage_rows)

    summary_long = (fold_metrics.melt(id_vars=["config_id"], value_vars=METRICS,
                                      var_name="metric", value_name="value")
                    .groupby(["config_id", "metric"], sort=False)["value"]
                    .agg(mean="mean", std_ddof1=lambda s: s.std(ddof=1), min="min", max="max")
                    .reset_index())
    summary_long["metric"] = pd.Categorical(summary_long["metric"], METRICS, ordered=True)
    summary_long = summary_long.sort_values(["config_id", "metric"]).reset_index(drop=True)

    wide = fold_metrics.groupby(["config_id"] + list(GRID), sort=False)[METRICS].agg(["mean", "std"])
    wide.columns = [f"{m}_{s.replace('std', 'std_ddof1')}" for m, s in wide.columns]
    config_results = wide.reset_index()
    config_results["rank_by_RMSE"] = config_results["RMSE_mean"].rank(method="min").astype(int)
    config_results = config_results.sort_values("RMSE_mean").reset_index(drop=True)

    # ---- 기준 모델들 ---------------------------------------------------
    s16_fold = pd.read_csv(STEP16_DIR / "fold_metrics.csv", float_precision="round_trip")
    s16_sum = pd.read_csv(STEP16_DIR / "summary.csv", float_precision="round_trip")
    s20_fold = pd.read_csv(STEP20_DIR / "fold_metrics.csv", float_precision="round_trip")
    s20_sum = pd.read_csv(STEP20_DIR / "summary_long.csv", float_precision="round_trip")
    s20_stage = pd.read_csv(STEP20_DIR / "stage_metrics.csv", dtype={"STAGE": "string"},
                            float_precision="round_trip")

    def s16_metric(model, metric, col):
        return float(s16_sum[(s16_sum.model == model) & (s16_sum.metric == metric)][col].iloc[0])

    def s20_metric(metric, col):
        r = s20_sum[(s20_sum.config_id == HYBRID_CONFIG) & (s20_sum.metric == f"mrr_{metric}")]
        return float(r[col].iloc[0])

    hybrid_fold = (s20_fold[s20_fold.config_id == HYBRID_CONFIG]
                   .sort_values("FOLD")[["FOLD", "mrr_R2", "mrr_MAE", "mrr_RMSE"]]
                   .rename(columns={f"mrr_{m}": m for m in METRICS}).reset_index(drop=True))

    best = config_results.iloc[0]
    best_id = best.config_id
    best_fold = fold_metrics[fold_metrics.config_id == best_id].sort_values("FOLD").reset_index(drop=True)

    rows = [dict(label="A. 기존 Pure ML XGBoost (Step 16 기본 설정)", model="Pure ML", source="Step 16",
                 config="XGBRegressor 기본값 (subsample / colsample 미설정)",
                 n_estimators=np.nan, max_depth=np.nan,
                 **{f"{m}_mean": s16_metric("Model1_XGB", m, "mean") for m in METRICS},
                 **{f"{m}_std_ddof1": s16_metric("Model1_XGB", m, "std_ddof1") for m in METRICS})]
    for rank in range(N_TOP):
        r = config_results.iloc[rank]
        rows.append(dict(label=f"B{rank + 1}. Pure ML tuning {rank + 1}위", model="Pure ML",
                         source="Step 24", config=r.config_id,
                         n_estimators=int(r.n_estimators), max_depth=int(r.max_depth),
                         **{f"{m}_mean": float(r[f"{m}_mean"]) for m in METRICS},
                         **{f"{m}_std_ddof1": float(r[f"{m}_std_ddof1"]) for m in METRICS}))
    rows.append(dict(label="C. Step 20 대표 Hybrid", model="Hybrid", source="Step 20",
                     config=f"{HYBRID_CONFIG}, PV = P9 x V3, standardized k",
                     n_estimators=900, max_depth=2,
                     **{f"{m}_mean": s20_metric(m, "mean") for m in METRICS},
                     **{f"{m}_std_ddof1": s20_metric(m, "std_ddof1") for m in METRICS}))
    rows.append(dict(label="D. fixed-k Preston", model="Model 2", source="Step 16",
                     config="절편 없는 fold 별 고정 k", n_estimators=np.nan, max_depth=np.nan,
                     **{f"{m}_mean": s16_metric("Model2_fixed_k", m, "mean") for m in METRICS},
                     **{f"{m}_std_ddof1": s16_metric("Model2_fixed_k", m, "std_ddof1") for m in METRICS}))
    comparison = pd.DataFrame(rows)

    # ---- 최상위 Pure ML 과 대표 Hybrid 의 fold 단위 비교 -----------------
    paired = best_fold[["FOLD"] + METRICS].rename(columns={m: f"{m}_PureML_best" for m in METRICS})
    paired = paired.merge(hybrid_fold.rename(columns={m: f"{m}_Hybrid" for m in METRICS}), on="FOLD")
    paired["RMSE_diff_Hybrid_minus_PureML"] = paired.RMSE_Hybrid - paired.RMSE_PureML_best
    paired["lower"] = np.where(paired.RMSE_diff_Hybrid_minus_PureML < 0, "Hybrid", "Pure ML")

    head_to_head = pd.DataFrame([
        dict(item="비교한 Pure ML 설정", value=best_id),
        dict(item="비교한 Hybrid 설정", value=f"{HYBRID_CONFIG} (P9 x V3)"),
        dict(item="Hybrid 가 더 낮은 fold 수", value=str(int((paired.lower == "Hybrid").sum()))),
        dict(item="Pure ML 이 더 낮은 fold 수", value=str(int((paired.lower == "Pure ML").sum()))),
        dict(item="평균 RMSE (Pure ML 최상위)", value=f"{best.RMSE_mean:.4f}"),
        dict(item="평균 RMSE (대표 Hybrid)", value=f"{s20_metric('RMSE', 'mean'):.4f}"),
        dict(item="평균 RMSE 차이 (Hybrid - Pure ML)",
             value=f"{s20_metric('RMSE', 'mean') - best.RMSE_mean:+.4f}"),
        dict(item="fold 별 차이의 표준편차",
             value=f"{paired.RMSE_diff_Hybrid_minus_PureML.std(ddof=1):.4f}"),
    ])

    # ---- 상위 설정끼리의 차이와 fold 변동 비교 ---------------------------
    top = config_results.head(N_TOP)
    spread = pd.DataFrame([
        dict(item=f"상위 {N_TOP}개 평균 RMSE 최소", value=f"{top.RMSE_mean.min():.4f}"),
        dict(item=f"상위 {N_TOP}개 평균 RMSE 최대", value=f"{top.RMSE_mean.max():.4f}"),
        dict(item=f"상위 {N_TOP}개 평균 RMSE 폭", value=f"{top.RMSE_mean.max() - top.RMSE_mean.min():.4f}"),
        dict(item=f"상위 {N_TOP}개 설정의 fold 표준편차 평균", value=f"{top.RMSE_std_ddof1.mean():.4f}"),
        dict(item="폭 / fold 표준편차 평균",
             value=f"{(top.RMSE_mean.max() - top.RMSE_mean.min()) / top.RMSE_std_ddof1.mean():.4f}"),
        dict(item="1위와 2위의 평균 RMSE 차이",
             value=f"{config_results.RMSE_mean.iloc[1] - config_results.RMSE_mean.iloc[0]:.4f}"),
        dict(item=f"상위 {N_TOP}개 중 가장 적은 tree 수", value=str(int(top.n_estimators.min()))),
        dict(item=f"상위 {N_TOP}개 중 가장 얕은 depth", value=str(int(top.max_depth.min()))),
    ])

    # ---- 일관성 확인 ---------------------------------------------------
    folds_file = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds_file["STAGE"] = folds_file["STAGE"].astype(str)
    merged = base[KEY + ["FOLD"]].merge(folds_file, on=KEY, suffixes=("", "_S8"))
    checks = pd.DataFrame([
        dict(item="sample 수가 1,609개인가", value=str(len(base) == 1609)),
        dict(item="feature 가 31개이고 WAFER_ID / CHAMBER_PATH 가 빠졌는가",
             value=str(len(features) == 31 and "WAFER_ID" not in features
                       and "CHAMBER_PATH" not in features and TARGET not in features)),
        dict(item="FOLD 가 Step 8 배정과 같은가", value=str(bool(merged.eval("FOLD == FOLD_S8").all()))),
        dict(item="설정이 54개인가", value=str(len(config_results) == 54)),
        dict(item="설정마다 1,609개 예측이 있는가",
             value=str(bool((oof.groupby("config_id").size() == 1609).all()))),
        dict(item="각 sample 이 정확히 한 fold 에서만 평가되는가",
             value=str(bool((oof[oof.config_id == best_id].groupby(KEY).size() == 1).all()))),
        dict(item="기존 Pure ML 평균 RMSE 가 Step 16 저장값과 같은가",
             value=f"{s16_metric('Model1_XGB', 'RMSE', 'mean'):.6f}"),
        dict(item="대표 Hybrid 평균 RMSE 가 Step 20 저장값과 같은가",
             value=f"{s20_metric('RMSE', 'mean'):.6f}"),
        dict(item="Hybrid 의 fold 별 RMSE 를 Step 20 에서 그대로 읽었는가",
             value=str(bool(np.allclose(
                 hybrid_fold.RMSE.to_numpy(),
                 s20_fold[s20_fold.config_id == HYBRID_CONFIG].sort_values("FOLD").mrr_RMSE.to_numpy(),
                 rtol=0, atol=0)))),
    ])

    # ---- 저장 -----------------------------------------------------------
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    stage_metrics.to_csv(OUT_DIR / "stage_metrics.csv", index=False)
    summary_long.to_csv(OUT_DIR / "summary_long.csv", index=False)
    config_results.to_csv(OUT_DIR / "config_results.csv", index=False)
    config_results.head(N_TOP).to_csv(OUT_DIR / "top10_configs.csv", index=False)
    comparison.to_csv(OUT_DIR / "comparison_with_reference_models.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_fold_comparison_vs_hybrid.csv", index=False)
    head_to_head.to_csv(OUT_DIR / "head_to_head_summary.csv", index=False)
    spread.to_csv(OUT_DIR / "top_config_spread.csv", index=False)
    checks.to_csv(OUT_DIR / "consistency_checks.csv", index=False)
    stage_metrics[stage_metrics.config_id.isin(config_results.config_id.head(N_TOP))].to_csv(
        OUT_DIR / "stage_metrics_top10.csv", index=False)
    s20_stage[s20_stage.config_id == HYBRID_CONFIG].to_csv(
        OUT_DIR / "hybrid_stage_metrics_reference.csv", index=False)
    for cid in config_results.config_id.head(N_SAVE_OOF):
        g = oof[oof.config_id == cid]
        g[KEY + ["FOLD", TARGET, "MRR_PRED"]].to_csv(OOF_DIR / f"oof_{cid}.csv", index=False)

    # ---- 화면 출력 -------------------------------------------------------
    print("\n[일관성 확인]")
    print(checks.to_string(index=False))
    print(f"\n[Pure ML 상위 {N_TOP} 설정]")
    cols = ["config_id", "n_estimators", "max_depth", "learning_rate", "min_child_weight",
            "RMSE_mean", "RMSE_std_ddof1", "MAE_mean", "R2_mean", "rank_by_RMSE"]
    print(config_results[cols].head(N_TOP).to_string(index=False))
    print("\n[상위 설정 차이와 fold 변동]")
    print(spread.to_string(index=False))
    print("\n[기준 모델과 비교]")
    print(comparison.to_string(index=False))
    print("\n[최상위 Pure ML 과 대표 Hybrid 의 fold 별 비교]")
    print(paired.to_string(index=False))
    print(head_to_head.to_string(index=False))
    print(f"\n[Stage 별 (상위 {N_TOP} 설정)]")
    print(stage_metrics[stage_metrics.config_id.isin(config_results.config_id.head(N_TOP))]
          .to_string(index=False))
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

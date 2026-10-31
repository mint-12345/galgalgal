"""Step 26: training 에서 고정한 세 모델을 PHM validation 데이터로 한 번 평가한다.

- 이 단계에서 처음으로 validation 을 사용한다. validation 결과를 보고 feature, P/V 정의,
  preprocessing, 모델 설정, hyperparameter 를 바꾸지 않는다. 평가만 한다.
- 세 모델은 training 4-5-6 의 1,609개 sample 전체로 한 번 학습한다.
    Model 1 Pure ML : XGBRegressor(n_estimators=900, max_depth=4, learning_rate=0.05,
                                   min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
                                   random_state=42)  (Step 24 대표)
    Model 2 Fixed-k : MRR = k x PV_proxy, 절편 없이 k = sum(PV*MRR) / sum(PV^2)
                      training 4-5-6 전체로 한 번 계산하고 validation 에서 다시 맞추지 않는다.
    Model 3 Hybrid  : MRR = k(x) x PV_proxy, k 는 training 통계로만 표준화한다 (Step 20~25 대표)
- PV_proxy = P9 x V3. P9 = pressure 6개의 raw row 별 평균, V3 = |HEAD_ROTATION|,
  두 값이 모두 0 이 아닌 raw row 에서만 P x V 를 평균한다 (training 과 동일).
- validation 전처리는 training 과 같은 방식만 적용한다.
    같은 wafer + Stage 단위, 같은 raw feature 계산, 같은 feature 정의, 같은 P9 x V3 정의,
    같은 연속 완전중복 행 제거 규칙.
  training 의 "MRR > 1,000 인 wafer+Stage 제외" 는 정답을 보고 표본을 고르는 규칙이므로
  평가 대상인 validation 에는 적용하지 않는다. 해당 sample 이 있는지는 개수만 보고한다.
- 4-5-6 이 아닌 validation sample 은 삭제하지 않고 개수와 구조만 보고한다.
  PV_proxy 를 계산할 수 없는 sample 에 Preston / Hybrid 를 억지로 적용하지 않는다.
출력: reports/step26_validation/ (CSV, 그림)
실행: python -B scripts/step26_validation_evaluation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
RAW = ROOT / "data" / "raw" / "PHM2016"
TRAIN_DIR = RAW / "2016 PHM DATA CHALLENGE CMP DATA SET" / "CMP-data" / "training"
TRAIN_MRR = RAW / "2016 PHM DATA CHALLENGE CMP DATA SET" / "CMP-training-removalrate.csv"
VAL_DIR = RAW / "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET" / "validation"
VAL_MRR = RAW / "PHM16TestValidationAnswers" / "PHM16TestValidationAnswers" / "orig_CMP-validation-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
STEP20_OOF = ROOT / "reports" / "step20_hybrid_xgb_tuning" / "oof_top" / "oof_n900_d2_lr0.1_mcw1.csv"
STEP24_OOF = ROOT / "reports" / "step24_pure_ml_tuning" / "oof_top" / "oof_n900_d4_lr0.05_mcw3.csv"
OUT_DIR = ROOT / "reports" / "step26_validation"
FIG_DIR = OUT_DIR / "figures"
VAL_TABLE = ROOT / "data" / "processed" / "step26" / "step26_validation_wafer_stage_table.csv"

KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
SEED = 42
FIXED = dict(subsample=0.9, colsample_bytree=0.9)
PURE_PARAMS = dict(n_estimators=900, max_depth=4, learning_rate=0.05, min_child_weight=3)
HYBRID_PARAMS = dict(n_estimators=900, max_depth=2, learning_rate=0.1, min_child_weight=1)
PRESSURES = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
             "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE"]
USAGE_MEAN = ["USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE", "USAGE_OF_DRESSER_TABLE",
              "USAGE_OF_BACKING_FILM"]
MEAN_STD = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
            "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE",
            "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
            "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION", "DRESSING_WATER_STATUS"]
MODELS = ["Pure ML", "Fixed-k Preston", "Hybrid"]
PRED_COL = {"Pure ML": "PRED_PURE", "Fixed-k Preston": "PRED_FIXED_K", "Hybrid": "PRED_HYBRID"}

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES1, SERIES2, SERIES3 = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834", "#898781"
MODEL_COLOR = {"Pure ML": SERIES2, "Hybrid": SERIES1, "Fixed-k Preston": SERIES3}


# ---------------------------------------------------------------- 전처리 (training 과 동일)
def load_raw(folder, pattern):
    frames = []
    for f in sorted(folder.glob(pattern)):
        df = pd.read_csv(f, dtype={"WAFER_ID": "int64", "STAGE": "string"})
        if df.empty:
            continue
        df["FILE"] = f.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    cols = [c for c in raw.columns if c != "FILE"]
    same_prev = raw[cols].eq(raw[cols].shift()).all(axis=1) & raw["FILE"].eq(raw["FILE"].shift())
    return raw[~same_prev], int(same_prev.sum()), len(raw), len(frames)


def chamber_path(chambers):
    order = [int(c) for c in chambers[chambers.ne(chambers.shift())]]
    if order == [1, 2, 3]:
        return "1-2-3"
    if order == [4, 5, 6]:
        return "4-5-6"
    return "other"


def build_table(data, mrr):
    """Step 7 과 같은 방식으로 wafer + Stage 한 행짜리 표를 만든다."""
    data = data.sort_values(KEY + ["TIMESTAMP"], kind="stable")
    g = data.groupby(KEY, sort=False)
    table = pd.DataFrame({"CHAMBER_PATH": g["CHAMBER"].agg(chamber_path)})
    for c in USAGE_MEAN:
        table[f"{c}_MEAN"] = g[c].mean()
    for c in MEAN_STD:
        table[f"{c}_MEAN"] = g[c].mean()
        table[f"{c}_STD"] = g[c].std(ddof=0)
    first_time = g["TIMESTAMP"].min()
    table = table.join(mrr.set_index(KEY)[TARGET], how="left")
    table = table.loc[first_time.sort_values(kind="stable").index].reset_index()
    table["STAGE"] = table["STAGE"].astype(str)
    return table


def pv_proxy(data, table):
    """PV_proxy = P9 x V3 를 P 와 V 가 모두 0 이 아닌 raw row 에서만 평균한다."""
    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(table["WAFER_ID"], table["STAGE"]))}
    keys = list(zip(data["WAFER_ID"].astype(int), data["STAGE"].astype(str)))
    keep = np.array([k in order for k in keys])
    rows = data[keep]
    codes = np.array([order[k] for k, ok in zip(keys, keep) if ok], dtype=np.int64)
    p_row = np.mean([rows[c].to_numpy() for c in PRESSURES], axis=0)
    v_row = np.abs(rows["HEAD_ROTATION"].to_numpy())
    mask = (p_row != 0) & (v_row != 0)
    n_used = np.bincount(codes, weights=mask.astype(float), minlength=len(table))
    sum_pv = np.bincount(codes, weights=p_row * v_row * mask, minlength=len(table))
    pv = np.where(n_used > 0, sum_pv / np.where(n_used > 0, n_used, 1), np.nan)
    return pv, n_used


def build(params, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["STAGE"]),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep),
                     ("reg", XGBRegressor(random_state=SEED, n_jobs=-1, **FIXED, **params))])


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=len(y), R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))))


# ---------------------------------------------------------------- 실행
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_TABLE.parent.mkdir(parents=True, exist_ok=True)
    checks = []

    # ---- training 쪽 (기존 확정 표 사용) ----
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                        float_precision="round_trip")
    prep_tbl = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                           float_precision="round_trip")
    hyb_oof = pd.read_csv(STEP20_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                          float_precision="round_trip")
    pure_oof = pd.read_csv(STEP24_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                           float_precision="round_trip")
    for d in (step7, prep_tbl, hyb_oof, pure_oof):
        d["STAGE"] = d["STAGE"].astype(str)

    train = prep_tbl[KEY + ["FOLD", TARGET]].merge(step7.drop(columns=["CHAMBER_PATH", TARGET]),
                                                   on=KEY, how="left", validate="one_to_one")
    train = train.merge(hyb_oof[KEY + ["PV_PROXY", "K_TARGET"]], on=KEY, how="left", validate="one_to_one")
    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    assert len(train) == 1609 and len(features) == 31 and train.notna().all().all()

    # ---- validation 전처리 ----
    val_raw, val_dups, val_rows, val_files = load_raw(VAL_DIR, "CMP-validation-*.csv")
    val_mrr = pd.read_csv(VAL_MRR, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                          float_precision="round_trip")
    val_mrr["STAGE"] = val_mrr["STAGE"].astype(str)
    val = build_table(val_raw, val_mrr)
    pv_val, n_used_val = pv_proxy(val_raw, val)
    val["PV_PROXY"] = pv_val
    val["N_PV_ROWS"] = n_used_val.astype(int)
    val.to_csv(VAL_TABLE, index=False)

    structure = (val.groupby("CHAMBER_PATH")
                 .agg(n_samples=("WAFER_ID", "size"), n_wafers=("WAFER_ID", "nunique"),
                      n_stage_A=("STAGE", lambda s: int((s == "A").sum())),
                      n_with_PV=("PV_PROXY", lambda s: int(s.notna().sum())),
                      n_missing_MRR=(TARGET, lambda s: int(s.isna().sum())),
                      mean_MRR=(TARGET, "mean"), min_MRR=(TARGET, "min"), max_MRR=(TARGET, "max"))
                 .reset_index())
    structure["n_stage_B"] = structure.n_samples - structure.n_stage_A

    val456 = val[val.CHAMBER_PATH == "4-5-6"].reset_index(drop=True)
    n_big = int((val456[TARGET] > MRR_LIMIT).sum())
    n_no_pv = int(val456.PV_PROXY.isna().sum())
    checks += [
        dict(item="validation 원본 파일 수 (폴더 전체)",
             value=str(len(list(VAL_DIR.glob("CMP-validation-*.csv"))))),
        dict(item="validation 원본 파일 수 (내용이 있는 파일)", value=str(val_files)),
        dict(item="validation 원본 행 수", value=f"{val_rows:,}"),
        dict(item="연속 완전중복으로 제외한 행 수", value=f"{val_dups:,}"),
        dict(item="validation wafer+Stage 수", value=str(len(val))),
        dict(item="정답 파일 행 수", value=str(len(val_mrr))),
        dict(item="MRR 이 비어 있는 wafer+Stage 수", value=str(int(val[TARGET].isna().sum()))),
        dict(item="validation 4-5-6 sample 수", value=str(len(val456))),
        dict(item="4-5-6 중 MRR > 1,000 인 sample 수 (제외하지 않음)", value=str(n_big)),
        dict(item="4-5-6 중 PV_proxy 를 계산할 수 없는 sample 수", value=str(n_no_pv)),
        dict(item="validation feature 열이 training 과 같은가",
             value=str(all(f in val.columns for f in features))),
    ]

    # ---- 세 모델을 training 전체로 한 번 학습 ----
    pure = build(PURE_PARAMS, numeric).fit(train[features], train[TARGET])
    k_fixed = float((train.PV_PROXY * train[TARGET]).sum() / (train.PV_PROXY ** 2).sum())
    mean_k, std_k = float(train.K_TARGET.mean()), float(train.K_TARGET.std(ddof=0))
    hybrid = build(HYBRID_PARAMS, numeric).fit(train[features], (train.K_TARGET - mean_k) / std_k)

    pred = val456[KEY + ["CHAMBER_PATH", TARGET, "PV_PROXY"]].copy()
    # XGBoost 는 float32 로 예측한다. 연산 자체는 Step 20~25 와 똑같이 두고 결과만 float64 로 넓혀
    # 저장한다. 그래야 CSV 에 적었다 읽어도 PRED_HYBRID = K_PRED x PV_PROXY 가 그대로 성립한다.
    pred["PRED_PURE"] = np.asarray(pure.predict(val456[features]), dtype=np.float64)
    pred["PRED_FIXED_K"] = k_fixed * val456.PV_PROXY
    pred["K_PRED"] = np.asarray(hybrid.predict(val456[features]) * std_k + mean_k, dtype=np.float64)
    pred["PRED_HYBRID"] = pred.K_PRED * val456.PV_PROXY
    for m in MODELS:
        pred[f"ERROR_{PRED_COL[m]}"] = pred[PRED_COL[m]] - pred[TARGET]

    # 세 모델을 모두 적용할 수 있는 sample (PV_proxy 가 있는 4-5-6)
    common = pred[pred.PV_PROXY.notna()].reset_index(drop=True)

    # ---- validation 성능 ----
    rows = []
    for m in MODELS:
        rows.append(dict(model=m, scope="validation 4-5-6 (three-model common set)",
                         **metrics(common[TARGET], common[PRED_COL[m]])))
    rows.append(dict(model="Pure ML", scope="validation 4-5-6 (all samples)",
                     **metrics(pred[TARGET], pred.PRED_PURE)))
    val_metrics = pd.DataFrame(rows)

    stage_rows = []
    for m in MODELS:
        for st in ("A", "B"):
            sub = common[common.STAGE == st]
            stage_rows.append(dict(model=m, STAGE=st, **metrics(sub[TARGET], sub[PRED_COL[m]])))
    stage_metrics = pd.DataFrame(stage_rows)

    # MRR > 1,000 sample 을 뺐을 때의 민감도 (참고용, 주 지표 아님)
    sens = []
    if n_big:
        keep = common[common[TARGET] <= MRR_LIMIT]
        for m in MODELS:
            sens.append(dict(model=m, scope=f"MRR <= {MRR_LIMIT} 만 (참고)",
                             **metrics(keep[TARGET], keep[PRED_COL[m]])))
    sensitivity = pd.DataFrame(sens)

    # ---- training OOF 기준값 ----
    folds = sorted(train.FOLD.unique())
    k_oof = np.empty(len(train))
    for f in folds:
        te = train.FOLD.to_numpy() == f
        tr = ~te
        k_f = float((train.PV_PROXY[tr] * train[TARGET][tr]).sum() / (train.PV_PROXY[tr] ** 2).sum())
        k_oof[te] = k_f
    train_oof = train[KEY + ["FOLD", TARGET, "PV_PROXY"]].copy()
    train_oof["PRED_PURE"] = pure_oof.set_index(KEY).loc[
        list(zip(train.WAFER_ID, train.STAGE)), "MRR_PRED"].to_numpy()
    train_oof["PRED_HYBRID"] = hyb_oof.set_index(KEY).loc[
        list(zip(train.WAFER_ID, train.STAGE)), "MRR_PRED"].to_numpy()
    train_oof["PRED_FIXED_K"] = k_oof * train.PV_PROXY
    oof_rows = [dict(model=m, scope="training 5-fold OOF", **metrics(train_oof[TARGET],
                                                                    train_oof[PRED_COL[m]]))
                for m in MODELS]
    train_metrics = pd.DataFrame(oof_rows)

    # ---- generalization gap ----
    t = train_metrics.set_index("model")
    v = val_metrics[val_metrics.scope.str.startswith("validation 4-5-6 (three")].set_index("model")
    gap = pd.DataFrame([dict(
        model=m, training_OOF_RMSE=float(t.loc[m, "RMSE"]), validation_RMSE=float(v.loc[m, "RMSE"]),
        RMSE_gap=float(v.loc[m, "RMSE"] - t.loc[m, "RMSE"]),
        RMSE_ratio=float(v.loc[m, "RMSE"] / t.loc[m, "RMSE"]),
        training_OOF_MAE=float(t.loc[m, "MAE"]), validation_MAE=float(v.loc[m, "MAE"]),
        MAE_gap=float(v.loc[m, "MAE"] - t.loc[m, "MAE"]),
        training_OOF_R2=float(t.loc[m, "R2"]), validation_R2=float(v.loc[m, "R2"]),
        R2_gap=float(v.loc[m, "R2"] - t.loc[m, "R2"])) for m in MODELS])

    # ---- 분포 비교 ----
    dist_vars = [TARGET, "PV_PROXY"] + [f"{c}_MEAN" for c in USAGE_MEAN] + \
                ["RETAINER_RING_PRESSURE_MEAN", "PRESSURIZED_CHAMBER_PRESSURE_MEAN",
                 "HEAD_ROTATION_MEAN", "WAFER_ROTATION_MEAN", "STAGE_ROTATION_MEAN"]
    tr456 = train
    va456 = val456[val456.PV_PROXY.notna()].reset_index(drop=True)
    dist = []
    for var in dist_vars:
        a, b = tr456[var].dropna(), va456[var].dropna()
        pooled_sd = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2)
        dist.append(dict(variable=var, training_mean=float(a.mean()), validation_mean=float(b.mean()),
                         mean_gap=float(b.mean() - a.mean()),
                         training_std=float(a.std(ddof=0)), validation_std=float(b.std(ddof=0)),
                         training_median=float(a.median()), validation_median=float(b.median()),
                         standardized_gap=float((b.mean() - a.mean()) / pooled_sd) if pooled_sd else np.nan))
    dist = pd.DataFrame(dist).sort_values("standardized_gap", key=abs, ascending=False).reset_index(drop=True)
    stage_ratio = pd.DataFrame([
        dict(dataset="training 4-5-6", n=len(tr456), stage_A=int((tr456.STAGE == "A").sum()),
             stage_B=int((tr456.STAGE == "B").sum()),
             stage_B_ratio=float((tr456.STAGE == "B").mean())),
        dict(dataset="validation 4-5-6", n=len(va456), stage_A=int((va456.STAGE == "A").sum()),
             stage_B=int((va456.STAGE == "B").sum()),
             stage_B_ratio=float((va456.STAGE == "B").mean()))])

    checks += [
        dict(item="fixed k (training 4-5-6 전체)", value=f"{k_fixed:.10g}"),
        dict(item="k 표준화 평균 (training)", value=f"{mean_k:.10g}"),
        dict(item="k 표준화 표준편차 (training)", value=f"{std_k:.10g}"),
        dict(item="Hybrid 예측 = k_pred x PV_proxy 인가",
             value=str(bool(np.allclose(common.PRED_HYBRID, common.K_PRED * common.PV_PROXY,
                                        rtol=0, atol=1e-12)))),
        # 같은 기록이 양쪽에 들어갔는지(키 겹침)와, 같은 wafer 가 다른 Stage 로 양쪽에 나타나는지를 나눠서 기록한다
        dict(item="validation 과 training 이 공유하는 wafer+Stage 키 수",
             value=str(len(set(map(tuple, val[KEY].itertuples(index=False))) &
                           set(map(tuple, train[KEY].itertuples(index=False)))))),
        dict(item="validation 과 training 이 공유하는 WAFER_ID 수 (Stage 는 서로 다름)",
             value=str(len(set(val.WAFER_ID) & set(train.WAFER_ID)))),
    ]
    checks = pd.DataFrame(checks)

    # ---- 저장 ----
    structure.to_csv(OUT_DIR / "validation_structure.csv", index=False)
    val_metrics.to_csv(OUT_DIR / "validation_metrics.csv", index=False)
    stage_metrics.to_csv(OUT_DIR / "validation_stage_metrics.csv", index=False)
    train_metrics.to_csv(OUT_DIR / "training_oof_reference.csv", index=False)
    gap.to_csv(OUT_DIR / "generalization_gap.csv", index=False)
    dist.to_csv(OUT_DIR / "distribution_comparison.csv", index=False)
    stage_ratio.to_csv(OUT_DIR / "stage_ratio_comparison.csv", index=False)
    pred.to_csv(OUT_DIR / "validation_predictions.csv", index=False)
    checks.to_csv(OUT_DIR / "consistency_checks.csv", index=False)
    if len(sensitivity):
        sensitivity.to_csv(OUT_DIR / "sensitivity_without_high_mrr.csv", index=False)

    # ---- 그림 ----
    def style(ax, axis="y"):
        ax.set_facecolor(SURFACE)
        ax.grid(axis=axis, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)

    def legend(ax, loc="upper left"):
        leg = ax.legend(frameon=False, fontsize=8.5, loc=loc)
        for x in leg.get_texts():
            x.set_color(INK2)

    # 그림 1: training OOF vs validation RMSE
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    xx = np.arange(len(MODELS))
    ymax = max(gap.validation_RMSE.max(), gap.training_OOF_RMSE.max()) * 1.25
    for i, (label, col) in enumerate([("Training 5-fold OOF", "training_OOF_RMSE"),
                                      ("Validation (held out)", "validation_RMSE")]):
        off = (i - 0.5) * 0.36
        vals = gap[col].to_numpy()
        ax.bar(xx + off, vals, width=0.34, color=[SERIES1, SERIES2][i], label=label)
        for x, val in zip(xx + off, vals):
            ax.text(x, val + ymax * 0.015, f"{val:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xx, gap.model)
    style(ax)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Model (frozen before validation was opened)", fontsize=9, color=INK2)
    ax.set_title("Training out-of-fold vs held-out validation RMSE\n"
                 "same CHAMBER_PATH = 4-5-6 samples, models fitted once on all 1,609 training samples",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_training_vs_validation_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 2: validation R2 / RMSE
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    vm = val_metrics[val_metrics.scope.str.startswith("validation 4-5-6 (three")].set_index("model")
    for ax, col, title, fmt in [(axes[0], "R2", "Validation R²  (higher is better)", "{:.3f}"),
                                (axes[1], "RMSE", "Validation RMSE  (lower is better)", "{:.2f}")]:
        vals = [float(vm.loc[m, col]) for m in MODELS]
        ax.bar(np.arange(len(MODELS)), vals, width=0.55, color=[MODEL_COLOR[m] for m in MODELS])
        lo = min(0, min(vals) * 1.2)
        hi = max(vals) * 1.25 if max(vals) > 0 else 1
        for x, v in zip(np.arange(len(MODELS)), vals):
            ax.text(x, v + (hi - lo) * 0.02, fmt.format(v), ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(np.arange(len(MODELS)), MODELS, fontsize=8)
        style(ax)
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Model", fontsize=8.5, color=INK2)
    fig.suptitle("Held-out validation performance of the three frozen models",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig2_validation_metrics.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 3: 실제 vs 예측
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    lo = float(min(common[TARGET].min(), common.PRED_PURE.min(), common.PRED_HYBRID.min())) - 2
    hi = float(max(common[TARGET].max(), common.PRED_PURE.max(), common.PRED_HYBRID.max())) + 2
    for ax, m in zip(axes, ["Pure ML", "Hybrid"]):
        ax.scatter(common[TARGET], common[PRED_COL[m]], s=12, color=MODEL_COLOR[m], alpha=0.55,
                   edgecolors=SURFACE, linewidths=0.4)
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1.0, ls="--")
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual MRR (validation)", fontsize=9, color=INK2)
        ax.set_ylabel("Predicted MRR", fontsize=9, color=INK2)
        r = float(vm.loc[m, "RMSE"])
        ax.set_title(f"{m}   RMSE = {r:.2f}", fontsize=10, color=INK, loc="left")
    fig.suptitle("Validation: actual vs predicted MRR (dashed line = perfect prediction)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig3_actual_vs_predicted.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 4: 분포 비교
    show = [TARGET, "PV_PROXY", "USAGE_OF_DRESSER_MEAN", "USAGE_OF_DRESSER_TABLE_MEAN",
            "RETAINER_RING_PRESSURE_MEAN", "HEAD_ROTATION_MEAN"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, var in zip(axes.ravel(), show):
        a, b = tr456[var].dropna(), va456[var].dropna()
        bins = np.histogram_bin_edges(np.concatenate([a, b]), bins=30)
        ax.hist(a, bins=bins, color=SERIES1, alpha=0.55, label="training 4-5-6", density=True)
        ax.hist(b, bins=bins, color=SERIES2, alpha=0.55, label="validation 4-5-6", density=True)
        style(ax)
        d = float(dist[dist.variable == var].standardized_gap.iloc[0])
        ax.set_title(f"{var}\nstandardized mean gap = {d:+.2f}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Value", fontsize=8, color=INK2)
        ax.set_ylabel("Density", fontsize=8, color=INK2)
    legend(axes[0, 0], loc="upper right")
    fig.suptitle("Training vs validation distributions (CHAMBER_PATH = 4-5-6)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig4_distribution_comparison.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---- 출력 ----
    print("[일관성 확인 / validation 구조]")
    print(checks.to_string(index=False))
    print("\n[CHAMBER_PATH 별 구조]")
    print(structure.to_string(index=False))
    print("\n[validation 성능]")
    print(val_metrics.to_string(index=False))
    if len(sensitivity):
        print("\n[참고: MRR > 1,000 제외 시]")
        print(sensitivity.to_string(index=False))
    print("\n[Stage 별]")
    print(stage_metrics.to_string(index=False))
    print("\n[training OOF 기준]")
    print(train_metrics.to_string(index=False))
    print("\n[generalization gap]")
    print(gap.to_string(index=False))
    print("\n[Stage 비율]")
    print(stage_ratio.to_string(index=False))
    print("\n[분포 비교]")
    print(dist.to_string(index=False))
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

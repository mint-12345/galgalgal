"""Step 22: Pure ML 과 Hybrid 의 robustness(조건 변화에 대한 안정성) 비교.

- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold. training 만 사용한다.
  test / validation 은 사용하지 않는다.
- 새 feature, 모델 구조 변경, 추가 튜닝을 하지 않는다. 새로 학습하지도 않는다.
  이미 저장된 5-fold 예측(Step 16 의 Model1_XGB, Step 20 의 n900_d2_lr0.1_mcw1)을
  조건 그룹별로 다시 집계하기만 한다.
- Pure ML : Step 16 Model 1 XGBoost (공정 feature x -> MRR 직접 예측)
- Hybrid  : Step 20~21 대표 Hybrid, MRR = k(x) x PV_proxy, PV_proxy = P9 x V3
            k(x): n_estimators 900, max_depth 2, learning_rate 0.1, min_child_weight 1,
                  subsample 0.9, colsample_bytree 0.9, random_state 42
            k 표준화는 Step 18~21 과 같이 training fold 통계만 사용한 값이 이미 반영되어 있다.
- 조건 그룹: Stage(A/B), usage 4개 변수의 low/mid/high(전체 33%/67% 분위수),
            시간 초기/중기/후기(Step 7 의 첫 기록 시각 순으로 3등분).
출력: reports/step22_robustness/ (CSV, 그림)
실행: python -B scripts/step22_robustness_comparison.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
STEP16_DIR = ROOT / "reports" / "step16_model3_prep"
STEP20_DIR = ROOT / "reports" / "step20_hybrid_xgb_tuning"
STEP21_DIR = ROOT / "reports" / "step21_interpretation"
OUT_DIR = ROOT / "reports" / "step22_robustness"
FIG_DIR = OUT_DIR / "figures"
TIME_FILE = ROOT / "data" / "processed" / "step22" / "step22_first_timestamp.csv"

KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
HYBRID_CONFIG = "n900_d2_lr0.1_mcw1"
USAGE_VARS = ["USAGE_OF_DRESSER_MEAN", "USAGE_OF_POLISHING_TABLE_MEAN",
              "USAGE_OF_DRESSER_TABLE_MEAN", "USAGE_OF_BACKING_FILM_MEAN"]
USAGE_LEVELS = ["low", "mid", "high"]
TIME_LEVELS = ["early", "middle", "late"]
MODELS = {"Pure ML": "MRR_PRED_PURE", "Hybrid": "MRR_PRED_HYBRID"}
Q_LOW, Q_HIGH = 0.33, 0.67

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES1, SERIES2 = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"
MODEL_COLOR = {"Pure ML": SERIES2, "Hybrid": SERIES1}


# ---------------------------------------------------------------- 입력 준비
def first_timestamp():
    """Step 7 과 같은 제외 규칙을 적용해 wafer+Stage 별 첫 기록 시각을 구한다."""
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
    drop = data.set_index(KEY).index.isin(big.set_index(KEY).index)
    data = data[~drop]
    assert int(same_prev.sum()) == 1819 and len(big) == 4
    print(f"원본 행 {len(raw):,} → 연속 완전 중복 {int(same_prev.sum()):,}행 제외 → "
          f"MRR>{MRR_LIMIT:,} 묶음 {len(big)}개 제외 → 사용 행 {len(data):,}")
    t = data.groupby(KEY, sort=False)["TIMESTAMP"].min().rename("FIRST_TIMESTAMP").reset_index()
    t["STAGE"] = t["STAGE"].astype(str)
    return t


def load_inputs():
    pure = pd.read_csv(STEP16_DIR / "oof_predictions.csv", dtype={"STAGE": "string"},
                       float_precision="round_trip")
    hyb = pd.read_csv(STEP20_DIR / "oof_top" / f"oof_{HYBRID_CONFIG}.csv", dtype={"STAGE": "string"},
                      float_precision="round_trip")
    step21 = pd.read_csv(STEP21_DIR / "oof_predictions_both_models.csv", dtype={"STAGE": "string"},
                         float_precision="round_trip")
    step7 = pd.read_csv(STEP7_FILE, dtype={"STAGE": "string"}, float_precision="round_trip")
    folds = pd.read_csv(FOLD_FILE, dtype={"STAGE": "string"})
    for d in (pure, hyb, step21, step7, folds):
        d["STAGE"] = d["STAGE"].astype(str)

    assert (hyb["CONFIG"] == HYBRID_CONFIG).all()
    # PV_PROXY 는 Hybrid 가 쓰는 P9 x V3 를 가져온다.
    # Step 16 파일의 PV_PROXY 는 그 단계에서 쓰던 다른 P x V 조합이므로 여기서는 사용하지 않는다.
    df = pure[KEY + ["CHAMBER_PATH", "FOLD", TARGET, "PRED_Model1_XGB"]].rename(
        columns={"PRED_Model1_XGB": "MRR_PRED_PURE"})
    df = df.merge(hyb[KEY + ["PV_PROXY", "MRR_PRED", "K_TARGET", "K_PRED"]].rename(
        columns={"MRR_PRED": "MRR_PRED_HYBRID"}), on=KEY, how="left", validate="one_to_one")
    df = df.merge(step7[KEY + USAGE_VARS], on=KEY, how="left", validate="one_to_one")

    if TIME_FILE.exists():
        t = pd.read_csv(TIME_FILE, dtype={"STAGE": "string"}, float_precision="round_trip")
        t["STAGE"] = t["STAGE"].astype(str)
    else:
        t = first_timestamp()
        TIME_FILE.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(TIME_FILE, index=False)
    df = df.merge(t, on=KEY, how="left", validate="one_to_one")

    assert len(df) == 1609 and (df["CHAMBER_PATH"] == "4-5-6").all()
    assert df.notna().all().all()
    return df, step21, folds, step7, t


# ---------------------------------------------------------------- 지표
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=len(y), R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))))


def group_rows(df, group_type, group_col, order):
    out = []
    for level in order:
        sub = df[df[group_col] == level]
        for model, col in MODELS.items():
            out.append(dict(group_type=group_type, group_key=group_col, group=level, model=model,
                            **metrics(sub[TARGET], sub[col])))
    return out


def assign_bins(df):
    edges = []
    for v in USAGE_VARS:
        lo, hi = df[v].quantile(Q_LOW), df[v].quantile(Q_HIGH)
        df[f"BIN_{v}"] = np.where(df[v] <= lo, "low", np.where(df[v] <= hi, "mid", "high"))
        counts = df[f"BIN_{v}"].value_counts()
        edges.append(dict(usage_variable=v, q33_boundary=lo, q67_boundary=hi,
                          n_low=int(counts.get("low", 0)), n_mid=int(counts.get("mid", 0)),
                          n_high=int(counts.get("high", 0)),
                          min_value=df[v].min(), max_value=df[v].max()))

    order = df.sort_values(["FIRST_TIMESTAMP"] + KEY, kind="stable").index
    parts = np.array_split(np.arange(len(order)), 3)
    df["TIME_BIN"] = pd.Series("", index=df.index, dtype=object)
    time_info = []
    for name, idx in zip(TIME_LEVELS, parts):
        rows = order[idx]
        df.loc[rows, "TIME_BIN"] = name
        time_info.append(dict(time_bin=name, n=len(rows),
                              first_timestamp_min=df.loc[rows, "FIRST_TIMESTAMP"].min(),
                              first_timestamp_max=df.loc[rows, "FIRST_TIMESTAMP"].max()))
    df["TIME_ORDER"] = pd.Series(np.arange(len(order)), index=order)
    return df, pd.DataFrame(edges), pd.DataFrame(time_info)


# ---------------------------------------------------------------- 실행
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, step21, folds, step7, tstamp = load_inputs()

    # ---- 일관성 확인 -------------------------------------------------
    s21 = step21.set_index(KEY).loc[list(zip(df.WAFER_ID, df.STAGE))]
    checks = [
        dict(item="sample 수가 1,609개인가", value=str(len(df) == 1609)),
        dict(item="CHAMBER_PATH 가 모두 4-5-6 인가", value=str(bool((df["CHAMBER_PATH"] == "4-5-6").all()))),
        dict(item="FOLD 가 Step 8 배정과 같은가",
             value=str(bool(df.merge(folds, on=KEY, suffixes=("", "_S8"))
                            .eval("FOLD == FOLD_S8").all()))),
        dict(item="Pure ML 예측이 Step 21 저장값과 같은가",
             value=str(bool(np.array_equal(df.MRR_PRED_PURE.to_numpy(), s21.MRR_PRED_PURE.to_numpy())))),
        dict(item="Hybrid 예측이 Step 21 저장값과 같은가",
             value=str(bool(np.array_equal(df.MRR_PRED_HYBRID.to_numpy(), s21.MRR_PRED_HYBRID.to_numpy())))),
        dict(item="Hybrid 예측 = k_pred x PV_proxy 인가",
             value=str(bool(np.allclose(df.MRR_PRED_HYBRID, df.K_PRED * df.PV_PROXY, rtol=0, atol=1e-12)))),
    ]

    # 전체 성능 재계산 후 저장된 값과 대조
    overall = []
    for model, col in MODELS.items():
        overall.append(dict(model=model, scope="pooled_out_of_fold", **metrics(df[TARGET], df[col])))
    overall = pd.DataFrame(overall)
    o = overall.set_index("model")

    saved_pure = pd.read_csv(STEP16_DIR / "pooled_metrics.csv", float_precision="round_trip")
    saved_pure = saved_pure[saved_pure.model == "Model1_XGB"].iloc[0]
    checks.append(dict(item="Pure ML 전체 지표가 Step 16 저장값과 같은가",
                       value=str(bool(np.allclose([o.loc["Pure ML", m] for m in ("R2", "MAE", "RMSE")],
                                                  [saved_pure.R2, saved_pure.MAE, saved_pure.RMSE],
                                                  rtol=0, atol=1e-12)))))
    s20_stage = pd.read_csv(STEP20_DIR / "stage_metrics.csv", dtype={"STAGE": "string"},
                            float_precision="round_trip")
    s20_stage = s20_stage[s20_stage.config_id == HYBRID_CONFIG].copy()
    s20_stage["STAGE"] = s20_stage["STAGE"].astype(str)

    # ---- 그룹별 지표 -------------------------------------------------
    df, usage_edges, time_info = assign_bins(df)
    rows = group_rows(df, "stage", "STAGE", ["A", "B"])
    for v in USAGE_VARS:
        rows += group_rows(df, "usage", f"BIN_{v}", USAGE_LEVELS)
    rows += group_rows(df, "time", "TIME_BIN", TIME_LEVELS)
    g = pd.DataFrame(rows)
    g["group_label"] = np.where(g.group_type == "usage",
                                g.group_key.str.replace("BIN_", "", regex=False) + " = " + g.group,
                                g.group_type + " = " + g.group)
    g = g.merge(overall[["model", "RMSE", "MAE", "R2"]].rename(
        columns={"RMSE": "overall_RMSE", "MAE": "overall_MAE", "R2": "overall_R2"}), on="model")
    g["delta_RMSE"] = g.RMSE - g.overall_RMSE
    g["delta_RMSE_pct"] = g.delta_RMSE / g.overall_RMSE * 100
    g["delta_MAE"] = g.MAE - g.overall_MAE
    g = g[["group_type", "group_key", "group", "group_label", "model", "n", "R2", "MAE", "RMSE",
           "overall_RMSE", "delta_RMSE", "delta_RMSE_pct", "delta_MAE"]]

    # Stage 지표를 Step 20 저장값과 대조 (Hybrid)
    st = g[(g.group_type == "stage") & (g.model == "Hybrid")].set_index("group")
    checks.append(dict(item="Hybrid Stage별 지표가 Step 20 저장값과 같은가",
                       value=str(bool(np.allclose(
                           [st.loc[s, m] for s in ("A", "B") for m in ("R2", "MAE", "RMSE")],
                           [s20_stage.set_index("STAGE").loc[s, f"mrr_{m}"] for s in ("A", "B")
                            for m in ("R2", "MAE", "RMSE")], rtol=0, atol=1e-12)))))
    # 시간 순서가 Step 7 표의 행 순서와 같은지
    step7_order = {k: i for i, k in enumerate(zip(step7.WAFER_ID, step7.STAGE))}
    mine = df.sort_values("TIME_ORDER")
    checks.append(dict(item="첫 기록 시각 정렬이 Step 7 표의 행 순서와 같은가",
                       value=str(bool(pd.Series([step7_order[k] for k in zip(mine.WAFER_ID, mine.STAGE)])
                                      .is_monotonic_increasing))))
    checks = pd.DataFrame(checks)

    # ---- robustness 요약 --------------------------------------------
    summary = []
    for model in MODELS:
        d = g[g.model == model]
        worst = d.loc[d.RMSE.idxmax()]
        best = d.loc[d.RMSE.idxmin()]
        summary.append(dict(
            model=model, n_groups=len(d), overall_RMSE=float(o.loc[model, "RMSE"]),
            overall_MAE=float(o.loc[model, "MAE"]), overall_R2=float(o.loc[model, "R2"]),
            min_group_RMSE=float(d.RMSE.min()), max_group_RMSE=float(d.RMSE.max()),
            range_group_RMSE=float(d.RMSE.max() - d.RMSE.min()),
            std_group_RMSE=float(d.RMSE.std(ddof=0)),
            mean_abs_delta_RMSE=float(d.delta_RMSE.abs().mean()),
            max_delta_RMSE=float(d.delta_RMSE.max()),
            worst_group=worst.group_label, worst_group_n=int(worst.n), worst_group_RMSE=float(worst.RMSE),
            best_group=best.group_label, best_group_RMSE=float(best.RMSE)))
    summary = pd.DataFrame(summary)

    worst_rows = []
    for model in MODELS:
        d = g[g.model == model].sort_values("RMSE", ascending=False)
        for _, r in d.head(5).iterrows():
            other = g[(g.model != model) & (g.group_label == r.group_label)].iloc[0]
            worst_rows.append(dict(model=model, rank=len(worst_rows) % 5 + 1, group_type=r.group_type,
                                   group_label=r.group_label, n=int(r.n), RMSE=r.RMSE, MAE=r.MAE, R2=r.R2,
                                   overall_RMSE=r.overall_RMSE, delta_RMSE=r.delta_RMSE,
                                   other_model=other.model, other_model_RMSE=other.RMSE,
                                   RMSE_difference=r.RMSE - other.RMSE))
    worst = pd.DataFrame(worst_rows)

    # 모델 간 그룹별 차이 (Hybrid - Pure ML)
    wide = g.pivot_table(index=["group_type", "group_label", "n"], columns="model",
                         values=["RMSE", "MAE", "R2", "delta_RMSE"]).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1].replace(' ', '')}" for c in wide.columns]
    wide["RMSE_diff_Hybrid_minus_PureML"] = wide["RMSE_Hybrid"] - wide["RMSE_PureML"]
    wide["MAE_diff_Hybrid_minus_PureML"] = wide["MAE_Hybrid"] - wide["MAE_PureML"]
    wide = wide.sort_values(["group_type", "group_label"]).reset_index(drop=True)

    # ---- 저장 -------------------------------------------------------
    overall.to_csv(OUT_DIR / "overall_metrics.csv", index=False)
    g.to_csv(OUT_DIR / "group_metrics_all.csv", index=False)
    g[g.group_type == "stage"].to_csv(OUT_DIR / "stage_metrics.csv", index=False)
    g[g.group_type == "usage"].to_csv(OUT_DIR / "usage_group_metrics.csv", index=False)
    g[g.group_type == "time"].to_csv(OUT_DIR / "time_group_metrics.csv", index=False)
    usage_edges.to_csv(OUT_DIR / "usage_bin_edges.csv", index=False)
    time_info.to_csv(OUT_DIR / "time_bin_info.csv", index=False)
    summary.to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    worst.to_csv(OUT_DIR / "worst_groups.csv", index=False)
    wide.to_csv(OUT_DIR / "model_difference_by_group.csv", index=False)
    checks.to_csv(OUT_DIR / "consistency_checks.csv", index=False)
    keep = KEY + ["FOLD", "TIME_ORDER", "FIRST_TIMESTAMP", "TIME_BIN"] + \
        [f"BIN_{v}" for v in USAGE_VARS] + USAGE_VARS + [TARGET, "MRR_PRED_PURE", "MRR_PRED_HYBRID"]
    df.sort_values("TIME_ORDER")[keep].to_csv(OUT_DIR / "sample_group_assignment.csv", index=False)

    # ---- 그림 -------------------------------------------------------
    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)

    def grouped_bars(ax, labels, values_by_model, ymax=None, annotate=True):
        xx = np.arange(len(labels))
        for i, (model, vals) in enumerate(values_by_model.items()):
            off = (i - 0.5) * 0.36
            ax.bar(xx + off, vals, width=0.34, color=MODEL_COLOR[model], label=model)
            if annotate:
                for x, v in zip(xx + off, vals):
                    # 전체 RMSE 점선이 값 라벨 위를 지나가지 않도록 라벨에 배경을 깔고 위로 올린다
                    ax.text(x, v + (ymax or max(vals)) * 0.015, f"{v:.2f}", ha="center",
                            fontsize=7.5, color=INK2, zorder=4,
                            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        ax.set_xticks(xx, labels)
        style(ax)
        if ymax:
            ax.set_ylim(0, ymax)

    def model_values(sub, order, col="RMSE"):
        return {m: [float(sub[(sub.model == m) & (sub.group == lv)][col].iloc[0]) for lv in order]
                for m in MODELS}

    # 그림 1: Stage 별 RMSE
    sub = g[g.group_type == "stage"]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    vals = model_values(sub, ["A", "B"])
    grouped_bars(ax, [f"Stage A  (n={int(sub[sub.group == 'A'].n.iloc[0])})",
                      f"Stage B  (n={int(sub[sub.group == 'B'].n.iloc[0])})"], vals, ymax=6.0)
    for model in MODELS:
        ax.axhline(float(o.loc[model, "RMSE"]), color=MODEL_COLOR[model], lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Polishing stage", fontsize=9, color=INK2)
    ax.set_title("Robustness by stage: Pure ML vs Hybrid\ndashed line = each model's overall RMSE on all 1,609 samples",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_stage_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 2: usage 구간별 RMSE (4개 변수)
    usub = g[g.group_type == "usage"]
    ymax = float(usub.RMSE.max()) * 1.25
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, v in zip(axes.ravel(), USAGE_VARS):
        s = usub[usub.group_key == f"BIN_{v}"]
        labels = [f"{lv}\n(n={int(s[s.group == lv].n.iloc[0])})" for lv in USAGE_LEVELS]
        grouped_bars(ax, labels, model_values(s, USAGE_LEVELS), ymax=ymax)
        ax.set_title(v, fontsize=9.5, color=INK, loc="left")
        ax.set_ylabel("RMSE of predicted MRR", fontsize=8, color=INK2)
        ax.set_xlabel("Usage level  (split at the 33% and 67% quantiles)", fontsize=8, color=INK2)
    leg = axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle("Robustness by consumable-usage level: Pure ML vs Hybrid",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "fig2_usage_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 3: 시간 구간별 RMSE
    sub = g[g.group_type == "time"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = [f"{lv}\n(n={int(sub[sub.group == lv].n.iloc[0])})" for lv in TIME_LEVELS]
    grouped_bars(ax, labels, model_values(sub, TIME_LEVELS), ymax=6.0)
    for model in MODELS:
        ax.axhline(float(o.loc[model, "RMSE"]), color=MODEL_COLOR[model], lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Time segment  (samples ordered by first recorded timestamp, split into thirds)",
                  fontsize=9, color=INK2)
    ax.set_title("Robustness over process time: Pure ML vs Hybrid\n"
                 "same 5-fold predictions re-aggregated by time segment; no time-based retraining",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_time_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 4: 전체 RMSE vs worst-group RMSE
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    xx = np.arange(2)
    s = summary.set_index("model")
    for i, model in enumerate(MODELS):
        off = (i - 0.5) * 0.36
        vals = [float(s.loc[model, "overall_RMSE"]), float(s.loc[model, "worst_group_RMSE"])]
        ax.bar(xx + off, vals, width=0.34, color=MODEL_COLOR[model], label=model)
        for x, v in zip(xx + off, vals):
            ax.text(x, v + 0.07, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
        ax.text(xx[1] + off, 0.25, s.loc[model, "worst_group"], ha="center", fontsize=7,
                color=SURFACE, rotation=90, va="bottom")
    ax.set_xticks(xx, ["All 1,609 samples", "Worst condition group"])
    style(ax)
    ax.set_ylim(0, max(summary.worst_group_RMSE) * 1.25)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Evaluation scope", fontsize=9, color=INK2)
    ax.set_title("Overall RMSE vs worst condition group: Pure ML vs Hybrid\n"
                 "worst group chosen across stage, usage-level and time-segment groups",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_overall_vs_worst_group.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---- 화면 출력 ---------------------------------------------------
    print("\n[일관성 확인]")
    print(checks.to_string(index=False))
    print("\n[전체 성능]")
    print(overall.to_string(index=False))
    print("\n[Stage 별]")
    print(g[g.group_type == "stage"].drop(columns=["group_key", "group_type"]).to_string(index=False))
    print("\n[usage 구간 경계]")
    print(usage_edges.to_string(index=False))
    print("\n[usage 구간별]")
    print(g[g.group_type == "usage"].drop(columns=["group_type", "group_key"]).to_string(index=False))
    print("\n[시간 구간]")
    print(time_info.to_string(index=False))
    print(g[g.group_type == "time"].drop(columns=["group_type", "group_key"]).to_string(index=False))
    print("\n[robustness 요약]")
    print(summary.to_string(index=False))
    print("\n[worst group 상위 5개]")
    print(worst.to_string(index=False))
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

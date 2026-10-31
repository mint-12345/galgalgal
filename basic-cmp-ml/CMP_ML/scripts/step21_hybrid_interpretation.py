"""Step 21: 대표 Hybrid 모델이 k(x) 예측에 쓰는 공정정보 해석 (Pure ML 과 비교).

- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold.
  training 만 사용한다. test / validation 은 사용하지 않는다. 성능 개선이나 feature 추가는 하지 않는다.
- 대표 Hybrid: MRR = k(x) x PV_proxy, PV_proxy = P9 x V3 (Step 19~20 과 동일)
    k(x) = XGBRegressor(n_estimators=900, max_depth=2, learning_rate=0.1, min_child_weight=1,
                        subsample=0.9, colsample_bytree=0.9, random_state=42)
    Step 20 상위 후보 중 하나이며 1위와 평균 RMSE 가 사실상 같다. 유일한 최적 설정이라는 뜻이 아니다.
    k 는 fold 마다 학습 sample 로만 표준화해 학습하고 예측 후 원래 단위로 되돌린다.
- Pure ML: Step 16 의 Model 1 XGBoost (기본 설정) 로 MRR 을 직접 예측한다. feature 는 같은 31개.
- 기여도: 각 fold 모델의 booster 에서 pred_contribs (SHAP 기여도) 를 평가 fold sample 에만 계산해 5개 fold 를 합친다.
    Hybrid 는 표준화된 k 를 학습하므로 기여도에 그 fold 의 training_std 를 곱해 k 단위로 되돌린다.
    Pure ML 과 비교할 때는 k 기여도에 sample 의 PV_proxy 를 곱해 MRR 단위로 환산한다 (MRR = k x PV).
- 인과가 아니라 모델이 사용한 연관으로 해석한다.
출력: reports/step21_interpretation/ (CSV, 그림)
실행: python -B scripts/step21_hybrid_interpretation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
STEP16_DIR = ROOT / "reports" / "step16_model3_prep"
STEP19_DIR = ROOT / "reports" / "step19_hybrid_pv_comparison"
STEP20_DIR = ROOT / "reports" / "step20_hybrid_xgb_tuning"
OUT_DIR = ROOT / "reports" / "step21_interpretation"
FIG_DIR = OUT_DIR / "figures"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
SEED = 42
N_TOP = 15
HYBRID_PARAMS = dict(n_estimators=900, max_depth=2, learning_rate=0.1, min_child_weight=1,
                     subsample=0.9, colsample_bytree=0.9)
PRESSURES = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
             "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE"]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES1, SERIES2 = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"


def group_of(feature):
    if feature.startswith("STAGE"):
        return "STAGE"
    if feature.startswith("USAGE_"):
        return "usage"
    if feature.startswith("SLURRY_"):
        return "slurry"
    if feature.startswith("DRESSING_WATER"):
        return "dressing water"
    if any(feature.startswith(p) for p in PRESSURES):
        return "pressure"
    if "ROTATION" in feature:
        return "rotation"
    raise ValueError(feature)


def load_filtered_raw():
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


def build(params, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["STAGE"]),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep), ("reg", xgb.XGBRegressor(random_state=SEED, n_jobs=-1, **params))])


def contributions(model, X):
    """fitted pipeline 의 booster 에서 pred_contribs 를 구한다. 마지막 열은 bias."""
    transformed = model[:-1].transform(X)
    # ColumnTransformer 는 이름에 cat__ / num__ 접두사를 붙이므로 원래 열 이름만 남긴다
    names = [n.split("__", 1)[1] if "__" in n else n for n in model[:-1].get_feature_names_out()]
    dm = xgb.DMatrix(transformed, feature_names=[f"f{i}" for i in range(transformed.shape[1])])
    contrib = model.named_steps["reg"].get_booster().predict(dm, pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1], names, transformed


def main():
    raw = load_filtered_raw()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    prep_tbl = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    base = prep_tbl[KEY + ["FOLD", TARGET]].merge(step7.drop(columns=["CHAMBER_PATH", TARGET]), on=KEY,
                                                  how="left", validate="one_to_one")
    base["STAGE"] = base["STAGE"].astype(str)
    assert len(base) == 1609

    # PV_proxy = P9 x V3 (nonzero rows)
    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(base["WAFER_ID"], base["STAGE"]))}
    keys = list(zip(raw["WAFER_ID"].astype(int), raw["STAGE"].astype(str)))
    keep = np.array([k in order for k in keys])
    rows = raw[keep]
    codes = np.array([order[k] for k, ok in zip(keys, keep) if ok], dtype=np.int64)
    p_row = np.mean([rows[c].to_numpy() for c in PRESSURES], axis=0)
    v_row = np.abs(rows["HEAD_ROTATION"].to_numpy())
    mask = (p_row != 0) & (v_row != 0)
    n_used = np.bincount(codes, weights=mask.astype(float), minlength=len(base))
    pv = np.bincount(codes, weights=p_row * v_row * mask, minlength=len(base)) / n_used
    base["PV_PROXY"] = pv
    base["K_TARGET"] = base[TARGET] / pv
    prev = pd.read_csv(STEP19_DIR / "oof_predictions.csv", float_precision="round_trip")
    prev = prev[prev.COMBO == "P9xV3"].set_index(KEY).loc[list(zip(base.WAFER_ID, base.STAGE))]
    assert np.allclose(prev["PV_PROXY"].to_numpy(), pv, rtol=0, atol=0)

    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    assert len(features) == 31
    folds = sorted(base["FOLD"].unique())
    y = base[TARGET].to_numpy()
    k_target = base["K_TARGET"].to_numpy()
    sample_fold = base["FOLD"].to_numpy()

    hybrid_contrib = {}   # feature -> per-sample contribution (k 단위)
    pure_contrib = {}     # feature -> per-sample contribution (MRR 단위)
    k_pred = np.full(len(base), np.nan)
    mrr_pred_pure = np.full(len(base), np.nan)
    feature_names = None
    fold_info = []
    for f in folds:
        is_test = sample_fold == f
        tr = ~is_test
        # Hybrid
        mean_k, std_k = float(k_target[tr].mean()), float(k_target[tr].std(ddof=0))
        hyb = build(HYBRID_PARAMS, numeric).fit(base.loc[tr, features], (k_target[tr] - mean_k) / std_k)
        k_pred[is_test] = hyb.predict(base.loc[is_test, features]) * std_k + mean_k
        c_h, bias_h, names, _ = contributions(hyb, base.loc[is_test, features])
        feature_names = names if feature_names is None else feature_names
        assert names == feature_names
        c_h = c_h * std_k  # 표준화된 k -> 원래 k 단위
        # Pure ML (Step 16 Model1_XGB 와 같은 기본 설정)
        pure = build({}, numeric).fit(base.loc[tr, features], y[tr])
        mrr_pred_pure[is_test] = pure.predict(base.loc[is_test, features])
        c_p, bias_p, names_p, _ = contributions(pure, base.loc[is_test, features])
        assert names_p == feature_names
        idx = np.flatnonzero(is_test)
        for j, name in enumerate(feature_names):
            hybrid_contrib.setdefault(name, np.full(len(base), np.nan))[idx] = c_h[:, j]
            pure_contrib.setdefault(name, np.full(len(base), np.nan))[idx] = c_p[:, j]
        fold_info.append(dict(FOLD=int(f), training_mean_k=mean_k, training_std_k=std_k,
                              n_train=int(tr.sum()), n_test=int(is_test.sum()),
                              hybrid_bias_k=float(np.mean(bias_h) * std_k), pure_bias_mrr=float(np.mean(bias_p))))

    mrr_pred_hybrid = k_pred * pv
    # 이전 단계 결과와 같은지 확인
    s20 = pd.read_csv(STEP20_DIR / "oof_top" / "oof_n900_d2_lr0.1_mcw1.csv", float_precision="round_trip")
    s20 = s20.set_index(KEY).loc[list(zip(base.WAFER_ID, base.STAGE))]
    same_hybrid = np.allclose(s20["MRR_PRED"].to_numpy(), mrr_pred_hybrid, rtol=0, atol=1e-12)
    s16 = pd.read_csv(STEP16_DIR / "oof_predictions.csv", dtype={"STAGE": "string"}, float_precision="round_trip")
    s16 = s16.set_index(KEY).loc[list(zip(base.WAFER_ID, base.STAGE))]
    same_pure = np.allclose(s16["PRED_Model1_XGB"].to_numpy(), mrr_pred_pure, rtol=0, atol=1e-12)
    print(f"대표 Hybrid 예측이 Step 20 저장값과 같은가: {same_hybrid}")
    print(f"Pure ML 예측이 Step 16 Model1_XGB 저장값과 같은가: {same_pure}")
    assert same_hybrid and same_pure
    print("대표 Hybrid 성능:", {k: round(v, 4) for k, v in metrics(y, mrr_pred_hybrid).items()})
    print("Pure ML 성능:", {k: round(v, 4) for k, v in metrics(y, mrr_pred_pure).items()})

    stage = base["STAGE"].to_numpy()
    rows_imp = []
    for name in feature_names:
        h = hybrid_contrib[name]
        p = pure_contrib[name]
        h_mrr = h * pv  # MRR 단위 환산
        rows_imp.append(dict(
            feature=name, group=group_of(name),
            hybrid_mean_abs_contrib_k=float(np.mean(np.abs(h))),
            hybrid_mean_abs_contrib_mrr=float(np.mean(np.abs(h_mrr))),
            hybrid_mean_abs_contrib_k_stage_A=float(np.mean(np.abs(h[stage == "A"]))),
            hybrid_mean_abs_contrib_k_stage_B=float(np.mean(np.abs(h[stage == "B"]))),
            pure_mean_abs_contrib_mrr=float(np.mean(np.abs(p))),
            pure_mean_abs_contrib_mrr_stage_A=float(np.mean(np.abs(p[stage == "A"]))),
            pure_mean_abs_contrib_mrr_stage_B=float(np.mean(np.abs(p[stage == "B"]))),
            spearman_value_vs_hybrid_contrib=float(spearmanr(base[name] if name in base else
                                                             (base["STAGE"] == "B").astype(float), h).statistic),
            spearman_value_vs_pure_contrib=float(spearmanr(base[name] if name in base else
                                                           (base["STAGE"] == "B").astype(float), p).statistic),
        ))
    imp = pd.DataFrame(rows_imp)
    imp["hybrid_rank"] = imp["hybrid_mean_abs_contrib_k"].rank(ascending=False, method="first").astype(int)
    imp["pure_rank"] = imp["pure_mean_abs_contrib_mrr"].rank(ascending=False, method="first").astype(int)
    imp["hybrid_rank_stage_A"] = imp["hybrid_mean_abs_contrib_k_stage_A"].rank(ascending=False, method="first").astype(int)
    imp["hybrid_rank_stage_B"] = imp["hybrid_mean_abs_contrib_k_stage_B"].rank(ascending=False, method="first").astype(int)
    imp["hybrid_share_mrr"] = imp["hybrid_mean_abs_contrib_mrr"] / imp["hybrid_mean_abs_contrib_mrr"].sum()
    imp["pure_share_mrr"] = imp["pure_mean_abs_contrib_mrr"] / imp["pure_mean_abs_contrib_mrr"].sum()
    imp["rank_gap_pure_minus_hybrid"] = imp["pure_rank"] - imp["hybrid_rank"]
    imp = imp.sort_values("hybrid_rank")

    groups = (imp.groupby("group")[["hybrid_mean_abs_contrib_mrr", "pure_mean_abs_contrib_mrr",
                                    "hybrid_mean_abs_contrib_k"]].sum().reset_index())
    groups["hybrid_share"] = groups["hybrid_mean_abs_contrib_mrr"] / groups["hybrid_mean_abs_contrib_mrr"].sum()
    groups["pure_share"] = groups["pure_mean_abs_contrib_mrr"] / groups["pure_mean_abs_contrib_mrr"].sum()
    groups["n_features"] = imp.groupby("group").size().reindex(groups["group"]).to_numpy()
    groups = groups.sort_values("hybrid_share", ascending=False)

    top_features = imp.head(N_TOP)["feature"].tolist()
    long_rows = []
    for name in top_features:
        value = base[name].to_numpy() if name in base else (base["STAGE"] == "B").astype(float).to_numpy()
        long_rows.append(pd.DataFrame(dict(WAFER_ID=base["WAFER_ID"], STAGE=base["STAGE"], FOLD=base["FOLD"],
                                           feature=name, feature_value=value,
                                           hybrid_contrib_k=hybrid_contrib[name],
                                           hybrid_contrib_mrr=hybrid_contrib[name] * pv,
                                           pure_contrib_mrr=pure_contrib[name])))
    contrib_long = pd.concat(long_rows, ignore_index=True)

    direction = []
    for name in top_features:
        value = base[name].to_numpy() if name in base else (base["STAGE"] == "B").astype(float).to_numpy()
        h = hybrid_contrib[name]
        if len(np.unique(value)) == 2:
            # 0/1 값은 중앙값으로 나누면 한쪽이 비므로 값 자체로 나눈다
            lo_v, hi_v = np.unique(value)
            low, high = value == lo_v, value == hi_v
        else:
            med = np.median(value)
            low, high = value <= med, value > med
        direction.append(dict(feature=name, group=group_of(name),
                              spearman=float(spearmanr(value, h).statistic),
                              mean_contrib_low_half_k=float(np.mean(h[low])) if low.any() else np.nan,
                              mean_contrib_high_half_k=float(np.mean(h[high])) if high.any() else np.nan,
                              tendency="feature 값이 클수록 k_pred 를 높이는 경향" if spearmanr(value, h).statistic > 0
                              else "feature 값이 클수록 k_pred 를 낮추는 경향"))
    direction = pd.DataFrame(direction)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    groups.to_csv(OUT_DIR / "group_contribution.csv", index=False)
    direction.to_csv(OUT_DIR / "top_feature_direction.csv", index=False)
    contrib_long.to_csv(OUT_DIR / "top_feature_contributions_long.csv", index=False)
    pd.DataFrame(fold_info).to_csv(OUT_DIR / "fold_scaling_and_bias.csv", index=False)
    pd.DataFrame(dict(WAFER_ID=base["WAFER_ID"], STAGE=base["STAGE"], FOLD=base["FOLD"], PV_PROXY=pv,
                      K_TARGET=k_target, K_PRED=k_pred, AVG_REMOVAL_RATE=y,
                      MRR_PRED_HYBRID=mrr_pred_hybrid, MRR_PRED_PURE=mrr_pred_pure)).to_csv(
        OUT_DIR / "oof_predictions_both_models.csv", index=False)

    # ---------------- 그림 ----------------
    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)

    def bar_plot(df, value_col, title, xlabel, path, color=SERIES1):
        d = df.head(N_TOP).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        bars = ax.barh(d["feature"], d[value_col], color=color, height=0.62)
        style(ax)
        ax.grid(axis="y", visible=False)
        for b, v in zip(bars, d[value_col]):
            ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2, f"{v:.3g}",
                    va="center", fontsize=7.5, color=INK2)
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=12)
        ax.set_xlim(0, d[value_col].max() * 1.15)
        fig.tight_layout()
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    bar_plot(imp, "hybrid_mean_abs_contrib_k",
             "Hybrid k(x) prediction: top 15 features by mean |SHAP contribution|",
             "Mean |contribution| to predicted k  (k units)", FIG_DIR / "fig1_hybrid_k_importance.png")
    pure_sorted = imp.sort_values("pure_rank")
    bar_plot(pure_sorted, "pure_mean_abs_contrib_mrr",
             "Pure ML MRR prediction: top 15 features by mean |SHAP contribution|",
             "Mean |contribution| to predicted MRR", FIG_DIR / "fig2_pure_ml_mrr_importance.png", color=SERIES2)

    g = groups.copy()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    yy = np.arange(len(g))
    ax.barh(yy + 0.19, g["hybrid_share"] * 100, height=0.34, color=SERIES1, label="Hybrid  k(x) contribution")
    ax.barh(yy - 0.19, g["pure_share"] * 100, height=0.34, color=SERIES2, label="Pure ML  MRR contribution")
    ax.set_yticks(yy, g["group"])
    style(ax)
    ax.grid(axis="y", visible=False)
    for i, (h, p) in enumerate(zip(g["hybrid_share"] * 100, g["pure_share"] * 100)):
        ax.text(h + 0.6, i + 0.19, f"{h:.1f}%", va="center", fontsize=7.5, color=INK2)
        ax.text(p + 0.6, i - 0.19, f"{p:.1f}%", va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("Share of total mean |contribution|  (%, MRR units)", fontsize=9, color=INK2)
    ax.set_title("Feature-group contribution share: Hybrid k(x) vs Pure ML", fontsize=11.5, color=INK, loc="left", pad=12)
    ax.set_xlim(0, max(g["hybrid_share"].max(), g["pure_share"].max()) * 130)
    # 범례가 가장 긴 막대(usage)의 값 라벨과 겹치지 않도록 위쪽 빈 공간에 둔다
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_group_share_comparison.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    show = top_features[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes.ravel(), show):
        d = contrib_long[contrib_long.feature == name]
        ax.scatter(d["feature_value"], d["hybrid_contrib_k"], s=9, color=SERIES1, alpha=0.55,
                   edgecolors=SURFACE, linewidths=0.4)
        ax.axhline(0, color=AXIS, lw=0.8)
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        rho = direction.loc[direction.feature == name, "spearman"].iloc[0]
        ax.set_title(f"{name}\nSpearman(value, contribution) = {rho:+.2f}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Feature value", fontsize=8, color=INK2)
        ax.set_ylabel("Contribution to k (k units)", fontsize=8, color=INK2)
    fig.suptitle("Hybrid k(x): feature value vs SHAP contribution (association, not causation)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig4_value_vs_contribution.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Hybrid k(x) 상위 {N_TOP} feature]")
    print(imp.head(N_TOP)[["hybrid_rank", "feature", "group", "hybrid_mean_abs_contrib_k",
                           "hybrid_mean_abs_contrib_mrr", "hybrid_rank_stage_A", "hybrid_rank_stage_B",
                           "pure_rank"]].to_string(index=False))
    print(f"\n[Pure ML 상위 {N_TOP} feature]")
    print(pure_sorted.head(N_TOP)[["pure_rank", "feature", "group", "pure_mean_abs_contrib_mrr",
                                   "hybrid_rank"]].to_string(index=False))
    print("\n[그룹별 기여 비중]")
    print(groups[["group", "n_features", "hybrid_share", "pure_share", "hybrid_mean_abs_contrib_mrr",
                  "pure_mean_abs_contrib_mrr"]].round(4).to_string(index=False))
    print("\n[상위 feature 영향 방향]")
    print(direction.round(4).to_string(index=False))
    print("\n[공통 / 한쪽에서만 중요한 feature]")
    both = imp[(imp.hybrid_rank <= N_TOP) & (imp.pure_rank <= N_TOP)]["feature"].tolist()
    only_h = imp[(imp.hybrid_rank <= N_TOP) & (imp.pure_rank > N_TOP)]["feature"].tolist()
    only_p = imp[(imp.pure_rank <= N_TOP) & (imp.hybrid_rank > N_TOP)]["feature"].tolist()
    print("  공통:", both)
    print("  Hybrid 에서만:", only_h)
    print("  Pure ML 에서만:", only_p)
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

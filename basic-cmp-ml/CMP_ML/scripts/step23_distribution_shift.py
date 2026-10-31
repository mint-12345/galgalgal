"""Step 23: 학습 조건과 평가 조건을 실제로 분리한 distribution-shift stress test.

- Step 22 와의 차이: Step 22 는 기존 random 5-fold 의 out-of-fold 예측을 조건별로 나눈
  within-distribution subgroup 분석이다. Step 23 은 과거 -> 미래, 낮은 usage -> 높은 usage 로
  학습/평가를 실제로 분리해 새로 학습한다.
- 대상: CHAMBER_PATH = 4-5-6 의 training 1,609개 sample. test / validation 은 사용하지 않는다.
- 모델 구조, feature, P/V, hyperparameter 를 바꾸거나 튜닝하지 않는다.
    Pure ML : Step 16 의 Model 1 XGBoost = XGBRegressor(random_state=42, n_jobs=-1) 기본 설정
    Hybrid  : Step 20~22 의 대표 Hybrid. MRR = k(x) x PV_proxy, PV_proxy = P9 x V3
              k(x) = XGBRegressor(n_estimators=900, max_depth=2, learning_rate=0.1,
                                  min_child_weight=1, subsample=0.9, colsample_bytree=0.9,
                                  random_state=42, n_jobs=-1), feature 는 C_full 31개
              k_target 표준화는 학습 데이터의 평균/표준편차만 사용한다.
- 모든 split 은 wafer 단위다. 같은 WAFER_ID 가 학습과 평가에 동시에 들어가지 않는다.
- 평가 데이터의 실제 MRR / k_target 으로 예측을 보정하거나 clipping 하지 않는다.
출력: reports/step23_distribution_shift/ (CSV, 그림)
실행: python -B scripts/step23_distribution_shift.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 250)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
TIME_FILE = ROOT / "data" / "processed" / "step22" / "step22_first_timestamp.csv"
STEP20_OOF = ROOT / "reports" / "step20_hybrid_xgb_tuning" / "oof_top" / "oof_n900_d2_lr0.1_mcw1.csv"
STEP22_DIR = ROOT / "reports" / "step22_robustness"
OUT_DIR = ROOT / "reports" / "step23_distribution_shift"
FIG_DIR = OUT_DIR / "figures"

KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
SEED = 42
N_BLOCKS = 5
TRAIN_WAFER_FRACTION = 2 / 3
HYBRID_PARAMS = dict(n_estimators=900, max_depth=2, learning_rate=0.1, min_child_weight=1,
                     subsample=0.9, colsample_bytree=0.9)
USAGE_VARS = ["USAGE_OF_DRESSER_MEAN", "USAGE_OF_POLISHING_TABLE_MEAN",
              "USAGE_OF_DRESSER_TABLE_MEAN", "USAGE_OF_BACKING_FILM_MEAN"]
MODELS = ["Pure ML", "Hybrid"]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES1, SERIES2 = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"
MODEL_COLOR = {"Pure ML": SERIES2, "Hybrid": SERIES1}


# ---------------------------------------------------------------- 모델 (Step 16 / 20~21 과 동일)
def build(params, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["STAGE"]),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep), ("reg", xgb.XGBRegressor(random_state=SEED, n_jobs=-1, **params))])


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def fit_predict(train, evaluate, features, numeric):
    """두 모델을 학습 데이터에만 맞추고 평가 데이터를 예측한다."""
    pure = build({}, numeric).fit(train[features], train[TARGET])
    pred_pure = pure.predict(evaluate[features])

    mean_k, std_k = train["K_TARGET"].mean(), train["K_TARGET"].std(ddof=0)
    scaled = (train["K_TARGET"] - mean_k) / std_k
    hyb = build(HYBRID_PARAMS, numeric).fit(train[features], scaled)
    k_pred = hyb.predict(evaluate[features]) * std_k + mean_k
    pred_hyb = k_pred * evaluate["PV_PROXY"].to_numpy()
    return pred_pure, pred_hyb, k_pred, float(mean_k), float(std_k)


def stage_mix(df, part):
    n = len(df)
    a = int((df["STAGE"] == "A").sum())
    return {f"{part}_n": n, f"{part}_stage_A": a, f"{part}_stage_B": n - a,
            f"{part}_stage_B_ratio": (n - a) / n}


# ---------------------------------------------------------------- 실행
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                        float_precision="round_trip")
    hyb_oof = pd.read_csv(STEP20_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                          float_precision="round_trip")
    tstamp = pd.read_csv(TIME_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                         float_precision="round_trip")
    for d in (step7, hyb_oof, tstamp):
        d["STAGE"] = d["STAGE"].astype(str)

    base = hyb_oof[KEY + ["PV_PROXY", "K_TARGET", TARGET]].merge(
        step7.drop(columns=["CHAMBER_PATH", TARGET]), on=KEY, how="left", validate="one_to_one")
    base = base.merge(tstamp, on=KEY, how="left", validate="one_to_one")
    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    assert len(base) == 1609 and len(features) == 31 and base.notna().all().all()

    # 각 wafer 의 가장 이른 기록 시각 (여기서 쓰는 4-5-6 sample 기준)
    wafer = base.groupby("WAFER_ID", as_index=False).agg(
        FIRST_TIMESTAMP=("FIRST_TIMESTAMP", "min"), n_samples=("WAFER_ID", "size"))
    wafer = wafer.sort_values(["FIRST_TIMESTAMP", "WAFER_ID"], kind="stable").reset_index(drop=True)
    blocks = np.array_split(np.arange(len(wafer)), N_BLOCKS)
    wafer["BLOCK"] = 0
    for i, idx in enumerate(blocks, start=1):
        wafer.loc[idx, "BLOCK"] = i
    base = base.merge(wafer[["WAFER_ID", "BLOCK"]], on="WAFER_ID", how="left", validate="many_to_one")

    block_info = []
    for b in range(1, N_BLOCKS + 1):
        w = wafer[wafer.BLOCK == b]
        s = base[base.BLOCK == b]
        block_info.append(dict(block=b, n_wafers=len(w), n_samples=len(s),
                               first_timestamp_min=float(w.FIRST_TIMESTAMP.min()),
                               first_timestamp_max=float(w.FIRST_TIMESTAMP.max()),
                               **stage_mix(s, "block")))
    block_info = pd.DataFrame(block_info)

    rows, stage_rows, pred_rows, stagewise, overlap_counts, dist_rows = [], [], [], [], [], []

    def run_split(test_id, kind, description, train, evaluate, extra=None):
        # 같은 wafer 가 학습과 평가에 동시에 들어가지 않는지 실제로 세어서 확인한다
        overlap_counts.append(len(set(train.WAFER_ID) & set(evaluate.WAFER_ID)))
        assert overlap_counts[-1] == 0
        pred_pure, pred_hyb, k_pred, mean_k, std_k = fit_predict(train, evaluate, features, numeric)
        y = evaluate[TARGET].to_numpy()
        m = {"Pure ML": metrics(y, pred_pure), "Hybrid": metrics(y, pred_hyb)}
        for model in MODELS:
            rows.append(dict(test_id=test_id, kind=kind, description=description, model=model,
                             n_train_wafers=train.WAFER_ID.nunique(), n_train_samples=len(train),
                             n_eval_wafers=evaluate.WAFER_ID.nunique(), n_eval_samples=len(evaluate),
                             **m[model], **(extra or {})))
        stage_rows.append(dict(test_id=test_id, kind=kind, description=description,
                               **stage_mix(train, "train"), **stage_mix(evaluate, "eval"),
                               stage_B_ratio_gap=stage_mix(evaluate, "eval")["eval_stage_B_ratio"] -
                               stage_mix(train, "train")["train_stage_B_ratio"]))
        # 학습 구간과 평가 구간의 정답 분포가 얼마나 다른지 기록한다 (모델과 무관한 기술 통계)
        dist_rows.append(dict(
            test_id=test_id, kind=kind, description=description,
            train_MRR_mean=float(train[TARGET].mean()), train_MRR_std=float(train[TARGET].std(ddof=0)),
            eval_MRR_mean=float(evaluate[TARGET].mean()), eval_MRR_std=float(evaluate[TARGET].std(ddof=0)),
            MRR_mean_gap=float(evaluate[TARGET].mean() - train[TARGET].mean()),
            eval_MRR_min=float(evaluate[TARGET].min()), eval_MRR_max=float(evaluate[TARGET].max()),
            train_MRR_min=float(train[TARGET].min()), train_MRR_max=float(train[TARGET].max()),
            train_k_mean=mean_k, train_k_std=std_k,
            eval_k_mean=float(evaluate["K_TARGET"].mean()), eval_k_std=float(evaluate["K_TARGET"].std(ddof=0)),
            train_PV_mean=float(train["PV_PROXY"].mean()), eval_PV_mean=float(evaluate["PV_PROXY"].mean())))
        out = evaluate[KEY + ["BLOCK", TARGET, "PV_PROXY", "K_TARGET"]].copy()
        out.insert(0, "test_id", test_id)
        out["K_PRED_HYBRID"] = k_pred
        out["MRR_PRED_PURE"] = pred_pure
        out["MRR_PRED_HYBRID"] = pred_hyb
        out["train_k_mean"], out["train_k_std"] = mean_k, std_k
        pred_rows.append(out)
        for st in ("A", "B"):
            sel = (evaluate["STAGE"] == st).to_numpy()
            if sel.sum() > 1:
                for model, p in (("Pure ML", pred_pure), ("Hybrid", pred_hyb)):
                    stagewise.append(dict(test_id=test_id, kind=kind, STAGE=st, model=model,
                                          n=int(sel.sum()), **metrics(y[sel], p[sel])))

    # ---- 2. 시간 변화 stress test (과거 -> 미래) ----------------------
    for i, k in enumerate((2, 3, 4), start=1):
        train = base[base.BLOCK <= k]
        evaluate = base[base.BLOCK == k + 1]
        run_split(f"T{i}", "temporal", f"block 1-{k} 학습 → block {k + 1} 평가", train, evaluate,
                  extra=dict(train_blocks=f"1-{k}", eval_block=k + 1,
                             train_time_max=float(train.FIRST_TIMESTAMP.max()),
                             eval_time_min=float(evaluate.FIRST_TIMESTAMP.min())))

    # ---- 3. usage 변화 stress test (낮은 usage -> 높은 usage) ---------
    usage_info = []
    for j, v in enumerate(USAGE_VARS, start=1):
        rep = base.groupby("WAFER_ID", as_index=False)[v].mean().rename(columns={v: "USAGE_REP"})
        rep = rep.sort_values(["USAGE_REP", "WAFER_ID"], kind="stable").reset_index(drop=True)
        n_train = int(round(len(rep) * TRAIN_WAFER_FRACTION))
        train_w = set(rep.WAFER_ID[:n_train])
        eval_w = set(rep.WAFER_ID[n_train:])
        train = base[base.WAFER_ID.isin(train_w)]
        evaluate = base[base.WAFER_ID.isin(eval_w)]
        boundary = float(rep.USAGE_REP.iloc[n_train - 1]), float(rep.USAGE_REP.iloc[n_train])
        run_split(f"U{j}", "usage", f"{v} 하위 wafer 학습 → 상위 wafer 평가", train, evaluate,
                  extra=dict(usage_variable=v))
        usage_info.append(dict(
            test_id=f"U{j}", usage_variable=v, n_wafers=len(rep), n_train_wafers=n_train,
            n_eval_wafers=len(rep) - n_train, n_train_samples=len(train), n_eval_samples=len(evaluate),
            train_wafer_usage_min=float(rep.USAGE_REP[:n_train].min()),
            train_wafer_usage_max=boundary[0], eval_wafer_usage_min=boundary[1],
            eval_wafer_usage_max=float(rep.USAGE_REP[n_train:].max()),
            train_sample_usage_min=float(train[v].min()), train_sample_usage_max=float(train[v].max()),
            eval_sample_usage_min=float(evaluate[v].min()), eval_sample_usage_max=float(evaluate[v].max())))
    usage_info = pd.DataFrame(usage_info)

    results = pd.DataFrame(rows)
    stage_comp = pd.DataFrame(stage_rows)
    dist_info = pd.DataFrame(dist_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    stagewise = pd.DataFrame(stagewise)

    # ---- 모델 비교 (wide) --------------------------------------------
    wide = results.pivot_table(index=["test_id", "kind", "description", "n_train_samples",
                                      "n_eval_wafers", "n_eval_samples"],
                               columns="model", values=["R2", "MAE", "RMSE"]).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1].replace(' ', '')}" for c in wide.columns]
    wide["RMSE_diff_Hybrid_minus_PureML"] = wide["RMSE_Hybrid"] - wide["RMSE_PureML"]
    wide["MAE_diff_Hybrid_minus_PureML"] = wide["MAE_Hybrid"] - wide["MAE_PureML"]
    wide["hybrid_lower_RMSE"] = wide["RMSE_diff_Hybrid_minus_PureML"] < 0
    wide = wide.sort_values("test_id").reset_index(drop=True)

    # ---- random 5-fold (Step 22) 와 비교 ------------------------------
    random_ref = pd.read_csv(STEP22_DIR / "overall_metrics.csv", float_precision="round_trip")
    ref = random_ref.set_index("model")
    comp = [dict(scenario="Random 5-fold (Step 22)", kind="within-distribution", test_id="CV",
                 n_eval_samples=1609,
                 RMSE_PureML=float(ref.loc["Pure ML", "RMSE"]), RMSE_Hybrid=float(ref.loc["Hybrid", "RMSE"]),
                 RMSE_diff_Hybrid_minus_PureML=float(ref.loc["Hybrid", "RMSE"] - ref.loc["Pure ML", "RMSE"]))]
    for _, r in wide.iterrows():
        comp.append(dict(scenario=r.description, kind=f"shift ({r.kind})", test_id=r.test_id,
                         n_eval_samples=int(r.n_eval_samples), RMSE_PureML=r.RMSE_PureML,
                         RMSE_Hybrid=r.RMSE_Hybrid,
                         RMSE_diff_Hybrid_minus_PureML=r.RMSE_diff_Hybrid_minus_PureML))
    comp = pd.DataFrame(comp)
    for m in ("PureML", "Hybrid"):
        comp[f"RMSE_{m}_minus_random"] = comp[f"RMSE_{m}"] - comp.loc[0, f"RMSE_{m}"]

    # ---- 요약 --------------------------------------------------------
    t = wide[wide.kind == "temporal"]
    u = wide[wide.kind == "usage"]
    summary = pd.DataFrame([
        dict(item="temporal test 중 Hybrid 의 RMSE 가 더 낮은 경우",
             value=f"{int(t.hybrid_lower_RMSE.sum())} / {len(t)}"),
        dict(item="usage test 중 Hybrid 의 RMSE 가 더 낮은 경우",
             value=f"{int(u.hybrid_lower_RMSE.sum())} / {len(u)}"),
        dict(item="전체 shift test 중 Hybrid 의 RMSE 가 더 낮은 경우",
             value=f"{int(wide.hybrid_lower_RMSE.sum())} / {len(wide)}"),
        dict(item="Pure ML 의 shift RMSE 평균", value=f"{wide.RMSE_PureML.mean():.4f}"),
        dict(item="Hybrid 의 shift RMSE 평균", value=f"{wide.RMSE_Hybrid.mean():.4f}"),
        dict(item="Pure ML 의 shift RMSE 최대", value=f"{wide.RMSE_PureML.max():.4f}"),
        dict(item="Hybrid 의 shift RMSE 최대", value=f"{wide.RMSE_Hybrid.max():.4f}"),
        dict(item="Pure ML 의 random 5-fold 대비 평균 증가",
             value=f"{(wide.RMSE_PureML - comp.loc[0, 'RMSE_PureML']).mean():+.4f}"),
        dict(item="Hybrid 의 random 5-fold 대비 평균 증가",
             value=f"{(wide.RMSE_Hybrid - comp.loc[0, 'RMSE_Hybrid']).mean():+.4f}"),
        dict(item="RMSE 차이(Hybrid - Pure ML) 최소 / 최대",
             value=f"{wide.RMSE_diff_Hybrid_minus_PureML.min():+.4f} / "
                   f"{wide.RMSE_diff_Hybrid_minus_PureML.max():+.4f}"),
        dict(item="Stage B 비율 차이(평가 - 학습)의 최대 절대값",
             value=f"{stage_comp.stage_B_ratio_gap.abs().max():.4f}"),
    ])

    # ---- 일관성 확인 -------------------------------------------------
    s20 = hyb_oof.set_index(KEY).loc[list(zip(base.WAFER_ID, base.STAGE))]
    checks = pd.DataFrame([
        dict(item="sample 수가 1,609개인가", value=str(len(base) == 1609)),
        dict(item="feature 가 31개인가", value=str(len(features) == 31)),
        dict(item="PV_PROXY 가 Step 20 저장값과 같은가",
             value=str(bool(np.array_equal(base.PV_PROXY.to_numpy(), s20.PV_PROXY.to_numpy())))),
        dict(item="k_target = MRR / PV_proxy 인가",
             value=str(bool(np.allclose(base.K_TARGET, base[TARGET] / base.PV_PROXY, rtol=0, atol=1e-12)))),
        dict(item="모든 split 에서 학습과 평가의 wafer 가 겹치지 않는가", value=str(max(overlap_counts) == 0)),
        dict(item="시간 block 의 sample 합이 1,609 인가",
             value=str(int(block_info.n_samples.sum()) == 1609)),
        dict(item="시간 block 이 시간순으로 겹치지 않는가",
             value=str(bool((block_info.first_timestamp_min.values[1:] >
                             block_info.first_timestamp_max.values[:-1]).all()))),
        dict(item="usage test 에서 평가 wafer 의 usage 가 학습 wafer 보다 높은가",
             value=str(bool((usage_info.eval_wafer_usage_min > usage_info.train_wafer_usage_max).all()))),
        dict(item="Hybrid 예측 = k_pred x PV_proxy 인가",
             value=str(bool(np.allclose(predictions.MRR_PRED_HYBRID,
                                        predictions.K_PRED_HYBRID * predictions.PV_PROXY,
                                        rtol=0, atol=1e-12)))),
    ])

    # ---- 저장 -------------------------------------------------------
    block_info.to_csv(OUT_DIR / "temporal_block_info.csv", index=False)
    results[results.kind == "temporal"].to_csv(OUT_DIR / "temporal_test_results.csv", index=False)
    results[results.kind == "usage"].to_csv(OUT_DIR / "usage_test_results.csv", index=False)
    usage_info.to_csv(OUT_DIR / "usage_split_info.csv", index=False)
    wide.to_csv(OUT_DIR / "model_comparison_by_test.csv", index=False)
    stage_comp.to_csv(OUT_DIR / "stage_composition.csv", index=False)
    dist_info.to_csv(OUT_DIR / "split_distribution_info.csv", index=False)
    stagewise.to_csv(OUT_DIR / "stagewise_eval_metrics.csv", index=False)
    comp.to_csv(OUT_DIR / "shift_vs_random_comparison.csv", index=False)
    predictions.to_csv(OUT_DIR / "eval_predictions.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_counts.csv", index=False)
    checks.to_csv(OUT_DIR / "consistency_checks.csv", index=False)

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

    def bars(ax, labels, sub, ymax):
        xx = np.arange(len(labels))
        for i, model in enumerate(MODELS):
            off = (i - 0.5) * 0.36
            vals = sub[f"RMSE_{model.replace(' ', '')}"].to_numpy()
            ax.bar(xx + off, vals, width=0.34, color=MODEL_COLOR[model], label=model)
            for x, v in zip(xx + off, vals):
                ax.text(x, v + ymax * 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2,
                        zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        ax.set_xticks(xx, labels)
        style(ax)
        ax.set_ylim(0, ymax)

    ymax = max(wide.RMSE_PureML.max(), wide.RMSE_Hybrid.max()) * 1.3

    # 그림 1: 시간 forward test
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = [f"Test {i}\ntrain blocks 1-{k}\n→ block {k + 1}  (n={int(t.n_eval_samples.iloc[i - 1])})"
              for i, k in enumerate((2, 3, 4), start=1)]
    bars(ax, labels, t, ymax)
    ax.axhline(comp.loc[0, "RMSE_PureML"], color=MODEL_COLOR["Pure ML"], lw=1.0, ls="--", alpha=0.7)
    ax.axhline(comp.loc[0, "RMSE_Hybrid"], color=MODEL_COLOR["Hybrid"], lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Temporal forward test  (train on past wafers, evaluate on later wafers)",
                  fontsize=9, color=INK2)
    ax.set_title("Temporal forward stress test: Pure ML vs Hybrid\n"
                 "dashed line = each model's random 5-fold RMSE on all 1,609 samples (Step 22)",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for x in leg.get_texts():
        x.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_temporal_forward_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 2: usage stress test
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = [v.replace("USAGE_OF_", "").replace("_MEAN", "").replace("_", " ").lower() +
              f"\n(n={int(u.n_eval_samples.iloc[i])})" for i, v in enumerate(USAGE_VARS)]
    bars(ax, labels, u, ymax)
    ax.axhline(comp.loc[0, "RMSE_PureML"], color=MODEL_COLOR["Pure ML"], lw=1.0, ls="--", alpha=0.7)
    ax.axhline(comp.loc[0, "RMSE_Hybrid"], color=MODEL_COLOR["Hybrid"], lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Usage variable used to split wafers  (train on the lower 2/3, evaluate on the upper 1/3)",
                  fontsize=9, color=INK2)
    ax.set_title("Usage-shift stress test: Pure ML vs Hybrid\n"
                 "dashed line = each model's random 5-fold RMSE on all 1,609 samples (Step 22)",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for x in leg.get_texts():
        x.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_usage_shift_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 3: random 5-fold vs distribution shift
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    short = ["Random\n5-fold"] + [f"{r.test_id}\n{'time' if 'temporal' in r.kind else 'usage'}"
                                  for _, r in comp.iloc[1:].iterrows()]
    bars(ax, short, comp, ymax)
    ax.axvline(0.5, color=AXIS, lw=0.8, ls=":")
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Left of the dotted line: within-distribution 5-fold (Step 22).  "
                  "Right: distribution-shift tests (Step 23)", fontsize=9, color=INK2)
    ax.set_title("Within-distribution vs distribution-shift RMSE: Pure ML vs Hybrid\n"
                 "T1-T3 = temporal forward tests, U1-U4 = usage-shift tests; evaluation sets differ in size",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for x in leg.get_texts():
        x.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_random_vs_shift_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---- 화면 출력 ---------------------------------------------------
    print("[일관성 확인]")
    print(checks.to_string(index=False))
    print("\n[시간 block]")
    print(block_info.to_string(index=False))
    print("\n[시간 forward test]")
    print(t.to_string(index=False))
    print("\n[usage split]")
    print(usage_info.to_string(index=False))
    print("\n[usage stress test]")
    print(u.to_string(index=False))
    print("\n[Stage 구성]")
    print(stage_comp.to_string(index=False))
    print("\n[random 5-fold 와 비교]")
    print(comp.to_string(index=False))
    print("\n[Stage 별 평가 성능]")
    print(stagewise.to_string(index=False))
    print("\n[요약]")
    print(summary.to_string(index=False))
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

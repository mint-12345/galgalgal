"""Step 25: tuning 이 끝난 최종 후보끼리 robustness 를 다시 공정하게 비교한다.

- Step 22 / 23 의 비교는 기본 설정 Pure ML 을 썼다. 여기서는 Step 24 에서 tuning 한 Pure ML 로 다시 본다.
- 새 feature, P/V 변경, hyperparameter tuning 을 하지 않는다. 두 설정 모두 이미 확정된 값이다.
    Pure ML : XGBRegressor(n_estimators=900, max_depth=4, learning_rate=0.05, min_child_weight=3,
                           subsample=0.9, colsample_bytree=0.9, random_state=42)  (Step 24 1위)
    Hybrid  : MRR = k(x) x PV_proxy, PV_proxy = P9 x V3, k(x) = XGBRegressor(
              n_estimators=900, max_depth=2, learning_rate=0.1, min_child_weight=1,
              subsample=0.9, colsample_bytree=0.9, random_state=42)  (Step 20~23 대표)
              k_target 표준화는 학습 데이터 통계만 사용한다.
- 대상: CHAMBER_PATH = 4-5-6 의 training 1,609개. test / validation 은 사용하지 않는다.
- 5-fold 성능과 Step 22 방식 그룹 분석은 이미 저장된 out-of-fold 예측을 재집계한다 (재학습 없음).
- Step 23 방식 shift 시험만 확정된 두 설정으로 split 마다 새로 학습한다.
  split 정의는 Step 23 과 같은 규칙으로 다시 만들고, 평가 sample 이 Step 23 과 같은지 대조한다.
- 평가 데이터의 실제 값으로 예측을 보정하거나 clipping 하지 않는다.
출력: reports/step25_final_robustness/ (CSV, 그림)
실행: python -B scripts/step25_final_robustness.py
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
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
TIME_FILE = ROOT / "data" / "processed" / "step22" / "step22_first_timestamp.csv"
STEP20_OOF = ROOT / "reports" / "step20_hybrid_xgb_tuning" / "oof_top" / "oof_n900_d2_lr0.1_mcw1.csv"
STEP20_SUM = ROOT / "reports" / "step20_hybrid_xgb_tuning" / "summary_long.csv"
STEP22_DIR = ROOT / "reports" / "step22_robustness"
STEP23_DIR = ROOT / "reports" / "step23_distribution_shift"
STEP24_DIR = ROOT / "reports" / "step24_pure_ml_tuning"
PURE_CONFIG = "n900_d4_lr0.05_mcw3"
HYBRID_CONFIG = "n900_d2_lr0.1_mcw1"
STEP24_OOF = STEP24_DIR / "oof_top" / f"oof_{PURE_CONFIG}.csv"
OUT_DIR = ROOT / "reports" / "step25_final_robustness"
FIG_DIR = OUT_DIR / "figures"

KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
SEED = 42
N_BLOCKS = 5
TRAIN_WAFER_FRACTION = 2 / 3
FIXED = dict(subsample=0.9, colsample_bytree=0.9)
PURE_PARAMS = dict(n_estimators=900, max_depth=4, learning_rate=0.05, min_child_weight=3)
HYBRID_PARAMS = dict(n_estimators=900, max_depth=2, learning_rate=0.1, min_child_weight=1)
USAGE_VARS = ["USAGE_OF_DRESSER_MEAN", "USAGE_OF_POLISHING_TABLE_MEAN",
              "USAGE_OF_DRESSER_TABLE_MEAN", "USAGE_OF_BACKING_FILM_MEAN"]
USAGE_LEVELS = ["low", "mid", "high"]
TIME_LEVELS = ["early", "middle", "late"]
MODELS = {"Pure ML": "MRR_PRED_PURE", "Hybrid": "MRR_PRED_HYBRID"}

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES1, SERIES2 = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"
MODEL_COLOR = {"Pure ML": SERIES2, "Hybrid": SERIES1}


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))))


def build(params, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["STAGE"]),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep),
                     ("reg", XGBRegressor(random_state=SEED, n_jobs=-1, **FIXED, **params))])


def stage_mix(df, part):
    n = len(df)
    a = int((df["STAGE"] == "A").sum())
    return {f"{part}_n": n, f"{part}_stage_B_ratio": (n - a) / n}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 입력 ----------------
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                        float_precision="round_trip")
    prep = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                       float_precision="round_trip")
    hyb_oof = pd.read_csv(STEP20_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                          float_precision="round_trip")
    pure_oof = pd.read_csv(STEP24_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                           float_precision="round_trip")
    groups22 = pd.read_csv(STEP22_DIR / "sample_group_assignment.csv",
                           dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    tstamp = pd.read_csv(TIME_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                         float_precision="round_trip")
    for d in (step7, prep, hyb_oof, pure_oof, groups22, tstamp):
        d["STAGE"] = d["STAGE"].astype(str)

    base = prep[KEY + ["FOLD", TARGET]].merge(step7.drop(columns=["CHAMBER_PATH", TARGET]), on=KEY,
                                              how="left", validate="one_to_one")
    base = base.merge(hyb_oof[KEY + ["PV_PROXY", "K_TARGET"]], on=KEY, how="left", validate="one_to_one")
    base = base.merge(tstamp, on=KEY, how="left", validate="one_to_one")
    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    assert len(base) == 1609 and len(features) == 31 and base.notna().all().all()

    # 5-fold out-of-fold 예측 (저장된 값 재사용)
    oof = base[KEY + ["FOLD", TARGET, "PV_PROXY"]].merge(
        pure_oof[KEY + ["MRR_PRED"]].rename(columns={"MRR_PRED": "MRR_PRED_PURE"}), on=KEY,
        how="left", validate="one_to_one")
    oof = oof.merge(hyb_oof[KEY + ["MRR_PRED", "K_PRED"]].rename(
        columns={"MRR_PRED": "MRR_PRED_HYBRID", "K_PRED": "K_PRED_HYBRID"}), on=KEY,
        how="left", validate="one_to_one")
    # Step 22 와 완전히 같은 구간 배정을 그대로 가져온다
    gcols = ["TIME_BIN"] + [f"BIN_{v}" for v in USAGE_VARS]
    oof = oof.merge(groups22[KEY + gcols + ["FIRST_TIMESTAMP"]], on=KEY, how="left", validate="one_to_one")
    assert oof.notna().all().all()

    checks = []

    # ---------------- 3. 기본 5-fold 성능 ----------------
    overall_rows, fold_rows = [], []
    for model, col in MODELS.items():
        overall_rows.append(dict(model=model, scope="pooled_out_of_fold", n=len(oof),
                                 **metrics(oof[TARGET], oof[col])))
        for f, g in oof.groupby("FOLD", sort=True):
            fold_rows.append(dict(model=model, FOLD=int(f), n=len(g), **metrics(g[TARGET], g[col])))
    overall = pd.DataFrame(overall_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_summary = (fold_metrics.groupby("model", sort=False)[["R2", "MAE", "RMSE"]]
                    .agg(["mean", "std"]))
    fold_summary.columns = [f"{m}_{s.replace('std', 'std_ddof1')}" for m, s in fold_summary.columns]
    fold_summary = fold_summary.reset_index()

    s24 = pd.read_csv(STEP24_DIR / "config_results.csv", float_precision="round_trip")
    s24 = s24[s24.config_id == PURE_CONFIG].iloc[0]
    s20 = pd.read_csv(STEP20_SUM, float_precision="round_trip")
    s20_rmse = float(s20[(s20.config_id == HYBRID_CONFIG) & (s20.metric == "mrr_RMSE")]["mean"].iloc[0])
    s22_overall = pd.read_csv(STEP22_DIR / "overall_metrics.csv", float_precision="round_trip")
    fs = fold_summary.set_index("model")
    checks += [
        dict(item="sample 수가 1,609개인가", value=str(len(oof) == 1609)),
        dict(item="Pure ML 5-fold 평균 RMSE 가 Step 24 저장값과 같은가",
             value=str(bool(np.isclose(fs.loc["Pure ML", "RMSE_mean"], s24.RMSE_mean, rtol=0, atol=1e-12)))),
        dict(item="Hybrid 5-fold 평균 RMSE 가 Step 20 저장값과 같은가",
             value=str(bool(np.isclose(fs.loc["Hybrid", "RMSE_mean"], s20_rmse, rtol=0, atol=1e-12)))),
        dict(item="Hybrid pooled RMSE 가 Step 22 저장값과 같은가",
             value=str(bool(np.isclose(
                 float(overall[overall.model == "Hybrid"].RMSE.iloc[0]),
                 float(s22_overall[s22_overall.model == "Hybrid"].RMSE.iloc[0]), rtol=0, atol=1e-12)))),
        dict(item="Hybrid 예측 = k_pred x PV_proxy 인가",
             value=str(bool(np.allclose(oof.MRR_PRED_HYBRID, oof.K_PRED_HYBRID * oof.PV_PROXY,
                                        rtol=0, atol=1e-12)))),
    ]

    # ---------------- 4. Step 22 방식 그룹 분석 ----------------
    o = overall.set_index("model")
    rows = []

    def add_groups(group_type, col, order, label_prefix=None):
        for level in order:
            sub = oof[oof[col] == level]
            label = f"{label_prefix} = {level}" if label_prefix else f"{group_type} = {level}"
            for model, pcol in MODELS.items():
                m = metrics(sub[TARGET], sub[pcol])
                rows.append(dict(group_type=group_type, group_label=label, group=level, model=model,
                                 n=len(sub), **m,
                                 overall_RMSE=float(o.loc[model, "RMSE"]),
                                 delta_RMSE=m["RMSE"] - float(o.loc[model, "RMSE"])))

    add_groups("stage", "STAGE", ["A", "B"])
    for v in USAGE_VARS:
        add_groups("usage", f"BIN_{v}", USAGE_LEVELS, label_prefix=v)
    add_groups("time", "TIME_BIN", TIME_LEVELS)
    gm = pd.DataFrame(rows)
    assert gm.group_label.nunique() == 17

    gwide = gm.pivot_table(index=["group_type", "group_label", "n"], columns="model",
                           values=["R2", "MAE", "RMSE", "delta_RMSE"]).reset_index()
    gwide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1].replace(' ', '')}" for c in gwide.columns]
    gwide["RMSE_diff_Hybrid_minus_PureML"] = gwide.RMSE_Hybrid - gwide.RMSE_PureML
    gwide["lower"] = np.where(gwide.RMSE_diff_Hybrid_minus_PureML < 0, "Hybrid", "Pure ML")
    order_map = {lab: i for i, lab in enumerate(gm.group_label.unique())}
    gwide = gwide.sort_values("group_label", key=lambda s: s.map(order_map)).reset_index(drop=True)

    group_summary = []
    for model in MODELS:
        d = gm[gm.model == model]
        worst = d.loc[d.RMSE.idxmax()]
        group_summary.append(dict(
            model=model, n_groups=len(d), overall_RMSE=float(o.loc[model, "RMSE"]),
            min_group_RMSE=float(d.RMSE.min()), max_group_RMSE=float(d.RMSE.max()),
            range_group_RMSE=float(d.RMSE.max() - d.RMSE.min()),
            std_group_RMSE=float(d.RMSE.std(ddof=0)),
            mean_abs_delta_RMSE=float(d.delta_RMSE.abs().mean()),
            worst_group=worst.group_label, worst_group_n=int(worst.n),
            worst_group_RMSE=float(worst.RMSE),
            groups_lower_than_other=int((gwide.lower == model).sum())))
    group_summary = pd.DataFrame(group_summary)

    # ---------------- 5. Step 23 방식 shift 시험 ----------------
    wafer = base.groupby("WAFER_ID", as_index=False).agg(FIRST_TIMESTAMP=("FIRST_TIMESTAMP", "min"))
    wafer = wafer.sort_values(["FIRST_TIMESTAMP", "WAFER_ID"], kind="stable").reset_index(drop=True)
    wafer["BLOCK"] = 0
    for i, idx in enumerate(np.array_split(np.arange(len(wafer)), N_BLOCKS), start=1):
        wafer.loc[idx, "BLOCK"] = i
    base = base.merge(wafer[["WAFER_ID", "BLOCK"]], on="WAFER_ID", how="left", validate="many_to_one")

    s23_eval = pd.read_csv(STEP23_DIR / "eval_predictions.csv",
                           dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    s23_eval["STAGE"] = s23_eval["STAGE"].astype(str)

    shift_rows, shift_pred, overlap_counts, same_split, same_hybrid = [], [], [], [], []

    def run_split(test_id, kind, description, train, evaluate):
        overlap_counts.append(len(set(train.WAFER_ID) & set(evaluate.WAFER_ID)))
        assert overlap_counts[-1] == 0
        pure = build(PURE_PARAMS, numeric).fit(train[features], train[TARGET])
        pred_pure = pure.predict(evaluate[features])
        mean_k, std_k = train.K_TARGET.mean(), train.K_TARGET.std(ddof=0)
        hyb = build(HYBRID_PARAMS, numeric).fit(train[features], (train.K_TARGET - mean_k) / std_k)
        k_pred = hyb.predict(evaluate[features]) * std_k + mean_k
        pred_hyb = k_pred * evaluate.PV_PROXY.to_numpy()
        y = evaluate[TARGET].to_numpy()

        prev = s23_eval[s23_eval.test_id == test_id]
        same_split.append(sorted(map(tuple, prev[KEY].itertuples(index=False))) ==
                          sorted(map(tuple, evaluate[KEY].itertuples(index=False))))
        prev_idx = prev.set_index(KEY).loc[list(zip(evaluate.WAFER_ID, evaluate.STAGE))]
        same_hybrid.append(bool(np.allclose(pred_hyb, prev_idx.MRR_PRED_HYBRID.to_numpy(),
                                            rtol=0, atol=1e-9)))

        m = {"Pure ML": metrics(y, pred_pure), "Hybrid": metrics(y, pred_hyb)}
        for model in MODELS:
            shift_rows.append(dict(test_id=test_id, kind=kind, description=description, model=model,
                                   n_train_samples=len(train), n_eval_wafers=evaluate.WAFER_ID.nunique(),
                                   n_eval_samples=len(evaluate), **m[model],
                                   **stage_mix(train, "train"), **stage_mix(evaluate, "eval")))
        out = evaluate[KEY + [TARGET, "PV_PROXY"]].copy()
        out.insert(0, "test_id", test_id)
        out["MRR_PRED_PURE"], out["MRR_PRED_HYBRID"], out["K_PRED_HYBRID"] = pred_pure, pred_hyb, k_pred
        shift_pred.append(out)

    for i, k in enumerate((2, 3, 4), start=1):
        run_split(f"T{i}", "temporal", f"block 1-{k} 학습 → block {k + 1} 평가",
                  base[base.BLOCK <= k], base[base.BLOCK == k + 1])
    for j, v in enumerate(USAGE_VARS, start=1):
        rep = base.groupby("WAFER_ID", as_index=False)[v].mean().rename(columns={v: "USAGE_REP"})
        rep = rep.sort_values(["USAGE_REP", "WAFER_ID"], kind="stable").reset_index(drop=True)
        n_train = int(round(len(rep) * TRAIN_WAFER_FRACTION))
        train_w = set(rep.WAFER_ID[:n_train])
        run_split(f"U{j}", "usage", f"{v} 하위 wafer 학습 → 상위 wafer 평가",
                  base[base.WAFER_ID.isin(train_w)], base[~base.WAFER_ID.isin(train_w)])

    shift = pd.DataFrame(shift_rows)
    shift_predictions = pd.concat(shift_pred, ignore_index=True)
    swide = shift.pivot_table(index=["test_id", "kind", "description", "n_train_samples",
                                     "n_eval_samples"], columns="model",
                              values=["R2", "MAE", "RMSE"]).reset_index()
    swide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1].replace(' ', '')}" for c in swide.columns]
    swide["RMSE_diff_Hybrid_minus_PureML"] = swide.RMSE_Hybrid - swide.RMSE_PureML
    swide["lower"] = np.where(swide.RMSE_diff_Hybrid_minus_PureML < 0, "Hybrid", "Pure ML")
    swide = swide.sort_values("test_id").reset_index(drop=True)

    # 구간 정의가 Step 22 와 같은지: 17개 그룹의 sample 수를 Step 22 저장값과 대조한다
    s22_groups = pd.read_csv(STEP22_DIR / "group_metrics_all.csv", float_precision="round_trip")
    s22_n = s22_groups[s22_groups.model == "Pure ML"].set_index("group_label").n
    mine_n = gm[gm.model == "Pure ML"].set_index("group_label").n
    same_group_sizes = bool(len(mine_n) == len(s22_n) == 17 and
                            (mine_n.reindex(s22_n.index) == s22_n).all())

    checks += [
        dict(item="모든 split 에서 학습과 평가의 wafer 가 겹치지 않는가",
             value=str(max(overlap_counts) == 0)),
        dict(item="7개 split 의 평가 sample 이 Step 23 과 같은가", value=str(all(same_split))),
        dict(item="Hybrid 재학습 결과가 Step 23 저장값과 같은가", value=str(all(same_hybrid))),
        dict(item="17개 그룹의 sample 수가 Step 22 저장값과 같은가", value=str(same_group_sizes)),
    ]
    checks = pd.DataFrame(checks)

    # ---------------- 6. 핵심 비교 ----------------
    nominal = {m: float(o.loc[m, "RMSE"]) for m in MODELS}
    head = []
    for model in MODELS:
        d = swide[f"RMSE_{model.replace(' ', '')}"]
        head.append(dict(
            model=model,
            nominal_pooled_RMSE=nominal[model],
            nominal_fold_mean_RMSE=float(fs.loc[model, "RMSE_mean"]),
            within_groups_lower=int((gwide.lower == model).sum()),
            within_group_RMSE_range=float(group_summary.set_index("model").loc[model, "range_group_RMSE"]),
            within_group_RMSE_std=float(group_summary.set_index("model").loc[model, "std_group_RMSE"]),
            within_mean_abs_delta_RMSE=float(
                group_summary.set_index("model").loc[model, "mean_abs_delta_RMSE"]),
            within_worst_group_RMSE=float(
                group_summary.set_index("model").loc[model, "worst_group_RMSE"]),
            shift_tests_lower=int((swide.lower == model).sum()),
            shift_mean_RMSE=float(d.mean()), shift_max_RMSE=float(d.max()),
            shift_min_RMSE=float(d.min()),
            shift_mean_increase_vs_nominal=float(d.mean() - nominal[model]),
            shift_max_increase_vs_nominal=float(d.max() - nominal[model])))
    headline = pd.DataFrame(head)

    hw = headline.set_index("model")
    verdict = pd.DataFrame([
        dict(question="within-distribution 17개 그룹에서 더 낮은 RMSE 그룹 수",
             answer=f"Pure ML {int(hw.loc['Pure ML', 'within_groups_lower'])} / "
                    f"Hybrid {int(hw.loc['Hybrid', 'within_groups_lower'])}"),
        dict(question="distribution-shift 7개 test 에서 더 낮은 RMSE test 수",
             answer=f"Pure ML {int(hw.loc['Pure ML', 'shift_tests_lower'])} / "
                    f"Hybrid {int(hw.loc['Hybrid', 'shift_tests_lower'])}"),
        dict(question="shift RMSE 평균",
             answer=f"Pure ML {hw.loc['Pure ML', 'shift_mean_RMSE']:.4f} / "
                    f"Hybrid {hw.loc['Hybrid', 'shift_mean_RMSE']:.4f}"),
        dict(question="shift RMSE 최대값",
             answer=f"Pure ML {hw.loc['Pure ML', 'shift_max_RMSE']:.4f} / "
                    f"Hybrid {hw.loc['Hybrid', 'shift_max_RMSE']:.4f}"),
        dict(question="within-distribution worst-group RMSE",
             answer=f"Pure ML {hw.loc['Pure ML', 'within_worst_group_RMSE']:.4f} / "
                    f"Hybrid {hw.loc['Hybrid', 'within_worst_group_RMSE']:.4f}"),
        dict(question="nominal 대비 shift 평균 증가폭",
             answer=f"Pure ML {hw.loc['Pure ML', 'shift_mean_increase_vs_nominal']:+.4f} / "
                    f"Hybrid {hw.loc['Hybrid', 'shift_mean_increase_vs_nominal']:+.4f}"),
        dict(question="그룹 RMSE 범위 (작을수록 흔들림이 작음)",
             answer=f"Pure ML {hw.loc['Pure ML', 'within_group_RMSE_range']:.4f} / "
                    f"Hybrid {hw.loc['Hybrid', 'within_group_RMSE_range']:.4f}"),
        dict(question="그룹 RMSE 표준편차",
             answer=f"Pure ML {hw.loc['Pure ML', 'within_group_RMSE_std']:.4f} / "
                    f"Hybrid {hw.loc['Hybrid', 'within_group_RMSE_std']:.4f}"),
        dict(question="평균 절대 delta_RMSE",
             answer=f"Pure ML {hw.loc['Pure ML', 'within_mean_abs_delta_RMSE']:.4f} / "
                    f"Hybrid {hw.loc['Hybrid', 'within_mean_abs_delta_RMSE']:.4f}"),
    ])

    # ---------------- 저장 ----------------
    overall.to_csv(OUT_DIR / "overall_metrics.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    gm.to_csv(OUT_DIR / "group_metrics_all.csv", index=False)
    gwide.to_csv(OUT_DIR / "group_comparison.csv", index=False)
    group_summary.to_csv(OUT_DIR / "group_summary.csv", index=False)
    shift.to_csv(OUT_DIR / "shift_test_results.csv", index=False)
    swide.to_csv(OUT_DIR / "shift_comparison.csv", index=False)
    shift_predictions.to_csv(OUT_DIR / "shift_eval_predictions.csv", index=False)
    headline.to_csv(OUT_DIR / "headline_comparison.csv", index=False)
    verdict.to_csv(OUT_DIR / "key_questions.csv", index=False)
    oof.to_csv(OUT_DIR / "oof_predictions_both_models.csv", index=False)
    checks.to_csv(OUT_DIR / "consistency_checks.csv", index=False)

    # ---------------- 그림 ----------------
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
        for t in leg.get_texts():
            t.set_color(INK2)

    # 그림 1: 전체 5-fold 성능
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    xx = np.arange(2)
    for i, model in enumerate(MODELS):
        off = (i - 0.5) * 0.36
        vals = [nominal[model], float(fs.loc[model, "RMSE_mean"])]
        ax.bar(xx + off, vals, width=0.34, color=MODEL_COLOR[model], label=model)
        for x, v in zip(xx + off, vals):
            ax.text(x, v + 0.05, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xx, ["Pooled over all 1,609 samples", "Mean of the 5 folds"])
    style(ax)
    ax.set_ylim(0, 4.4)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("How the 5-fold predictions are aggregated", fontsize=9, color=INK2)
    ax.set_title("Final tuned models, nominal 5-fold performance\n"
                 "Pure ML n900_d4_lr0.05_mcw3 vs representative Hybrid n900_d2_lr0.1_mcw1",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_nominal_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 2: 17개 조건 그룹
    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = gwide.group_label.tolist()[::-1]
    yy = np.arange(len(labels))
    for i, model in enumerate(MODELS):
        vals = gwide[f"RMSE_{model.replace(' ', '')}"].tolist()[::-1]
        ax.barh(yy + (0.19 if i == 0 else -0.19), vals, height=0.34, color=MODEL_COLOR[model], label=model)
        for y, v in zip(yy + (0.19 if i == 0 else -0.19), vals):
            ax.text(v + 0.04, y, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(yy, [f"{lab}  (n={int(n)})" for lab, n in zip(labels, gwide.n.tolist()[::-1])])
    style(ax, axis="x")
    ax.set_xlim(0, max(gwide.RMSE_PureML.max(), gwide.RMSE_Hybrid.max()) * 1.18)
    ax.set_xlabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_ylabel("Condition group  (same definitions as Step 22)", fontsize=9, color=INK2)
    ax.set_title("Within-distribution robustness: 17 condition groups\n"
                 "same 5-fold predictions re-aggregated by stage, usage level and time segment",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    legend(ax, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_group_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 3: shift 7개 test
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    xx = np.arange(len(swide))
    ymax = max(swide.RMSE_PureML.max(), swide.RMSE_Hybrid.max()) * 1.25
    for i, model in enumerate(MODELS):
        off = (i - 0.5) * 0.36
        vals = swide[f"RMSE_{model.replace(' ', '')}"].to_numpy()
        ax.bar(xx + off, vals, width=0.34, color=MODEL_COLOR[model], label=model)
        for x, v in zip(xx + off, vals):
            ax.text(x, v + ymax * 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.set_xticks(xx, [f"{t}\n{'time' if k == 'temporal' else 'usage'}  (n={n})"
                       for t, k, n in zip(swide.test_id, swide.kind, swide.n_eval_samples)])
    style(ax)
    ax.set_ylim(0, ymax)
    for model in MODELS:
        ax.axhline(nominal[model], color=MODEL_COLOR[model], lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Distribution-shift test  (T1-T3 train on past wafers, U1-U4 train on lower-usage wafers)",
                  fontsize=9, color=INK2)
    ax.set_title("Distribution-shift robustness: 7 stress tests, models retrained per split\n"
                 "dashed line = each model's nominal 5-fold RMSE on all 1,609 samples",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_shift_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 4: nominal vs 평균 shift vs 최악 shift
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    xx = np.arange(3)
    ymax = max(hw.shift_max_RMSE) * 1.22
    for i, model in enumerate(MODELS):
        off = (i - 0.5) * 0.36
        vals = [nominal[model], float(hw.loc[model, "shift_mean_RMSE"]),
                float(hw.loc[model, "shift_max_RMSE"])]
        ax.bar(xx + off, vals, width=0.34, color=MODEL_COLOR[model], label=model)
        for x, v in zip(xx + off, vals):
            ax.text(x, v + ymax * 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xx, ["Nominal 5-fold", "Average of 7 shift tests", "Worst shift test"])
    style(ax)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Evaluation regime", fontsize=9, color=INK2)
    ax.set_title("Cost of leaving the training distribution\n"
                 "nominal, average shift and worst shift RMSE for the two final models",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_nominal_vs_shift.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 출력 ----------------
    print("[일관성 확인]")
    print(checks.to_string(index=False))
    print("\n[전체 5-fold 성능]")
    print(overall.to_string(index=False))
    print(fold_summary.to_string(index=False))
    print("\n[17개 조건 그룹]")
    print(gwide.to_string(index=False))
    print("\n[그룹 요약]")
    print(group_summary.to_string(index=False))
    print("\n[shift 7개 test]")
    print(swide.to_string(index=False))
    print("\n[핵심 비교]")
    print(headline.to_string(index=False))
    print(verdict.to_string(index=False))
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

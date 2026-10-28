"""Step 28: Step 1~27 의 저장된 결과만으로 최종 연구 결과를 한 곳에 모은다.

- 새로운 학습, tuning, feature engineering, P/V 변경을 하지 않는다.
- 모든 수치는 기존 결과 CSV 에서 읽는다. 손으로 입력하지 않는다.
- 요청서에 적힌 기대값과 저장값을 대조해 불일치가 있으면 고치지 않고 그대로 보고한다.
- 최종 그림은 기존 단계에서 저장한 PNG 를 골라 복사하고, 개념도 한 장만 새로 그린다
  (개념도는 분석 결과가 아니라 구조 설명 그림이다).
출력: reports/final_summary/ (CSV, 그림), 화면에 문서용 markdown 표
실행: python -B scripts/step28_final_summary.py
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT_DIR = REP / "final_summary"
FIG_DIR = OUT_DIR / "figures"
MODELS = ["Pure ML", "Fixed-k Preston", "Hybrid"]
SETS = ["training", "validation", "test"]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES1, SERIES2, SERIES3 = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834", "#898781"
MODEL_COLOR = {"Pure ML": SERIES2, "Hybrid": SERIES1, "Fixed-k Preston": SERIES3}

f4 = lambda v: f"{float(v):.4f}"
sg = lambda v: f"{float(v):+.4f}"


def read(rel):
    return pd.read_csv(REP / rel, float_precision="round_trip")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tw = read("step27_test/three_way_comparison.csv").set_index("model")
    s16 = read("step16_model3_prep/summary.csv")
    s21_imp = read("step21_interpretation/feature_importance.csv")
    s21_grp = read("step21_interpretation/group_contribution.csv")
    s22_sum = read("step22_robustness/robustness_summary.csv").set_index("model")
    s22_diff = read("step22_robustness/model_difference_by_group.csv")
    s23_cmp = read("step23_distribution_shift/model_comparison_by_test.csv").set_index("test_id")
    s24_cfg = read("step24_pure_ml_tuning/config_results.csv")
    s24_spread = read("step24_pure_ml_tuning/top_config_spread.csv")
    s25_head = read("step25_final_robustness/headline_comparison.csv").set_index("model")
    s25_grp = read("step25_final_robustness/group_comparison.csv")
    s25_shift = read("step25_final_robustness/shift_comparison.csv").set_index("test_id")
    s26_val = read("step26_validation/validation_metrics.csv")
    s27_ind = read("step27_test/data_independence.csv")
    s27_struct = read("step27_test/dataset_structure.csv")
    s27_dist = read("step27_test/distribution_comparison.csv")

    # ---------------- master metrics ----------------
    rows = []
    for m in MODELS:
        for s in SETS:
            rows.append(dict(model=m, dataset=s, n=int(tw.loc[m, f"{s}_n"]),
                             R2=float(tw.loc[m, f"{s}_R2"]), MAE=float(tw.loc[m, f"{s}_MAE"]),
                             RMSE=float(tw.loc[m, f"{s}_RMSE"]),
                             source="Step 27 three_way_comparison.csv"))
    master = pd.DataFrame(rows)
    master["PV_definition"] = np.where(master.model == "Pure ML", "사용하지 않음", "P9 x V3")

    # ---------------- recovery: fixed-k -> hybrid ----------------
    rec = []
    for s in SETS:
        fk = float(tw.loc["Fixed-k Preston", f"{s}_RMSE"])
        hy = float(tw.loc["Hybrid", f"{s}_RMSE"])
        rec.append(dict(dataset=s, n=int(tw.loc["Hybrid", f"{s}_n"]),
                        fixed_k_RMSE=fk, hybrid_RMSE=hy,
                        absolute_RMSE_reduction=fk - hy,
                        relative_RMSE_reduction_pct=(fk - hy) / fk * 100,
                        fixed_k_R2=float(tw.loc["Fixed-k Preston", f"{s}_R2"]),
                        hybrid_R2=float(tw.loc["Hybrid", f"{s}_R2"]),
                        fixed_k_MAE=float(tw.loc["Fixed-k Preston", f"{s}_MAE"]),
                        hybrid_MAE=float(tw.loc["Hybrid", f"{s}_MAE"])))
    recovery = pd.DataFrame(rec)

    # ---------------- trade-off: hybrid vs pure ML ----------------
    tr = []
    for s in SETS:
        pu = float(tw.loc["Pure ML", f"{s}_RMSE"])
        hy = float(tw.loc["Hybrid", f"{s}_RMSE"])
        tr.append(dict(dataset=s, n=int(tw.loc["Hybrid", f"{s}_n"]),
                       pure_ml_RMSE=pu, hybrid_RMSE=hy,
                       hybrid_minus_pure_RMSE=hy - pu,
                       relative_difference_pct=(hy / pu - 1) * 100,
                       pure_ml_MAE=float(tw.loc["Pure ML", f"{s}_MAE"]),
                       hybrid_MAE=float(tw.loc["Hybrid", f"{s}_MAE"]),
                       hybrid_minus_pure_MAE=float(tw.loc["Hybrid", f"{s}_MAE"] -
                                                   tw.loc["Pure ML", f"{s}_MAE"]),
                       lower_RMSE="Pure ML" if pu < hy else "Hybrid"))
    tradeoff = pd.DataFrame(tr)

    # ---------------- 요청서 기대값과 저장값 대조 ----------------
    expected = [("training", "Pure ML", 3.32, 2), ("training", "Hybrid", 3.57, 2),
                ("training", "Fixed-k Preston", 8.65, 2),
                ("validation", "Pure ML", 3.2560, 4), ("validation", "Hybrid", 3.4142, 4),
                ("validation", "Fixed-k Preston", 9.1130, 4),
                ("test", "Pure ML", 3.1856, 4), ("test", "Hybrid", 3.4650, 4),
                ("test", "Fixed-k Preston", 9.1367, 4)]
    checks = []
    for s, m, exp, nd in expected:
        saved = float(tw.loc[m, f"{s}_RMSE"])
        checks.append(dict(dataset=s, model=m, expected_in_request=exp,
                           saved_value=saved, rounded_saved=round(saved, nd),
                           matches=bool(round(saved, nd) == exp)))
    consistency = pd.DataFrame(checks)
    s16_fixed = float(s16[(s16.model == "Model2_fixed_k") & (s16.metric == "RMSE")]["mean"].iloc[0])
    variants = pd.DataFrame([
        dict(item="training fixed-k RMSE, PV = P9 x V3 (Step 26/27 에서 사용, master table 값)",
             value=float(tw.loc["Fixed-k Preston", "training_RMSE"])),
        dict(item="training fixed-k RMSE, Step 16 당시의 PV 조합 (5-fold 평균)", value=s16_fixed)])

    # validation 이 Step 26 저장값과 같은지
    v26 = s26_val[s26_val.scope.str.startswith("validation 4-5-6 (three")].set_index("model")
    val_match = bool(all(np.isclose(tw.loc[m, "validation_RMSE"], v26.loc[m, "RMSE"],
                                    rtol=0, atol=1e-9) for m in MODELS))

    # ---------------- robustness 통합 ----------------
    rob = pd.DataFrame([
        dict(analysis="Step 22 (within-distribution, 기본 설정 Pure ML)",
             comparison="Hybrid vs 기본 Pure ML",
             hybrid_better=int((s22_diff.RMSE_diff_Hybrid_minus_PureML < 0).sum()),
             total=len(s22_diff),
             hybrid_spread=float(s22_sum.loc["Hybrid", "range_group_RMSE"]),
             pure_spread=float(s22_sum.loc["Pure ML", "range_group_RMSE"])),
        dict(analysis="Step 23 (distribution shift, 기본 설정 Pure ML)",
             comparison="Hybrid vs 기본 Pure ML",
             hybrid_better=int((s23_cmp.RMSE_diff_Hybrid_minus_PureML < 0).sum()),
             total=len(s23_cmp), hybrid_spread=np.nan, pure_spread=np.nan),
        dict(analysis="Step 25 (within-distribution, tuned Pure ML)",
             comparison="Hybrid vs tuned Pure ML",
             hybrid_better=int((s25_grp.RMSE_diff_Hybrid_minus_PureML < 0).sum()),
             total=len(s25_grp),
             hybrid_spread=float(s25_head.loc["Hybrid", "within_group_RMSE_range"]),
             pure_spread=float(s25_head.loc["Pure ML", "within_group_RMSE_range"])),
        dict(analysis="Step 25 (distribution shift, tuned Pure ML)",
             comparison="Hybrid vs tuned Pure ML",
             hybrid_better=int((s25_shift.RMSE_diff_Hybrid_minus_PureML < 0).sum()),
             total=len(s25_shift), hybrid_spread=np.nan, pure_spread=np.nan)])

    shift_summary = pd.DataFrame([dict(
        metric="shift RMSE 평균", pure_ml=float(s25_head.loc["Pure ML", "shift_mean_RMSE"]),
        hybrid=float(s25_head.loc["Hybrid", "shift_mean_RMSE"])),
        dict(metric="shift RMSE 최대", pure_ml=float(s25_head.loc["Pure ML", "shift_max_RMSE"]),
             hybrid=float(s25_head.loc["Hybrid", "shift_max_RMSE"])),
        dict(metric="worst-group RMSE", pure_ml=float(s25_head.loc["Pure ML", "within_worst_group_RMSE"]),
             hybrid=float(s25_head.loc["Hybrid", "within_worst_group_RMSE"])),
        dict(metric="T1 RMSE", pure_ml=float(s25_shift.loc["T1", "RMSE_PureML"]),
             hybrid=float(s25_shift.loc["T1", "RMSE_Hybrid"]))])
    shift_summary["hybrid_minus_pure"] = shift_summary.hybrid - shift_summary.pure_ml

    # ---------------- 저장 ----------------
    master.to_csv(OUT_DIR / "master_metrics.csv", index=False)
    recovery.to_csv(OUT_DIR / "recovery_analysis.csv", index=False)
    tradeoff.to_csv(OUT_DIR / "pureml_hybrid_tradeoff.csv", index=False)
    consistency.to_csv(OUT_DIR / "consistency_with_request.csv", index=False)
    variants.to_csv(OUT_DIR / "fixed_k_training_variants.csv", index=False)
    rob.to_csv(OUT_DIR / "robustness_summary_across_steps.csv", index=False)
    shift_summary.to_csv(OUT_DIR / "shift_summary_step25.csv", index=False)
    s21_grp.to_csv(OUT_DIR / "interpretation_group_shares.csv", index=False)
    s21_imp.sort_values("hybrid_rank").head(15).to_csv(
        OUT_DIR / "interpretation_top15_features.csv", index=False)

    # ---------------- 그림: 개념도 ----------------
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    linewidth=1.2, edgecolor=color, facecolor=SURFACE))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)

    def arrow(x1, y1, x2, y2, color):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.1, color=color))

    ax.text(0.1, 4.3, "Three model structures compared on the same wafer+Stage samples",
            fontsize=12, color=INK, ha="left")
    # Model 1
    ax.text(0.1, 3.75, "Model 1  Pure ML", fontsize=9.5, color=SERIES2, ha="left")
    box(0.1, 2.95, 2.3, 0.65, "process features x\n(31 features)", SERIES2)
    arrow(2.5, 3.28, 3.4, 3.28, SERIES2)
    box(3.5, 2.95, 2.3, 0.65, "XGBoost", SERIES2)
    arrow(5.9, 3.28, 6.8, 3.28, SERIES2)
    box(6.9, 2.95, 2.0, 0.65, "MRR", SERIES2)
    # Model 2
    ax.text(0.1, 2.55, "Model 2  Fixed-k Preston", fontsize=9.5, color=SERIES3, ha="left")
    box(0.1, 1.75, 2.3, 0.65, "PV_proxy = P9 x V3\n(sensor-based)", SERIES3)
    arrow(2.5, 2.08, 3.4, 2.08, SERIES3)
    box(3.5, 1.75, 2.3, 0.65, "one constant k\nfitted on training", SERIES3)
    arrow(5.9, 2.08, 6.8, 2.08, SERIES3)
    box(6.9, 1.75, 2.0, 0.65, "MRR = k x PV", SERIES3)
    # Model 3
    ax.text(0.1, 1.35, "Model 3  Hybrid", fontsize=9.5, color=SERIES1, ha="left")
    box(0.1, 0.45, 2.3, 0.75, "process features x\n+ PV_proxy = P9 x V3", SERIES1)
    arrow(2.5, 0.83, 3.4, 0.83, SERIES1)
    box(3.5, 0.45, 2.3, 0.75, "XGBoost learns k(x)\n(standardized target)", SERIES1)
    arrow(5.9, 0.83, 6.8, 0.83, SERIES1)
    box(6.9, 0.45, 2.0, 0.75, "MRR = k(x) x PV", SERIES1)
    ax.text(9.15, 2.1, "P9 = mean of 6 pressure channels per raw row\n"
                       "V3 = abs(HEAD_ROTATION)\n\n"
                       "PV_proxy is a sensor-based proxy,\nnot a calibrated physical\n"
                       "pressure or linear velocity.",
            fontsize=8, color=INK2, ha="left", va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_model_concept.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 기존 그림 복사 ----------------
    copies = [
        ("step27_test/figures/fig1_train_val_test_rmse.png", "fig2_train_val_test_rmse.png"),
        ("step27_test/figures/fig4_fixedk_hybrid_pure.png", "fig3_fixedk_hybrid_pure.png"),
        ("step27_test/figures/fig3_test_actual_vs_predicted.png", "fig4_test_actual_vs_predicted.png"),
        ("step21_interpretation/figures/fig1_hybrid_k_importance.png", "fig5_hybrid_k_importance.png"),
        ("step21_interpretation/figures/fig3_group_share_comparison.png", "fig6_feature_group_share.png"),
        ("step25_final_robustness/figures/fig3_shift_rmse.png", "fig7_distribution_shift.png"),
    ]
    copied = []
    for src, dst in copies:
        shutil.copyfile(REP / src, FIG_DIR / dst)
        copied.append(dict(final_figure=dst, source=f"reports/{src}"))
    figure_index = pd.DataFrame([dict(final_figure="fig1_model_concept.png",
                                      source="Step 28 에서 새로 그린 구조 개념도")] + copied)
    figure_index.to_csv(OUT_DIR / "figure_index.csv", index=False)

    # ---------------- 문서용 출력 ----------------
    print("### MASTER")
    print("| 모델 | 데이터 | sample 수 | R² | MAE | RMSE |")
    print("|---|---|---|---|---|---|")
    for _, r in master.iterrows():
        print(f"| {r.model} | {r.dataset} | {int(r.n)} | {f4(r.R2)} | {f4(r.MAE)} | {f4(r.RMSE)} |")

    print("\n### RECOVERY")
    print("| 데이터 | sample 수 | Fixed-k RMSE | Hybrid RMSE | 절대 감소 | 상대 감소 (%) | Fixed-k R² | Hybrid R² |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in recovery.iterrows():
        print(f"| {r.dataset} | {int(r.n)} | {f4(r.fixed_k_RMSE)} | {f4(r.hybrid_RMSE)} | "
              f"{f4(r.absolute_RMSE_reduction)} | {f4(r.relative_RMSE_reduction_pct)} | "
              f"{f4(r.fixed_k_R2)} | {f4(r.hybrid_R2)} |")

    print("\n### TRADEOFF")
    print("| 데이터 | sample 수 | Pure ML RMSE | Hybrid RMSE | Hybrid − Pure ML | 상대 차이 (%) | MAE 차이 | 더 낮은 쪽 |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in tradeoff.iterrows():
        print(f"| {r.dataset} | {int(r.n)} | {f4(r.pure_ml_RMSE)} | {f4(r.hybrid_RMSE)} | "
              f"{sg(r.hybrid_minus_pure_RMSE)} | {f4(r.relative_difference_pct)} | "
              f"{sg(r.hybrid_minus_pure_MAE)} | {r.lower_RMSE} |")

    print("\n### CONSISTENCY")
    print("| 데이터 | 모델 | 요청서 기대값 | 저장값 | 반올림 | 일치 |")
    print("|---|---|---|---|---|---|")
    for _, r in consistency.iterrows():
        print(f"| {r.dataset} | {r.model} | {r.expected_in_request} | {r.saved_value:.6f} | "
              f"{r.rounded_saved} | {r.matches} |")
    print("\n### VARIANTS")
    print("| 항목 | 값 |"); print("|---|---|")
    for _, r in variants.iterrows():
        print(f"| {r['item']} | {f4(r['value'])} |")

    print("\n### ROBUSTNESS")
    print("| 분석 | 비교 대상 | Hybrid 가 더 좋은 경우 | 전체 |")
    print("|---|---|---|---|")
    for _, r in rob.iterrows():
        print(f"| {r.analysis} | {r.comparison} | {int(r.hybrid_better)} | {int(r.total)} |")
    print("\n### SHIFTSUMMARY")
    print("| 지표 | Pure ML | Hybrid | Hybrid − Pure ML |")
    print("|---|---|---|---|")
    for _, r in shift_summary.iterrows():
        print(f"| {r.metric} | {f4(r.pure_ml)} | {f4(r.hybrid)} | {sg(r.hybrid_minus_pure)} |")

    print("\n### GROUPSHARES")
    print("| feature 그룹 | feature 수 | Hybrid k(x) 비중 | Pure ML 비중 |")
    print("|---|---|---|---|")
    for _, r in s21_grp.iterrows():
        print(f"| {r.group} | {int(r.n_features)} | {r.hybrid_share * 100:.1f}% | {r.pure_share * 100:.1f}% |")

    print("\n### TOP10FEATURES")
    top = s21_imp.sort_values("hybrid_rank").head(10)
    print("| 순위 | feature | 그룹 | 평균 절대 기여도 (k 단위) | Pure ML 순위 |")
    print("|---|---|---|---|---|")
    for _, r in top.iterrows():
        print(f"| {int(r.hybrid_rank)} | {r.feature} | {r.group} | {r.hybrid_mean_abs_contrib_k:.6f} | "
              f"{int(r.pure_rank)} |")

    print("\n### INDEPENDENCE")
    print("| 비교 | wafer+Stage 키 겹침 | 공유 WAFER_ID 수 |")
    print("|---|---|---|")
    for _, r in s27_ind.iterrows():
        print(f"| {r.pair} | {int(r.shared_wafer_stage_keys)} | {int(r.shared_wafer_ids)} |")

    print("\n### derived")
    print("validation 이 Step 26 저장값과 같은가:", val_match)
    print("요청서 기대값과 불일치한 항목:",
          consistency[~consistency.matches][["dataset", "model", "expected_in_request",
                                             "rounded_saved"]].to_dict("records") or "없음")
    pv_feats = ["RETAINER_RING_PRESSURE_MEAN", "RETAINER_RING_PRESSURE_STD",
                "HEAD_ROTATION_MEAN", "HEAD_ROTATION_STD"]
    pv_share_h = s21_imp[s21_imp.feature.isin(pv_feats)].hybrid_mean_abs_contrib_mrr.sum() / \
        s21_imp.hybrid_mean_abs_contrib_mrr.sum() * 100
    pv_share_p = s21_imp[s21_imp.feature.isin(pv_feats)].pure_mean_abs_contrib_mrr.sum() / \
        s21_imp.pure_mean_abs_contrib_mrr.sum() * 100
    print(f"PV 를 구성하는 4개 feature 비중: Hybrid {pv_share_h:.1f}% / Pure ML {pv_share_p:.1f}%")
    print(f"PV 밖 정보 비중 (Hybrid): {100 - pv_share_h:.1f}%")
    print("Step 24 상위 10개 평균 RMSE 폭 / fold 표준편차:",
          s24_spread.set_index("item").loc["상위 10개 평균 RMSE 폭", "value"], "/",
          s24_spread.set_index("item").loc["상위 10개 설정의 fold 표준편차 평균", "value"])
    print("Step 24 1위 설정:", s24_cfg.iloc[0].config_id, round(float(s24_cfg.iloc[0].RMSE_mean), 4))
    print("test 4-5-6 / 1-2-3 sample 수:",
          int(s27_struct[(s27_struct.dataset == "test") & (s27_struct.CHAMBER_PATH == "4-5-6")].n_samples.iloc[0]),
          int(s27_struct[(s27_struct.dataset == "test") & (s27_struct.CHAMBER_PATH == "1-2-3")].n_samples.iloc[0]))
    print("test 최대 |표준화 분포차|:", s27_dist.iloc[0].variable,
          round(float(s27_dist.iloc[0].test_standardized_gap), 4))
    print("세 데이터 모두에서 Pure ML 이 더 낮은가:", bool((tradeoff.lower_RMSE == "Pure ML").all()))
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

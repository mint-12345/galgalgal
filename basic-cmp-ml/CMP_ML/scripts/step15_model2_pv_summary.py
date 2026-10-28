"""Step 15: Model 2 - PV 를 어떤 raw row 에서 요약하느냐에 따른 차이 비교.

- P/V 조합은 6개만 사용한다 (Step 14 확장판과 정의 동일):
    P4  RETAINER_RING_PRESSURE
    P8  air-bag 4개 + retainer ring 의 행별 평균 (raw)
    P11 위 5개를 각각 "학습용 4개 fold raw 행의 0 이 아닌 값 중앙값" 으로 나눈 뒤 행별 평균 (index)
    V5  |WAFER_ROTATION - HEAD_ROTATION|,  V6  |STAGE_ROTATION - HEAD_ROTATION|  (상대회전 proxy)
- PV 요약 방식 3가지
    A  모든 raw row 의 mean(P x V)                         (Step 14 방식)
    B  P x V 가 0 이 아닌 row 만의 mean(P x V)
    C  P 가 0 이 아니고 V 도 0 이 아닌 row 만의 mean(P x V)
  B 와 C 가 실제로 같은 값이 되는지 확인한다. 이 row 들을 polishing 구간이라고 부르지 않는다.
  조건을 만족하는 row 가 없는 wafer+Stage 는 PV_proxy 를 0 으로 만들지 않고 빈 값(NaN)으로 둔다.
- 모델: MRR = k x PV_proxy. fold 마다 학습용 4개 fold 의 "PV_proxy 가 있는" sample 로만
  절편 없는 최소제곱 k 하나를 구해 평가 fold 에 그대로 적용한다 (effective k).
  PV_proxy 가 빈 값인 sample 은 예측을 만들 수 없으므로 지표 계산에서 빠지고, 개수를 따로 보고한다.
- training 만 사용하고 raw 파일은 수정하지 않는다. 제외 기준은 앞 단계와 같다.
- 최종 P/V 나 요약 방식은 선택하지 않는다.
출력: data/processed/step15/, reports/step15_model2_pv_summary/
실행: python -B scripts/step15_model2_pv_summary.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 250)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
STEP14B_COMBOS = ROOT / "reports" / "step14b_model2_pv_representation" / "combo_results.csv"
PROC_DIR = ROOT / "data" / "processed" / "step15"
OUT_DIR = ROOT / "reports" / "step15_model2_pv_summary"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000

CHAMBER_P = "PRESSURIZED_CHAMBER_PRESSURE"
RETAINER = "RETAINER_RING_PRESSURE"
AIR_BAGS = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE",
            "EDGE_AIR_BAG_PRESSURE"]
PRESSURES = [CHAMBER_P] + AIR_BAGS + [RETAINER]
P8_COLS = AIR_BAGS + [RETAINER]
P_DEFINITION = {"P4": "RETAINER_RING_PRESSURE",
                "P8": "mean of 4 air-bag + retainer ring (raw)",
                "P11": "mean of 4 air-bag + retainer ring, each divided by train-fold nonzero median (index)"}
V_DEFINITION = {"V5": "|WAFER_ROTATION - HEAD_ROTATION| (relative-rotation proxy)",
                "V6": "|STAGE_ROTATION - HEAD_ROTATION| (relative-rotation proxy)"}
COMBOS = [(p, v) for p in ["P4", "P8", "P11"] for v in ["V5", "V6"]]
METHOD_LABEL = {"A": "all rows: mean(P x V)",
                "B": "rows with P x V != 0: mean(P x V)",
                "C": "rows with P != 0 and V != 0: mean(P x V)"}
SCOPES = {"all": None, "paths_123_456": ["1-2-3", "4-5-6"], "path_1-2-3": ["1-2-3"], "path_4-5-6": ["4-5-6"]}
SCOPE_ROWS = {"all": 1977, "paths_123_456": 1973, "path_1-2-3": 364, "path_4-5-6": 1609}
METRICS = ["R2", "MAE", "RMSE"]
MAIN_SCOPE = "paths_123_456"


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
    drop = data.set_index(KEY).index.isin(big.set_index(KEY).index)
    data = data[~drop]
    assert int(same_prev.sum()) == 1819 and len(big) == 4
    assert data[cols].notna().all().all() and (data[PRESSURES].to_numpy() >= 0).all()
    print(f"사용 행 {len(data):,} / wafer+Stage {data.groupby(KEY).ngroups:,}")
    return data


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def scoped(frame):
    """예측이 있는 sample 로만 지표를 계산하고, 범위별 사용/미사용 개수를 함께 남긴다."""
    out = {}
    for scope, paths in SCOPES.items():
        part = frame if paths is None else frame[frame["CHAMBER_PATH"].isin(paths)]
        ok = part[part["MRR_PRED"].notna()]
        out[scope] = metrics(ok["MRR_TRUE"], ok["MRR_PRED"]) | {"n_used": len(ok), "n_missing": len(part) - len(ok)}
    return out


def main():
    data = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    base = step7[KEY + ["CHAMBER_PATH", TARGET]].merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(base) == 1977

    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(base["WAFER_ID"], base["STAGE"]))}
    codes = np.array([order[(int(w), str(s))] for w, s in zip(data["WAFER_ID"], data["STAGE"])], dtype=np.int64)
    n_samples = len(base)
    counts_all = np.bincount(codes, minlength=n_samples).astype(float)
    row_fold = base["FOLD"].to_numpy()[codes]
    sample_fold = base["FOLD"].to_numpy()
    y = base[TARGET].to_numpy()
    fold_list = sorted(base["FOLD"].unique())

    rot = {k: data[f"{k}_ROTATION"].to_numpy() for k in ["WAFER", "STAGE", "HEAD"]}
    V_ROW = {"V5": np.abs(rot["WAFER"] - rot["HEAD"]), "V6": np.abs(rot["STAGE"] - rot["HEAD"])}
    press = {c: data[c].to_numpy() for c in PRESSURES}
    P_ROW_FIXED = {"P4": press[RETAINER], "P8": np.mean([press[c] for c in P8_COLS], axis=0)}

    median_rows, P11_BY_FOLD = [], {}
    for k_fold in fold_list:
        train_rows = row_fold != k_fold
        med = {}
        for c in P8_COLS:
            v = press[c][train_rows]
            nz = v[v != 0]
            med[c] = float(np.median(nz))
            median_rows.append(dict(FOLD=int(k_fold), pressure=c, train_nonzero_median=med[c]))
        P11_BY_FOLD[k_fold] = np.mean([press[c] / med[c] for c in P8_COLS], axis=0)

    def summarize(pv_row, mask=None):
        """mask 가 있으면 그 row 만 평균. 해당 row 가 없는 sample 은 NaN."""
        if mask is None:
            return np.bincount(codes, weights=pv_row, minlength=n_samples) / counts_all
        n = np.bincount(codes, weights=mask.astype(float), minlength=n_samples)
        s = np.bincount(codes, weights=pv_row * mask, minlength=n_samples)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n > 0, s / np.where(n > 0, n, 1.0), np.nan)
        return out

    fold_rows, k_rows, pooled_rows, case_rows, preds, oof_cols = [], [], [], [], {}, {}
    bc_identical = {}
    for p, v in COMBOS:
        combo = f"{p}x{v}"
        v_row = V_ROW[v]
        for method in ["A", "B", "C"]:
            case = f"{combo}_{method}"
            pv_by_fold = {}
            for k_fold in fold_list:
                p_row = P11_BY_FOLD[k_fold] if p == "P11" else P_ROW_FIXED[p]
                pv_row = p_row * v_row
                if method == "A":
                    pv_by_fold[k_fold] = summarize(pv_row)
                elif method == "B":
                    pv_by_fold[k_fold] = summarize(pv_row, pv_row != 0)
                else:
                    pv_by_fold[k_fold] = summarize(pv_row, (p_row != 0) & (v_row != 0))
            pred = np.full(n_samples, np.nan)
            k_used = np.full(n_samples, np.nan)
            oof_pv = np.full(n_samples, np.nan)
            for k_fold in fold_list:
                pv = pv_by_fold[k_fold]
                is_test = sample_fold == k_fold
                train_ok = (~is_test) & np.isfinite(pv)
                denom = float(np.sum(pv[train_ok] ** 2))
                assert denom > 0
                k = float(np.sum(pv[train_ok] * y[train_ok])) / denom
                oof_pv[is_test] = pv[is_test]
                pred[is_test] = k * pv[is_test]
                k_used[is_test] = k
                k_rows.append(dict(case=case, combo=combo, method=method, FOLD=int(k_fold), effective_k=k,
                                   n_train_used=int(train_ok.sum()), n_train_missing_pv=int(((~is_test) & ~np.isfinite(pv)).sum())))
                frame = base.loc[is_test, KEY + ["CHAMBER_PATH"]].assign(MRR_TRUE=y[is_test], MRR_PRED=pred[is_test])
                for scope, m in scoped(frame).items():
                    fold_rows.append(dict(case=case, combo=combo, method=method, FOLD=int(k_fold), scope=scope, **m))
            oof = base[KEY + ["CHAMBER_PATH", "FOLD"]].assign(CASE=case, COMBO=combo, METHOD=method, PV_PROXY=oof_pv,
                                                              EFFECTIVE_K_USED=k_used, MRR_TRUE=y, MRR_PRED=pred,
                                                              ERROR=pred - y)
            preds[case] = oof
            oof_cols[f"PV_{p}_{v}_{method}"] = oof_pv
            for scope, m in scoped(oof).items():
                assert m["n_used"] + m["n_missing"] == SCOPE_ROWS[scope]
                pooled_rows += [dict(case=case, scope=scope, metric=x, pooled_out_of_fold=m[x]) for x in METRICS]
            zero = np.isfinite(oof_pv) & (oof_pv == 0)
            miss = ~np.isfinite(oof_pv)
            corr = np.nan
            sel = (base["CHAMBER_PATH"] == "4-5-6").to_numpy() & np.isfinite(oof_pv)
            if sel.sum() > 2:
                corr = float(np.corrcoef(oof_pv[sel], y[sel])[0, 1])
            case_rows.append(dict(case=case, combo=combo, method=method, P=p, V=v,
                                  P_definition=P_DEFINITION[p], V_definition=V_DEFINITION[v],
                                  method_definition=METHOD_LABEL[method],
                                  n_zero_pv=int(zero.sum()), n_missing_pv=int(miss.sum()),
                                  n_missing_pv_1_2_3=int((miss & (base["CHAMBER_PATH"] == "1-2-3")).sum()),
                                  n_missing_pv_4_5_6=int((miss & (base["CHAMBER_PATH"] == "4-5-6")).sum()),
                                  n_missing_pv_other=int((miss & (base["CHAMBER_PATH"] == "other")).sum()),
                                  corr_pv_mrr_4_5_6=corr))
        b, c = oof_cols[f"PV_{p}_{v}_B"], oof_cols[f"PV_{p}_{v}_C"]
        same = bool(np.array_equal(np.isnan(b), np.isnan(c)) and np.allclose(b[~np.isnan(b)], c[~np.isnan(c)], rtol=0, atol=0))
        bc_identical[combo] = same
    print("B 와 C 의 PV_proxy 가 완전히 같은가 (조합별):", bc_identical)

    fold_metrics = pd.DataFrame(fold_rows)
    long = fold_metrics.melt(id_vars=["case", "combo", "method", "FOLD", "scope"], value_vars=METRICS, var_name="metric")
    summary = (long.groupby(["case", "combo", "method", "scope", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index()
               .merge(pd.DataFrame(pooled_rows), on=["case", "scope", "metric"]))
    k_values = pd.DataFrame(k_rows)
    k_summary = k_values.groupby("case", sort=False)["effective_k"].agg(
        k_mean="mean", k_std_ddof1="std", k_min="min", k_max="max").reset_index()
    k_summary["k_cv"] = k_summary["k_std_ddof1"] / k_summary["k_mean"].abs()
    wide = summary.pivot_table(index="case", columns=["scope", "metric"], values=["mean", "std_ddof1"])
    wide.columns = [f"{scope}_{metric}_{stat}" for stat, scope, metric in wide.columns]
    case_table = pd.DataFrame(case_rows).set_index("case").join(k_summary.set_index("case")).join(wide).reset_index()
    case_table["rank_by_paths_RMSE"] = case_table[f"{MAIN_SCOPE}_RMSE_mean"].rank(method="first").astype(int)
    case_table = case_table.sort_values(f"{MAIN_SCOPE}_RMSE_mean")

    # 같은 fold 기준 A 대비 차이
    a_long = long[long["method"] == "A"].drop(columns=["case", "method"]).rename(columns={"value": "method_A"})
    paired = long[long["method"] != "A"].merge(a_long, on=["combo", "FOLD", "scope", "metric"])
    paired["diff_vs_A"] = paired["value"] - paired["method_A"]
    paired["better"] = np.where(paired["metric"] == "R2", paired["diff_vs_A"] > 0, paired["diff_vs_A"] < 0)
    paired_summary = (paired.groupby(["case", "combo", "method", "scope", "metric"], sort=False)
                      .agg(mean_diff_vs_A=("diff_vs_A", "mean"), std_diff=("diff_vs_A", "std"),
                           folds_better_than_A=("better", "sum")).reset_index())

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(median_rows).to_csv(PROC_DIR / "step15_normalization_medians.csv", index=False)
    pd.concat([base, pd.DataFrame(oof_cols)], axis=1).to_csv(PROC_DIR / "step15_pv_proxy_oof_table.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_long.csv", index=False)
    k_values.to_csv(OUT_DIR / "effective_k_by_fold.csv", index=False)
    case_table.to_csv(OUT_DIR / "case_results.csv", index=False)
    paired_summary.to_csv(OUT_DIR / "paired_difference_vs_method_A.csv", index=False)
    pd.concat(preds.values(), ignore_index=True).to_csv(OUT_DIR / "oof_predictions_all_cases.csv", index=False)
    pd.DataFrame([dict(combo=c, B_equals_C=v) for c, v in bc_identical.items()]).to_csv(OUT_DIR / "b_vs_c_check.csv", index=False)

    # 방식 A 가 Step 14 확장판과 같은지 확인
    prev = pd.read_csv(STEP14B_COMBOS).set_index("combo")
    a_rows = case_table[case_table["method"] == "A"].set_index("combo")
    cols = [f"{s}_{m}_mean" for s in SCOPES for m in METRICS] + ["k_mean"]
    same_prev = all(np.allclose(a_rows.loc[[c for c, _ in COMBOS[:0]] + list(a_rows.index), col].to_numpy(),
                                prev.loc[list(a_rows.index), col].to_numpy(), rtol=0, atol=1e-9) for col in cols)
    print(f"방식 A 의 지표와 k 가 Step 14 확장판과 같은가: {same_prev}")
    assert same_prev

    show_cols = ["rank_by_paths_RMSE", "case", "method", f"{MAIN_SCOPE}_RMSE_mean", f"{MAIN_SCOPE}_R2_mean",
                 "path_1-2-3_RMSE_mean", "path_4-5-6_RMSE_mean", "all_RMSE_mean", "n_zero_pv", "n_missing_pv",
                 "corr_pv_mrr_4_5_6", "k_mean", "k_cv"]
    print("\n[18개 경우: 일반 경로 RMSE 평균 순]")
    print(case_table[show_cols].round(4).to_string(index=False))
    print("\n[조합별 A / B / C 비교: 일반 경로와 경로별 RMSE 평균]")
    piv = case_table.pivot_table(index="combo", columns="method",
                                 values=[f"{MAIN_SCOPE}_RMSE_mean", "path_1-2-3_RMSE_mean", "path_4-5-6_RMSE_mean"])
    print(piv.round(4).to_string())
    print("\n[A 대비 같은 fold RMSE 차이 (B, C) - 음수면 A 보다 좋음, 괄호는 A 보다 좋은 fold 수]")
    pv = paired_summary[paired_summary["metric"] == "RMSE"].assign(
        cell=lambda d: d["mean_diff_vs_A"].map("{:+.4f}".format) + " (" + d["folds_better_than_A"].astype(int).astype(str) + "/5)")
    print(pv.pivot_table(index=["combo", "method"], columns="scope", values="cell", aggfunc="first")[list(SCOPES)].to_string())
    print("\n[PV_proxy 가 0 인 sample 수 / 계산 불가 sample 수]")
    print(case_table[["case", "n_zero_pv", "n_missing_pv", "n_missing_pv_1_2_3", "n_missing_pv_4_5_6",
                      "n_missing_pv_other"]].sort_values("case").to_string(index=False))
    print("\n[4-5-6 에서 PV_proxy 와 실제 MRR 의 상관]")
    print(case_table.pivot_table(index="combo", columns="method", values="corr_pv_mrr_4_5_6").round(4).to_string())
    print("\n[effective k]")
    print(case_table[["case", "k_mean", "k_std_ddof1", "k_min", "k_max", "k_cv"]].sort_values("case").to_string(index=False))
    print(f"\n저장: {PROC_DIR.relative_to(ROOT).as_posix()}, {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

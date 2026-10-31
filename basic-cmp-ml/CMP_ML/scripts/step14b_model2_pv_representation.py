"""Step 14 (확장): Model 2 고정 k Preston 모델에서 P, V 표현 방식 비교 (P 후보 12개 x V 후보 6개 = 72조합).

- training 원데이터만 사용한다. test / validation 은 사용하지 않는다. raw 파일은 수정하지 않는다.
- 제외: 연속 완전 중복 1,819행, MRR > 1,000 인 wafer+Stage 4개. 여러 CSV 에 걸친 wafer+Stage 는 합쳐 계산한다.
- fold 는 Step 8 의 같은 5-fold.

P 후보
  개별            P1 PRESSURIZED_CHAMBER_PRESSURE, P2 MAIN_OUTER_AIR_BAG, P3 CENTER_AIR_BAG,
                  P4 RETAINER_RING, P5 RIPPLE_AIR_BAG, P6 EDGE_AIR_BAG
  raw 합성(행별 평균, 가중치 없음)
                  P7  air-bag 4개 (MAIN_OUTER, CENTER, RIPPLE, EDGE)
                  P8  air-bag 4개 + RETAINER_RING (5개)
                  P9  6개 전체
  정규화 합성(행별 평균) - 각 pressure 를 "학습용 4개 fold raw 행의 0 이 아닌 값 중앙값"으로 나눈 뒤 평균
                  P10 air-bag 4개, P11 air-bag 4개 + RETAINER_RING (5개 모두 정규화), P12 6개 전체
                  → 물리 단위 pressure 가 아니라 dimensionless pressure index
V 후보            V1 |WAFER_ROTATION|, V2 |STAGE_ROTATION|, V3 |HEAD_ROTATION|,
                  V4 |WAFER-STAGE|, V5 |WAFER-HEAD|, V6 |STAGE-HEAD|   (V4~V6 은 상대회전 proxy)

PV_proxy = wafer+Stage 안 모든 행의 mean(P(t) x V(t)).  P10~P12 는 정규화 기준이 fold 마다 다르므로
PV_proxy 도 fold 마다 다르다. 저장하는 값은 그 sample 이 평가 fold 에 속했을 때 쓰인 값(OOF PV_proxy)이다.
k: fold 마다 학습용 4개 fold 로만 절편 없는 최소제곱 k = sum(PV x MRR)/sum(PV^2). 평가 fold 에 그대로 적용.
   k 는 물리 단위 Preston coefficient 가 아니라 이 연구 안에서 쓰는 effective k.
출력: data/processed/step14b/, reports/step14b_model2_pv_representation/
실행: python -B scripts/step14b_model2_pv_representation.py
"""
from pathlib import Path

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
STEP14_COMBOS = ROOT / "reports" / "step14_model2_fixed_k" / "combo_results.csv"
PROC_DIR = ROOT / "data" / "processed" / "step14b"
OUT_DIR = ROOT / "reports" / "step14b_model2_pv_representation"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
N_TOP = 10

CHAMBER_P = "PRESSURIZED_CHAMBER_PRESSURE"
RETAINER = "RETAINER_RING_PRESSURE"
AIR_BAGS = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE",
            "EDGE_AIR_BAG_PRESSURE"]
PRESSURES = [CHAMBER_P] + AIR_BAGS + [RETAINER]
SINGLE_P = {"P1": CHAMBER_P, "P2": "MAIN_OUTER_AIR_BAG_PRESSURE", "P3": "CENTER_AIR_BAG_PRESSURE",
            "P4": RETAINER, "P5": "RIPPLE_AIR_BAG_PRESSURE", "P6": "EDGE_AIR_BAG_PRESSURE"}
RAW_COMPOSITE = {"P7": AIR_BAGS, "P8": AIR_BAGS + [RETAINER], "P9": AIR_BAGS + [RETAINER, CHAMBER_P]}
NORM_COMPOSITE = {"P10": AIR_BAGS, "P11": AIR_BAGS + [RETAINER], "P12": AIR_BAGS + [RETAINER, CHAMBER_P]}
P_FAMILY = {**{p: "individual" for p in SINGLE_P}, **{p: "raw_composite" for p in RAW_COMPOSITE},
            **{p: "normalized_composite" for p in NORM_COMPOSITE}}
P_DEFINITION = {
    **{p: c for p, c in SINGLE_P.items()},
    "P7": "mean of 4 air-bag pressures (raw)",
    "P8": "mean of 4 air-bag + retainer ring (raw)",
    "P9": "mean of all 6 pressures (raw)",
    "P10": "mean of 4 air-bag pressures, each divided by train-fold nonzero median (index)",
    "P11": "mean of 4 air-bag + retainer ring, each divided by train-fold nonzero median (index)",
    "P12": "mean of all 6 pressures, each divided by train-fold nonzero median (index)",
}
V_DEFINITION = {"V1": "|WAFER_ROTATION|", "V2": "|STAGE_ROTATION|", "V3": "|HEAD_ROTATION|",
                "V4": "|WAFER_ROTATION - STAGE_ROTATION| (relative-rotation proxy)",
                "V5": "|WAFER_ROTATION - HEAD_ROTATION| (relative-rotation proxy)",
                "V6": "|STAGE_ROTATION - HEAD_ROTATION| (relative-rotation proxy)"}
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
    assert data[cols].notna().all().all()
    assert (data[PRESSURES].to_numpy() >= 0).all(), "pressure 에 음수가 있습니다"
    print(f"사용 행 {len(data):,} / wafer+Stage {data.groupby(KEY).ngroups:,} / "
          f"여러 파일에 걸친 wafer+Stage {int((data.groupby(KEY)['FILE'].nunique() > 1).sum())}개")
    return data


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def scoped(frame):
    out = {}
    for scope, paths in SCOPES.items():
        part = frame if paths is None else frame[frame["CHAMBER_PATH"].isin(paths)]
        out[scope] = metrics(part["MRR_TRUE"], part["MRR_PRED"]) | {"n": len(part)}
    return out


def main():
    data = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    base = step7[KEY + ["CHAMBER_PATH", TARGET]].merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(base) == 1977 and base["FOLD"].notna().all()

    # 행 -> sample(=Step 7 표 행 순서) 인덱스
    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(base["WAFER_ID"], base["STAGE"]))}
    codes = np.array([order[(int(w), str(s))] for w, s in zip(data["WAFER_ID"], data["STAGE"])], dtype=np.int64)
    counts = np.bincount(codes, minlength=len(base)).astype(float)
    assert (counts > 0).all()
    row_fold = base["FOLD"].to_numpy()[codes]  # 각 raw 행이 속한 sample 의 fold
    y = base[TARGET].to_numpy()
    sample_fold = base["FOLD"].to_numpy()
    fold_list = sorted(base["FOLD"].unique())

    rot = {"WAFER": data["WAFER_ROTATION"].to_numpy(), "STAGE": data["STAGE_ROTATION"].to_numpy(),
           "HEAD": data["HEAD_ROTATION"].to_numpy()}
    V_ROW = {"V1": np.abs(rot["WAFER"]), "V2": np.abs(rot["STAGE"]), "V3": np.abs(rot["HEAD"]),
             "V4": np.abs(rot["WAFER"] - rot["STAGE"]), "V5": np.abs(rot["WAFER"] - rot["HEAD"]),
             "V6": np.abs(rot["STAGE"] - rot["HEAD"])}
    press = {c: data[c].to_numpy() for c in PRESSURES}

    P_ROW_FIXED = {p: press[c] for p, c in SINGLE_P.items()}
    for p, cols in RAW_COMPOSITE.items():
        P_ROW_FIXED[p] = np.mean([press[c] for c in cols], axis=0)

    # fold 별 정규화 기준값: 학습용 4개 fold raw 행의 0 이 아닌 값 중앙값
    median_rows, P_ROW_BY_FOLD = [], {}
    for k_fold in fold_list:
        train_rows = row_fold != k_fold
        med = {}
        for c in PRESSURES:
            v = press[c][train_rows]
            nz = v[v != 0]
            assert nz.size > 0
            med[c] = float(np.median(nz))
            median_rows.append(dict(FOLD=int(k_fold), pressure=c, train_nonzero_median=med[c], n_train_nonzero=int(nz.size)))
        scaled = {c: press[c] / med[c] for c in PRESSURES}
        P_ROW_BY_FOLD[k_fold] = {p: np.mean([scaled[c] for c in cols], axis=0) for p, cols in NORM_COMPOSITE.items()}
    medians = pd.DataFrame(median_rows)

    def group_mean(row_values):
        return np.bincount(codes, weights=row_values, minlength=len(base)) / counts

    # 고정 P 후보의 PV_proxy (fold 무관)
    pv_fixed = {(p, v): group_mean(P_ROW_FIXED[p] * V_ROW[v]) for p in P_ROW_FIXED for v in V_ROW}
    # 정규화 P 후보의 PV_proxy (fold 마다 다름)
    pv_by_fold = {k_fold: {(p, v): group_mean(P_ROW_BY_FOLD[k_fold][p] * V_ROW[v])
                           for p in NORM_COMPOSITE for v in V_ROW} for k_fold in fold_list}

    combos = [(p, v) for p in list(SINGLE_P) + list(RAW_COMPOSITE) + list(NORM_COMPOSITE) for v in V_DEFINITION]
    fold_rows, k_rows, pooled_rows, zero_rows, oof_pv_cols, preds = [], [], [], [], {}, {}
    for p, v in combos:
        combo = f"{p}x{v}"
        pred = np.full(len(base), np.nan)
        k_used = np.full(len(base), np.nan)
        oof_pv = np.full(len(base), np.nan)
        for k_fold in fold_list:
            pv = pv_fixed[(p, v)] if p in P_ROW_FIXED else pv_by_fold[k_fold][(p, v)]
            is_test = sample_fold == k_fold
            denom = float(np.sum(pv[~is_test] ** 2))
            assert denom > 0
            k = float(np.sum(pv[~is_test] * y[~is_test])) / denom
            pred[is_test] = k * pv[is_test]
            k_used[is_test] = k
            oof_pv[is_test] = pv[is_test]
            k_rows.append(dict(combo=combo, P=p, V=v, FOLD=int(k_fold), effective_k=k, n_train=int((~is_test).sum())))
            frame = base.loc[is_test, KEY + ["CHAMBER_PATH"]].assign(MRR_TRUE=y[is_test], MRR_PRED=pred[is_test])
            for scope, m in scoped(frame).items():
                fold_rows.append(dict(combo=combo, P=p, V=v, FOLD=int(k_fold), scope=scope, **m))
        oof = base[KEY + ["CHAMBER_PATH", "FOLD"]].assign(COMBO=combo, PV_PROXY=oof_pv, EFFECTIVE_K_USED=k_used,
                                                          MRR_TRUE=y, MRR_PRED=pred, ERROR=pred - y)
        preds[combo] = oof
        oof_pv_cols[f"PV_{p}_{v}"] = oof_pv
        for scope, m in scoped(oof).items():
            assert m["n"] == SCOPE_ROWS[scope]
            pooled_rows += [dict(combo=combo, scope=scope, metric=x, pooled_out_of_fold=m[x]) for x in METRICS]
        zero = oof_pv == 0
        zero_rows.append(dict(combo=combo, P=p, V=v, n_zero_pv=int(zero.sum()),
                              n_zero_pv_1_2_3=int((zero & (base["CHAMBER_PATH"] == "1-2-3")).sum()),
                              n_zero_pv_4_5_6=int((zero & (base["CHAMBER_PATH"] == "4-5-6")).sum()),
                              n_zero_pv_other=int((zero & (base["CHAMBER_PATH"] == "other")).sum())))

    fold_metrics = pd.DataFrame(fold_rows)
    long = fold_metrics.melt(id_vars=["combo", "P", "V", "FOLD", "scope"], value_vars=METRICS, var_name="metric")
    summary = (long.groupby(["combo", "P", "V", "scope", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index()
               .merge(pd.DataFrame(pooled_rows), on=["combo", "scope", "metric"]))
    k_values = pd.DataFrame(k_rows)
    k_summary = k_values.groupby("combo", sort=False)["effective_k"].agg(
        k_mean="mean", k_std_ddof1="std", k_min="min", k_max="max").reset_index()
    k_summary["k_cv"] = k_summary["k_std_ddof1"] / k_summary["k_mean"].abs()
    wide = summary.pivot_table(index="combo", columns=["scope", "metric"], values=["mean", "std_ddof1"])
    wide.columns = [f"{scope}_{metric}_{stat}" for stat, scope, metric in wide.columns]
    combo_table = pd.DataFrame(zero_rows).set_index("combo").join(k_summary.set_index("combo")).join(wide).reset_index()
    combo_table["P_family"] = combo_table["P"].map(P_FAMILY)
    combo_table["P_definition"] = combo_table["P"].map(P_DEFINITION)
    combo_table["V_definition"] = combo_table["V"].map(V_DEFINITION)
    combo_table["rank_by_paths_RMSE"] = combo_table[f"{MAIN_SCOPE}_RMSE_mean"].rank(method="first").astype(int)
    combo_table = combo_table.sort_values("rank_by_paths_RMSE")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pv_table = pd.concat([base, pd.DataFrame(oof_pv_cols)], axis=1)
    pv_table.to_csv(PROC_DIR / "step14b_pv_proxy_oof_table.csv", index=False)
    medians.to_csv(PROC_DIR / "step14b_normalization_medians.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_long.csv", index=False)
    k_values.to_csv(OUT_DIR / "effective_k_by_fold.csv", index=False)
    pd.DataFrame(zero_rows).to_csv(OUT_DIR / "zero_pv_counts.csv", index=False)
    combo_table.to_csv(OUT_DIR / "combo_results.csv", index=False)

    top = combo_table.head(N_TOP)
    family_best = combo_table.sort_values("rank_by_paths_RMSE").groupby("P_family", sort=False).head(1)
    keep = pd.concat([top, family_best]).drop_duplicates("combo")
    pd.concat([preds[c] for c in keep["combo"]], ignore_index=True).to_csv(OUT_DIR / "oof_predictions_top_combos.csv", index=False)
    family_best.to_csv(OUT_DIR / "family_best.csv", index=False)

    # 이전 Step 14 (P1~P6 x V1~V5) 결과와 같은지 확인
    prev = pd.read_csv(STEP14_COMBOS).set_index("combo")
    shared = [c for c in combo_table["combo"] if c in prev.index]
    now = combo_table.set_index("combo").loc[shared]
    cols = [f"{s}_{m}_mean" for s in SCOPES for m in METRICS] + ["k_mean", "n_zero_pv"]
    same = all(np.allclose(now[c].to_numpy(), prev.loc[shared, c].to_numpy(), rtol=0, atol=1e-9) for c in cols)
    print(f"이전 Step 14 와 겹치는 {len(shared)}개 조합의 지표, k, 0 개수가 같은가: {same}")
    assert same and len(shared) == 30

    print(f"\n[상위 {N_TOP}개 조합: 일반 경로 RMSE 평균 기준]")
    show = top.assign(
        RMSE_all=top["all_RMSE_mean"].round(4), RMSE_paths=top[f"{MAIN_SCOPE}_RMSE_mean"].round(4),
        RMSE_123=top["path_1-2-3_RMSE_mean"].round(4), RMSE_456=top["path_4-5-6_RMSE_mean"].round(4),
        R2_paths=top[f"{MAIN_SCOPE}_R2_mean"].round(4), MAE_paths=top[f"{MAIN_SCOPE}_MAE_mean"].round(4),
        k=top["k_mean"], k_std=top["k_std_ddof1"])
    print(show[["rank_by_paths_RMSE", "combo", "P_family", "RMSE_paths", "R2_paths", "MAE_paths", "RMSE_all",
                "RMSE_123", "RMSE_456", "k", "k_std", "n_zero_pv"]].to_string(index=False))
    print("\n상위 10개의 P, V 정의")
    print(top[["combo", "P_definition", "V_definition"]].to_string(index=False))
    print("\n[P 종류별 최고 조합]")
    print(family_best[["P_family", "combo", "rank_by_paths_RMSE", f"{MAIN_SCOPE}_RMSE_mean", f"{MAIN_SCOPE}_R2_mean",
                       "path_1-2-3_RMSE_mean", "path_4-5-6_RMSE_mean", "k_mean", "n_zero_pv"]].round(4).to_string(index=False))

    # 상위 후보들 사이의 차이가 얼마나 작은지 (같은 fold 끼리 뺀 차이)
    piv = fold_metrics[(fold_metrics.scope == MAIN_SCOPE) & (fold_metrics.combo.isin(top["combo"]))] \
        .pivot_table(index="FOLD", columns="combo", values="RMSE")
    first = top["combo"].iloc[0]
    diff_rows = [dict(combo=c, mean_diff_vs_top1=float((piv[c] - piv[first]).mean()),
                      std_diff=float((piv[c] - piv[first]).std()),
                      folds_worse_than_top1=int((piv[c] > piv[first]).sum())) for c in top["combo"].iloc[1:]]
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUT_DIR / "top_combo_paired_difference.csv", index=False)
    print(f"\n[상위 10개: 1위({first}) 대비 같은 fold 일반 경로 RMSE 차이]")
    print(diff.round(4).to_string(index=False))
    print(f"1위 조합의 fold 간 일반 경로 RMSE 표준편차: {piv[first].std():.4f}")
    print("\n[72개 조합 전체 요약]")
    print(combo_table.groupby("P_family")[f"{MAIN_SCOPE}_RMSE_mean"].agg(["count", "min", "max"]).round(4).to_string())
    print("P 별 최소 일반 경로 RMSE:", combo_table.groupby("P")[f"{MAIN_SCOPE}_RMSE_mean"].min().round(4).to_dict())
    print("V 별 최소 일반 경로 RMSE:", combo_table.groupby("V")[f"{MAIN_SCOPE}_RMSE_mean"].min().round(4).to_dict())
    print(f"\n저장: {PROC_DIR.relative_to(ROOT).as_posix()}, {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

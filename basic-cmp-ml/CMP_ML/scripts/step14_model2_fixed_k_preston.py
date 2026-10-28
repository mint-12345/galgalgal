"""Step 14: Model 2 - 고정 k Preston 모델 (MRR = k x P x V), P 후보 6개 x V 후보 5개 = 30개 조합 비교.

- training 원데이터만 읽는다. test / validation 은 사용하지 않는다. raw 파일은 수정하지 않는다.
- Step 7 과 같은 기준으로 제외: 연속 완전 중복 1,819행, MRR > 1,000 인 wafer+Stage 4개.
  여러 CSV 에 걸친 wafer+Stage 는 모든 행을 합쳐 계산한다.
- P 후보: P1 PRESSURIZED_CHAMBER_PRESSURE, P2 MAIN_OUTER_AIR_BAG_PRESSURE, P3 CENTER_AIR_BAG_PRESSURE,
          P4 RETAINER_RING_PRESSURE, P5 RIPPLE_AIR_BAG_PRESSURE, P6 EDGE_AIR_BAG_PRESSURE (서로 섞지 않음)
- V 후보 (rotation 기반 proxy, 실제 wafer-pad 선속도가 아님):
          V1 |WAFER_ROTATION|, V2 |STAGE_ROTATION|, V3 |HEAD_ROTATION|,
          V4 |WAFER_ROTATION - STAGE_ROTATION|, V5 |WAFER_ROTATION - HEAD_ROTATION|  (V4, V5 는 상대회전 proxy)
- PV_proxy = wafer+Stage 안 모든 행의 mean(P(t) x V(t))  (mean(P) x mean(V) 가 아님)
- k: fold 마다 학습용 4개 fold 만으로 절편 없는 최소제곱 k = sum(PV x MRR) / sum(PV^2) 하나를 구해
     평가 fold 에 그대로 적용한다. k 는 원래 물리 단위의 Preston coefficient 가 아닌 effective k 이다.
- 평가 범위: 전체 1,977 / 일반 경로(1-2-3, 4-5-6) 1,973 / 1-2-3 364 / 4-5-6 1,609 의 R², MAE, RMSE
- 최종 P, V 는 선택하지 않는다.
출력: data/processed/step14/step14_pv_proxy_table.csv, reports/step14_model2_fixed_k/
실행: python -B scripts/step14_model2_fixed_k_preston.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
PV_TABLE = ROOT / "data" / "processed" / "step14" / "step14_pv_proxy_table.csv"
OUT_DIR = ROOT / "reports" / "step14_model2_fixed_k"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000

P_CANDIDATES = {
    "P1": "PRESSURIZED_CHAMBER_PRESSURE",
    "P2": "MAIN_OUTER_AIR_BAG_PRESSURE",
    "P3": "CENTER_AIR_BAG_PRESSURE",
    "P4": "RETAINER_RING_PRESSURE",
    "P5": "RIPPLE_AIR_BAG_PRESSURE",
    "P6": "EDGE_AIR_BAG_PRESSURE",
}
V_CANDIDATES = {
    "V1": ("|WAFER_ROTATION|", lambda d: d["WAFER_ROTATION"].abs()),
    "V2": ("|STAGE_ROTATION|", lambda d: d["STAGE_ROTATION"].abs()),
    "V3": ("|HEAD_ROTATION|", lambda d: d["HEAD_ROTATION"].abs()),
    "V4": ("|WAFER_ROTATION - STAGE_ROTATION| (상대회전 proxy)", lambda d: (d["WAFER_ROTATION"] - d["STAGE_ROTATION"]).abs()),
    "V5": ("|WAFER_ROTATION - HEAD_ROTATION| (상대회전 proxy)", lambda d: (d["WAFER_ROTATION"] - d["HEAD_ROTATION"]).abs()),
}
SCOPES = {"all": None, "paths_123_456": ["1-2-3", "4-5-6"], "path_1-2-3": ["1-2-3"], "path_4-5-6": ["4-5-6"]}
SCOPE_ROWS = {"all": 1977, "paths_123_456": 1973, "path_1-2-3": 364, "path_4-5-6": 1609}
METRICS = ["R2", "MAE", "RMSE"]
MAIN_SCOPE = "paths_123_456"
N_TOP = 5


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
    print(f"사용 행 {len(data):,} / wafer+Stage {data.groupby(KEY).ngroups:,} / "
          f"여러 파일에 걸친 wafer+Stage {int((data.groupby(KEY)['FILE'].nunique() > 1).sum())}개")
    return data


def build_pv_table(data):
    keys = [data["WAFER_ID"], data["STAGE"]]
    print("P 후보 최소값:", {p: float(data[c].min()) for p, c in P_CANDIDATES.items()})
    v_values = {v: fn(data) for v, (_, fn) in V_CANDIDATES.items()}
    pv = {}
    for p, pcol in P_CANDIDATES.items():
        for v in V_CANDIDATES:
            pv[f"PV_{p}_{v}"] = (data[pcol] * v_values[v]).groupby(keys).mean()  # mean(P x V)
    table = pd.DataFrame(pv)
    table.index = table.index.set_names(KEY)
    return table


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def scoped(frame):
    out = {}
    for scope, paths in SCOPES.items():
        part = frame if paths is None else frame[frame["CHAMBER_PATH"].isin(paths)]
        out[scope] = metrics(part["MRR_TRUE"], part["MRR_PRED"]) | {"n": len(part)}
    return out


def fit_k(pv, y):
    denom = float(np.sum(pv * pv))
    assert denom > 0, "학습 데이터의 PV_proxy 가 모두 0 입니다"
    return float(np.sum(pv * y)) / denom  # 절편 없는 최소제곱 해


def main():
    data = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    pv = build_pv_table(data)

    # Step 7 표와 같은 wafer+Stage, 같은 순서, 같은 MRR 로 맞춘다
    base = step7[KEY + ["CHAMBER_PATH", TARGET]].merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(base) == 1977 and base["FOLD"].notna().all()
    assert len(pv) == 1977 and pv.index.isin(base.set_index(KEY).index).all()
    table = base.join(pv, on=KEY)
    pv_cols = list(pv.columns)
    assert table[pv_cols].notna().all().all()
    same_mean = np.allclose(data.groupby(KEY)["MAIN_OUTER_AIR_BAG_PRESSURE"].mean().loc[list(map(tuple, step7[KEY].to_numpy()))].to_numpy(),
                            step7["MAIN_OUTER_AIR_BAG_PRESSURE_MEAN"].to_numpy(), rtol=0, atol=1e-9)
    print(f"같은 필터 데이터인지 확인 (MAIN_OUTER_AIR_BAG_PRESSURE 평균이 Step 7 표와 같음): {same_mean}")
    assert same_mean

    PV_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.drop(columns="FOLD").to_csv(PV_TABLE, index=False)

    combos = [(p, v) for p in P_CANDIDATES for v in V_CANDIDATES]
    fold_rows, k_rows, pooled_rows, zero_rows, preds = [], [], [], [], []
    for p, v in combos:
        combo, col = f"{p}x{v}", f"PV_{p}_{v}"
        zero = table[col] == 0
        zero_rows.append(dict(combo=combo, P=P_CANDIDATES[p], V=V_CANDIDATES[v][0], n_zero_pv=int(zero.sum()),
                              n_zero_pv_1_2_3=int((zero & (table["CHAMBER_PATH"] == "1-2-3")).sum()),
                              n_zero_pv_4_5_6=int((zero & (table["CHAMBER_PATH"] == "4-5-6")).sum()),
                              n_zero_pv_other=int((zero & (table["CHAMBER_PATH"] == "other")).sum())))
        pred = pd.Series(np.nan, index=table.index)
        k_used = pd.Series(np.nan, index=table.index)
        for k_fold in sorted(table["FOLD"].unique()):
            train, test = table[table["FOLD"] != k_fold], table[table["FOLD"] == k_fold]
            assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])
            k = fit_k(train[col].to_numpy(), train[TARGET].to_numpy())
            pred.loc[test.index] = k * test[col].to_numpy()
            k_used.loc[test.index] = k
            k_rows.append(dict(combo=combo, FOLD=int(k_fold), effective_k=k, n_train=len(train)))
            frame = test[KEY + ["CHAMBER_PATH"]].assign(MRR_TRUE=test[TARGET], MRR_PRED=pred.loc[test.index])
            for scope, m in scoped(frame).items():
                fold_rows.append(dict(combo=combo, P=p, V=v, FOLD=int(k_fold), scope=scope, **m))
        oof = table[KEY + ["CHAMBER_PATH", "FOLD"]].assign(
            COMBO=combo, PV_PROXY=table[col], EFFECTIVE_K_USED=k_used, MRR_TRUE=table[TARGET],
            MRR_PRED=pred, ERROR=pred - table[TARGET])
        preds.append(oof)
        for scope, m in scoped(oof).items():
            assert m["n"] == SCOPE_ROWS[scope]
            pooled_rows += [dict(combo=combo, scope=scope, metric=x, pooled_out_of_fold=m[x]) for x in METRICS]

    fold_metrics = pd.DataFrame(fold_rows)
    long = fold_metrics.melt(id_vars=["combo", "P", "V", "FOLD", "scope"], value_vars=METRICS, var_name="metric")
    summary = (long.groupby(["combo", "P", "V", "scope", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index()
               .merge(pd.DataFrame(pooled_rows), on=["combo", "scope", "metric"]))
    k_values = pd.DataFrame(k_rows)
    k_summary = k_values.groupby("combo", sort=False)["effective_k"].agg(
        k_mean="mean", k_std_ddof1="std", k_min="min", k_max="max").reset_index()
    k_summary["k_cv"] = k_summary["k_std_ddof1"] / k_summary["k_mean"].abs()
    zero_counts = pd.DataFrame(zero_rows)

    wide = summary.pivot_table(index="combo", columns=["scope", "metric"], values=["mean", "std_ddof1"])
    wide.columns = [f"{scope}_{metric}_{stat}" for stat, scope, metric in wide.columns]
    combo_table = (zero_counts.set_index("combo").join(k_summary.set_index("combo")).join(wide).reset_index())
    combo_table["rank_by_paths_RMSE"] = combo_table[f"{MAIN_SCOPE}_RMSE_mean"].rank(method="first").astype(int)
    combo_table = combo_table.sort_values("rank_by_paths_RMSE")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_long.csv", index=False)
    k_values.to_csv(OUT_DIR / "effective_k_by_fold.csv", index=False)
    zero_counts.to_csv(OUT_DIR / "zero_pv_counts.csv", index=False)
    combo_table.to_csv(OUT_DIR / "combo_results.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT_DIR / "oof_predictions_all_combos.csv", index=False)

    cell = lambda r, s, x: f"{r[f'{s}_{x}_mean']:.4f} ± {r[f'{s}_{x}_std_ddof1']:.4f}"
    top = combo_table.head(N_TOP)
    for metric in METRICS:
        t = pd.DataFrame({s: [cell(r, s, metric) for _, r in top.iterrows()] for s in SCOPES},
                         index=[f"{r.rank_by_paths_RMSE}. {r.combo} ({r.P} x {r.V})" for r in top.itertuples()])
        print(f"\n[상위 {N_TOP}개 {metric}] 5-fold 평균 ± 표준편차")
        print(t.to_string())
    print("\n[상위 5개 effective k 와 PV_proxy = 0 인 sample 수]")
    print(top[["rank_by_paths_RMSE", "combo", "k_mean", "k_std_ddof1", "k_cv", "k_min", "k_max",
               "n_zero_pv", "n_zero_pv_1_2_3", "n_zero_pv_4_5_6", "n_zero_pv_other"]].to_string(index=False))
    print("\n[30개 조합 전체: 일반 경로 RMSE 평균 순]")
    print(combo_table[["rank_by_paths_RMSE", "combo", f"{MAIN_SCOPE}_RMSE_mean", f"{MAIN_SCOPE}_R2_mean",
                       "path_1-2-3_RMSE_mean", "path_4-5-6_RMSE_mean", "all_RMSE_mean", "k_mean", "k_cv",
                       "n_zero_pv"]].round(4).to_string(index=False))
    print(f"\n저장: {PV_TABLE.relative_to(ROOT).as_posix()}, {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

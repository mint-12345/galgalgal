"""Step 16: Model 3 준비 - 공통 데이터(4-5-6 subset)와 k_target, 같은 subset 기준 Model 1 / Model 2 성능.

- 대상: training 1,977개 중 CHAMBER_PATH = 4-5-6 인 1,609개 wafer+Stage 만 사용한다.
  1-2-3 데이터는 오류가 아니며 삭제하지 않는다. 이번 strict Preston / Hybrid 비교에서만 제외한다.
- fold 는 Step 8 의 번호를 그대로 쓴다. test / validation 은 사용하지 않는다. raw 파일은 수정하지 않는다.
- 제외 기준은 앞 단계와 같다: 연속 완전 중복 1,819행, MRR > 1,000 인 wafer+Stage 4개.
- 주 PV proxy (Step 15 의 P4 x V5, nonzero 방식과 같은 정의):
    P = RETAINER_RING_PRESSURE, V = |WAFER_ROTATION - HEAD_ROTATION|
    PV_proxy = P 와 V 가 모두 0 이 아닌 row 만의 mean(P(t) x V(t))
- k_target = 실제 MRR / PV_proxy. 실제 단위의 Preston coefficient 가 아니라
  현재 센서 기반 PV proxy 에 대응하는 effective k target 이다. 이상값은 삭제하지 않는다.
- 같은 1,609개에서 기준 성능을 다시 계산한다. 각 fold 의 학습 데이터도 4-5-6 sample 만 사용한다.
    Model 1 A: RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_leaf=1,
                                     max_features=1.0, random_state=42, n_jobs=1)  (Step 13 과 동일)
    Model 1 B: XGBRegressor(random_state=42, n_jobs=-1)  (Step 10 기본 설정)
    Model 2  : MRR = k x PV_proxy, fold 마다 학습 sample 로만 절편 없는 최소제곱 k 하나
- Model 3 (k(x) 예측) 는 만들지 않는다. 어떤 모델이 최종적으로 좋은지도 결정하지 않는다.
출력: data/processed/step16/, reports/step16_model3_prep/
실행: python -B scripts/step16_model3_preparation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
STEP15_PV = ROOT / "data" / "processed" / "step15" / "step15_pv_proxy_oof_table.csv"
PROC_DIR = ROOT / "data" / "processed" / "step16"
OUT_DIR = ROOT / "reports" / "step16_model3_prep"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
PATH_USED = "4-5-6"
P_COL = "RETAINER_RING_PRESSURE"
CATEGORICAL = ["STAGE", "CHAMBER_PATH"]
SEED = 42
METRICS = ["R2", "MAE", "RMSE"]


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
    print(f"사용 행 {len(data):,} / wafer+Stage {data.groupby(KEY).ngroups:,}")
    return data


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def build_model(name, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", numeric)])
    reg = (RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_leaf=1, max_features=1.0,
                                 random_state=SEED, n_jobs=1) if name == "Model1_RF"
           else XGBRegressor(random_state=SEED, n_jobs=-1))
    return Pipeline([("prep", prep), ("reg", reg)])


def main():
    data = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    base = step7.merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(base) == 1977 and base["FOLD"].notna().all()

    # PV_proxy: P 와 V 가 모두 0 이 아닌 row 만의 mean(P x V)
    p_row = data[P_COL].to_numpy()
    v_row = np.abs(data["WAFER_ROTATION"].to_numpy() - data["HEAD_ROTATION"].to_numpy())
    mask = (p_row != 0) & (v_row != 0)
    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(base["WAFER_ID"], base["STAGE"]))}
    codes = np.array([order[(int(w), str(s))] for w, s in zip(data["WAFER_ID"], data["STAGE"])], dtype=np.int64)
    n_rows = np.bincount(codes, weights=mask.astype(float), minlength=len(base))
    sum_pv = np.bincount(codes, weights=p_row * v_row * mask, minlength=len(base))
    with np.errstate(invalid="ignore", divide="ignore"):
        pv_all = np.where(n_rows > 0, sum_pv / np.where(n_rows > 0, n_rows, 1.0), np.nan)
    base["PV_PROXY"] = pv_all
    base["N_ROWS_USED_FOR_PV"] = n_rows.astype(int)

    # Step 15 의 P4 x V5 nonzero (방식 B) 와 같은 값인지 확인
    prev = pd.read_csv(STEP15_PV, dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    assert prev[KEY].astype(str).equals(base[KEY].astype(str))
    b = prev["PV_P4_V5_B"].to_numpy()
    same = np.array_equal(np.isnan(b), np.isnan(pv_all)) and np.allclose(b[~np.isnan(b)], pv_all[~np.isnan(pv_all)],
                                                                        rtol=0, atol=0)
    print(f"PV_proxy 가 Step 15 의 P4xV5 nonzero 값과 완전히 같은가: {same}")
    assert same

    sub = base[base["CHAMBER_PATH"] == PATH_USED].reset_index(drop=True)
    print(f"\n[대상] CHAMBER_PATH={PATH_USED} sample {len(sub):,}개 "
          f"(전체 {len(base):,}개 중, 1-2-3 {int((base['CHAMBER_PATH'] == '1-2-3').sum())}개와 "
          f"other {int((base['CHAMBER_PATH'] == 'other').sum())}개는 이번 비교에서만 제외, 파일에서는 그대로 보존)")
    assert len(sub) == 1609
    print(f"PV_proxy 를 계산할 수 없는 sample: {int(sub['PV_PROXY'].isna().sum())}개 / "
          f"PV_proxy 가 0 인 sample: {int((sub['PV_PROXY'] == 0).sum())}개")
    assert sub["PV_PROXY"].notna().all() and (sub["PV_PROXY"] > 0).all()
    print("fold 별 sample 수:", sub["FOLD"].value_counts().sort_index().to_dict())
    print("PV 계산에 쓰인 row 수 요약:", sub["N_ROWS_USED_FOR_PV"].describe().round(1).to_dict())

    # k_target
    sub["K_TARGET"] = sub[TARGET] / sub["PV_PROXY"]
    k = sub["K_TARGET"]
    q = k.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    stats_rows = [dict(group="all", n=len(k), min=k.min(), p01=q[0.01], p05=q[0.05], median=k.median(),
                       mean=k.mean(), p95=q[0.95], p99=q[0.99], max=k.max(), std_ddof1=k.std(),
                       max_over_median=k.max() / k.median(), min_over_median=k.min() / k.median())]
    for stage, g in sub.groupby("STAGE"):
        kk = g["K_TARGET"]
        qq = kk.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        stats_rows.append(dict(group=f"STAGE {stage}", n=len(kk), min=kk.min(), p01=qq[0.01], p05=qq[0.05],
                               median=kk.median(), mean=kk.mean(), p95=qq[0.95], p99=qq[0.99], max=kk.max(),
                               std_ddof1=kk.std(), max_over_median=kk.max() / kk.median(),
                               min_over_median=kk.min() / kk.median()))
    k_stats = pd.DataFrame(stats_rows)
    print("\n[k_target 분포] effective k target (실제 단위의 Preston coefficient 아님)")
    print(k_stats.round(6).to_string(index=False))
    far = sub.loc[(k > q[0.99]) | (k < q[0.01]), KEY + ["FOLD", "PV_PROXY", TARGET, "K_TARGET"]].sort_values("K_TARGET")
    print(f"\n1% 지점 아래 또는 99% 지점 위인 sample: {len(far)}개 (삭제하지 않음)")
    print(far.head(8).round(6).to_string(index=False))
    print("...")
    print(far.tail(8).round(6).to_string(index=False))

    # 같은 1,609개에서 Model 1 / Model 2 기준 성능 (학습도 4-5-6 만)
    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET)]
    numeric = [c for c in features if c not in CATEGORICAL]
    assert len(features) == 32
    work = sub.copy()
    for c in CATEGORICAL:
        work[c] = work[c].astype(str)

    fold_rows, k_rows, pred_cols = [], [], {}
    for name in ["Model1_RF", "Model1_XGB", "Model2_fixed_k"]:
        pred = pd.Series(np.nan, index=work.index)
        for f in sorted(work["FOLD"].unique()):
            train, test = work[work["FOLD"] != f], work[work["FOLD"] == f]
            assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])
            assert (train["CHAMBER_PATH"] == PATH_USED).all() and (test["CHAMBER_PATH"] == PATH_USED).all()
            if name == "Model2_fixed_k":
                pv, y = train["PV_PROXY"].to_numpy(), train[TARGET].to_numpy()
                k_fold = float(np.sum(pv * y) / np.sum(pv * pv))
                pred.loc[test.index] = k_fold * test["PV_PROXY"].to_numpy()
                k_rows.append(dict(model=name, FOLD=int(f), effective_k=k_fold, n_train=len(train)))
            else:
                model = build_model(name, numeric).fit(train[features], train[TARGET])
                pred.loc[test.index] = model.predict(test[features])
                if f == sorted(work["FOLD"].unique())[0]:
                    n_in = model[:-1].transform(train[features].iloc[:1]).shape[1]
                    print(f"{name}: 모델 입력 열 {n_in}개 (CHAMBER_PATH 는 이 subset 에서 값이 하나뿐이라 열이 생기지 않음)")
            fold_rows.append(dict(model=name, FOLD=int(f), n_train=len(train), n_test=len(test),
                                  **metrics(test[TARGET], pred.loc[test.index])))
        assert pred.notna().all()
        pred_cols[name] = pred

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (fold_metrics.melt(id_vars=["model", "FOLD"], value_vars=METRICS, var_name="metric")
               .groupby(["model", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index())
    oof = work[KEY + ["CHAMBER_PATH", "FOLD", "PV_PROXY", TARGET, "K_TARGET"]].copy()
    for name, pred in pred_cols.items():
        oof[f"PRED_{name}"] = pred
        oof[f"ERROR_{name}"] = pred - work[TARGET]
    pooled = pd.DataFrame([dict(model=name, scope="pooled_out_of_fold", **metrics(oof[TARGET], oof[f"PRED_{name}"]))
                           for name in pred_cols])

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep_table = work[KEY + ["CHAMBER_PATH", "FOLD", "PV_PROXY", TARGET, "K_TARGET"]]
    prep_table.to_csv(PROC_DIR / "step16_model3_prep_table.csv", index=False)
    k_stats.to_csv(OUT_DIR / "k_target_stats.csv", index=False)
    far.to_csv(OUT_DIR / "k_target_tail_samples.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pooled.to_csv(OUT_DIR / "pooled_metrics.csv", index=False)
    pd.DataFrame(k_rows).to_csv(OUT_DIR / "model2_effective_k_by_fold.csv", index=False)
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    print("\n[같은 1,609개 기준 성능: fold 별]")
    print(fold_metrics.round(4).to_string(index=False))
    print("\n[5-fold 평균 ± 표준편차]")
    view = summary.assign(cell=summary["mean"].map("{:.4f}".format) + " ± " + summary["std_ddof1"].map("{:.4f}".format))
    print(view.pivot(index="model", columns="metric", values="cell")[METRICS].to_string())
    print("\n[5개 fold 예측을 모아 계산한 값]")
    print(pooled.round(4).to_string(index=False))
    print("\n[Model 2 의 fold 별 effective k]")
    print(pd.DataFrame(k_rows).round(8).to_string(index=False))
    print(f"\n저장: {PROC_DIR.relative_to(ROOT).as_posix()}, {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

"""Step 13: Model 1 (Pure ML) - Baseline feature 로 RF / XGBoost 설정을 제한적으로 비교.

- 입력 표: data/processed/step7_wafer_stage_table_v1.csv (feature 32개 그대로), fold: step8_cv_folds_v1.csv 그대로
- test / validation 은 사용하지 않는다. 최종 모델은 선택하지 않는다.

RF  (54개): n_estimators {300, 600} x max_depth {None, 10, 20} x min_samples_leaf {1, 2, 4}
            x max_features {"sqrt", 0.7, 1.0}
            나머지는 Step 10 과 같다: one-hot + 숫자 그대로 → RandomForestRegressor(random_state=42, n_jobs=1)
            설정 x fold 작업을 여러 프로세스로 동시에 실행하지만, 각 학습은 단일 스레드 + 고정 seed 라 결과는 같다.
XGB (18개): n_estimators {300, 600} x max_depth {2, 3, 4} x learning_rate {0.03, 0.05, 0.1}
            고정: subsample=0.9, colsample_bytree=0.9
            나머지는 Step 10 과 같다: one-hot + 숫자 그대로 → XGBRegressor(random_state=42, n_jobs=-1), 순서대로 실행
기준: Step 10 기본 설정(RF_default_step10, XGB_default_step10)을 같은 코드로 다시 계산하고 Step 10 예측과 같은지 확인

평가 범위: 전체 1,977 / 일반 경로(1-2-3, 4-5-6) 1,973 / 1-2-3 364 / 4-5-6 1,609 의 R², MAE, RMSE
주 비교 기준(정렬용): 일반 경로 1,973행의 5-fold 평균 RMSE
출력: reports/step13_model1_tuning/
실행: python -B scripts/step13_model1_tuning.py
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 60)

ROOT = Path(__file__).resolve().parents[1]
TABLE_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
FOLD_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
STEP10_DIR = ROOT / "reports" / "step10_model1_compare"
OUT_DIR = ROOT / "reports" / "step13_model1_tuning"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
CATEGORICAL = ["STAGE", "CHAMBER_PATH"]
SEED = 42
N_TOP = 5
MAIN_SCOPE = "paths_123_456"
SCOPES = {"all": None, "paths_123_456": ["1-2-3", "4-5-6"], "path_1-2-3": ["1-2-3"], "path_4-5-6": ["4-5-6"]}
SCOPE_ROWS = {"all": 1977, "paths_123_456": 1973, "path_1-2-3": 364, "path_4-5-6": 1609}
METRICS = ["R2", "MAE", "RMSE"]

RF_GRID = dict(n_estimators=[300, 600], max_depth=[None, 10, 20], min_samples_leaf=[1, 2, 4],
               max_features=["sqrt", 0.7, 1.0])
XGB_GRID = dict(n_estimators=[300, 600], max_depth=[2, 3, 4], learning_rate=[0.03, 0.05, 0.1])
XGB_FIXED = dict(subsample=0.9, colsample_bytree=0.9)


def load():
    table = pd.read_csv(TABLE_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    data = table.merge(folds[KEY + ["FOLD"]], on=KEY, how="left", validate="one_to_one")
    assert len(data) == 1977 and data["FOLD"].notna().all()
    for c in CATEGORICAL:
        data[c] = data[c].astype(str)
    features = [c for c in table.columns if c not in ("WAFER_ID", TARGET)]
    assert len(features) == 32
    numeric = [c for c in features if c not in CATEGORICAL]
    return data, features, numeric


def make_pipeline(model_name, params, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", numeric)])
    if model_name == "RF":
        reg = RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
    else:
        reg = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    return Pipeline([("prep", prep), ("reg", reg)])


def config_id(model_name, params):
    if not params:
        return f"{model_name}_default_step10"
    short = {"n_estimators": "n", "max_depth": "d", "min_samples_leaf": "leaf", "max_features": "mf",
             "learning_rate": "lr"}
    # XGB 의 subsample / colsample_bytree 는 모든 조정 설정에서 같으므로 이름에는 넣지 않는다 (params 열에는 기록)
    varying = {k: v for k, v in params.items() if not (model_name == "XGB" and k in XGB_FIXED)}
    return model_name + "_" + "_".join(f"{short[k]}{v}" for k, v in varying.items())


def expand(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def fit_predict_fold(model_name, params, numeric, train_X, train_y, test_X):
    t0 = time.perf_counter()
    pred = make_pipeline(model_name, params, numeric).fit(train_X, train_y).predict(test_X)
    return pred, time.perf_counter() - t0


def metrics(y_true, y_pred):
    return dict(R2=r2_score(y_true, y_pred), MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def scoped(frame):
    out = {}
    for scope, paths in SCOPES.items():
        part = frame if paths is None else frame[frame["CHAMBER_PATH"].isin(paths)]
        out[scope] = metrics(part["MRR_TRUE"], part["MRR_PRED"]) | {"n": len(part)}
    return out


def run_all(data, features, numeric):
    configs = [("RF", {})] + [("RF", p) for p in expand(RF_GRID)] + \
              [("XGB", {})] + [("XGB", XGB_FIXED | p) for p in expand(XGB_GRID)]
    folds = sorted(data["FOLD"].unique())
    split = {k: (data[data["FOLD"] != k], data[data["FOLD"] == k]) for k in folds}
    for k, (train, test) in split.items():
        assert not set(train["WAFER_ID"]) & set(test["WAFER_ID"])

    tasks = [(m, p, k) for m, p in configs for k in folds]
    rf_tasks = [t for t in tasks if t[0] == "RF"]
    xgb_tasks = [t for t in tasks if t[0] == "XGB"]
    print(f"설정 수: RF {len(rf_tasks) // len(folds)} (기본 포함), XGB {len(xgb_tasks) // len(folds)} (기본 포함) "
          f"/ 학습 작업 {len(tasks)}개")

    t0 = time.perf_counter()
    rf_out = Parallel(n_jobs=-1, backend="loky")(
        delayed(fit_predict_fold)(m, p, numeric, split[k][0][features], split[k][0][TARGET], split[k][1][features])
        for m, p, k in rf_tasks)
    print(f"RF 완료: {time.perf_counter() - t0:.0f}초")
    t0 = time.perf_counter()
    xgb_out = [fit_predict_fold(m, p, numeric, split[k][0][features], split[k][0][TARGET], split[k][1][features])
               for m, p, k in xgb_tasks]
    print(f"XGB 완료: {time.perf_counter() - t0:.0f}초")

    preds, timing = {}, []
    for (m, p, k), (pred, sec) in zip(rf_tasks + xgb_tasks, rf_out + xgb_out):
        cid = config_id(m, p)
        preds.setdefault((m, cid, json_params(p)), pd.Series(np.nan, index=data.index))
        preds[(m, cid, json_params(p))].loc[split[k][1].index] = pred
        timing.append(dict(model=m, config_id=cid, FOLD=int(k), fit_predict_seconds=round(sec, 3)))
    return preds, pd.DataFrame(timing)


def json_params(p):
    return ";".join(f"{k}={v}" for k, v in p.items()) if p else "Step 10 기본값"


OOF_ALL_FILE = OUT_DIR / "oof_all_configs.csv"
PARAMS_FILE = OUT_DIR / "config_params.csv"


def save_predictions(data, preds, timing):
    """학습 직후 모든 설정의 OOF 예측을 저장 (이후 집계 단계에서 오류가 나도 다시 학습하지 않도록)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide = data[KEY + ["CHAMBER_PATH", "FOLD"]].assign(MRR_TRUE=data[TARGET])
    wide = pd.concat([wide, pd.DataFrame({cid: pred for (_, cid, _), pred in preds.items()})], axis=1)
    wide.to_csv(OOF_ALL_FILE, index=False)
    pd.DataFrame([dict(model=m, config_id=cid, params=ptxt) for (m, cid, ptxt) in preds]).to_csv(PARAMS_FILE, index=False)
    timing.to_csv(OUT_DIR / "run_time_reference.csv", index=False)


def load_predictions(data):
    wide = pd.read_csv(OOF_ALL_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"}, float_precision="round_trip")
    # 값 기준으로 행 순서가 같은지 확인 (STAGE 는 읽는 방식에 따라 dtype 이 string / object 로 달라지므로 값만 비교)
    assert np.array_equal(wide["WAFER_ID"].to_numpy(), data["WAFER_ID"].to_numpy())
    assert (wide["STAGE"].astype(str).to_numpy() == data["STAGE"].astype(str).to_numpy()).all()
    params = pd.read_csv(PARAMS_FILE)
    return {(r.model, r.config_id, r.params): pd.Series(wide[r.config_id].to_numpy(), index=data.index)
            for r in params.itertuples()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="이미 저장된 oof_all_configs.csv 로 집계만 다시 한다 (학습하지 않음)")
    args = parser.parse_args()

    data, features, numeric = load()
    if args.reuse_predictions:
        preds = load_predictions(data)
        print(f"저장된 예측 사용: {len(preds)}개 설정")
    else:
        preds, timing = run_all(data, features, numeric)
        save_predictions(data, preds, timing)
    base_cols = data[KEY + ["CHAMBER_PATH", "FOLD"]].assign(MRR_TRUE=data[TARGET])

    for m in ("RF", "XGB"):  # 기본 설정이 Step 10 과 같은지 확인
        pred = preds[(m, f"{m}_default_step10", "Step 10 기본값")]
        ref = pd.read_csv(STEP10_DIR / f"oof_predictions_{m}.csv", float_precision="round_trip")
        same = np.array_equal(pred.to_numpy(), ref["MRR_PRED"].to_numpy())
        print(f"{m} 기본 설정 예측이 Step 10 저장 예측과 완전히 같은가: {same}")
        assert same

    fold_rows, other_rows, pooled_rows, params_rows = [], [], [], []
    oofs = {}
    for (m, cid, ptxt), pred in preds.items():
        assert pred.notna().all()
        oof = base_cols.assign(MRR_PRED=pred, ERROR=pred - base_cols["MRR_TRUE"])
        oofs[cid] = oof
        params_rows.append(dict(model=m, config_id=cid, params=ptxt))
        for k, part in oof.groupby("FOLD"):
            for scope, v in scoped(part).items():
                fold_rows.append(dict(model=m, config_id=cid, FOLD=int(k), scope=scope, **v))
        for scope, v in scoped(oof).items():
            assert v["n"] == SCOPE_ROWS[scope]
            pooled_rows += [dict(model=m, config_id=cid, scope=scope, metric=x, pooled_out_of_fold=v[x]) for x in METRICS]
        for r in oof[oof["CHAMBER_PATH"] == "other"].itertuples():
            other_rows.append(dict(model=m, config_id=cid, WAFER_ID=r.WAFER_ID, STAGE=r.STAGE, FOLD=r.FOLD,
                                   MRR_TRUE=r.MRR_TRUE, MRR_PRED=r.MRR_PRED, ABS_ERROR=abs(r.ERROR)))

    fold_metrics = pd.DataFrame(fold_rows)
    long = fold_metrics.melt(id_vars=["model", "config_id", "FOLD", "scope"], value_vars=METRICS, var_name="metric")
    summary = (long.groupby(["model", "config_id", "scope", "metric"], sort=False)["value"]
               .agg(mean="mean", std_ddof1="std", min="min", max="max").reset_index()
               .merge(pd.DataFrame(pooled_rows), on=["model", "config_id", "scope", "metric"]))

    # 같은 fold 기준 기본 설정(Step 10)과의 차이. better = RMSE/MAE 는 작을 때, R² 는 클 때
    base = long[long["config_id"].str.endswith("_default_step10")].drop(columns="config_id") \
        .rename(columns={"value": "default"})
    paired = long.merge(base, on=["model", "FOLD", "scope", "metric"])
    paired["diff"] = paired["value"] - paired["default"]
    paired["better"] = np.where(paired["metric"] == "R2", paired["diff"] > 0, paired["diff"] < 0)
    paired_summary = (paired.groupby(["model", "config_id", "scope", "metric"], sort=False)
                      .agg(mean_diff=("diff", "mean"), std_diff_ddof1=("diff", "std"),
                           folds_better=("better", "sum")).reset_index())

    wide = summary.pivot_table(index=["model", "config_id"], columns=["scope", "metric"], values=["mean", "std_ddof1"])
    wide.columns = [f"{scope}_{metric}_{stat}" for stat, scope, metric in wide.columns]
    pw = paired_summary.pivot_table(index=["model", "config_id"], columns=["scope", "metric"],
                                    values=["mean_diff", "folds_better"])
    pw.columns = [f"{scope}_{metric}_{stat}_vs_default" for stat, scope, metric in pw.columns]
    config_table = pd.DataFrame(params_rows).set_index(["model", "config_id"]).join(wide).join(pw).reset_index()
    config_table["rank_in_model_by_paths_RMSE"] = (config_table.groupby("model")[f"{MAIN_SCOPE}_RMSE_mean"]
                                                   .rank(method="first").astype(int))
    config_table = config_table.sort_values(["model", "rank_in_model_by_paths_RMSE"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "oof_top").mkdir(exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_long.csv", index=False)
    paired_summary.to_csv(OUT_DIR / "paired_difference_vs_default.csv", index=False)
    config_table.to_csv(OUT_DIR / "config_results.csv", index=False)
    pd.DataFrame(other_rows).to_csv(OUT_DIR / "other_path_rows.csv", index=False)

    for m in ("RF", "XGB"):
        top = config_table[(config_table["model"] == m) & ~config_table["config_id"].str.endswith("_default_step10")]
        top = top.nsmallest(N_TOP, f"{MAIN_SCOPE}_RMSE_mean")
        for cid in top["config_id"]:
            oofs[cid].to_csv(OUT_DIR / "oof_top" / f"oof_{cid}.csv", index=False)

        show = config_table[config_table["model"] == m]
        default = show[show["config_id"].str.endswith("_default_step10")]
        rows = pd.concat([top, default])
        cell = lambda r, s, x: f"{r[f'{s}_{x}_mean']:.4f} ± {r[f'{s}_{x}_std_ddof1']:.4f}"
        print(f"\n===== {m}: 일반 경로 RMSE 상위 {N_TOP}개 + Step 10 기본 설정 =====")
        for metric in METRICS:
            t = pd.DataFrame({s: [cell(r, s, metric) for _, r in rows.iterrows()] for s in SCOPES},
                             index=[f"{r['rank_in_model_by_paths_RMSE']:>2}. {r['config_id']}" for _, r in rows.iterrows()])
            print(f"\n[{metric}] 평균 ± 표준편차")
            print(t.to_string())
        print("\n[기본 설정 대비 같은 fold RMSE 차이 평균 (기본보다 좋아진 fold 수)]")
        t = pd.DataFrame({s: [f"{r[f'{s}_RMSE_mean_diff_vs_default']:+.4f} ({int(r[f'{s}_RMSE_folds_better_vs_default'])}/5)"
                              for _, r in top.iterrows()] for s in SCOPES}, index=top["config_id"])
        print(t.to_string())
        print(f"\n[{m} 전체 설정의 일반 경로 RMSE 평균 분포]",
              show[f"{MAIN_SCOPE}_RMSE_mean"].describe().round(4).to_dict())

    print(f"\n저장 폴더: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

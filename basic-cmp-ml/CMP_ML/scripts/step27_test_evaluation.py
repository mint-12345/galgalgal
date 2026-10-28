"""Step 27: 고정된 세 모델의 최종 external test 평가.

- test 데이터는 이번 단계에서 처음 사용한다. 결과를 보고 feature, P/V, preprocessing,
  모델 구조, hyperparameter 를 바꾸지 않는다. 평가만 한다.
- 학습 데이터는 Step 26 과 완전히 같다. training 4-5-6 의 1,609개 sample 뿐이다.
  validation 을 학습에 추가하지 않는다. validation 과 test 를 같은 모델로 평가하기 위해서다.
    Model 1 Pure ML : XGBRegressor(900, depth 4, lr 0.05, mcw 3, subsample 0.9, colsample 0.9, seed 42)
    Model 2 Fixed-k : MRR = k x PV_proxy, 절편 없이 k = sum(PV*MRR)/sum(PV^2), training 으로만 계산
    Model 3 Hybrid  : MRR = k(x) x PV_proxy, k(x) = XGBRegressor(900, depth 2, lr 0.1, mcw 1, ...)
                      k 표준화 평균/표준편차도 training 1,609개에서만 계산한다.
- test 의 MRR 이나 k_target 은 학습, scaling, preprocessing 에 쓰지 않는다. 평가에만 쓴다.
- test 전처리는 Step 26 의 validation 처리와 완전히 같다.
  validation 표를 같은 코드로 다시 만들어 Step 26 저장본과 일치하는지 대조한다.
- 4-5-6 이 아닌 sample 은 삭제하지 않고 개수와 구조만 보고한다.
  PV_proxy 를 계산할 수 없는 sample 에 억지로 예측을 만들지 않는다.
출력: reports/step27_test/ (CSV, 그림)
실행: python -B scripts/step27_test_evaluation.py
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
RAW = ROOT / "data" / "raw" / "PHM2016"
DS = RAW / "2016 PHM DATA CHALLENGE CMP DATA SET"
TEST_DIR = DS / "CMP-data" / "test"
VAL_DIR = RAW / "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET" / "validation"
ANSWERS = RAW / "PHM16TestValidationAnswers" / "PHM16TestValidationAnswers"
TEST_MRR = ANSWERS / "orig_CMP-test-removalrate.csv"
VAL_MRR = ANSWERS / "orig_CMP-validation-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
PREP_FILE = ROOT / "data" / "processed" / "step16" / "step16_model3_prep_table.csv"
STEP20_OOF = ROOT / "reports" / "step20_hybrid_xgb_tuning" / "oof_top" / "oof_n900_d2_lr0.1_mcw1.csv"
STEP24_OOF = ROOT / "reports" / "step24_pure_ml_tuning" / "oof_top" / "oof_n900_d4_lr0.05_mcw3.csv"
STEP26_DIR = ROOT / "reports" / "step26_validation"
STEP26_TABLE = ROOT / "data" / "processed" / "step26" / "step26_validation_wafer_stage_table.csv"
OUT_DIR = ROOT / "reports" / "step27_test"
FIG_DIR = OUT_DIR / "figures"
TEST_TABLE = ROOT / "data" / "processed" / "step27" / "step27_test_wafer_stage_table.csv"

KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
SEED = 42
FIXED = dict(subsample=0.9, colsample_bytree=0.9)
PURE_PARAMS = dict(n_estimators=900, max_depth=4, learning_rate=0.05, min_child_weight=3)
HYBRID_PARAMS = dict(n_estimators=900, max_depth=2, learning_rate=0.1, min_child_weight=1)
PRESSURES = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
             "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE"]
USAGE_MEAN = ["USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE", "USAGE_OF_DRESSER_TABLE",
              "USAGE_OF_BACKING_FILM"]
MEAN_STD = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
            "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE",
            "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
            "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION", "DRESSING_WATER_STATUS"]
MODELS = ["Pure ML", "Fixed-k Preston", "Hybrid"]
PRED_COL = {"Pure ML": "PRED_PURE", "Fixed-k Preston": "PRED_FIXED_K", "Hybrid": "PRED_HYBRID"}
DIST_VARS = [TARGET, "PV_PROXY"] + [f"{c}_MEAN" for c in USAGE_MEAN] + \
            ["RETAINER_RING_PRESSURE_MEAN", "PRESSURIZED_CHAMBER_PRESSURE_MEAN",
             "HEAD_ROTATION_MEAN", "WAFER_ROTATION_MEAN", "STAGE_ROTATION_MEAN"]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES1, SERIES2, SERIES3 = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834", "#898781"
MODEL_COLOR = {"Pure ML": SERIES2, "Hybrid": SERIES1, "Fixed-k Preston": SERIES3}
SET_COLOR = {"training": SERIES3, "validation": SERIES1, "test": SERIES2}


# ---------------------------------------------------------------- 전처리 (Step 7 / 26 과 동일)
def load_raw(folder, pattern):
    frames = []
    for f in sorted(folder.glob(pattern)):
        df = pd.read_csv(f, dtype={"WAFER_ID": "int64", "STAGE": "string"})
        if df.empty:
            continue
        df["FILE"] = f.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    cols = [c for c in raw.columns if c != "FILE"]
    same_prev = raw[cols].eq(raw[cols].shift()).all(axis=1) & raw["FILE"].eq(raw["FILE"].shift())
    return raw[~same_prev], int(same_prev.sum()), len(raw), len(frames)


def chamber_path(chambers):
    order = [int(c) for c in chambers[chambers.ne(chambers.shift())]]
    if order == [1, 2, 3]:
        return "1-2-3"
    if order == [4, 5, 6]:
        return "4-5-6"
    return "other"


def build_table(data, mrr):
    data = data.sort_values(KEY + ["TIMESTAMP"], kind="stable")
    g = data.groupby(KEY, sort=False)
    table = pd.DataFrame({"CHAMBER_PATH": g["CHAMBER"].agg(chamber_path)})
    for c in USAGE_MEAN:
        table[f"{c}_MEAN"] = g[c].mean()
    for c in MEAN_STD:
        table[f"{c}_MEAN"] = g[c].mean()
        table[f"{c}_STD"] = g[c].std(ddof=0)
    first_time = g["TIMESTAMP"].min()
    table = table.join(mrr.set_index(KEY)[TARGET], how="left")
    table = table.loc[first_time.sort_values(kind="stable").index].reset_index()
    table["STAGE"] = table["STAGE"].astype(str)
    return table


def pv_proxy(data, table):
    order = {(int(w), str(s)): i for i, (w, s) in enumerate(zip(table["WAFER_ID"], table["STAGE"]))}
    keys = list(zip(data["WAFER_ID"].astype(int), data["STAGE"].astype(str)))
    keep = np.array([k in order for k in keys])
    rows = data[keep]
    codes = np.array([order[k] for k, ok in zip(keys, keep) if ok], dtype=np.int64)
    p_row = np.mean([rows[c].to_numpy() for c in PRESSURES], axis=0)
    v_row = np.abs(rows["HEAD_ROTATION"].to_numpy())
    mask = (p_row != 0) & (v_row != 0)
    n_used = np.bincount(codes, weights=mask.astype(float), minlength=len(table))
    sum_pv = np.bincount(codes, weights=p_row * v_row * mask, minlength=len(table))
    pv = np.where(n_used > 0, sum_pv / np.where(n_used > 0, n_used, 1), np.nan)
    return pv, n_used.astype(int)


def prepare(folder, pattern, mrr_file):
    raw, dups, n_rows, n_files = load_raw(folder, pattern)
    mrr = pd.read_csv(mrr_file, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                      float_precision="round_trip")
    mrr["STAGE"] = mrr["STAGE"].astype(str)
    table = build_table(raw, mrr)
    pv, n_used = pv_proxy(raw, table)
    table["PV_PROXY"] = pv
    table["N_PV_ROWS"] = n_used
    info = dict(n_files_in_folder=len(list(folder.glob(pattern))), n_files_read=n_files,
                n_raw_rows=n_rows, n_duplicate_rows=dups, n_wafer_stage=len(table),
                n_labels=len(mrr), n_missing_label=int(table[TARGET].isna().sum()))
    return table, info


def build(params, numeric):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["STAGE"]),
        ("num", "passthrough", numeric)])
    return Pipeline([("prep", prep),
                     ("reg", XGBRegressor(random_state=SEED, n_jobs=-1, **FIXED, **params))])


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=len(y), R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))))


def structure_of(table, name):
    s = (table.groupby("CHAMBER_PATH")
         .agg(n_samples=("WAFER_ID", "size"), n_wafers=("WAFER_ID", "nunique"),
              n_stage_A=("STAGE", lambda x: int((x == "A").sum())),
              n_with_PV=("PV_PROXY", lambda x: int(x.notna().sum())),
              n_missing_MRR=(TARGET, lambda x: int(x.isna().sum())),
              mean_MRR=(TARGET, "mean"), min_MRR=(TARGET, "min"), max_MRR=(TARGET, "max"))
         .reset_index())
    s.insert(0, "dataset", name)
    s["n_stage_B"] = s.n_samples - s.n_stage_A
    return s


# ---------------------------------------------------------------- 실행
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_TABLE.parent.mkdir(parents=True, exist_ok=True)
    checks = []

    # ---- training (확정된 표) ----
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                        float_precision="round_trip")
    prep_tbl = pd.read_csv(PREP_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                           float_precision="round_trip")
    hyb_oof = pd.read_csv(STEP20_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                          float_precision="round_trip")
    pure_oof = pd.read_csv(STEP24_OOF, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                           float_precision="round_trip")
    for d in (step7, prep_tbl, hyb_oof, pure_oof):
        d["STAGE"] = d["STAGE"].astype(str)
    train = prep_tbl[KEY + ["FOLD", TARGET]].merge(step7.drop(columns=["CHAMBER_PATH", TARGET]),
                                                   on=KEY, how="left", validate="one_to_one")
    train = train.merge(hyb_oof[KEY + ["PV_PROXY", "K_TARGET"]], on=KEY, how="left",
                        validate="one_to_one")
    features = [c for c in step7.columns if c not in ("WAFER_ID", TARGET, "CHAMBER_PATH")]
    numeric = [c for c in features if c != "STAGE"]
    assert len(train) == 1609 and len(features) == 31 and train.notna().all().all()

    # ---- validation 을 같은 코드로 다시 만들어 Step 26 과 대조 ----
    val, val_info = prepare(VAL_DIR, "CMP-validation-*.csv", VAL_MRR)
    saved_val = pd.read_csv(STEP26_TABLE, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                            float_precision="round_trip")
    saved_val["STAGE"] = saved_val["STAGE"].astype(str)
    same_val = bool(val.shape == saved_val.shape and
                    np.array_equal(val.WAFER_ID.to_numpy(), saved_val.WAFER_ID.to_numpy()) and
                    np.array_equal(val.STAGE.to_numpy(), saved_val.STAGE.to_numpy()) and
                    np.allclose(val.select_dtypes("number").to_numpy(),
                                saved_val.select_dtypes("number").to_numpy(),
                                rtol=0, atol=0, equal_nan=True))

    # ---- test 전처리 ----
    test, test_info = prepare(TEST_DIR, "CMP-test-*.csv", TEST_MRR)
    test.to_csv(TEST_TABLE, index=False)

    root_test_mrr = pd.read_csv(DS / "CMP-test-removalrate.csv",
                                dtype={"WAFER_ID": "int64", "STAGE": "string"},
                                float_precision="round_trip")
    answers_test_mrr = pd.read_csv(TEST_MRR, dtype={"WAFER_ID": "int64", "STAGE": "string"},
                                   float_precision="round_trip")
    same_labels = bool(root_test_mrr.equals(answers_test_mrr))

    structure = pd.concat([structure_of(val, "validation"), structure_of(test, "test")],
                          ignore_index=True)
    test456 = test[test.CHAMBER_PATH == "4-5-6"].reset_index(drop=True)
    val456 = val[val.CHAMBER_PATH == "4-5-6"].reset_index(drop=True)

    checks += [
        dict(item="test 원본 파일 수 (폴더 전체)", value=str(test_info["n_files_in_folder"])),
        dict(item="test 원본 파일 수 (내용이 있는 파일)", value=str(test_info["n_files_read"])),
        dict(item="test 원본 행 수", value=f"{test_info['n_raw_rows']:,}"),
        dict(item="연속 완전중복으로 제외한 행 수", value=f"{test_info['n_duplicate_rows']:,}"),
        dict(item="test wafer+Stage 수", value=str(test_info["n_wafer_stage"])),
        dict(item="test 정답 행 수", value=str(test_info["n_labels"])),
        dict(item="MRR 이 비어 있는 test wafer+Stage 수", value=str(test_info["n_missing_label"])),
        dict(item="test 4-5-6 sample 수", value=str(len(test456))),
        dict(item="4-5-6 중 PV_proxy 를 계산할 수 없는 sample 수",
             value=str(int(test456.PV_PROXY.isna().sum()))),
        dict(item=f"4-5-6 중 MRR > {MRR_LIMIT:,} 인 sample 수 (제외하지 않음)",
             value=str(int((test456[TARGET] > MRR_LIMIT).sum()))),
        dict(item="두 test 정답 파일이 동일한가 (데이터셋 루트 / 정답 폴더)", value=str(same_labels)),
        dict(item="validation 표를 다시 만들어 Step 26 저장본과 같은가", value=str(same_val)),
    ]

    # ---- 세 모델 학습 (training 1,609 만 사용) ----
    pure = build(PURE_PARAMS, numeric).fit(train[features], train[TARGET])
    k_fixed = float((train.PV_PROXY * train[TARGET]).sum() / (train.PV_PROXY ** 2).sum())
    mean_k, std_k = float(train.K_TARGET.mean()), float(train.K_TARGET.std(ddof=0))
    hybrid = build(HYBRID_PARAMS, numeric).fit(train[features], (train.K_TARGET - mean_k) / std_k)

    s26_checks = pd.read_csv(STEP26_DIR / "consistency_checks.csv")
    s26_map = {r["item"]: str(r["value"]) for _, r in s26_checks.iterrows()}
    checks += [
        dict(item="fixed k (training 4-5-6 전체)", value=f"{k_fixed:.10g}"),
        dict(item="fixed k 가 Step 26 값과 같은가",
             value=str(f"{k_fixed:.10g}" == s26_map["fixed k (training 4-5-6 전체)"])),
        dict(item="k 표준화 평균/표준편차가 Step 26 값과 같은가",
             value=str(f"{mean_k:.10g}" == s26_map["k 표준화 평균 (training)"] and
                       f"{std_k:.10g}" == s26_map["k 표준화 표준편차 (training)"])),
    ]

    def predict_on(table456):
        out = table456[KEY + ["CHAMBER_PATH", TARGET, "PV_PROXY"]].copy()
        out["PRED_PURE"] = np.asarray(pure.predict(table456[features]), dtype=np.float64)
        out["PRED_FIXED_K"] = k_fixed * table456.PV_PROXY
        out["K_PRED"] = np.asarray(hybrid.predict(table456[features]) * std_k + mean_k,
                                   dtype=np.float64)
        out["PRED_HYBRID"] = out.K_PRED * table456.PV_PROXY
        for m in MODELS:
            out[f"ERROR_{PRED_COL[m]}"] = out[PRED_COL[m]] - out[TARGET]
        return out

    test_pred = predict_on(test456)
    val_pred = predict_on(val456)
    test_common = test_pred[test_pred.PV_PROXY.notna()].reset_index(drop=True)
    val_common = val_pred[val_pred.PV_PROXY.notna()].reset_index(drop=True)

    # Step 26 의 validation 결과와 같은지 대조
    s26_val = pd.read_csv(STEP26_DIR / "validation_metrics.csv", float_precision="round_trip")
    s26_val = s26_val[s26_val.scope.str.startswith("validation 4-5-6 (three")].set_index("model")
    val_now = {m: metrics(val_common[TARGET], val_common[PRED_COL[m]]) for m in MODELS}
    checks.append(dict(item="validation 성능이 Step 26 결과와 같은가",
                       value=str(bool(all(np.isclose(val_now[m]["RMSE"], s26_val.loc[m, "RMSE"],
                                                     rtol=0, atol=1e-9) for m in MODELS)))))

    # ---- test 성능 ----
    rows = [dict(model=m, scope="test 4-5-6", **metrics(test_common[TARGET], test_common[PRED_COL[m]]))
            for m in MODELS]
    test_metrics = pd.DataFrame(rows)
    stage_rows = []
    for m in MODELS:
        for st in ("A", "B"):
            sub = test_common[test_common.STAGE == st]
            if len(sub) > 1:
                stage_rows.append(dict(model=m, STAGE=st, **metrics(sub[TARGET], sub[PRED_COL[m]])))
    test_stage = pd.DataFrame(stage_rows)

    # ---- training OOF 기준 ----
    k_oof = np.empty(len(train))
    for f in sorted(train.FOLD.unique()):
        te = train.FOLD.to_numpy() == f
        tr_ = ~te
        k_oof[te] = float((train.PV_PROXY[tr_] * train[TARGET][tr_]).sum() /
                          (train.PV_PROXY[tr_] ** 2).sum())
    oof = train[KEY + [TARGET]].copy()
    oof["PRED_PURE"] = pure_oof.set_index(KEY).loc[
        list(zip(train.WAFER_ID, train.STAGE)), "MRR_PRED"].to_numpy()
    oof["PRED_HYBRID"] = hyb_oof.set_index(KEY).loc[
        list(zip(train.WAFER_ID, train.STAGE)), "MRR_PRED"].to_numpy()
    oof["PRED_FIXED_K"] = k_oof * train.PV_PROXY
    train_metrics = pd.DataFrame([dict(model=m, scope="training 5-fold OOF",
                                       **metrics(oof[TARGET], oof[PRED_COL[m]])) for m in MODELS])

    # ---- 세 데이터 비교표 ----
    t = train_metrics.set_index("model")
    v = pd.DataFrame([dict(model=m, **val_now[m]) for m in MODELS]).set_index("model")
    s = test_metrics.set_index("model")
    three_way = pd.DataFrame([dict(
        model=m,
        training_n=int(t.loc[m, "n"]), training_R2=float(t.loc[m, "R2"]),
        training_MAE=float(t.loc[m, "MAE"]), training_RMSE=float(t.loc[m, "RMSE"]),
        validation_n=int(v.loc[m, "n"]), validation_R2=float(v.loc[m, "R2"]),
        validation_MAE=float(v.loc[m, "MAE"]), validation_RMSE=float(v.loc[m, "RMSE"]),
        test_n=int(s.loc[m, "n"]), test_R2=float(s.loc[m, "R2"]),
        test_MAE=float(s.loc[m, "MAE"]), test_RMSE=float(s.loc[m, "RMSE"]),
        val_minus_train_RMSE=float(v.loc[m, "RMSE"] - t.loc[m, "RMSE"]),
        test_minus_train_RMSE=float(s.loc[m, "RMSE"] - t.loc[m, "RMSE"]),
        test_minus_val_RMSE=float(s.loc[m, "RMSE"] - v.loc[m, "RMSE"])) for m in MODELS])

    # ---- 분포 비교 (세 데이터의 4-5-6) ----
    sets = {"training": train, "validation": val456, "test": test456}
    dist = []
    for var in DIST_VARS:
        row = dict(variable=var)
        for name, df in sets.items():
            a = df[var].dropna()
            row[f"{name}_mean"] = float(a.mean())
            row[f"{name}_std"] = float(a.std(ddof=0))
            row[f"{name}_median"] = float(a.median())
        for name in ("validation", "test"):
            a, b = train[var].dropna(), sets[name][var].dropna()
            sd = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2)
            row[f"{name}_standardized_gap"] = float((b.mean() - a.mean()) / sd) if sd else np.nan
        dist.append(row)
    dist = pd.DataFrame(dist).sort_values("test_standardized_gap", key=abs, ascending=False)
    dist = dist.reset_index(drop=True)

    stage_ratio = pd.DataFrame([
        dict(dataset=name, n=len(df), stage_A=int((df.STAGE == "A").sum()),
             stage_B=int((df.STAGE == "B").sum()), stage_B_ratio=float((df.STAGE == "B").mean()))
        for name, df in sets.items()])

    # ---- 데이터 독립성 ----
    def keys_of(df):
        return set(map(tuple, df[KEY].itertuples(index=False)))

    pairs = [("training", train, "validation", val), ("training", train, "test", test),
             ("validation", val, "test", test)]
    independence = pd.DataFrame([
        dict(pair=f"{n1} vs {n2}", n1=len(d1), n2=len(d2),
             shared_wafer_stage_keys=len(keys_of(d1) & keys_of(d2)),
             shared_wafer_ids=len(set(d1.WAFER_ID) & set(d2.WAFER_ID)))
        for n1, d1, n2, d2 in pairs])
    checks.append(dict(item="세 데이터 사이에 겹치는 wafer+Stage 키가 있는가",
                       value=str(bool(independence.shared_wafer_stage_keys.sum() > 0))))
    checks.append(dict(item="test 예측 = k_pred x PV_proxy 인가",
                       value=str(bool(np.allclose(test_common.PRED_HYBRID,
                                                  test_common.K_PRED * test_common.PV_PROXY,
                                                  rtol=0, atol=1e-12)))))
    checks = pd.DataFrame(checks)

    # ---- 저장 ----
    structure.to_csv(OUT_DIR / "dataset_structure.csv", index=False)
    test_metrics.to_csv(OUT_DIR / "test_metrics.csv", index=False)
    test_stage.to_csv(OUT_DIR / "test_stage_metrics.csv", index=False)
    train_metrics.to_csv(OUT_DIR / "training_oof_reference.csv", index=False)
    three_way.to_csv(OUT_DIR / "three_way_comparison.csv", index=False)
    dist.to_csv(OUT_DIR / "distribution_comparison.csv", index=False)
    stage_ratio.to_csv(OUT_DIR / "stage_ratio_comparison.csv", index=False)
    independence.to_csv(OUT_DIR / "data_independence.csv", index=False)
    test_pred.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    checks.to_csv(OUT_DIR / "consistency_checks.csv", index=False)

    # ---- 그림 ----
    def style(ax, axis="y"):
        ax.set_facecolor(SURFACE)
        ax.grid(axis=axis, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)

    def legend(ax, loc="upper left"):
        leg = ax.legend(frameon=False, fontsize=8.5, loc=loc)
        for x in leg.get_texts():
            x.set_color(INK2)

    tw = three_way.set_index("model")

    # 그림 1: training / validation / test RMSE
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    xx = np.arange(len(MODELS))
    ymax = max(tw[[f"{s_}_RMSE" for s_ in ("training", "validation", "test")]].max()) * 1.25
    for i, s_ in enumerate(("training", "validation", "test")):
        off = (i - 1) * 0.28
        vals = [float(tw.loc[m, f"{s_}_RMSE"]) for m in MODELS]
        label = {"training": "Training 5-fold OOF", "validation": "Validation (Step 26)",
                 "test": "Test (final)"}[s_]
        ax.bar(xx + off, vals, width=0.26, color=SET_COLOR[s_], label=label)
        for x, v_ in zip(xx + off, vals):
            ax.text(x, v_ + ymax * 0.015, f"{v_:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xx, MODELS)
    style(ax)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_xlabel("Model (frozen before validation and test were opened)", fontsize=9, color=INK2)
    ax.set_title("Training, validation and test RMSE of the three frozen models\n"
                 "all three fitted once on the same 1,609 training samples (CHAMBER_PATH = 4-5-6)",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    legend(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_train_val_test_rmse.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 2: test 성능 (R2 / RMSE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, col, title, fmt in [(axes[0], "R2", "Test R²  (higher is better)", "{:.3f}"),
                                (axes[1], "RMSE", "Test RMSE  (lower is better)", "{:.2f}")]:
        vals = [float(s.loc[m, col]) for m in MODELS]
        ax.bar(np.arange(len(MODELS)), vals, width=0.55, color=[MODEL_COLOR[m] for m in MODELS])
        lo = min(0, min(vals) * 1.2)
        hi = max(vals) * 1.25 if max(vals) > 0 else 1
        for x, v_ in zip(np.arange(len(MODELS)), vals):
            ax.text(x, v_ + (hi - lo) * 0.02, fmt.format(v_), ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(np.arange(len(MODELS)), MODELS, fontsize=8)
        style(ax)
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Model", fontsize=8.5, color=INK2)
    fig.suptitle(f"Final held-out test performance ({int(s.loc['Pure ML', 'n'])} samples, "
                 "CHAMBER_PATH = 4-5-6)", fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig2_test_metrics.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 3: test 실제 vs 예측
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    lo = float(min(test_common[TARGET].min(), test_common.PRED_PURE.min(),
                   test_common.PRED_HYBRID.min())) - 2
    hi = float(max(test_common[TARGET].max(), test_common.PRED_PURE.max(),
                   test_common.PRED_HYBRID.max())) + 2
    for ax, m in zip(axes, ["Pure ML", "Hybrid"]):
        ax.scatter(test_common[TARGET], test_common[PRED_COL[m]], s=12, color=MODEL_COLOR[m],
                   alpha=0.55, edgecolors=SURFACE, linewidths=0.4)
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1.0, ls="--")
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual MRR (test)", fontsize=9, color=INK2)
        ax.set_ylabel("Predicted MRR", fontsize=9, color=INK2)
        ax.set_title(f"{m}   RMSE = {float(s.loc[m, 'RMSE']):.2f}", fontsize=10, color=INK, loc="left")
    fig.suptitle("Test: actual vs predicted MRR (dashed line = perfect prediction)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig3_test_actual_vs_predicted.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 4: fixed-k -> Hybrid -> Pure ML
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    order = ["Fixed-k Preston", "Hybrid", "Pure ML"]
    vals = [float(s.loc[m, "RMSE"]) for m in order]
    yy = np.arange(len(order))[::-1]
    ax.barh(yy, vals, height=0.5, color=[MODEL_COLOR[m] for m in order])
    for y_, v_, m in zip(yy, vals, order):
        ax.text(v_ + max(vals) * 0.012, y_, f"{v_:.2f}", va="center", fontsize=9, color=INK2)
    drop1 = vals[0] - vals[1]
    drop2 = vals[1] - vals[2]
    ax.annotate(f"k → k(x):  RMSE −{drop1:.2f}", xy=(vals[1] + max(vals) * 0.20, yy[1] + 0.42),
                fontsize=9, color=INK2)
    ax.annotate(f"k(x) × PV → direct MRR:  RMSE −{drop2:.2f}",
                xy=(vals[2] + max(vals) * 0.20, yy[2] + 0.42), fontsize=9, color=INK2)
    ax.set_yticks(yy, order)
    style(ax, axis="x")
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("Test RMSE of predicted MRR  (lower is better)", fontsize=9, color=INK2)
    ax.set_title("From a constant Preston k to a learned k(x) to direct prediction\n"
                 "held-out test set, three models frozen before the test data was opened",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_fixedk_hybrid_pure.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 그림 5: 세 데이터 분포 비교
    show = [TARGET, "PV_PROXY", "USAGE_OF_DRESSER_MEAN", "USAGE_OF_DRESSER_TABLE_MEAN",
            "RETAINER_RING_PRESSURE_MEAN", "HEAD_ROTATION_MEAN"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, var in zip(axes.ravel(), show):
        arrays = {n: df[var].dropna() for n, df in sets.items()}
        bins = np.histogram_bin_edges(np.concatenate(list(arrays.values())), bins=30)
        for n, a in arrays.items():
            ax.hist(a, bins=bins, color=SET_COLOR[n], alpha=0.45, label=n, density=True)
        style(ax)
        g = float(dist[dist.variable == var].test_standardized_gap.iloc[0])
        ax.set_title(f"{var}\ntest vs training standardized mean gap = {g:+.2f}",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Value", fontsize=8, color=INK2)
        ax.set_ylabel("Density", fontsize=8, color=INK2)
    legend(axes[0, 0], loc="upper right")
    fig.suptitle("Training, validation and test distributions (CHAMBER_PATH = 4-5-6)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig5_distribution_comparison.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---- 출력 ----
    print("[일관성 확인]")
    print(checks.to_string(index=False))
    print("\n[데이터 구조]")
    print(structure.to_string(index=False))
    print("\n[test 성능]")
    print(test_metrics.to_string(index=False))
    print("\n[test Stage 별]")
    print(test_stage.to_string(index=False))
    print("\n[training / validation / test 비교]")
    print(three_way.to_string(index=False))
    print("\n[Stage 비율]")
    print(stage_ratio.to_string(index=False))
    print("\n[데이터 독립성]")
    print(independence.to_string(index=False))
    print("\n[분포 비교]")
    print(dist.to_string(index=False))
    print(f"\n저장: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

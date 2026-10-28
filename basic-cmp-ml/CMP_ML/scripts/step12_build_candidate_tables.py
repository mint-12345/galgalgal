"""Step 12-1: Model 1 두 번째 feature 후보 표 만들기 (Candidate 5~7).

- training 원데이터만 읽는다. test / validation 은 열지 않는다. raw 파일은 수정하지 않는다.
- Step 7 과 같은 기준으로 제외: 연속 완전 중복 1,819행, MRR > 1,000 인 wafer+Stage 4개
- 여러 CSV 에 걸친 wafer+Stage 는 모든 행을 합쳐 계산한다.
- Baseline 은 Step 7 표 그대로이며, 같은 필터 데이터로 다시 계산해 Step 7 값과 같은지 확인한다.

대상 변수 10개: 압력 6개, SLURRY_FLOW_LINE_A, SLURRY_FLOW_LINE_C, WAFER_ROTATION, STAGE_ROTATION
Candidate 5 (nonzero-state): 변수마다 0 이 아닌 행의 비율, 0 이 아닌 행만의 평균 (20열).
                             0 이 아닌 행이 없으면 평균은 빈 값(NaN). 공정 상태나 연마로 해석하지 않는다.
Candidate 6 (distribution) : 변수마다 25% 지점, 중앙값, 75% 지점 (30열). pandas 기본(선형 보간) 분위수.
Candidate 7 (combined)     : Candidate 5 + 6 추가 열 모두 (50열).
출력: data/processed/step12/
실행: python -B scripts/step12_build_candidate_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
OUT_DIR = ROOT / "data" / "processed" / "step12"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000

TARGET_VARS = [
    "PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE",
    "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_C", "WAFER_ROTATION", "STAGE_ROTATION",
]
USAGE = ["USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE", "USAGE_OF_DRESSER_TABLE", "USAGE_OF_BACKING_FILM"]
BASE_MEAN_STD = [
    "PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE",
    "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION", "DRESSING_WATER_STATUS",
]
CANDIDATE_FILES = {
    "C5_nonzero": "step12_candidate5_nonzero.csv",
    "C6_distribution": "step12_candidate6_distribution.csv",
    "C7_combined": "step12_candidate7_combined.csv",
}


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
    data = data[~drop].sort_values(KEY + ["TIMESTAMP"], kind="stable")
    assert int(same_prev.sum()) == 1819 and len(big) == 4
    assert data[cols].notna().all().all()
    print(f"사용 행 {len(data):,} / wafer+Stage {data.groupby(KEY).ngroups:,} / "
          f"여러 파일에 걸친 wafer+Stage {int((data.groupby(KEY)['FILE'].nunique() > 1).sum())}개")
    return data


def check_baseline(data, step7):
    g = data.groupby(KEY)
    re = pd.DataFrame({f"{c}_MEAN": g[c].mean() for c in USAGE})
    for c in BASE_MEAN_STD:
        re[f"{c}_MEAN"] = g[c].mean()
        re[f"{c}_STD"] = g[c].std(ddof=0)
    s7 = step7.set_index(KEY)
    re = re.loc[s7.index, [c for c in s7.columns if c not in ("CHAMBER_PATH", TARGET)]]
    same = np.allclose(re.to_numpy(), s7[re.columns].to_numpy(), rtol=0, atol=1e-9)
    print(f"같은 필터 데이터로 다시 계산한 Baseline 숫자 {re.shape[1]}열이 Step 7 표와 같은가: {same}")
    assert same


def candidate5(data):
    nonzero = data[TARGET_VARS].ne(0)
    keys = [data["WAFER_ID"], data["STAGE"]]
    share = nonzero.groupby(keys).mean()
    nz_mean = data[TARGET_VARS].where(nonzero).groupby(keys).mean()  # 0 이 아닌 행이 없으면 NaN
    out = pd.DataFrame(index=share.index)
    for v in TARGET_VARS:
        out[f"{v}_NONZERO_ROW_SHARE"] = share[v]
        out[f"{v}_NONZERO_MEAN"] = nz_mean[v]
    return out


def candidate6(data):
    q = data.groupby(KEY)[TARGET_VARS].quantile([0.25, 0.5, 0.75])
    q.index = q.index.set_names(KEY + ["q"])
    q = q.unstack("q")
    out = pd.DataFrame(index=q.index)
    for v in TARGET_VARS:
        out[f"{v}_P25"] = q[(v, 0.25)]
        out[f"{v}_MEDIAN"] = q[(v, 0.5)]
        out[f"{v}_P75"] = q[(v, 0.75)]
    return out


def main():
    data = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    assert len(step7) == 1977
    check_baseline(data, step7)

    extras = {"C5_nonzero": candidate5(data), "C6_distribution": candidate6(data)}
    extras["C7_combined"] = extras["C5_nonzero"].join(extras["C6_distribution"])

    base = step7.set_index(KEY)
    base_features = [c for c in base.columns if c != TARGET]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dictionary = [dict(feature_set="Baseline", column=c, part="baseline") for c in ["STAGE"] + base_features]
    for name, extra in extras.items():
        assert len(extra) == len(base) and extra.index.isin(base.index).all()
        table = base[base_features].join(extra.loc[base.index]).join(base[[TARGET]]).reset_index()
        table.to_csv(OUT_DIR / CANDIDATE_FILES[name], index=False)
        dictionary += [dict(feature_set=name, column=c, part="added") for c in extra.columns]
        na = table[extra.columns].isna()
        print(f"\n[{name}] {CANDIDATE_FILES[name]}: {table.shape[0]:,}행 x {table.shape[1]}열 "
              f"(feature = Baseline 32 + 추가 {extra.shape[1]} = {32 + extra.shape[1]})")
        print(f"  추가 열 중 빈 값이 있는 열 {int(na.any().sum())}개, 빈 칸 {int(na.to_numpy().sum()):,}개")
    pd.DataFrame(dictionary).to_csv(OUT_DIR / "step12_feature_columns.csv", index=False)

    c5, c6 = extras["C5_nonzero"], extras["C6_distribution"]
    path = base["CHAMBER_PATH"]
    print("\n[C5] 0 이 아닌 행이 하나도 없어 NONZERO_MEAN 이 빈 값인 wafer+Stage 수 (CHAMBER_PATH 별)")
    empty = c5[[f"{v}_NONZERO_MEAN" for v in TARGET_VARS]].isna()
    print(empty.groupby(path.loc[empty.index]).sum().T.rename(index=lambda s: s.replace("_NONZERO_MEAN", "")).to_string())
    print("\n[C5] NONZERO_ROW_SHARE 요약")
    print(c5[[f"{v}_NONZERO_ROW_SHARE" for v in TARGET_VARS]].describe().round(3).T.to_string())
    print("\n[C6] 분위수 요약")
    print(c6.describe().round(3).T.to_string())
    print(f"\n저장 폴더: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

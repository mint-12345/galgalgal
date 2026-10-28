"""Step 11-1: Model 1 feature 후보 표 만들기 (Baseline + Candidate 1~4).

- training 원데이터만 읽는다. test / validation 은 열지 않는다. raw 파일은 수정하지 않는다.
- Step 7 과 같은 기준으로 제외: 연속 완전 중복 1,819행, MRR > 1,000 인 wafer+Stage 4개
- 여러 CSV 에 걸친 wafer+Stage 는 모든 파일을 파일 순서대로 합친 뒤 TIMESTAMP 순으로 정렬해 계산한다
  (같은 TIMESTAMP 는 원래 행 순서 유지).
- Baseline 은 Step 7 표 그대로이며, 같은 필터 데이터로 다시 계산해 Step 7 값과 같은지 확인한다.

Candidate 1 (Chamber 정보): Chamber 1~6 별 평균 (압력 6개, SLURRY A/B/C, WAFER/STAGE ROTATION = 11개 x 6 = 66열)
                            + Chamber 1~6 별 행 비율 (6열).
                            지나지 않은 Chamber 의 평균은 0 이 아니라 빈 값(NaN)으로 둔다.
                            행 비율은 해당 Chamber 기록 행이 없으면 실제로 0 이다.
Candidate 2 (air-bag 상태): 네 air-bag pressure 가 모두 0 이 아닌 행의 비율 (1열). 연마 여부로 해석하지 않는다.
Candidate 3 (usage 상태) : usage 4개의 시간순 마지막 값, 마지막 값 - 처음 값 (8열).
Candidate 4 (Combined)   : Candidate 1, 2, 3 추가 열 모두 (81열).
출력: data/processed/step11/
실행: python -B scripts/step11_build_candidate_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
STEP7_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
OUT_DIR = ROOT / "data" / "processed" / "step11"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
MRR_LIMIT = 1000
CHAMBERS = [1, 2, 3, 4, 5, 6]

PRESSURE = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
            "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE"]
SLURRY = ["SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C"]
CHAMBER_VARS = PRESSURE + SLURRY + ["WAFER_ROTATION", "STAGE_ROTATION"]
AIR_BAG = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE",
           "EDGE_AIR_BAG_PRESSURE"]
USAGE = ["USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE", "USAGE_OF_DRESSER_TABLE", "USAGE_OF_BACKING_FILM"]
BASE_MEAN_STD = PRESSURE + SLURRY + ["WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION", "DRESSING_WATER_STATUS"]

CANDIDATE_FILES = {
    "C1_chamber": "step11_candidate1_chamber.csv",
    "C2_airbag": "step11_candidate2_airbag.csv",
    "C3_usage": "step11_candidate3_usage.csv",
    "C4_combined": "step11_candidate4_combined.csv",
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
    assert data[cols].notna().all().all(), "raw 공정 데이터에 빈 값이 있습니다"
    data = data.assign(CHAMBER=data["CHAMBER"].astype(int))
    n_multi_file = int((data.groupby(KEY)["FILE"].nunique() > 1).sum())
    print(f"사용 행 {len(data):,} / wafer+Stage {data.groupby(KEY).ngroups:,} / 여러 파일에 걸친 wafer+Stage {n_multi_file}개")
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


def candidate1(data):
    g = data.groupby(KEY + ["CHAMBER"])[CHAMBER_VARS].mean().unstack("CHAMBER")
    cols = {}
    for c in CHAMBERS:
        for v in CHAMBER_VARS:
            cols[f"{v}_CH{c}_MEAN"] = g[(v, c)] if (v, c) in g.columns else np.nan
    means = pd.DataFrame(cols)
    share = (data.groupby(KEY)["CHAMBER"].value_counts(normalize=True).unstack(fill_value=0.0)
             .reindex(columns=CHAMBERS, fill_value=0.0))
    share.columns = [f"CHAMBER_{c}_ROW_SHARE" for c in CHAMBERS]
    return means.join(share)


def candidate2(data):
    on = data[AIR_BAG].ne(0).all(axis=1)
    return on.groupby([data["WAFER_ID"], data["STAGE"]]).mean().rename("AIR_BAG_ALL_NONZERO_ROW_SHARE").to_frame()


def candidate3(data):
    g = data.groupby(KEY)[USAGE]
    first, last = g.first(), g.last()  # data 는 KEY + TIMESTAMP 순으로 정렬되어 있음 (빈 값 없음)
    out = pd.DataFrame(index=first.index)
    for c in USAGE:
        out[f"{c}_LAST"] = last[c]
        out[f"{c}_LAST_MINUS_FIRST"] = last[c] - first[c]
    return out


def main():
    data = load_filtered()
    step7 = pd.read_csv(STEP7_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    assert len(step7) == 1977
    check_baseline(data, step7)

    extras = {"C1_chamber": candidate1(data), "C2_airbag": candidate2(data), "C3_usage": candidate3(data)}
    extras["C4_combined"] = extras["C1_chamber"].join(extras["C2_airbag"]).join(extras["C3_usage"])

    base = step7.set_index(KEY)
    base_features = [c for c in base.columns if c != TARGET]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dictionary = [dict(feature_set="Baseline", column=c, part="baseline") for c in ["STAGE"] + base_features]
    for name, extra in extras.items():
        assert extra.index.isin(base.index).all() and len(extra) == len(base)
        table = base[base_features].join(extra.loc[base.index]).join(base[[TARGET]]).reset_index()
        table.to_csv(OUT_DIR / CANDIDATE_FILES[name], index=False)
        dictionary += [dict(feature_set=name, column=c, part="added") for c in extra.columns]

        na = table[extra.columns].isna()
        print(f"\n[{name}] {CANDIDATE_FILES[name]}: {table.shape[0]:,}행 x {table.shape[1]}열 "
              f"(feature = Baseline 32 + 추가 {extra.shape[1]} = {32 + extra.shape[1]})")
        print(f"  추가 열 중 빈 값이 있는 열 {int(na.any().sum())}개, 빈 칸 {int(na.to_numpy().sum()):,}개")
        if na.any().any():
            per_path = na.any(axis=1).groupby(table["CHAMBER_PATH"]).mean().round(3).to_dict()
            print("  빈 값이 하나 이상 있는 행 비율 (CHAMBER_PATH 별):", per_path)
            by_ch = {c: int(table[f"PRESSURIZED_CHAMBER_PRESSURE_CH{c}_MEAN"].isna().sum()) for c in CHAMBERS}
            print("  Chamber 별 평균이 빈 값인 행 수:", by_ch)
    pd.DataFrame(dictionary).to_csv(OUT_DIR / "step11_feature_columns.csv", index=False)

    print("\n[추가 값 요약]")
    print(extras["C1_chamber"][[f"CHAMBER_{c}_ROW_SHARE" for c in CHAMBERS]].describe().round(3).to_string())
    print(extras["C2_airbag"].describe().round(3).to_string())
    c3 = extras["C3_usage"]
    print(c3.describe().round(3).T.to_string())
    print("usage 마지막-처음 값이 음수인 wafer+Stage 수:",
          {c: int((c3[f"{c}_LAST_MINUS_FIRST"] < 0).sum()) for c in USAGE})
    print(f"\n저장 폴더: {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

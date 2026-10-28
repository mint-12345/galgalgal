"""Step 7: training 데이터로 wafer+Stage 하나당 한 행인 첫 번째 표 만들기.

- training 원데이터만 읽는다. test / validation 은 열지 않는다. raw 파일은 수정하지 않는다.
- 분석에서만 제외: (1) 같은 파일에서 바로 윗줄과 25개 컬럼이 모두 같은 1,819행
                   (2) training MRR 이 1,000 을 넘는 wafer+Stage 4개
- MACHINE_ID, MACHINE_DATA, USAGE_OF_MEMBRANE, USAGE_OF_PRESSURIZED_SHEET 는 사용하지 않는다.
- 요청한 열 외의 feature 는 추가하지 않는다. 모델은 만들지 않는다.
출력: data/processed/step7_wafer_stage_table_v1.csv
실행: python -B scripts/step7_build_wafer_stage_table.py
"""
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
OUT_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
KEY = ["WAFER_ID", "STAGE"]
MRR_LIMIT = 1000

USAGE_MEAN = ["USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE", "USAGE_OF_DRESSER_TABLE", "USAGE_OF_BACKING_FILM"]
MEAN_STD = [
    "PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE",
    "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION",
    "DRESSING_WATER_STATUS",
]
TARGET = "AVG_REMOVAL_RATE"


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
    print(f"원본 행 {len(raw):,} → 연속 완전 중복 {int(same_prev.sum()):,}행 제외 → "
          f"MRR>{MRR_LIMIT:,} 묶음 {len(big)}개({int(drop.sum()):,}행) 제외 → 사용 행 {len(data):,}")
    # 파일 순서를 유지한 채 TIMESTAMP 순으로 정렬 (같은 TIMESTAMP 는 원래 행 순서 유지)
    return data.sort_values(KEY + ["TIMESTAMP"], kind="stable"), mrr


def chamber_path(chambers):
    """시간순 Chamber 기록에서 연속된 같은 번호를 하나로 줄여 경로를 만든다."""
    order = [int(c) for c in chambers[chambers.ne(chambers.shift())]]
    if order == [1, 2, 3]:
        return "1-2-3"
    if order == [4, 5, 6]:
        return "4-5-6"
    return "other"


def build(data, mrr):
    g = data.groupby(KEY, sort=False)
    table = pd.DataFrame({"CHAMBER_PATH": g["CHAMBER"].agg(chamber_path)})
    first_time = g["TIMESTAMP"].min()

    for c in USAGE_MEAN:
        table[f"{c}_MEAN"] = g[c].mean()
    for c in MEAN_STD:
        table[f"{c}_MEAN"] = g[c].mean()
        table[f"{c}_STD"] = g[c].std(ddof=0)  # 해당 wafer+Stage 에 기록된 행 전체의 표준편차

    table = table.join(mrr.set_index(KEY)[TARGET], how="left")
    # 행 순서: 각 wafer+Stage 의 첫 기록 시각 순 (열로는 추가하지 않음)
    table = table.loc[first_time.sort_values(kind="stable").index].reset_index()
    return table


def main():
    data, mrr = load_filtered()
    table = build(data, mrr)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_FILE, index=False)
    print(f"저장: {OUT_FILE.relative_to(ROOT).as_posix()}")

    features = [c for c in table.columns if c not in ("WAFER_ID", TARGET)]
    numeric = [c for c in features if c not in ("STAGE", "CHAMBER_PATH")]
    print(f"\n행 수: {len(table):,}")
    print(f"전체 열 수: {table.shape[1]} (WAFER_ID 1 + feature {len(features)} + MRR 정답 1)")
    print(f"feature {len(features)}개 = 구분형 2개(STAGE, CHAMBER_PATH) + 숫자형 {len(numeric)}개 "
          f"(usage 평균 {len(USAGE_MEAN)} + 공정 변수 {len(MEAN_STD)}개 x 평균/표준편차)")
    print(f"중복 (WAFER_ID, STAGE) 행: {table.duplicated(KEY).sum()}")

    na = table.isna().sum()
    print(f"\n빈 값이 있는 열: {na[na > 0].to_dict() or '없음'} / 전체 빈 칸 수: {int(na.sum())}")
    print("STAGE 별 행 수:", table["STAGE"].value_counts().sort_index().to_dict())
    print("CHAMBER_PATH 별 행 수:", table["CHAMBER_PATH"].value_counts().to_dict())
    other = table.loc[table["CHAMBER_PATH"] == "other", KEY]
    print("CHAMBER_PATH=other 인 wafer+Stage:", list(map(tuple, other.itertuples(index=False))))

    print("\n열 목록:", list(table.columns))
    example = table.iloc[0]
    print(f"\n예시 한 행 (첫 기록 시각이 가장 이른 행):")
    print(example.to_string())


if __name__ == "__main__":
    main()

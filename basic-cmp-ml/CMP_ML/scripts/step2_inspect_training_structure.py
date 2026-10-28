"""Step 2: PHM 2016 training 데이터의 기록 구조 확인 (읽기 전용).

- training 원데이터만 읽는다. test / validation 은 열지 않는다.
- 데이터를 가공·저장하지 않고, 구조 확인 결과만 화면에 출력한다.
실행: python scripts/step2_inspect_training_structure.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
KEY = ["WAFER_ID", "STAGE"]


def load_training():
    files = sorted(TRAIN_DIR.glob("CMP-training-*.csv"))
    frames, empty = [], []
    for f in files:
        df = pd.read_csv(f, dtype={"WAFER_ID": "int64", "STAGE": "string"})
        if df.empty:  # 헤더만 있는 파일: 합치면 숫자 컬럼이 object 가 되므로 제외하고 보고
            empty.append(f.name)
            continue
        df["SOURCE_FILE"] = f.name
        frames.append(df)
    return files, empty, pd.concat(frames, ignore_index=True)


def main():
    files, empty, raw = load_training()
    mrr = pd.read_csv(MRR_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})

    print("=" * 70)
    print("[0] 파일 / 행 수")
    print(f"training 공정 파일 수: {len(files)}")
    print(f"헤더만 있고 데이터 행이 0인 파일: {empty}")
    print(f"training 공정 데이터 전체 행 수: {len(raw):,}")
    print("컬럼 dtype:", raw.dtypes.astype(str).value_counts().to_dict())
    print(f"MRR 파일 행 수: {len(mrr):,}")
    print(f"컬럼 ({raw.shape[1] - 1}개): {list(raw.columns[:-1])}")

    print("=" * 70)
    print("[1] wafer 수, wafer+Stage 조합 수, MRR 정답 수")
    groups = raw.groupby(KEY, sort=False)
    print(f"공정 데이터 고유 WAFER_ID 수: {raw['WAFER_ID'].nunique():,}")
    print(f"공정 데이터 고유 (WAFER_ID, STAGE) 조합 수: {groups.ngroups:,}")
    print(f"MRR 정답 행 수: {len(mrr):,}")
    print(f"MRR 고유 WAFER_ID 수: {mrr['WAFER_ID'].nunique():,}")
    print(f"MRR 고유 (WAFER_ID, STAGE) 조합 수: {mrr[KEY].drop_duplicates().shape[0]:,}")
    stages_per_wafer = raw.groupby("WAFER_ID")["STAGE"].nunique()
    print("wafer 하나에 기록된 Stage 개수 분포:", stages_per_wafer.value_counts().sort_index().to_dict())

    print("=" * 70)
    print("[2] STAGE / CHAMBER 값")
    print("STAGE 값별 행 수:", raw["STAGE"].value_counts(dropna=False).sort_index().to_dict())
    print("STAGE 값별 wafer+Stage 조합 수:",
          raw[KEY].drop_duplicates()["STAGE"].value_counts().sort_index().to_dict())
    print("MRR 파일 STAGE 값별 행 수:", mrr["STAGE"].value_counts(dropna=False).sort_index().to_dict())
    print("CHAMBER 값별 행 수:", raw["CHAMBER"].value_counts(dropna=False).sort_index().to_dict())
    print("STAGE x CHAMBER 행 수 교차표:")
    print(pd.crosstab(raw["STAGE"], raw["CHAMBER"]))
    print("MACHINE_ID 값별 행 수:", raw["MACHINE_ID"].value_counts(dropna=False).sort_index().to_dict())

    print("=" * 70)
    print("[3] wafer+Stage 한 묶음의 기록 방식")
    summary = groups.agg(
        n_rows=("TIMESTAMP", "size"),
        n_unique_ts=("TIMESTAMP", "nunique"),
        n_files=("SOURCE_FILE", "nunique"),
        n_chambers=("CHAMBER", "nunique"),
        t_start=("TIMESTAMP", "min"),
        t_end=("TIMESTAMP", "max"),
    )
    summary["duration_s"] = summary["t_end"] - summary["t_start"]
    print("(a) 묶음당 행 수 분포:")
    print(summary["n_rows"].describe().round(1).to_string())
    print(f"    행이 1줄뿐인 묶음 수: {(summary['n_rows'] == 1).sum()}")
    print(f"    TIMESTAMP 중복이 있는 묶음 수: {(summary['n_rows'] != summary['n_unique_ts']).sum()}")
    dup_rows = raw.duplicated(KEY + ["TIMESTAMP"]).sum()
    dup_full = raw.drop(columns="SOURCE_FILE").duplicated().sum()
    print(f"    같은 묶음 안 TIMESTAMP 중복 행 수: {dup_rows:,} (그중 모든 컬럼까지 완전히 같은 행: {dup_full:,})")
    print("    묶음당 기록 시간 길이(초) 분포:")
    print(summary["duration_s"].describe().round(1).to_string())
    print("    행 수 상위 5개 묶음:")
    print(summary.sort_values("n_rows", ascending=False).head(5)[["n_rows", "duration_s", "n_chambers"]].to_string())
    print("    기록 시간 길이 상위 5개 묶음:")
    print(summary.sort_values("duration_s", ascending=False).head(5)[["n_rows", "duration_s", "n_chambers"]].to_string())
    order = raw["WAFER_ID"].astype(str) + "_" + raw["STAGE"].astype(str)
    block = order.ne(order.shift()).cumsum()
    n_blocks = block.groupby(order).nunique()
    print(f"    파일 순서대로 이어 붙였을 때 연속 구간이 2개 이상으로 끊긴 묶음 수: {(n_blocks > 1).sum()}")
    print("(b) 묶음이 걸쳐 있는 파일 수 분포:", summary["n_files"].value_counts().sort_index().to_dict())
    split = raw.groupby(KEY)["SOURCE_FILE"].agg(lambda x: sorted(set(x)))
    split = split[split.map(len) > 1]
    adjacent = all(int(fs[-1][-7:-4]) - int(fs[0][-7:-4]) == len(fs) - 1 for fs in split)
    print(f"    2개 이상 파일에 걸친 묶음이 모두 번호가 연속된 파일에 걸쳐 있는가: {adjacent}")
    print("    파일 하나에 들어 있는 묶음 수 분포:")
    print(raw[KEY + ["SOURCE_FILE"]].drop_duplicates().groupby("SOURCE_FILE").size()
          .describe().round(1).to_string())
    print("(c) 묶음이 거치는 CHAMBER 개수 분포:", summary["n_chambers"].value_counts().sort_index().to_dict())
    seq = (raw.sort_values(KEY + ["TIMESTAMP"])
              .groupby(KEY, sort=False)["CHAMBER"]
              .agg(lambda s: tuple(int(c) for c in s[s.ne(s.shift())])))
    print("    CHAMBER 방문 순서 패턴별 묶음 수:")
    print(seq.value_counts().to_string())
    odd = seq[~seq.isin([(1, 2, 3), (4, 5, 6)])]
    print("    일반 패턴(1-2-3 / 4-5-6)이 아닌 묶음:")
    print(summary.loc[odd.index, ["n_rows", "duration_s"]].assign(chamber_seq=odd).to_string())
    chamber_set = seq.map(lambda t: "1-3" if set(t) <= {1, 2, 3} else "4-6").rename("CHAMBER_SET")
    print("    STAGE x CHAMBER 그룹(1-3 / 4-6) 묶음 수:")
    print(pd.crosstab(chamber_set.index.get_level_values("STAGE"), chamber_set))

    print("=" * 70)
    print("[4] wafer+Stage 와 MRR 정답 연결")
    dup_mrr = mrr.duplicated(KEY, keep=False).sum()
    print(f"MRR 파일에서 같은 (WAFER_ID, STAGE)가 중복된 행 수: {dup_mrr}")
    print(f"MRR 결측값 수: {mrr['AVG_REMOVAL_RATE'].isna().sum()}")
    raw_keys = set(map(tuple, raw[KEY].drop_duplicates().itertuples(index=False)))
    mrr_keys = set(map(tuple, mrr[KEY].itertuples(index=False)))
    print(f"공정 데이터에만 있고 MRR 이 없는 조합 수: {len(raw_keys - mrr_keys)}")
    print(f"MRR 에만 있고 공정 데이터가 없는 조합 수: {len(mrr_keys - raw_keys)}")
    print(f"양쪽에 모두 있는 조합 수: {len(raw_keys & mrr_keys)}")
    if raw_keys - mrr_keys:
        print("  MRR 없는 조합 예:", sorted(raw_keys - mrr_keys)[:10])
    if mrr_keys - raw_keys:
        print("  공정 데이터 없는 조합 예:", sorted(mrr_keys - raw_keys)[:10])

    print("=" * 70)
    print("[5] 예시: training 첫 파일의 첫 번째 wafer+Stage")
    first = raw.iloc[0]
    key = (int(first["WAFER_ID"]), str(first["STAGE"]))
    ex = raw[(raw["WAFER_ID"] == key[0]) & (raw["STAGE"] == key[1])].sort_values("TIMESTAMP")
    print(f"WAFER_ID={key[0]}, STAGE={key[1]}")
    print(f"행 수: {len(ex)}")
    print(f"기록된 파일: {sorted(ex['SOURCE_FILE'].unique())}")
    print(f"TIMESTAMP 범위: {ex['TIMESTAMP'].min()} ~ {ex['TIMESTAMP'].max()} "
          f"(길이 {ex['TIMESTAMP'].max() - ex['TIMESTAMP'].min():.1f}초)")
    print("TIMESTAMP 간격(초) 분포:", ex["TIMESTAMP"].diff().dropna().round(3).value_counts().to_dict())
    run = ex["CHAMBER"].ne(ex["CHAMBER"].shift()).cumsum()
    print("CHAMBER 방문 순서 (구간별 행 수, 시작~끝 TIMESTAMP):")
    for _, seg in ex.groupby(run, sort=True):
        print(f"  CHAMBER {int(seg['CHAMBER'].iloc[0])}: {len(seg)}행, "
              f"{seg['TIMESTAMP'].min()} ~ {seg['TIMESTAMP'].max()}")
    ans = mrr[(mrr["WAFER_ID"] == key[0]) & (mrr["STAGE"] == key[1])]
    print("MRR 정답:", ans["AVG_REMOVAL_RATE"].tolist())


if __name__ == "__main__":
    main()

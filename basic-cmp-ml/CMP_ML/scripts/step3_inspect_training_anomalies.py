"""Step 3: Step 2 에서 발견한 training 이상 기록 확인 (읽기 전용).

- training 원데이터만 읽는다. test / validation 은 열지 않는다.
- 원데이터를 삭제하거나 수정하지 않는다. 확인 결과를 출력하고,
  완전 중복 행이 있는 묶음 목록만 reports/ 에 CSV 로 저장한다.
실행: python scripts/step3_inspect_training_anomalies.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
REPORT_DIR = ROOT / "reports"
KEY = ["WAFER_ID", "STAGE"]
GAP_S = 10.0  # 이 값(초)보다 긴 시간 간격은 "끊김"으로 표시

BIG_GROUPS = [(3021014294, "B"), (3015014228, "B")]
LONG_GROUP = (2921014258, "A")
ODD_GROUPS = [(-4019511766, "B"), (2665292924, "A"), (2939014292, "B"), (3015014228, "B")]


def load_training():
    frames = []
    for f in sorted(TRAIN_DIR.glob("CMP-training-*.csv")):
        df = pd.read_csv(f, dtype={"WAFER_ID": "int64", "STAGE": "string"})
        if df.empty:
            continue
        df["FILE"] = f.name
        df["FILE_ROW"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


RAW = load_training()
MRR = pd.read_csv(MRR_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
COLS = [c for c in RAW.columns if c not in ("FILE", "FILE_ROW")]  # 원본 25개 컬럼
SENS = [c for c in COLS if c != "TIMESTAMP"]                       # TIMESTAMP 제외 24개
USAGE = [c for c in COLS if c.startswith("USAGE")]


def group(key):
    w, s = key
    return RAW[(RAW["WAFER_ID"] == w) & (RAW["STAGE"] == s)]


def segments(g):
    """TIMESTAMP 순으로 정렬해 Chamber 가 바뀌거나 GAP_S 초 넘게 끊기는 곳에서 구간을 나눈다."""
    g = g.sort_values("TIMESTAMP", kind="stable")
    cut = g["CHAMBER"].ne(g["CHAMBER"].shift()) | (g["TIMESTAMP"].diff() > GAP_S)
    seg = g.groupby(cut.cumsum()).agg(
        CHAMBER=("CHAMBER", lambda x: int(x.iloc[0])),
        rows=("TIMESTAMP", "size"),
        t_start=("TIMESTAMP", "min"),
        t_end=("TIMESTAMP", "max"),
        file=("FILE", lambda x: ",".join(sorted(set(x)))),
        file_row_min=("FILE_ROW", "min"),
        file_row_max=("FILE_ROW", "max"),
    )
    seg["gap_before_s"] = seg["t_start"] - seg["t_end"].shift()
    seg.index = range(1, len(seg) + 1)
    return seg


def others_in_window(key, t0, t1):
    """시간 구간 (t0, t1) 안에 기록이 있는 다른 wafer+Stage 목록."""
    m = (RAW["TIMESTAMP"] > t0) & (RAW["TIMESTAMP"] < t1) & ~(
        (RAW["WAFER_ID"] == key[0]) & (RAW["STAGE"] == key[1]))
    return RAW[m].groupby(KEY).agg(rows=("TIMESTAMP", "size"),
                                   chambers=("CHAMBER", lambda x: sorted({int(c) for c in x})),
                                   files=("FILE", lambda x: sorted(set(x))))


def mrr_of(key):
    v = MRR[(MRR["WAFER_ID"] == key[0]) & (MRR["STAGE"] == key[1])]["AVG_REMOVAL_RATE"]
    return float(v.iloc[0])


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
def check_full_duplicates():
    header("[1] 모든 컬럼(원본 25개)이 완전히 같은 중복 행")
    dup = RAW.duplicated(COLS, keep="first")
    print(f"전체 행: {len(RAW):,} / 완전 중복(두 번째 이후) 행: {dup.sum():,} / 제외 후 남는 행: {len(RAW) - dup.sum():,}")

    same_file_prev = RAW["FILE"].eq(RAW["FILE"].shift())
    equal_prev = RAW[COLS].eq(RAW[COLS].shift()).all(axis=1) & same_file_prev
    print(f"중복 행 중 같은 파일에서 바로 윗줄과 똑같은 행: {(dup & equal_prev).sum():,}")
    sets = RAW[RAW.duplicated(COLS, keep=False)].groupby(COLS)["FILE"].agg(["size", "nunique"])
    print(f"중복 세트 수: {len(sets):,} / 한 세트의 복사본 수 분포: {sets['size'].value_counts().to_dict()} "
          f"/ 서로 다른 파일에 걸친 세트: {(sets['nunique'] > 1).sum()}")

    per = RAW.assign(DUP=dup).groupby(KEY).agg(rows=("DUP", "size"), full_dup_rows=("DUP", "sum"))
    per["rows_after"] = per["rows"] - per["full_dup_rows"]
    per = per[per["full_dup_rows"] > 0].sort_values("full_dup_rows", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "step3_full_duplicate_groups.csv"
    per.to_csv(out)
    print(f"완전 중복이 있는 wafer+Stage: {len(per)}개 (전체 목록: {out.relative_to(ROOT).as_posix()})")
    print("묶음당 완전 중복 행 수 분포:", per["full_dup_rows"].value_counts().sort_index().to_dict())

    heavy = per[per["full_dup_rows"] >= per["rows"] * 0.4]
    print(f"\n(a) 행 대부분이 두 번씩 기록된 묶음: {len(heavy)}개")
    rows = []
    for key in heavy.index:
        g = group(key)
        copies = g.groupby(COLS).size().value_counts().sort_index().to_dict()
        rows.append({"WAFER_ID": key[0], "STAGE": key[1], "rows": len(g),
                     "full_dup_rows": int(heavy.loc[key, "full_dup_rows"]),
                     "rows_after": int(heavy.loc[key, "rows_after"]),
                     "unique_TIMESTAMP": g["TIMESTAMP"].nunique(),
                     "copies_per_unique_row": copies, "files": sorted(g["FILE"].unique())})
    print(pd.DataFrame(rows).to_string(index=False))

    light = per.drop(heavy.index)
    print(f"\n(b) 나머지 {len(light)}개 묶음: 완전 중복 행 합계 {light['full_dup_rows'].sum():,}, "
          f"묶음당 {light['full_dup_rows'].min()}~{light['full_dup_rows'].max()}행")
    print("    상위 10개:")
    print(light.head(10).to_string())

    ts_dup = RAW.duplicated(KEY + ["TIMESTAMP"], keep="first") & ~dup
    print(f"\n참고: 완전 중복을 뺀 뒤에도 같은 묶음 안에서 TIMESTAMP 만 겹치는 행 {ts_dup.sum():,}개가 "
          f"{RAW[ts_dup].groupby(KEY).ngroups}개 묶음에 남습니다 (값은 다름).")


# ---------------------------------------------------------------------------
def repetition_stats(g):
    """같은 기록 블록이 반복되는지: TIMESTAMP 를 뺀 24개 값 기준으로 비교."""
    g = g.sort_values("TIMESTAMP", kind="stable")
    v = g[SENS].to_numpy()
    uniq = len(g[SENS].drop_duplicates())
    same_prev = int((v[1:] == v[:-1]).all(axis=1).sum())
    best_lag, best_frac = None, 0.0
    for lag in range(150, 1001):  # 일반 묶음 길이(약 180~800행) 범위의 주기로 반복되는지
        frac = (v[lag:] == v[:-lag]).all(axis=1).mean()
        if frac > best_frac:
            best_lag, best_frac = lag, frac
    return len(v), uniq, same_prev, best_lag, best_frac


def check_big_groups():
    header("[2] 5,000줄이 넘는 wafer+Stage 2개")
    all_groups = RAW.groupby(KEY)
    typical_rows = all_groups.size().median()
    typical_usage = all_groups["USAGE_OF_BACKING_FILM"].nunique().median()
    print(f"비교 기준(전체 1,981개 묶음 중앙값): 행 수 {typical_rows:.0f}, "
          f"USAGE_OF_BACKING_FILM 서로 다른 값 {typical_usage:.0f}개")

    for key in BIG_GROUPS:
        g = group(key).sort_values("TIMESTAMP", kind="stable")
        n, uniq, same_prev, lag, frac = repetition_stats(g)
        d = g["TIMESTAMP"].diff()
        print(f"\n--- WAFER_ID={key[0]}, STAGE={key[1]} (MRR {mrr_of(key)}) ---")
        print(f"행 {n:,} / 완전 중복 {g.duplicated(COLS).sum()} / 고유 TIMESTAMP {g['TIMESTAMP'].nunique():,} "
              f"/ 파일 {sorted(g['FILE'].unique())} / 파일 안 행 순서가 시간순: {group(key)['TIMESTAMP'].is_monotonic_increasing}")
        print(f"시간 범위 {g['TIMESTAMP'].min():.3f} ~ {g['TIMESTAMP'].max():.3f} "
              f"({g['TIMESTAMP'].max() - g['TIMESTAMP'].min():,.0f}초), 행 간격 중앙값 {d.median():.3f}초, "
              f"{GAP_S:.0f}초 넘는 간격 {int((d > GAP_S).sum())}개 {sorted(d[d > GAP_S].round(1).tolist(), reverse=True)}")
        print(f"반복 여부: TIMESTAMP 제외 서로 다른 행 {uniq:,}개 / 바로 윗줄과 값이 같은 행 {same_prev:,}개 / "
              f"150~1000행 주기로 같은 값이 반복되는 비율 최대 {frac:.1%} (주기 {lag})")
        print("usage 컬럼 서로 다른 값 개수 / 최소~최대:")
        for c in USAGE:
            print(f"  {c}: {g[c].nunique()}개, {g[c].min():.3f} ~ {g[c].max():.3f}")
        print("구간 (Chamber 변경 또는 10초 초과 끊김 기준):")
        print(segments(g).to_string())

    a, b = (group(k).sort_values("TIMESTAMP", kind="stable").drop_duplicates("TIMESTAMP").set_index("TIMESTAMP")
            for k in BIG_GROUPS)
    common = np.intersect1d(a.index.to_numpy(), b.index.to_numpy())
    print(f"\n두 묶음의 공통 TIMESTAMP: {len(common):,}개")
    cmp_cols = [c for c in SENS if c not in ("WAFER_ID", "STAGE", "CHAMBER", "MACHINE_DATA")]
    same = a.loc[common, cmp_cols] == b.loc[common, cmp_cols]
    print(f"공통 TIMESTAMP 에서 두 묶음의 값(식별·Chamber 컬럼 제외 {len(cmp_cols)}개)이 모두 같은 행: "
          f"{int(same.all(axis=1).sum()):,}개")
    print("컬럼별로 같은 비율:", same.mean().round(3).to_dict())
    print("같은 시각 두 묶음의 CHAMBER 조합 (행: 3021014294 B, 열: 3015014228 B):")
    print(pd.crosstab(a.loc[common, "CHAMBER"], b.loc[common, "CHAMBER"]).to_string())
    t0 = min(a.index.min(), b.index.min())
    t1 = max(a.index.max(), b.index.max())
    oth = others_in_window(BIG_GROUPS[0], t0, t1).drop(index=[BIG_GROUPS[1]], errors="ignore")
    print(f"두 묶음 시간 범위 안에 기록된 다른 wafer+Stage: {len(oth)}개")
    if len(oth):
        print(oth.to_string())


# ---------------------------------------------------------------------------
def check_long_group():
    header("[3] 시간 범위가 약 8시간인 wafer+Stage")
    key = LONG_GROUP
    g = group(key).sort_values("TIMESTAMP", kind="stable")
    d = g["TIMESTAMP"].diff()
    print(f"WAFER_ID={key[0]}, STAGE={key[1]} (MRR {mrr_of(key)})")
    print(f"행 {len(g)} / 파일 {sorted(g['FILE'].unique())} / 전체 시간 범위 "
          f"{g['TIMESTAMP'].max() - g['TIMESTAMP'].min():,.3f}초")
    print(f"{GAP_S:.0f}초 넘는 간격: {sorted(d[d > GAP_S].round(3).tolist(), reverse=True)}")
    print(f"{GAP_S:.0f}초 넘는 간격을 뺀 실제 기록 시간 합계: {d[d <= GAP_S].sum():,.3f}초")
    seg = segments(g)
    print("구간:")
    print(seg.to_string())
    for i in seg.index[seg["gap_before_s"] > GAP_S]:
        t0, t1 = seg.loc[i - 1, "t_end"], seg.loc[i, "t_start"]
        oth = others_in_window(key, t0, t1)
        print(f"\n구간 {i - 1}→{i} 사이 {t1 - t0:,.3f}초 동안 기록된 다른 wafer+Stage: {len(oth)}개, "
              f"행 {oth['rows'].sum():,}개, 파일 {sorted({f for fs in oth['files'] for f in fs})}")


# ---------------------------------------------------------------------------
def check_odd_chambers():
    header("[4] Chamber 순서가 1→2→3 / 4→5→6 이 아닌 wafer+Stage 4개")
    for key in ODD_GROUPS:
        g = group(key)
        seg = segments(g)
        print(f"\n--- WAFER_ID={key[0]}, STAGE={key[1]} (MRR {mrr_of(key)}) ---")
        print(f"행 {len(g):,} / Chamber 별 행 수 {g['CHAMBER'].astype(int).value_counts().sort_index().to_dict()} "
              f"/ 파일 안 행 순서가 시간순: {g['TIMESTAMP'].is_monotonic_increasing}")
        print(f"시간순 Chamber 순서: {' → '.join(str(c) for c in seg['CHAMBER'])}")
        print(seg.to_string())
        for i in seg.index[seg["gap_before_s"] > GAP_S]:
            t0, t1 = seg.loc[i - 1, "t_end"], seg.loc[i, "t_start"]
            oth = others_in_window(key, t0, t1)
            chambers = sorted({c for cs in oth["chambers"] for c in cs}) if len(oth) else []
            print(f"  구간 {i - 1}→{i} 사이 {t1 - t0:,.3f}초 동안 기록된 다른 wafer+Stage: {len(oth)}개"
                  + (f" (Chamber {chambers})" if chambers else ""))
        if not g["TIMESTAMP"].is_monotonic_increasing:
            fo = g.sort_values(["FILE", "FILE_ROW"])
            blk = fo["CHAMBER"].ne(fo["CHAMBER"].shift()).cumsum()
            print("  파일 행 순서대로 본 Chamber 구간:")
            print(fo.groupby(blk).agg(CHAMBER=("CHAMBER", lambda x: int(x.iloc[0])), rows=("CHAMBER", "size"),
                                      file_row_min=("FILE_ROW", "min"), file_row_max=("FILE_ROW", "max"),
                                      t_start=("TIMESTAMP", "min"), t_end=("TIMESTAMP", "max")).to_string(index=False))


if __name__ == "__main__":
    check_full_duplicates()
    check_big_groups()
    check_long_group()
    check_odd_chambers()

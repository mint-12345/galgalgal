"""웨이퍼 단위 집계표 생성 (CLAUDE.md 10절).

출력: data/processed/wafer_table.csv
  - 행: train·val·test 전체 웨이퍼 (키 = WAFER_ID, STAGE). 행은 지우지 않고 플래그만 붙인다.
  - MRR: train·val만 채우고 test는 비운다.
  - 피처: R 세트 ID 22~125 (6.1절), S3 소모품 상태 (6.3절).
컬럼 설명은 data/processed/COLUMNS.md.

실행: python src/aggregate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import KEY, ROOT, SPLITS, load_labels, load_sensors  # noqa: E402

OUT = ROOT / "data" / "processed" / "wafer_table.csv"

# 원본 컬럼 → 피처 이름에 쓰는 약어 (6.1절 이름 규칙 {통계}_{약어}_{구간})
USAGE = {
    "USAGE_OF_BACKING_FILM": "BACKING_FILM",
    "USAGE_OF_DRESSER": "DRESSER",
    "USAGE_OF_POLISHING_TABLE": "POLISH_TABLE",
    "USAGE_OF_DRESSER_TABLE": "DRESSER_TABLE",
    "USAGE_OF_MEMBRANE": "MEMBRANE",
    "USAGE_OF_PRESSURIZED_SHEET": "PSHEET",
}
PRESSURE = {
    "PRESSURIZED_CHAMBER_PRESSURE": "PCHAMBER",
    "MAIN_OUTER_AIR_BAG_PRESSURE": "MAIN_OUTER",
    "CENTER_AIR_BAG_PRESSURE": "CENTER",
    "RETAINER_RING_PRESSURE": "RETAINER",
    "RIPPLE_AIR_BAG_PRESSURE": "RIPPLE",
    "EDGE_AIR_BAG_PRESSURE": "EDGE",
}
FLOW = {
    "SLURRY_FLOW_LINE_A": "SLURRY_A",
    "SLURRY_FLOW_LINE_B": "SLURRY_B",
    "SLURRY_FLOW_LINE_C": "SLURRY_C",
    "DRESSING_WATER_STATUS": "DRESS_WATER",
}
ROTATION = {
    "WAFER_ROTATION": "WAFER_ROT",
    "STAGE_ROTATION": "STAGE_ROT",
    "HEAD_ROTATION": "HEAD_ROT",
}
# 전체-0 판정 대상 12개 (5.4절): 압력 6 + 슬러리 3 + 회전 3. DRESSING_WATER_STATUS 제외.
ALLZERO_SENSORS = list(PRESSURE) + ["SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C"] + list(ROTATION)

FIRST_CHAMBER = {4, 1}  # s1 = 챔버 4 또는 1, s2 = 나머지 두 챔버 (5.3절)
LONG_TRACE_ROWS = 2000  # 5.3절 "2,000행 이상"
OUTLIER_MRR = 400  # 5.2절
# TODO(결정 필요): 리셋 감지 임계값 (13절 미결정). 5.5절 사전 확인치의 "범위의 20% 이상 감소"를 임시로 쓴다.
RESET_FRAC = 0.20
# 리셋이 한 번도 없어 since_* 가 전 행 NaN인 변수 → 컬럼을 만들지 않는다.
NO_SINCE = {"DRESSER_TABLE"}


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """2단계: 완전히 같은 행 제거."""
    n = int(df.duplicated().sum())
    return df.drop_duplicates(ignore_index=True), n


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    """TIMESTAMP에 대한 선형회귀 기울기 (단위/초). 시각이 2개 미만이면 NaN."""
    if np.unique(t).size < 2:
        return np.nan
    return float(np.polyfit(t - t[0], y, 1)[0])


def _std(y: np.ndarray) -> float:
    return float(np.std(y, ddof=1)) if y.size >= 2 else np.nan


def _auc(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, t)) if y.size >= 2 else np.nan


def segment_features(t: np.ndarray, block: dict[str, np.ndarray], seg: str) -> dict[str, float]:
    """6단계: 한 구간의 R 피처 (ID 22~125 중 해당 구간 몫 52개)."""
    f: dict[str, float] = {}
    empty = t.size == 0
    f[f"dur_POLISH_{seg}"] = np.nan if empty else float(t[-1] - t[0])  # ID 22–23
    for col, ab in USAGE.items():  # ID 24–59
        y = block[col]
        f[f"mean_{ab}_{seg}"] = np.nan if empty else float(y.mean())
        f[f"std_{ab}_{seg}"] = _std(y)
        f[f"slope_{ab}_{seg}"] = np.nan if empty else _slope(t, y)
    for col, ab in {**PRESSURE, **FLOW}.items():  # ID 60–119
        y = block[col]
        f[f"mean_{ab}_{seg}"] = np.nan if empty else float(y.mean())
        f[f"std_{ab}_{seg}"] = _std(y)
        f[f"auc_{ab}_{seg}"] = _auc(t, y)
    for col, ab in ROTATION.items():  # ID 120–125
        y = block[col]
        f[f"mean_{ab}_{seg}"] = np.nan if empty else float(y.mean())
    return f


def r_feature_names() -> list[str]:
    """R 피처 104개 (ID 22~125) 이름을 6.1절 ID 블록 순서로. 블록 안에서는 변수 → 통계 → 구간 순."""
    segs = ("s1", "s2")
    names = [f"dur_POLISH_{s}" for s in segs]
    blocks = [(USAGE, ("mean", "std", "slope")), ({**PRESSURE, **FLOW}, ("mean", "std", "auc")),
              (ROTATION, ("mean",))]
    for vars_, stats in blocks:
        names += [f"{st}_{ab}_{s}" for ab in vars_.values() for st in stats for s in segs]
    assert len(names) == 104
    return names


def wafer_row(w: pd.DataFrame) -> dict:
    """3~6단계: 웨이퍼 하나 (TIMESTAMP로 정렬된 상태)를 한 행으로."""
    t = w["TIMESTAMP"].to_numpy(dtype=float)
    chambers = sorted(int(c) for c in w["CHAMBER"].unique())
    cset = "123" if set(chambers) <= {1, 2, 3} else "456" if set(chambers) <= {4, 5, 6} else "MIXED"
    row = {
        "WAFER_ID": w["WAFER_ID"].iat[0],
        "STAGE": w["STAGE"].iat[0],
        "split": w["split"].iat[0],
        "group": f"{w['STAGE'].iat[0]}-{cset}",
        "chambers": "-".join(map(str, chambers)),  # 예: "4-5-6" (CSV에서 숫자로 읽히지 않게)
        "t_start": t[0],
        "t_end": t[-1],
        "n_rows": len(w),
        # 5단계 플래그
        "flag_incomplete_chamber": int(len(chambers) < 3),
        "flag_long_trace": int(len(w) >= LONG_TRACE_ROWS),
        "n_allzero_sensors": int((w[ALLZERO_SENSORS] == 0).all().sum()),
        # TODO(결정 필요): 같은 (웨이퍼, TIMESTAMP)에 값이 다른 행 처리 (13절). 지금은 그대로 두고 개수만 기록한다.
        "n_dup_ts_rows": int(w.duplicated("TIMESTAMP", keep=False).sum()),
    }
    in_s1 = w["CHAMBER"].isin(FIRST_CHAMBER).to_numpy()
    cols = list(USAGE) + list(PRESSURE) + list(FLOW) + list(ROTATION)
    for seg, mask in (("s1", in_s1), ("s2", ~in_s1)):
        block = {c: w[c].to_numpy(dtype=float)[mask] for c in cols}
        row.update(segment_features(t[mask], block, seg))
    # S3 중 웨이퍼 내부 값 (구간 = w: 웨이퍼 전체)
    for col, ab in USAGE.items():
        y = w[col].to_numpy(dtype=float)
        row[f"start_{ab}_w"] = y[0]
        row[f"delta_{ab}_w"] = y[-1] - y[0]
        row[f"slope_{ab}_w"] = _slope(t, y)
    return row


def add_reset_features(tab: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    """7단계: 소모품 리셋 감지 → S3의 리셋 플래그·교체 이후 누적 사용량·교체 횟수.

    - 스테이지별로 train·val·test 웨이퍼를 합쳐 t_start 순으로 정렬한다 (라벨 불필요).
      스테이지 A의 A-123·A-456은 같은 카운터를 공유한다 (시작 시각 순으로 값이 이어짐).
    - 리셋 = 시작값이 직전 웨이퍼의 시작값보다 (스테이지 내 값 범위 × RESET_FRAC) 이상 줄어든 경우.
      웨이퍼끼리 시간이 겹치므로 직전 웨이퍼의 마지막 값이 아니라 시작값과 비교한다.
      웨이퍼 내부에서는 리셋이 관측되지 않았다.
    """
    tab = tab.sort_values(["STAGE", "t_start"], kind="mergesort").reset_index(drop=True)
    for col, ab in USAGE.items():
        start = tab[f"start_{ab}_w"]
        rng = sensors.groupby("STAGE")[col].agg(lambda x: x.max() - x.min())
        drop = start.groupby(tab["STAGE"]).diff()
        reset = (drop < -RESET_FRAC * tab["STAGE"].map(rng)).astype(int)
        tab[f"reset_{ab}_w"] = reset
        tab[f"nreset_{ab}_w"] = reset.groupby(tab["STAGE"]).cumsum()
        if ab in NO_SINCE:
            continue
        # 마지막 리셋 웨이퍼의 시작값 기준 누적 사용량. 스테이지 첫 리셋 이전 웨이퍼는 기준을 알 수 없어 NaN.
        # TODO(결정 필요): 첫 리셋 이전 웨이퍼의 기준값 (NaN 유지 / 카운터 값 그대로 사용)
        base = start.where(reset == 1).groupby(tab["STAGE"]).ffill()
        tab[f"since_{ab}_w"] = start - base
    return tab


def build() -> pd.DataFrame:
    frames = []
    for split in SPLITS:  # 1단계
        df, n_dup = dedup(load_sensors(split))  # 2단계
        print(f"[{split}] 행 {len(df):,} (완전 중복 {n_dup:,}행 제거)")
        frames.append(df)
    sensors = pd.concat(frames, ignore_index=True)
    sensors = sensors.sort_values(KEY + ["TIMESTAMP"], kind="mergesort")  # 3단계

    tab = pd.DataFrame([wafer_row(w) for _, w in sensors.groupby(KEY, sort=False)])
    tab = add_reset_features(tab, sensors)

    labels = pd.concat([load_labels("train"), load_labels("val")], ignore_index=True)
    labels = labels.rename(columns={"AVG_REMOVAL_RATE": "MRR"})
    tab = tab.merge(labels, on=KEY, how="left", validate="one_to_one")
    tab["is_outlier"] = ((tab["split"] == "train") & (tab["MRR"] > OUTLIER_MRR)).astype(int)

    meta = ["WAFER_ID", "STAGE", "split", "group", "chambers", "t_start", "t_end", "n_rows", "MRR",
            "is_outlier", "flag_incomplete_chamber", "flag_long_trace", "n_allzero_sensors", "n_dup_ts_rows"]
    r_cols = r_feature_names()
    s3_cols = [f"{p}_{ab}_w" for ab in USAGE.values()
               for p in ("start", "delta", "slope", "reset", "since", "nreset")
               if not (p == "since" and ab in NO_SINCE)]
    return tab[meta + r_cols + s3_cols]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다. 지우고 다시 실행한다.")
    tab = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT, index=False)
    print(f"저장: {OUT.relative_to(ROOT)} ({len(tab):,}행 × {tab.shape[1]}열)")


if __name__ == "__main__":
    main()

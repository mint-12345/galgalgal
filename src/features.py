"""우리 연구용 피처 (CLAUDE.md 14.1절). 새 피처는 이 파일에만 추가한다.

결과는 `data/processed/features_S.csv`에 웨이퍼 키 `(WAFER_ID, STAGE)`와 함께 저장한다 (14.2절).
센서만 쓰고 라벨은 읽지 않는다. 세 split에 같은 규칙을 적용한다.

피처
  press_on_s    압력 6개 중 하나라도 > 0 인 시간 (초). 행마다 다음 기록까지의 간격 dt를 붙여 더하고,
                dt > 2초(빈칸)는 0으로 센다. `src/hyp_mrr_time.py`의 `T_c_press_out`과 같은 값.
  press_long    456 그룹에서 press_on_s >= 220초면 1, 아니면 0. A-123은 NaN.
                456 그룹의 press_on_s는 약 150초와 약 285초 두 무리로 나뉘고 200~240초가 골짜기다
                (repro_김혜연.md 14·15번). 긴 무리는 챔버 4·5·6에서, 짧은 무리는 챔버 4에서만 압력이 걸린다.

실행: python src/features.py
출력: data/processed/features_S.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, SPLITS, load_sensors  # noqa: E402

OUT = ROOT / "data" / "processed" / "features_S.csv"
GAP_S = 2.0
PRESS_LONG_S = 220.0


def press_on_time(raw: pd.DataFrame) -> pd.Series:
    """웨이퍼별 압력 > 0 시간 (초, 빈칸 제외). raw는 완전 중복을 지운 센서 행."""
    raw = raw.sort_values(KEY + ["TIMESTAMP"], kind="stable")
    dt = (-raw.groupby(KEY)["TIMESTAMP"].diff(-1)).fillna(0.0)
    on = (raw[list(PRESSURE)] > 0).any(axis=1) & (dt <= GAP_S)
    return dt.where(on, 0.0).groupby([raw[k] for k in KEY]).sum().rename("press_on_s")


def press_cluster(raw: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """press_on_s와 press_long. group = 키 + `group` 컬럼 (집계표)."""
    f = group[KEY + ["group"]].merge(press_on_time(raw).reset_index(), on=KEY, how="left")
    is456 = f["group"].str.endswith("456")
    f["press_long"] = (f["press_on_s"] >= PRESS_LONG_S).astype(float).where(is456)
    return f[KEY + ["press_on_s", "press_long"]]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 14.2절: 새 버전은 _v2로)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    frames = []
    for split in SPLITS:
        df, n_dup = dedup(load_sensors(split))
        frames.append(df)
        print(f"  {split}: 중복 {n_dup:,}행 제거", flush=True)
    f = press_cluster(pd.concat(frames, ignore_index=True), tab)
    if len(f) != len(tab) or f["press_on_s"].isna().any():
        raise ValueError("집계표 웨이퍼와 센서 웨이퍼가 맞지 않는다")
    f.round(3).to_csv(OUT, index=False)
    print(f.merge(tab[KEY + ["group"]], on=KEY).groupby("group")["press_long"]
          .value_counts(dropna=False).to_string())
    print(f"저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""소모품 리셋 감지 임계값 민감도 (10% / 20% / 30%).

- train·val·test 센서를 합쳐 스테이지별 시작 시각 순으로 정렬한 뒤 리셋을 센다 (라벨 불필요).
- 판정 규칙은 `src/aggregate.py`와 같다: 웨이퍼 시작값이 직전 웨이퍼(같은 스테이지)의 시작값보다
  `frac × (스테이지 내 변수 범위)` 이상 줄면 리셋.
- 임계값은 확정하지 않는다 (CLAUDE.md 13절). 이 스크립트는 비교표만 만든다.

실행: python src/reset_sweep.py
출력: results/diagnostics/{날짜}_{이름}_reset_threshold.csv (+ 같은 이름의 .yaml). 진단 결과이므로 14.4 규칙을 따른다.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import USAGE, dedup  # noqa: E402
from load import KEY, ROOT, SPLITS, load_sensors  # noqa: E402

FRACS = [0.10, 0.20, 0.30]
AUTHOR = "김혜연"  # 파일명에 들어가는 작성자 (CLAUDE.md 14.4)
OUT = ROOT / "results" / "diagnostics" / f"{date.today().isoformat()}_{AUTHOR}_reset_threshold.csv"
SEED = 42  # 이 스크립트는 난수를 쓰지 않지만 설정 기록용으로 남긴다


def wafer_starts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """웨이퍼별 (스테이지, 시작 시각, 사용량 6개 시작값)과 스테이지별 변수 범위."""
    sensors = pd.concat([dedup(load_sensors(s))[0] for s in SPLITS], ignore_index=True)
    sensors = sensors.sort_values(KEY + ["TIMESTAMP"], kind="mergesort")
    g = sensors.groupby(KEY, sort=False)
    starts = g[list(USAGE)].first().reset_index()
    starts["t_start"] = g["TIMESTAMP"].min().values
    starts["split"] = g["split"].first().values
    ranges = sensors.groupby("STAGE")[list(USAGE)].agg(lambda x: x.max() - x.min())
    return starts.sort_values(["STAGE", "t_start"], kind="mergesort"), ranges


def count_resets(starts: pd.DataFrame, ranges: pd.DataFrame, frac: float) -> pd.DataFrame:
    rows = []
    for col, ab in USAGE.items():
        drop = starts.groupby("STAGE")[col].diff()
        reset = drop < -frac * starts["STAGE"].map(ranges[col])
        for stage, n in reset.groupby(starts["STAGE"]).sum().items():
            rows.append({"variable": ab, "stage": stage, "threshold_frac": frac, "n_reset": int(n),
                         "n_wafer": int((starts["STAGE"] == stage).sum()),
                         "stage_range": float(ranges.loc[stage, col])})
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절: 기존 결과를 덮어쓰지 않는다)")
    starts, ranges = wafer_starts()
    print(f"웨이퍼 {len(starts):,}장 (train·val·test 합산), 스테이지별 시작 시각 순 정렬")
    res = pd.concat([count_resets(starts, ranges, f) for f in FRACS], ignore_index=True)

    pivot = res.pivot_table(index="variable", columns=["threshold_frac", "stage"], values="n_reset")
    pivot = pivot.loc[[ab for ab in USAGE.values()]]
    print("\n임계값별 리셋 횟수 (행 = 변수, 열 = 임계값 × 스테이지)")
    print(pivot.to_string())

    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT, index=False)
    yaml = OUT.with_suffix(".yaml")
    yaml.write_text(
        "\n".join([
            f"date: {date.today().isoformat()}",
            "script: src/reset_sweep.py",
            "input: data/raw (train+val+test 센서, 라벨 미사용)",
            "wafers: %d" % len(starts),
            "rule: 시작값이 직전 웨이퍼(같은 스테이지, 시작 시각 순)의 시작값보다 frac x (스테이지 내 변수 범위) 이상 감소",
            "thresholds: [%s]" % ", ".join(str(f) for f in FRACS),
            "variables: [%s]" % ", ".join(USAGE.values()),
            f"seed: {SEED}",
            "note: 임계값 미확정 (CLAUDE.md 13절)",
            "",
        ]), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {yaml.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

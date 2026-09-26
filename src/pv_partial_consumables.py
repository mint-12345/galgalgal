"""P1_on 효과가 소모품 상태로 설명되는지 확인 (진단, CLAUDE.md 14.4). 새 파일로 저장한다.

대상 변수 (모두 clean, 압력 구간 기준):
  P1_on = RETAINER_RING_PRESSURE, "압력 > 0 이고 STAGE_ROTATION > 0" 인 행만 평균
  P1    = RETAINER_RING_PRESSURE, 압력 구간 전체 평균
  P3    = 에어백 4개(MAIN_OUTER, CENTER, RIPPLE, EDGE) 평균, 압력 구간 전체 평균
  T     = 압력 구간 중 STAGE_ROTATION > 0 인 행의 비율

통제 조건 4가지 (컬럼명은 data/processed/COLUMNS.md 4절에서 확인한 실제 이름):
  1 base        t_start + start_POLISH_TABLE_w                     (7번까지 쓰던 조건)
  2 usage6      t_start + 소모품 6개 시작값 start_{U}_w
  3 usage6+since  2번 + 마지막 교체 이후 누적량 since_{U}_w (DRESSER_TABLE은 컬럼 없음 → 5개)
  4 usage6+t23    2번 + t_start의 2차·3차 항

4번의 시간 항은 초 단위 3제곱이 1e20까지 커져 수치가 불안정하므로,
평균을 뺀 일(day) 단위로 바꾼 뒤 제곱·세제곱한다 (선형 통제와 동치인 재척도).
`since_*`는 스테이지 첫 리셋 이전 웨이퍼에서 NaN이라 3번의 표본이 줄어든다 → n_used를 같이 싣는다.

대상: train, 이상치 제외, 전체-0 센서 웨이퍼 제외(clean). MRR은 상관 계산에만 쓴다.

실행: python src/pv_partial_consumables.py
출력: results/diagnostics/2026-09-26_김혜연_pv_partial_consumables.csv (+ 같은 이름의 .yaml)
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402
from pv_check import AIRBAG, AUTHOR  # noqa: E402
from pv_check_followup import partial_corr  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"2026-09-26_{AUTHOR}_pv_partial_consumables.csv"
USAGE6 = ["BACKING_FILM", "DRESSER", "POLISH_TABLE", "DRESSER_TABLE", "MEMBRANE", "PSHEET"]
START6 = [f"start_{u}_w" for u in USAGE6]
SINCE5 = [f"since_{u}_w" for u in USAGE6 if u != "DRESSER_TABLE"]   # DRESSER_TABLE은 리셋 없음
T23 = ["t_day2", "t_day3"]
CONDITIONS = {
    "1_base": ["t_start", "start_POLISH_TABLE_w"],
    "2_usage6": ["t_start"] + START6,
    "3_usage6_since": ["t_start"] + START6 + SINCE5,
    "4_usage6_t23": ["t_start"] + START6 + T23,
}
TARGETS = ["P1_on", "P1", "P3", "T"]


def wafer_values() -> pd.DataFrame:
    raw, _ = dedup(load_sensors("train"))
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[(tab["split"] == "train") & (tab["is_outlier"] == 0) & (tab["n_allzero_sensors"] == 0)]
    missing = [c for c in ["t_start"] + START6 + SINCE5 if c not in tab.columns]
    if missing:
        raise KeyError(f"집계표에 없는 컬럼: {missing}")
    tab = tab[KEY + ["group", "MRR", "t_start"] + START6 + SINCE5].copy()
    raw = raw.merge(tab[KEY], on=KEY)

    polish = (raw[list(PRESSURE)] > 0).any(axis=1).to_numpy()
    on = polish & (raw["STAGE_ROTATION"] > 0).to_numpy()
    p = raw[polish]
    w = pd.DataFrame({"P1": p.groupby(KEY)["RETAINER_RING_PRESSURE"].mean()})
    w["P3"] = p[AIRBAG].mean(axis=1).groupby([p["WAFER_ID"], p["STAGE"]]).mean()
    w["T"] = (p[KEY + ["STAGE_ROTATION"]].assign(v=lambda d: d["STAGE_ROTATION"] > 0)
              .groupby(KEY)["v"].mean())
    w["P1_on"] = raw[on].groupby(KEY)["RETAINER_RING_PRESSURE"].mean()
    m = tab.merge(w, on=KEY, how="left")
    day = (m["t_start"] - m["t_start"].mean()) / 86400          # 4번용 시간 곡선 항
    m["t_day2"], m["t_day3"] = day ** 2, day ** 3
    print(f"clean 웨이퍼 {len(m):,}장 (train, 이상치·전체-0 센서 제외)")
    print("통제 변수 결측(NaN) 웨이퍼 수:")
    print(m[START6 + SINCE5].isna().sum().to_string())
    return m


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    m = wafer_values()

    rows = []
    for group, d in m.groupby("group"):
        for x in TARGETS:
            r = d[x].corr(d["MRR"])
            for cond, cols in CONDITIONS.items():
                z = d[cols]
                ok = d[x].notna() & d["MRR"].notna() & z.notna().all(axis=1)
                rows.append({"group": group, "x": x, "y": "MRR", "condition": cond,
                             "n_wafer": len(d), "n_valid_x": int(d[x].notna().sum()),
                             "n_used": int(ok.sum()), "corr": r,
                             "partial_corr": partial_corr(d[x], d["MRR"], z),
                             "n_controls": len(cols), "controls": "+".join(cols)})
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-26", f"author: {AUTHOR}", "script: src/pv_partial_consumables.py",
        "question: P1_on의 MRR 상관이 소모품 상태로 설명되는가",
        "input: data/raw/train 센서 + data/processed/wafer_table.csv",
        "wafers: train, is_outlier 제외, n_allzero_sensors == 0 (clean)",
        "agg: polish (압력 6개 중 하나라도 >0 인 행) / P1_on 은 그중 STAGE_ROTATION >0 인 행만",
        f"targets: {TARGETS}",
        f"usage_columns: {START6} (COLUMNS.md 4절 start_{{U}}_w = 웨이퍼 첫 행의 카운터 값)",
        f"since_columns: {SINCE5} (since_DRESSER_TABLE_w 는 리셋이 없어 집계표에 없음)",
        "t_curve: t_day = (t_start - 평균)/86400, 2차·3차 항 (초 단위 3제곱의 수치 불안정 회피)",
        "conditions:",
        *[f"  {k}: {v}" for k, v in CONDITIONS.items()],
        "metrics: Pearson. partial_corr = 통제 변수 선형 회귀 잔차끼리의 Pearson (완전 관측 행만)",
        "note: since_* 는 스테이지 첫 리셋 이전 웨이퍼가 NaN → 조건 3의 n_used 가 작다",
        "related: 2026-09-26_김혜연_p1_on_corr.csv (조건 1의 P1_on 값과 같아야 한다)",
        "note2: 진단 결과 (14.4). MRR은 상관 계산에만 사용",
        "",
    ]), encoding="utf-8")

    print("\n[부분상관] 행 = 변수 × 통제 조건, 열 = 그룹")
    print(res.pivot_table(index=["x", "condition"], columns="group",
                          values="partial_corr").reindex(TARGETS, level=0).round(3).to_string())
    print("\n[n_used]")
    print(res.pivot_table(index="condition", columns="group", values="n_used").to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

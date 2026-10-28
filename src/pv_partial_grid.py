"""P·V 후보 전체의 MRR 상관과 부분상관 (진단, CLAUDE.md 14.4). 결과는 새 파일로 저장한다.

1. P1, P2, V1~V5, P1×V1~V5 각각의 MRR 상관과 부분상관
   (통제: 웨이퍼 시작 시각 t_start, POLISHING_TABLE 사용량 start_POLISH_TABLE_w)
2. V3(압력 구간 STAGE 회전 평균)와 "압력 구간 중 STAGE_ROTATION이 켜진 시간 비율"의 상관
   → 높으면 V3는 속도가 아니라 켜진 시간을 재는 값이다.

대상: train, 이상치 제외, 전체-0 센서 웨이퍼 제외(clean). 압력 구간 = 압력 6개 중 하나라도 > 0.
MRR은 상관 계산에만 쓴다.

앞서 낸 `..._pv_check_followup.csv`(커밋됨)와 겹치는 항목이 있다. 그 파일은 그대로 두고
여기서는 P·V 후보 전체로 격자를 넓혀 새 파일에 저장한다.

실행: python src/pv_partial_grid.py
출력: results/diagnostics/{날짜}_김혜연_pv_partial_grid.csv (+ 같은 이름의 .yaml)
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402
from pv_check import AUTHOR, V_NAMES, row_candidates  # noqa: E402
from pv_check_followup import CONTROLS, partial_corr  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"{date.today().isoformat()}_{AUTHOR}_pv_partial_grid.csv"
CANDIDATES = ["P1", "P2"] + V_NAMES + [f"P1x{v}" for v in V_NAMES]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    raw, _ = dedup(load_sensors("train"))
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[(tab["split"] == "train") & (tab["is_outlier"] == 0) & (tab["n_allzero_sensors"] == 0)]
    tab = tab[KEY + ["group", "MRR"] + CONTROLS]
    raw = raw.merge(tab[KEY], on=KEY)

    polish = (raw[list(PRESSURE)] > 0).any(axis=1).to_numpy()
    w = pd.concat([raw[KEY], row_candidates(raw)], axis=1)[polish].groupby(KEY).mean()
    w["stage_on_frac"] = (raw.loc[polish, KEY + ["STAGE_ROTATION"]]
                          .assign(on=lambda d: d["STAGE_ROTATION"] != 0).groupby(KEY)["on"].mean())
    m = tab.merge(w, on=KEY, how="left")
    print(f"clean 웨이퍼 {len(m):,}장 (train, 이상치·전체-0 센서 제외), 압력 구간 평균 기준")

    rows = []
    for group, d in m.groupby("group"):
        z = d[CONTROLS]
        for c in CANDIDATES:  # 1번
            rows.append({"check": "1_corr_partial", "group": group, "x": c, "y": "MRR", "n": len(d),
                         "corr": d[c].corr(d["MRR"]),
                         "partial_corr": partial_corr(d[c], d["MRR"], z),
                         "controls": "+".join(CONTROLS)})
        for c in CONTROLS:  # 통제 변수 자체의 영향 크기
            rows.append({"check": "1_corr_partial", "group": group, "x": c, "y": "MRR", "n": len(d),
                         "corr": d[c].corr(d["MRR"]), "partial_corr": None, "controls": ""})
        rows.append({"check": "2_v3_is_time", "group": group, "x": "V3", "y": "stage_on_frac",  # 2번
                     "n": len(d), "corr": d["V3"].corr(d["stage_on_frac"]),
                     "partial_corr": partial_corr(d["V3"], d["stage_on_frac"], z),
                     "controls": "+".join(CONTROLS)})
        rows.append({"check": "2_v3_is_time", "group": group, "x": "stage_on_frac", "y": "MRR",
                     "n": len(d), "corr": d["stage_on_frac"].corr(d["MRR"]),
                     "partial_corr": partial_corr(d["stage_on_frac"], d["MRR"], z),
                     "controls": "+".join(CONTROLS)})
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {date.today().isoformat()}", f"author: {AUTHOR}", "script: src/pv_partial_grid.py",
        "input: data/raw/train 센서 + data/processed/wafer_table.csv",
        "wafers: train, is_outlier 제외, n_allzero_sensors == 0 (clean)",
        "agg: polish (압력 6개 중 하나라도 > 0 인 행만 평균)",
        f"candidates: {CANDIDATES}",
        f"controls: {CONTROLS}",
        "metrics: Pearson. partial_corr = 통제 변수 선형 회귀 잔차끼리의 Pearson",
        "related: 2026-09-26_김혜연_pv_check_followup.csv (항목 일부 중복, 그 파일은 유지)",
        "note: 진단 결과 (14.4). MRR은 상관 계산에만 사용",
        "",
    ]), encoding="utf-8")

    for check, d in res.groupby("check"):
        print(f"\n[{check}]")
        print(d.pivot_table(index=["x", "y"], columns="group", values=["corr", "partial_corr"],
                            dropna=False).round(3).to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

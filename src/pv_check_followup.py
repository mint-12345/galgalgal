"""pv_check 추가 확인 (진단, CLAUDE.md 14.4). 결과는 새 파일로 저장한다.

1. P1 단독 / V4 단독 / P1×V4 의 MRR 상관 (clean, 압력 구간)
2. V3(웨이퍼 평균)이 "압력 구간 중 STAGE_ROTATION이 켜진 시간 비율"과 얼마나 같은지
   → 높으면 V3는 속도가 아니라 '켜진 시간'을 재는 값이다.
3. P1×V3, P2, P3 와 MRR의 부분상관 (t_start와 POLISHING_TABLE 사용량의 영향 제거)
   → 시간 추세 때문에 생긴 상관인지 확인.
4. P3를 구성하는 에어백 4개(MAIN_OUTER, CENTER, RIPPLE, EDGE) 각각의 MRR 상관·부분상관
   → 평균으로 묶기 전에 개별 신호가 서로 다른 말을 하는지 확인.

대상: train, 이상치 제외, 전체-0 센서 웨이퍼 제외(clean). 압력 구간 = 압력 6개 중 하나라도 > 0.
MRR은 상관 계산에만 쓴다.

실행: python src/pv_check_followup.py
출력: results/diagnostics/{날짜}_김혜연_pv_check_followup.csv (+ 같은 이름의 .yaml)
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402
from pv_check import AUTHOR, row_candidates  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"{date.today().isoformat()}_{AUTHOR}_pv_check_followup.csv"
CONTROLS = ["t_start", "start_POLISH_TABLE_w"]
AIRBAG_NAMES = {"MAIN_OUTER_AIR_BAG_PRESSURE": "MAIN_OUTER", "CENTER_AIR_BAG_PRESSURE": "CENTER",
                "RIPPLE_AIR_BAG_PRESSURE": "RIPPLE", "EDGE_AIR_BAG_PRESSURE": "EDGE"}


def partial_corr(x: pd.Series, y: pd.Series, z: pd.DataFrame) -> float:
    """z의 선형 영향을 뺀 x, y 잔차의 Pearson 상관."""
    ok = x.notna() & y.notna() & z.notna().all(axis=1)
    if ok.sum() < len(z.columns) + 3:
        return np.nan
    Z = np.column_stack([np.ones(ok.sum()), z[ok].to_numpy(float)])
    rx = x[ok].to_numpy(float) - Z @ np.linalg.lstsq(Z, x[ok].to_numpy(float), rcond=None)[0]
    ry = y[ok].to_numpy(float) - Z @ np.linalg.lstsq(Z, y[ok].to_numpy(float), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    raw, _ = dedup(load_sensors("train"))
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[(tab["split"] == "train") & (tab["is_outlier"] == 0) & (tab["n_allzero_sensors"] == 0)]
    tab = tab[KEY + ["group", "MRR", "t_start", "start_POLISH_TABLE_w"]]
    raw = raw.merge(tab[KEY], on=KEY)

    polish = (raw[list(PRESSURE)] > 0).any(axis=1)
    cand = pd.concat([raw[KEY], row_candidates(raw)], axis=1)[polish.to_numpy()]
    w = cand.groupby(KEY).mean()
    # 4번용: 에어백 4개 각각의 압력 구간 평균
    w = w.join(raw.loc[polish.to_numpy(), KEY + list(AIRBAG_NAMES)].groupby(KEY).mean())
    # 압력 구간 중 STAGE_ROTATION이 켜진 행의 비율
    w["stage_on_frac"] = (raw.loc[polish, ["WAFER_ID", "STAGE", "STAGE_ROTATION"]]
                          .assign(on=lambda d: d["STAGE_ROTATION"] != 0)
                          .groupby(KEY)["on"].mean())
    m = tab.merge(w, on=KEY, how="left")
    print(f"clean 웨이퍼 {len(m):,}장 (train, 이상치·전체-0 센서 제외)")

    rows = []
    for group, d in m.groupby("group"):
        z = d[CONTROLS]
        for c in ["P1", "V4", "P1xV4"]:  # 1번
            rows.append({"check": "1_corr", "group": group, "x": c, "y": "MRR", "n": len(d),
                         "corr": d[c].corr(d["MRR"]), "partial_corr": np.nan, "controls": ""})
        rows.append({"check": "2_v3_is_time", "group": group, "x": "V3", "y": "stage_on_frac",  # 2번
                     "n": len(d), "corr": d["V3"].corr(d["stage_on_frac"]),
                     "partial_corr": np.nan, "controls": ""})
        rows.append({"check": "2_v3_is_time", "group": group, "x": "stage_on_frac", "y": "MRR",
                     "n": len(d), "corr": d["stage_on_frac"].corr(d["MRR"]),
                     "partial_corr": np.nan, "controls": ""})
        for c in ["P1xV3", "P2", "P3"]:  # 3번
            rows.append({"check": "3_partial", "group": group, "x": c, "y": "MRR", "n": len(d),
                         "corr": d[c].corr(d["MRR"]),
                         "partial_corr": partial_corr(d[c], d["MRR"], z),
                         "controls": "+".join(CONTROLS)})
        for col, name in AIRBAG_NAMES.items():  # 4번: P3를 구성하는 에어백 4개 각각
            rows.append({"check": "4_airbag", "group": group, "x": name, "y": "MRR", "n": len(d),
                         "corr": d[col].corr(d["MRR"]),
                         "partial_corr": partial_corr(d[col], d["MRR"], z),
                         "controls": "+".join(CONTROLS)})
        # 통제 변수 자체와 MRR의 상관 (시간 추세가 실제로 있는지)
        for c in CONTROLS:
            rows.append({"check": "3_partial", "group": group, "x": c, "y": "MRR", "n": len(d),
                         "corr": d[c].corr(d["MRR"]), "partial_corr": np.nan, "controls": ""})
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {date.today().isoformat()}", f"author: {AUTHOR}", "script: src/pv_check_followup.py",
        "base: results/diagnostics/{}_{}_pv_check.csv 의 추가 확인".format(date.today().isoformat(), AUTHOR),
        "input: data/raw/train 센서 + data/processed/wafer_table.csv",
        "wafers: train, is_outlier 제외, n_allzero_sensors == 0 (clean)",
        "agg: polish (압력 6개 중 하나라도 > 0 인 행만 평균)",
        "checks: {1: P1/V4/P1xV4 상관, 2: V3 vs 압력구간 내 STAGE 켜진 시간 비율, 3: 부분상관,"
        " 4: P3 구성 에어백 4개 각각의 상관·부분상관}",
        f"controls: {CONTROLS}",
        "metrics: Pearson. partial_corr = 통제 변수 선형 회귀 잔차끼리의 Pearson",
        "note: 진단 결과 (14.4). MRR은 상관 계산에만 사용",
        "",
    ]), encoding="utf-8")

    for check, d in res.groupby("check"):
        print(f"\n[{check}]")
        print(d.pivot_table(index=["x", "y"], columns="group",
                            values=["corr", "partial_corr"]).round(3).to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

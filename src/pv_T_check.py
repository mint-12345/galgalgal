"""압력 × '켜진 시간 비율 T' 조합의 MRR 상관·부분상관 (진단, CLAUDE.md 14.4). 새 파일로 저장한다.

T = 압력 구간(압력 6개 중 하나라도 > 0)에서 STAGE_ROTATION이 0이 아닌 행의 비율.
앞선 진단에서 V3(STAGE 회전 평균)가 T와 거의 같은 값(상관 0.97)으로 나왔으므로,
"압력 × 켜진 시간"이 MRR을 얼마나 설명하는지를 T로 직접 확인한다.

항목 (모두 clean, 압력 구간 평균, 그룹별):
  T 단독
  에어백 4개(MAIN_OUTER, CENTER, RIPPLE, EDGE) 각각 × T
  P3 × T, P1 × T, P2 × T
  비교용으로 각 압력 단독(×T 안 한 값)도 같이 낸다 (base).
통제: 웨이퍼 시작 시각 t_start, POLISHING_TABLE 사용량 start_POLISH_TABLE_w.

곱은 웨이퍼 단위로 만든다 (압력의 압력 구간 평균 × T). T 자체가 웨이퍼 단위 값이라
행 단위 곱이 정의되지 않기 때문이다.

대상: train, 이상치 제외, 전체-0 센서 웨이퍼 제외(clean). MRR은 상관 계산에만 쓴다.

실행: python src/pv_T_check.py
출력: results/diagnostics/{날짜}_김혜연_pv_T_check.csv (+ 같은 이름의 .yaml)
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402
from pv_check import AUTHOR, row_candidates  # noqa: E402
from pv_check_followup import AIRBAG_NAMES, CONTROLS, partial_corr  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"{date.today().isoformat()}_{AUTHOR}_pv_T_check.csv"
# ×T 를 붙일 압력들: 에어백 4개 각각 + P3(에어백 평균) + P1 + P2
BASES = list(AIRBAG_NAMES.values()) + ["P3", "P1", "P2"]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    raw, _ = dedup(load_sensors("train"))
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[(tab["split"] == "train") & (tab["is_outlier"] == 0) & (tab["n_allzero_sensors"] == 0)]
    tab = tab[KEY + ["group", "MRR"] + CONTROLS]
    raw = raw.merge(tab[KEY], on=KEY)

    polish = (raw[list(PRESSURE)] > 0).any(axis=1).to_numpy()
    w = pd.concat([raw[KEY], row_candidates(raw)], axis=1)[polish].groupby(KEY).mean()[["P1", "P2", "P3"]]
    w = w.join(raw[polish][KEY + list(AIRBAG_NAMES)].groupby(KEY).mean()
               .rename(columns=AIRBAG_NAMES))           # 에어백 4개 각각의 압력 구간 평균
    w["T"] = (raw[polish][KEY + ["STAGE_ROTATION"]]
              .assign(on=lambda d: d["STAGE_ROTATION"] != 0).groupby(KEY)["on"].mean())
    for b in BASES:
        w[f"{b}xT"] = w[b] * w["T"]
    m = tab.merge(w, on=KEY, how="left")
    print(f"clean 웨이퍼 {len(m):,}장 (train, 이상치·전체-0 센서 제외), 압력 구간 기준")

    rows = []
    for group, d in m.groupby("group"):
        z = d[CONTROLS]
        items = [("T_only", "T")] + [("base_xT", f"{b}xT") for b in BASES] \
            + [("base_only", b) for b in BASES]         # 비교용: ×T 안 한 값
        for check, c in items:
            rows.append({"check": check, "group": group, "x": c, "y": "MRR", "n": len(d),
                         "corr": d[c].corr(d["MRR"]),
                         "partial_corr": partial_corr(d[c], d["MRR"], z),
                         "controls": "+".join(CONTROLS)})
        for c in CONTROLS:                              # 통제 변수 자체와 MRR의 상관
            rows.append({"check": "control", "group": group, "x": c, "y": "MRR", "n": len(d),
                         "corr": d[c].corr(d["MRR"]), "partial_corr": None, "controls": ""})
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {date.today().isoformat()}", f"author: {AUTHOR}", "script: src/pv_T_check.py",
        "input: data/raw/train 센서 + data/processed/wafer_table.csv",
        "wafers: train, is_outlier 제외, n_allzero_sensors == 0 (clean)",
        "agg: polish (압력 6개 중 하나라도 > 0 인 행만 평균)",
        "T: 압력 구간에서 STAGE_ROTATION != 0 인 행의 비율 (웨이퍼 단위)",
        f"bases: {BASES} (에어백 4개는 각각의 압력 구간 평균, P3 = 에어백 4개 평균)",
        "products: 웨이퍼 단위 곱 (압력 구간 평균 × T). T가 웨이퍼 단위라 행 단위 곱은 정의되지 않음",
        f"controls: {CONTROLS}",
        "metrics: Pearson. partial_corr = 통제 변수 선형 회귀 잔차끼리의 Pearson",
        "related: 2026-09-26_김혜연_pv_check_followup.csv, _pv_partial_grid.csv (두 파일 모두 유지)",
        "note: 진단 결과 (14.4). MRR은 상관 계산에만 사용",
        "",
    ]), encoding="utf-8")

    for check, d in res.groupby("check"):
        print(f"\n[{check}]")
        print(d.pivot_table(index="x", columns="group", values=["corr", "partial_corr"],
                            dropna=False).round(3).to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

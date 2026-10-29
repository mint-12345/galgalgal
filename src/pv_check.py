"""P·V 후보의 그룹 내 변동계수와 MRR 상관 (진단, CLAUDE.md 14.4).

질문: P·V가 그룹 안에서 MRR 변동을 설명할 만큼 변하는가?

대상: train 웨이퍼, 이상치(is_outlier) 제외. 라벨은 MRR 상관 계산에만 쓰고 피처로 쓰지 않는다.
후보 값은 `data/processed/wafer_table.csv`에 없으므로 **원본 센서에서 웨이퍼 단위로 새로 계산**한다.

정의 (CLAUDE.md 8절 + docs/reports/repro_김혜연.md 물리 파트 제안):
  P1 = RETAINER_RING_PRESSURE
  P2 = PRESSURIZED_CHAMBER_PRESSURE
  P3 = MAIN_OUTER / CENTER / RIPPLE / EDGE 에어백 4개의 행 평균
  V1 = STAGE_ROTATION - WAFER_ROTATION
  V2 = HEAD_ROTATION - (STAGE_ROTATION - WAFER_ROTATION)
  V3 = STAGE_ROTATION
  V4 = HEAD_ROTATION                      (제안: 항상 켜져 있는 유일한 회전)
  V5 = 챔버별로 '켜진 회전' 한 개          (제안: 챔버 6은 STAGE, 나머지는 WAFER.
                                          2026-09-26 진단의 챔버별 비0 비율에서 더 높은 쪽)
  P×V = 행 단위로 곱한 뒤 평균 (∫P·V dt 의 이산 근사)

집계 두 가지:
  (a) whole   = 웨이퍼의 모든 행 평균
  (b) polish  = 압력이 걸린 행만 평균 (압력 6개 중 하나라도 > 0)

부분집합 두 가지:
  all       = 이상치 제외 train 전체
  clean     = 여기서 전체-0 센서 웨이퍼(5.4절)까지 제외

실행: python src/pv_check.py
출력: results/diagnostics/{날짜}_김혜연_pv_check.csv (+ 같은 이름의 .yaml)
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402

AUTHOR = "김혜연"
OUT = ROOT / "results" / "diagnostics" / f"{date.today().isoformat()}_{AUTHOR}_pv_check.csv"
AIRBAG = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
          "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE"]
V5_MAP = {1: "WAFER_ROTATION", 2: "WAFER_ROTATION", 3: "WAFER_ROTATION",
          4: "WAFER_ROTATION", 5: "WAFER_ROTATION", 6: "STAGE_ROTATION"}
P_NAMES = ["P1", "P2", "P3"]
V_NAMES = ["V1", "V2", "V3", "V4", "V5"]


def row_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    """행 단위 P·V 후보와 그 곱."""
    d = pd.DataFrame(index=raw.index)
    d["P1"] = raw["RETAINER_RING_PRESSURE"]
    d["P2"] = raw["PRESSURIZED_CHAMBER_PRESSURE"]
    d["P3"] = raw[AIRBAG].mean(axis=1)
    d["V1"] = raw["STAGE_ROTATION"] - raw["WAFER_ROTATION"]
    d["V2"] = raw["HEAD_ROTATION"] - (raw["STAGE_ROTATION"] - raw["WAFER_ROTATION"])
    d["V3"] = raw["STAGE_ROTATION"]
    d["V4"] = raw["HEAD_ROTATION"]
    v5 = raw["CHAMBER"].map(V5_MAP)
    d["V5"] = np.where(v5 == "STAGE_ROTATION", raw["STAGE_ROTATION"], raw["WAFER_ROTATION"])
    for p in P_NAMES:
        for v in V_NAMES:
            d[f"{p}x{v}"] = d[p] * d[v]
    return d


def wafer_level(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cand = pd.concat([raw[KEY], row_candidates(raw)], axis=1)
    polish = (raw[list(PRESSURE)] > 0).any(axis=1)  # 압력이 걸린 행
    return {"whole": cand.groupby(KEY).mean(),
            "polish": cand[polish].groupby(KEY).mean()}


def summarize(w: pd.DataFrame, tab: pd.DataFrame, agg: str) -> list[dict]:
    """그룹 × 부분집합별 변동계수와 MRR 상관."""
    m = tab.merge(w, on=KEY, how="left")
    rows = []
    cols = P_NAMES + V_NAMES + [f"{p}x{v}" for p in P_NAMES for v in V_NAMES]
    for subset, sel in {"all": m, "clean": m[m["n_allzero_sensors"] == 0]}.items():
        for group, d in sel.groupby("group"):
            rows.append({"agg": agg, "subset": subset, "group": group, "candidate": "MRR",
                         "n_wafer": len(d), "mean": d["MRR"].mean(), "std": d["MRR"].std(),
                         "cv": d["MRR"].std() / d["MRR"].mean(), "corr_mrr": 1.0, "n_valid": len(d)})
            for c in cols:
                x = d[c]
                ok = x.notna()
                mean, std = x.mean(), x.std()
                rows.append({
                    "agg": agg, "subset": subset, "group": group, "candidate": c,
                    "n_wafer": len(d), "mean": mean, "std": std,
                    "cv": std / mean if mean not in (0, np.nan) and abs(mean) > 1e-9 else np.nan,
                    "corr_mrr": x[ok].corr(d["MRR"][ok]) if ok.sum() > 2 and std > 0 else np.nan,
                    "n_valid": int(ok.sum()),
                })
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    raw, n_dup = dedup(load_sensors("train"))
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[(tab["split"] == "train") & (tab["is_outlier"] == 0)][KEY + ["group", "MRR", "n_allzero_sensors"]]
    raw = raw.merge(tab[KEY], on=KEY)  # 이상치 웨이퍼 제외
    print(f"train 센서 {len(raw):,}행 (완전 중복 {n_dup:,}행 제거), 웨이퍼 {len(tab):,}장")

    rows = []
    for agg, w in wafer_level(raw).items():
        rows += summarize(w, tab, agg)
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {date.today().isoformat()}", f"author: {AUTHOR}", "script: src/pv_check.py",
        "input: data/raw/train 센서 + data/processed/wafer_table.csv (group, MRR, n_allzero_sensors)",
        "wafers: train, is_outlier 제외",
        "question: P·V가 그룹 안에서 MRR 변동을 설명할 만큼 변하는가",
        "P: {P1: RETAINER_RING_PRESSURE, P2: PRESSURIZED_CHAMBER_PRESSURE, P3: 에어백 4개 행 평균}",
        "V: {V1: STAGE-WAFER, V2: HEAD-(STAGE-WAFER), V3: STAGE, V4: HEAD, V5: 챔버별 켜진 회전(6번은 STAGE, 나머지 WAFER)}",
        "PxV: 행 단위 곱의 웨이퍼 평균",
        "agg: {whole: 전 행 평균, polish: 압력 6개 중 하나라도 >0 인 행만}",
        "subset: {all: 이상치 제외 train, clean: 전체-0 센서 웨이퍼까지 제외}",
        "metrics: cv = std/mean (ddof=1), corr_mrr = Pearson",
        "note: 진단 결과 (14.4). MRR은 상관 계산에만 사용",
        "",
    ]), encoding="utf-8")

    show = res[(res["subset"] == "clean")].pivot_table(index="candidate", columns=["agg", "group"],
                                                       values=["cv", "corr_mrr"])
    order = ["MRR"] + P_NAMES + V_NAMES + [f"{p}x{v}" for p in P_NAMES for v in V_NAMES]
    print("\n[clean] 변동계수와 MRR 상관")
    print(show.reindex(order).round(3).to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""P1_on의 MRR 상관·부분상관을 val 웨이퍼로도 계산 (진단, CLAUDE.md 14.4). 새 파일로 저장한다.

train에서 P1_on이 소모품 6개와 시작 시각을 통제한 뒤에도 부분상관 0.33 / 0.34로 남았다
(`2026-09-26_김혜연_pv_partial_consumables.csv`). 다른 split에서도 같은 크기가 나오는지 본다.

P1_on = RETAINER_RING_PRESSURE를 "압력 6개 중 하나라도 > 0 이고 STAGE_ROTATION > 0"인 행에서만 평균.
대상: train과 val 각각, 이상치 제외, 전체-0 센서 웨이퍼 제외(clean).
통제: 1 base = t_start + start_POLISH_TABLE_w / 2 usage6 = t_start + 소모품 6개 start_{U}_w.

val 라벨은 `data/raw/labels/CMP-validation-removalrate.csv`에서만 읽는다. 집계표의 val MRR과 같은지
확인한 뒤 쓴다. test 라벨과 `data/locked/`는 어떤 경로로도 읽지 않는다.

val은 표본이 작다 (clean 기준 A-456 185 / B-456 172 / A-123 27장 부근). 상관의 95% 신뢰구간을
Fisher z로 같이 낸다.

실행: python src/p1_on_val.py
출력: results/diagnostics/2026-09-27_김혜연_p1_on_val.csv (+ 같은 이름의 .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import OUTLIER_MRR, PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_labels, load_sensors  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from pv_check_followup import partial_corr  # noqa: E402
from pv_partial_consumables import START6  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"2026-09-27_{AUTHOR}_p1_on_val.csv"
CONDITIONS = {"1_base": ["t_start", "start_POLISH_TABLE_w"], "2_usage6": ["t_start"] + START6}
GROUPS = ["A-456", "B-456", "A-123"]


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    """상관의 95% 신뢰구간 (Fisher z)."""
    if n < 4 or not np.isfinite(r):
        return (np.nan, np.nan)
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(n - 3)
    return tuple(float(np.tanh(z + s * 1.96 * se)) for s in (-1, 1))


def wafer_values(split: str, tab: pd.DataFrame) -> pd.DataFrame:
    """split 하나의 clean 웨이퍼에 P1_on과 통제 변수를 붙인다."""
    t = tab[(tab["split"] == split) & (tab["n_allzero_sensors"] == 0)]
    t = t[~(t["MRR"] > OUTLIER_MRR)][KEY + ["group", "MRR", "t_start"] + START6].copy()
    raw, _ = dedup(load_sensors(split))
    raw = raw.merge(t[KEY], on=KEY)
    on = ((raw[list(PRESSURE)] > 0).any(axis=1) & (raw["STAGE_ROTATION"] > 0)).to_numpy()
    p1 = raw[on].groupby(KEY)["RETAINER_RING_PRESSURE"].mean().rename("P1_on")
    m = t.merge(p1, on=KEY, how="left")
    print(f"{split}: clean {len(m):,}장, 압력+STAGE on 행 {on.sum():,}, "
          f"P1_on 계산된 웨이퍼 {int(m['P1_on'].notna().sum()):,}장", flush=True)
    return m


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    val = load_labels("val").rename(columns={"AVG_REMOVAL_RATE": "MRR_raw"})
    chk = tab[tab["split"] == "val"].merge(val, on=KEY)
    if not np.allclose(chk["MRR"], chk["MRR_raw"]):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")
    print(f"val 라벨 {len(val):,}장을 data/raw/labels/에서 읽어 집계표와 대조 완료", flush=True)

    rows = []
    for split in ("train", "val"):
        m = wafer_values(split, tab)
        for group, d in m.groupby("group"):
            ok = d["P1_on"].notna() & d["MRR"].notna()
            r = d.loc[ok, "P1_on"].corr(d.loc[ok, "MRR"])
            lo, hi = fisher_ci(r, int(ok.sum()))
            for cond, cols in CONDITIONS.items():
                rows.append({"split": split, "group": group, "x": "P1_on", "y": "MRR",
                             "condition": cond, "n_wafer": len(d), "n_valid": int(ok.sum()),
                             "n_used": int((ok & d[cols].notna().all(axis=1)).sum()),
                             "corr": r, "corr_ci_lo": lo, "corr_ci_hi": hi,
                             "partial_corr": partial_corr(d["P1_on"], d["MRR"], d[cols]),
                             "n_controls": len(cols), "controls": "+".join(cols)})
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-27", f"author: {AUTHOR}", "script: src/p1_on_val.py",
        "question: P1_on 의 MRR 상관이 val 에서도 train 과 같은 크기인가",
        "input: data/processed/wafer_table.csv + data/raw/{train,val} 센서 + "
        "data/raw/labels/CMP-validation-removalrate.csv",
        "P1_on: RETAINER_RING_PRESSURE, 압력 6개 중 하나라도 >0 이고 STAGE_ROTATION >0 인 행만 평균",
        "wafers: split 별 clean (MRR > 400 이상치 제외, n_allzero_sensors == 0)",
        "conditions:",
        *[f"  {k}: {v}" for k, v in CONDITIONS.items()],
        "metrics: Pearson, 95% CI 는 Fisher z. partial_corr = 통제 변수 선형 회귀 잔차끼리의 Pearson",
        "related: 2026-09-26_김혜연_pv_partial_consumables.csv (train 값, 같은 정의)",
        "locked: data/locked/ 와 test 라벨은 열지 않았다",
        "note: 진단 결과 (14.4). MRR 은 상관 계산에만 사용",
        "",
    ]), encoding="utf-8")

    print("\n[P1_on 대 MRR]  상관 (95% CI) / 부분상관")
    for group in GROUPS:
        for split in ("train", "val"):
            d = res[(res["group"] == group) & (res["split"] == split)]
            b = d[d["condition"] == "1_base"].iloc[0]
            u = d[d["condition"] == "2_usage6"].iloc[0]
            print(f"  {group} {split:5s} n={b['n_valid']:4d}  "
                  f"상관 {b['corr']:6.3f} [{b['corr_ci_lo']:6.3f}, {b['corr_ci_hi']:6.3f}]  "
                  f"부분(base) {b['partial_corr']:6.3f}  부분(소모품6) {u['partial_corr']:6.3f}")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

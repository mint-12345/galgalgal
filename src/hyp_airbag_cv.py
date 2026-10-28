"""가설 2: 에어백(압력)이 설정값으로 고정돼 있는가 (진단, CLAUDE.md 14.4).

압력 6개 각각의 "연마 중 평균"(압력 6개 중 하나라도 > 0인 행만 평균)을 웨이퍼마다 구하고,
그룹별로 웨이퍼 간 변동계수(CV = 표준편차 / 평균)를 낸다.

보조 지표
- cv_median: 웨이퍼별 연마 중 **중앙값**의 웨이퍼 간 CV. 평균은 단계별 체류 시간 차이에 흔들리므로
  (가설 1에서 리테이너가 그랬다) 설정값 자체가 움직이는지는 중앙값 쪽이 더 직접적이다.
- within_cv: 웨이퍼 **안** 연마 중 CV의 중앙값. 웨이퍼 간 CV가 이보다 훨씬 작으면 웨이퍼마다 같은 파형이다.
- 비교 기준으로 MRR(train·val 라벨)의 CV와 가설 1의 P1_on CV를 같은 표에 넣는다.

대상: 세 split 센서 전부(라벨 불필요), 전체-0 센서 웨이퍼와 train 이상치 제외. split별 값도 함께 저장한다.
val 라벨은 `data/raw/labels/`에서 읽어 집계표와 대조한 뒤 MRR CV에만 쓴다. test 라벨과 `data/locked/`는 읽지 않는다.

실행: python src/hyp_airbag_cv.py
출력: results/diagnostics/2026-10-06_김혜연_airbag_cv.csv (+ 같은 이름의 .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE  # noqa: E402
from hyp_r2r_retainer import DATE, load_raw  # noqa: E402
from load import KEY, ROOT, load_labels  # noqa: E402
from pv_check import AUTHOR  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"{DATE}_{AUTHOR}_airbag_cv.csv"
GROUPS = ["A-456", "B-456", "A-123"]


def cv(x: pd.Series) -> float:
    return float(x.std() / x.mean())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    val = load_labels("val").rename(columns={"AVG_REMOVAL_RATE": "MRR_raw"})
    chk = tab[tab["split"] == "val"].merge(val, on=KEY)
    if len(chk) != len(val) or not np.allclose(chk["MRR"], chk["MRR_raw"]):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")
    clean = tab[(tab["n_allzero_sensors"] == 0) & (tab["is_outlier"] == 0)]
    print(f"clean 웨이퍼 {len(clean):,}장 (세 split). 센서 읽는 중", flush=True)

    raw = load_raw(clean[KEY])
    pol = raw[(raw[list(PRESSURE)] > 0).any(axis=1)]
    g = pol.groupby(KEY)[list(PRESSURE)]
    mean, med, std = g.mean(), g.median(), g.std()
    on = (raw[list(PRESSURE)] > 0).any(axis=1) & (raw["STAGE_ROTATION"] > 0)
    p1_on = raw[on].groupby(KEY)["RETAINER_RING_PRESSURE"].mean().rename("P1_on")
    w = clean[KEY + ["group", "split", "MRR"]].set_index(KEY)
    w = w.join(mean).join(med.add_suffix("__med")).join((std / mean).add_suffix("__wcv")).join(p1_on)
    print(f"연마 행 {len(pol):,}개, 웨이퍼 {len(mean):,}장", flush=True)

    rows = []
    for split in ("all", "train", "val", "test"):
        ws = w if split == "all" else w[w["split"] == split]
        for grp in GROUPS:
            d = ws[ws["group"] == grp]
            for col, abbr in PRESSURE.items():
                rows.append({"split": split, "group": grp, "variable": abbr, "n": int(d[col].notna().sum()),
                             "mean": d[col].mean(), "std": d[col].std(), "cv_mean": cv(d[col]),
                             "cv_median": cv(d[f"{col}__med"]), "within_cv": d[f"{col}__wcv"].median()})
            rows.append({"split": split, "group": grp, "variable": "P1_on (참고)",
                         "n": int(d["P1_on"].notna().sum()), "mean": d["P1_on"].mean(),
                         "std": d["P1_on"].std(), "cv_mean": cv(d["P1_on"])})
            if split != "test":
                m = d["MRR"].dropna()
                rows.append({"split": split, "group": grp, "variable": "MRR (비교 기준)", "n": len(m),
                             "mean": m.mean(), "std": m.std(), "cv_mean": cv(m)})
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/hyp_airbag_cv.py",
        "question: 가설 2 — 에어백(압력 6개)이 설정값으로 고정돼 있는가",
        "input: data/processed/wafer_table.csv + data/raw/{train,val,test} 센서 + "
        "data/raw/labels/CMP-validation-removalrate.csv (MRR CV 에만 사용)",
        "wafers: 세 split, n_allzero_sensors == 0 이고 is_outlier == 0",
        "polish_rows: 압력 6개 중 하나라도 > 0 인 행",
        "metrics: {cv_mean: 웨이퍼별 연마 중 평균의 웨이퍼 간 CV, cv_median: 웨이퍼별 연마 중 중앙값의 웨이퍼 간 CV, "
        "within_cv: 웨이퍼 안 연마 중 std/mean 의 중앙값}",
        "reference_rows: {P1_on: 가설 1 정의 (압력>0 & STAGE on 행의 RETAINER 평균), MRR: train·val 라벨}",
        "locked: data/locked/ 와 test 라벨은 열지 않았다",
        "note: 진단 결과 (14.4)", "",
    ]), encoding="utf-8")
    pd.set_option("display.width", 200)
    print("\n" + res[res["split"] == "all"].to_string(index=False))
    print("\n[cv_mean split별]")
    print(res.pivot_table(index=["group", "variable"], columns="split", values="cv_mean").round(3).to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

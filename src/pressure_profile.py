"""그룹별·챔버별 압력이 0이 아닌 행의 비율 (진단, CLAUDE.md 14.4).

목적: A-123에서 주 연마가 챔버 1이 아니라 2·3에서 일어나는지 확인한다.
train·val·test 센서를 모두 쓴다. 라벨은 쓰지 않는다.

실행: python src/pressure_profile.py
출력: results/diagnostics/{날짜}_{이름}_pressure_by_chamber.csv (+ 같은 이름의 .yaml)
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import dedup  # noqa: E402
from load import KEY, ROOT, SPLITS, load_sensors  # noqa: E402

AUTHOR = "김혜연"
VARS = {"RETAINER_RING_PRESSURE": "RETAINER", "MAIN_OUTER_AIR_BAG_PRESSURE": "MAIN_OUTER"}
OUT = ROOT / "results" / "diagnostics" / f"{date.today().isoformat()}_{AUTHOR}_pressure_by_chamber.csv"


def build() -> pd.DataFrame:
    raw = pd.concat([dedup(load_sensors(s))[0] for s in SPLITS], ignore_index=True)
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    raw = raw.merge(tab[KEY + ["group", "n_allzero_sensors"]], on=KEY, how="left", validate="many_to_one")
    # 전체-0 센서 웨이퍼(A-123에 228+40+46장, 5.4절)가 비율을 끌어내리므로 두 경우를 함께 본다.
    subsets = {"전체": raw, "전체-0 센서 웨이퍼 제외": raw[raw["n_allzero_sensors"] == 0]}

    rows = []
    for label, sub in subsets.items():
        for (group, chamber), d in sub.groupby(["group", "CHAMBER"]):
            for col, name in VARS.items():
                nz = d[col] != 0
                rows.append({
                    "subset": label,
                    "group": group,
                    "chamber": int(chamber),
                    "variable": name,
                    "n_rows": len(d),
                    "n_wafer": d.groupby(KEY).ngroups,
                    "nonzero_pct": round(nz.mean() * 100, 2),
                    "median_nonzero": round(d.loc[nz, col].median(), 2) if nz.any() else None,
                    "median_all": round(d[col].median(), 2),
                })
    return pd.DataFrame(rows).sort_values(["subset", "group", "chamber", "variable"], ignore_index=True)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    res = build()
    for label, d in res.groupby("subset", sort=False):
        print(f"\n[{label}] 압력이 0이 아닌 행의 비율 (%)")
        print(d.pivot_table(index=["group", "chamber"], columns="variable",
                            values=["nonzero_pct", "median_nonzero"]).to_string())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {date.today().isoformat()}",
        f"author: {AUTHOR}",
        "script: src/pressure_profile.py",
        "input: data/raw (train+val+test 센서, 라벨 미사용) + data/processed/wafer_table.csv (group 컬럼)",
        "question: A-123의 주 연마 챔버가 1인가 2·3인가",
        "variables: [RETAINER_RING_PRESSURE, MAIN_OUTER_AIR_BAG_PRESSURE]",
        "note: 진단 결과 (14.4). featureset 컬럼 없음",
        "",
    ]), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

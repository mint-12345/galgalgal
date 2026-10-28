"""P·V 후보 대 MRR 산점도 (진단, CLAUDE.md 14.4). 점 색은 웨이퍼 시작 시각.

그림 3장, 각 그림에 그룹 3칸(A-456, B-456, A-123). 칸마다 상관과 부분상관을 적는다.
부분상관 통제 변수: t_start, start_POLISH_TABLE_w.

1. MRR 대 P3   (에어백 4개 평균, 압력 구간)          → 시간 추세가 만든 겉보기 상관을 보여주는 그림
2. MRR 대 P1_on (RETAINER_RING_PRESSURE를 "압력 > 0 이고 STAGE_ROTATION > 0"인 행에서만 평균)
                                                     → 456 그룹용. 6번 관찰(평균 구간을 좁히는 것이 이득의 원인)의 확인
3. MRR 대 T    (압력 구간 중 STAGE_ROTATION이 켜진 행의 비율) → A-123용

대상: train, 이상치 제외, 전체-0 센서 웨이퍼 제외(clean). MRR은 그림과 상관 계산에만 쓴다.
P1_on의 상관·부분상관은 표로도 따로 저장한다.

실행: python src/scatter_pv.py
출력: docs/reports/figs/김혜연_scatter_{p3,p1_on,t}.png
      results/diagnostics/{날짜}_김혜연_p1_on_corr.csv (+ 같은 이름의 .yaml)
"""
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.font_manager import findSystemFonts, FontProperties  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402
from pv_check import AIRBAG, AUTHOR  # noqa: E402
from pv_check_followup import CONTROLS, partial_corr  # noqa: E402

FIGS = ROOT / "docs" / "reports" / "figs"
OUT = ROOT / "results" / "diagnostics" / f"{date.today().isoformat()}_{AUTHOR}_p1_on_corr.csv"
GROUPS = ["A-456", "B-456", "A-123"]
# 시작 시각용 순차 컬러맵 (연한 파랑 → 진한 파랑, 한 색조)
CMAP = LinearSegmentedColormap.from_list("t_start", ["#cfe1f7", "#2a78d6", "#0b3c78"])
PANELS = {
    "p3": ("P3 = 에어백 4개 평균 (압력 구간)", "P3",
           "MRR 대 P3 — 시간을 통제하면 사라지는 겉보기 상관"),
    "p1_on": ("P1_on = 리테이너 링 압력 (압력>0 & STAGE 회전 on 행 평균)", "P1_on",
              "MRR 대 P1_on — 456 그룹에서 시간 통제 후에도 남는 압력 신호"),
    "t": ("T = 압력 구간 중 STAGE 회전이 켜진 시간 비율", "T",
          "MRR 대 T — A-123에서 MRR을 끌고 가는 것은 압력이 아니라 연마 시간"),
}


def korean_font() -> str:
    """설치된 한글 폰트 중 첫 번째. 없으면 기본 폰트."""
    want = ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "Gulim", "Batang", "AppleGothic"]
    have = {FontProperties(fname=f).get_name() for f in findSystemFonts()}
    for name in want:
        if name in have:
            return name
    print("한글 폰트를 찾지 못했다. 축 이름이 깨질 수 있다.")
    return plt.rcParams["font.family"][0]


def wafer_values() -> pd.DataFrame:
    raw, _ = dedup(load_sensors("train"))
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[(tab["split"] == "train") & (tab["is_outlier"] == 0) & (tab["n_allzero_sensors"] == 0)]
    tab = tab[KEY + ["group", "MRR"] + CONTROLS].copy()
    raw = raw.merge(tab[KEY], on=KEY)

    polish = (raw[list(PRESSURE)] > 0).any(axis=1).to_numpy()
    on = polish & (raw["STAGE_ROTATION"] > 0).to_numpy()   # 압력 구간 + STAGE 회전 on
    p = raw[polish]
    w = pd.DataFrame({"P3": p[AIRBAG].mean(axis=1).groupby([p["WAFER_ID"], p["STAGE"]]).mean()})
    w.index.names = KEY
    w["T"] = (p[KEY + ["STAGE_ROTATION"]].assign(v=lambda d: d["STAGE_ROTATION"] > 0)
              .groupby(KEY)["v"].mean())
    w["P1_on"] = raw[on][KEY + ["RETAINER_RING_PRESSURE"]].groupby(KEY)["RETAINER_RING_PRESSURE"].mean()
    m = tab.merge(w, on=KEY, how="left")
    m["day"] = (m["t_start"] - m["t_start"].min()) / 86400
    print(f"clean 웨이퍼 {len(m):,}장, 압력 구간 행 {polish.sum():,} / 그중 STAGE on {on.sum():,}")
    print("P1_on 계산된 웨이퍼 수:\n" + m.groupby("group")["P1_on"].agg(["count", "size"]).to_string())
    return m


def draw(m: pd.DataFrame, key: str, font: str) -> Path:
    xlabel, col, title = PANELS[key]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.055, right=0.925, wspace=0.22)
    vmin, vmax = m["day"].min(), m["day"].max()
    for ax, group in zip(axes, GROUPS):
        d = m[m["group"] == group]
        ok = d[col].notna()
        sc = ax.scatter(d.loc[ok, col], d.loc[ok, "MRR"], c=d.loc[ok, "day"], cmap=CMAP,
                        vmin=vmin, vmax=vmax, s=16, linewidths=0.3, edgecolors="white", alpha=0.85)
        r = d.loc[ok, col].corr(d.loc[ok, "MRR"])
        pr = partial_corr(d[col], d["MRR"], d[CONTROLS])
        ax.set_title(f"{group}  (n = {int(ok.sum()):,})", fontsize=11, color="#333333")
        ax.text(0.03, 0.97, f"r = {r:.3f}\n부분 r = {pr:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d0d0d0", lw=0.8, alpha=0.9))
        ax.set_xlabel(xlabel if group == "B-456" else "", fontsize=10, color="#555555")
        ax.grid(True, color="#e8e8e8", lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbbbb")
        ax.tick_params(colors="#666666", labelsize=9)
    axes[0].set_ylabel("MRR", fontsize=10, color="#555555")
    fig.text(0.02, 0.955, title, fontsize=13, color="#222222", ha="left", va="top")
    fig.text(0.02, 0.885, "train, 이상치·전체-0 센서 웨이퍼 제외. 부분상관 통제: 웨이퍼 시작 시각, "
             "POLISHING_TABLE 사용량", fontsize=9.5, color="#777777", ha="left", va="top")
    cb = fig.colorbar(sc, cax=fig.add_axes([0.945, 0.16, 0.011, 0.60]))
    cb.set_label("웨이퍼 시작 시각 (첫 웨이퍼 기준 일)", fontsize=9.5, color="#555555", labelpad=10)
    cb.ax.tick_params(colors="#666666", labelsize=9)
    cb.outline.set_visible(False)
    out = FIGS / f"{AUTHOR}_scatter_{key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    font = korean_font()
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    FIGS.mkdir(parents=True, exist_ok=True)
    m = wafer_values()

    for key in PANELS:
        print(f"저장: {draw(m, key, font).relative_to(ROOT)}")

    rows = []
    for group, d in m.groupby("group"):   # P1_on 표 (그림에 적은 값과 같은 수치)
        ok = d["P1_on"].notna()
        rows.append({"group": group, "x": "P1_on", "y": "MRR", "n_wafer": len(d),
                     "n_valid": int(ok.sum()),
                     "mean": d["P1_on"].mean(), "std": d["P1_on"].std(),
                     "cv": d["P1_on"].std() / d["P1_on"].mean(),
                     "corr": d.loc[ok, "P1_on"].corr(d.loc[ok, "MRR"]),
                     "partial_corr": partial_corr(d["P1_on"], d["MRR"], d[CONTROLS]),
                     "controls": "+".join(CONTROLS)})
    res = pd.DataFrame(rows).round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {date.today().isoformat()}", f"author: {AUTHOR}", "script: src/scatter_pv.py",
        "input: data/raw/train 센서 + data/processed/wafer_table.csv",
        "wafers: train, is_outlier 제외, n_allzero_sensors == 0 (clean)",
        "P1_on: RETAINER_RING_PRESSURE, 압력 6개 중 하나라도 >0 이고 STAGE_ROTATION >0 인 행만 평균",
        f"controls: {CONTROLS}",
        "metrics: Pearson. partial_corr = 통제 변수 선형 회귀 잔차끼리의 Pearson",
        "figs: docs/reports/figs/김혜연_scatter_p3.png, _p1_on.png, _t.png (점 색 = 웨이퍼 시작 시각)",
        "related: 2026-09-26_김혜연_pv_T_check.csv (T·P3 수치는 그 파일과 같은 정의)",
        "note: 진단 결과 (14.4). MRR은 그림과 상관 계산에만 사용",
        "",
    ]), encoding="utf-8")
    print("\n" + res.to_string(index=False))
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

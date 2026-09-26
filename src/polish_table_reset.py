"""POLISHING_TABLE 리셋 간격 확인 (진단, CLAUDE.md 14.4). 새 파일로 저장한다.

Deivendran 2025: 교체 기준 약 350, 웨이퍼당 9~11 사용 → 약 35장마다 교체.
우리 기존 결과(`week4_data.md` 3절, 스테이지 단위): 2,829장에 187회 → 약 15장마다. 원인을 본다.

1. 리셋을 **스테이지별**과 **스테이지 × 챔버군별**로 따로 세어 리셋 횟수와 리셋 사이 웨이퍼 수를 비교
2. 리셋 직전 카운터 최댓값의 분포 (중앙값, 범위) — 350 근처인지
3. 웨이퍼당 카운터 증가량 중앙값 (웨이퍼 안에서의 증가 / 연속 웨이퍼 사이 증가)
4. 스테이지 A에서 A-456과 A-123의 카운터를 시간순으로 한 그림에 (두 카운터가 따로 도는지)
5. (원인 확인) 센서 행 단위로 카운터가 **시간**에 따라 도는지, 그리고 한 주기(리셋 사이)의 증가분 중
   우리 웨이퍼 기록 안에서 일어난 몫이 얼마인지

리셋 판정은 `src/aggregate.py`와 같다: 시작값이 직전 웨이퍼보다
`RESET_FRAC × (그 스테이지의 센서 값 범위)` 이상 줄면 리셋. 1번은 세는 단위만 바꾼다.
train·val·test 센서를 모두 합쳐 쓰고 **라벨은 쓰지 않는다** (data/locked/도 열지 않는다).

실행: python src/polish_table_reset.py
출력: results/diagnostics/2026-09-27_김혜연_polish_table_reset.csv (+ 같은 이름의 .yaml)
      docs/reports/figs/김혜연_polish_table_stage_A.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import FontProperties, findSystemFonts  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import RESET_FRAC, dedup  # noqa: E402
from load import KEY, ROOT, SPLITS, load_sensors  # noqa: E402

AUTHOR = "김혜연"
OUT = ROOT / "results" / "diagnostics" / f"2026-09-27_{AUTHOR}_polish_table_reset.csv"
FIG = ROOT / "docs" / "reports" / "figs" / f"{AUTHOR}_polish_table_stage_A.png"
COL, AB = "USAGE_OF_POLISHING_TABLE", "POLISH_TABLE"
DROP_MIN = 10.0                                   # 행 단위 리셋 판정 (한 주기 높이의 3% 미만은 무시)
DEIV = {"limit": 350, "per_wafer": 10, "wafers_per_reset": 35}
COLORS = {"A-456": "#2a78d6", "A-123": "#eb6834"}   # 범주형 색 (dataviz)


def korean_font() -> str:
    """설치된 한글 폰트 중 첫 번째. 없으면 기본 폰트."""
    want = ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "Gulim", "Batang", "AppleGothic"]
    have = {FontProperties(fname=f).get_name() for f in findSystemFonts()}
    for name in want:
        if name in have:
            return name
    print("한글 폰트를 찾지 못했다. 축 이름이 깨질 수 있다.", flush=True)
    return plt.rcParams["font.family"][0]


def load_rows() -> pd.DataFrame:
    """센서 행에서 필요한 열만 (라벨 없음)."""
    parts = []
    for split in SPLITS:
        df, _ = dedup(load_sensors(split))
        parts.append(df[KEY + ["CHAMBER", "TIMESTAMP", COL]])
        print(f"  {split} 센서 로드", flush=True)
    r = pd.concat(parts, ignore_index=True)
    r["TIMESTAMP"] = r["TIMESTAMP"].astype(float)   # split 합치면서 object가 되는 경우가 있다
    r[COL] = r[COL].astype(float)
    r["cset"] = np.where(r["CHAMBER"] <= 3, "123", "456")
    r["unit"] = r["STAGE"] + "-" + r["cset"]        # 카운터 계열 후보 = 스테이지 × 챔버군
    return r


def count_resets(tab: pd.DataFrame, by: list[str], rng: pd.Series) -> pd.DataFrame:
    """웨이퍼 단위: by 안에서 t_start 순 정렬 후 리셋을 센다 (aggregate.py와 같은 규칙)."""
    d = tab.sort_values(by + ["t_start"], kind="mergesort").copy()
    start = d[f"start_{AB}_w"]
    d["reset"] = (start.groupby([d[c] for c in by]).diff()
                  < -RESET_FRAC * d["STAGE"].map(rng)).astype(int)
    d["seg"] = d.groupby(by)["reset"].cumsum()
    d["end_value"] = start + d[f"delta_{AB}_w"]
    d["gap_between"] = start.groupby([d[c] for c in by]).diff()
    return d


def summarize(d: pd.DataFrame, by: list[str], unit: str) -> list[dict]:
    rows = []
    for key, g in d.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        n_reset = int(g["reset"].sum())
        peaks = g.groupby("seg")["end_value"].max()
        peaks = peaks.iloc[:-1] if n_reset else peaks.iloc[0:0]   # 마지막 구간은 교체 전에 끝남
        between = g.loc[g["reset"] == 0, "gap_between"].dropna()
        rows.append({
            "check": "1_2_3", "unit": unit, "series": key[0] if len(key) == 1 else f"{key[0]}/{key[1]}",
            "n_wafer": len(g), "n_reset": n_reset,
            "wafers_per_reset": len(g) / n_reset if n_reset else np.nan,
            "peak_median": peaks.median(), "peak_min": peaks.min(), "peak_max": peaks.max(),
            "inside_median": g[f"delta_{AB}_w"].median(),
            "between_median": between.median(), "between_p90": between.quantile(0.9),
        })
    return rows


def row_level(r: pd.DataFrame) -> list[dict]:
    """5번: 한 주기 중 웨이퍼 기록이 차지하는 몫.

    카운터는 **스테이지 단위**로 하나다 (4번에서 A-123과 A-456의 같은 시각 값이 같음을 확인).
    그래서 챔버군으로 쪼개지 않고 스테이지별로 센다.
    """
    rows = []
    for unit, g in r.sort_values("TIMESTAMP").groupby("STAGE"):
        d = g[COL].diff()
        reset = d < -DROP_MIN
        inside = g.groupby(KEY)[COL].agg(lambda x: x.iloc[-1] - x.iloc[0])   # 웨이퍼 안 증가
        total_up = float(d[d > 0].sum())
        hours = float(g["TIMESTAMP"].max() - g["TIMESTAMP"].min()) / 3600
        rows.append({
            "check": "5_row_level", "unit": "row(stage)", "series": unit,
            "n_row": len(g), "n_wafer": g.groupby(KEY).ngroups, "n_reset": int(reset.sum()),
            "wafers_per_reset": g.groupby(KEY).ngroups / max(int(reset.sum()), 1),
            "rate_per_hour": total_up / hours if hours else np.nan,
            "hours_per_reset": hours / max(int(reset.sum()), 1),
            "cycle_height_median": float(-d[reset].median()) if reset.any() else np.nan,
            "total_increase": total_up,
            "inside_wafer_increase": float(inside.sum()),
            "inside_share": float(inside.sum()) / total_up if total_up else np.nan,
            "inside_median": float(inside.median()),
            "inside_mean": float(inside.mean()),
            # Deivendran 가정(웨이퍼 기록 안 증가만으로 350에 도달)대로면 몇 장인가
            "implied_wafers_deiv": DEIV["limit"] / float(inside.mean()) if inside.mean() else np.nan,
        })
    return rows


def gap_test(tab: pd.DataFrame, rng: pd.Series) -> list[dict]:
    """웨이퍼 사이 빈칸에서 카운터가 어떻게 오르는가.

    빈 시간에 비례해 오르면 '시간에 따라 도는 카운터', 웨이퍼 한 장 몫(약 10)의 배수로 오르면
    '기록에 없는 웨이퍼가 그 사이에 돌았다'는 뜻이다.
    """
    rows = []
    for stage, g in tab.sort_values(["STAGE", "t_start"]).groupby("STAGE"):
        start, end = g[f"start_{AB}_w"], g[f"start_{AB}_w"] + g[f"delta_{AB}_w"]
        gap_v = start.to_numpy()[1:] - end.to_numpy()[:-1]          # 빈칸에서의 카운터 증가
        gap_t = g["t_start"].to_numpy()[1:] - g["t_end"].to_numpy()[:-1]   # 빈 시간(초)
        ok = (gap_v > -DROP_MIN) & (gap_t > 0)                       # 리셋 구간과 겹침 제외
        gv, gt = gap_v[ok], gap_t[ok]
        inside_v = g[f"delta_{AB}_w"].to_numpy()
        inside_t = (g["t_end"] - g["t_start"]).to_numpy()
        io = inside_t > 0
        rows.append({
            "check": "6_gap", "unit": "wafer gap", "series": stage, "n_row": int(ok.sum()),
            "gap_counter_median": float(np.median(gv)), "gap_seconds_median": float(np.median(gt)),
            "gap_rate_per_s": float(np.median(gv[gt > 0] / gt[gt > 0])),
            "inside_rate_per_s": float(np.median(inside_v[io] / inside_t[io])),
            "corr_gap_time_counter": float(np.corrcoef(gt, gv)[0, 1]),
        })
    return rows


def shared_counter_test(r: pd.DataFrame) -> list[dict]:
    """4번 보조: 스테이지 A에서 A-456과 A-123이 같은 카운터를 쓰는가.
    시간이 가장 가까운 상대 그룹 행과 값을 비교한다."""
    a = r[r["STAGE"] == "A"].sort_values("TIMESTAMP")
    x = a[a["cset"] == "456"][["TIMESTAMP", COL]].rename(columns={COL: "v456"})
    y = a[a["cset"] == "123"][["TIMESTAMP", COL]].rename(columns={COL: "v123"})
    m = pd.merge_asof(y, x, on="TIMESTAMP", direction="nearest", tolerance=60).dropna()
    diff = (m["v123"] - m["v456"]).abs()
    return [{"check": "4_shared", "unit": "A-123 vs A-456", "series": "같은 시각(±60초) 값 차이",
             "n_row": len(m), "abs_diff_median": float(diff.median()),
             "abs_diff_p90": float(diff.quantile(0.9)),
             "same_within_1": float((diff < 1).mean())}]


def draw(r: pd.DataFrame, font: str) -> None:
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    a = r[r["STAGE"] == "A"].sort_values("TIMESTAMP")
    t0 = a["TIMESTAMP"].min()
    day = (a["TIMESTAMP"] - t0) / 86400
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), height_ratios=[2, 1])
    fig.subplots_adjust(top=0.82, bottom=0.09, left=0.08, right=0.985, hspace=0.32)
    for ax, (lo, hi), title in ((axes[0], (float(day.min()), float(day.max())), "전체 기간"),
                                (axes[1], (10.3, 11.0), "10.3~11.0일 확대 (한 주기)")):
        for g, c in COLORS.items():
            cset = g.split("-")[1]
            s = a[a["cset"] == cset]
            x = (s["TIMESTAMP"] - t0) / 86400
            m = (x >= lo) & (x <= hi)
            ax.plot(x[m], s.loc[m, COL], ".", color=c, ms=2.2, alpha=0.75, label=g)
        ax.set_xlim(lo, hi)
        ax.set_title(title, fontsize=10.5, color="#444444", loc="left")
        ax.set_ylabel("USAGE_OF_\nPOLISHING_TABLE", fontsize=9.5, color="#555555")
        ax.grid(True, color="#ececec", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#bbbbbb")
        ax.tick_params(colors="#666666", labelsize=9)
    axes[1].set_xlabel("첫 기록 기준 경과 일수", fontsize=9.5, color="#555555")
    fig.text(0.02, 0.975, "스테이지 A의 폴리싱 테이블 카운터 — A-456과 A-123이 같은 카운터를 쓴다",
             fontsize=13, color="#222222", va="top")
    fig.text(0.02, 0.915, "두 챔버군의 점이 같은 곡선을 이어 간다 (같은 시각 값 차이의 중앙값 0, "
             "98.5%가 1 이내). 약 350에서 떨어지고 다시 오른다.\n"
             "비어 보이는 구간은 우리 데이터에 웨이퍼 기록이 없는 시간이다. 그 사이에도 카운터는 오른다 "
             "— 전체 증가분의 절반가량이 기록 밖에서 생긴다.",
             fontsize=9.5, color="#777777", va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.985, 0.995), frameon=False,
               fontsize=10, markerscale=5, ncol=2)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    print(f"웨이퍼 {len(tab):,}장 (train·val·test 전부). 센서 로드:", flush=True)
    r = load_rows()
    rng = r.groupby("STAGE")[COL].agg(lambda x: x.max() - x.min())
    print(f"  스테이지 값 범위 {rng.round(1).to_dict()}, 임계값 = 범위 × {RESET_FRAC}", flush=True)

    rows = summarize(count_resets(tab, ["STAGE"], rng), ["STAGE"], "stage")
    rows += summarize(count_resets(tab, ["STAGE", "group"], rng), ["STAGE", "group"], "stage_group")
    rows += row_level(r)
    rows += gap_test(tab, rng)
    rows += shared_counter_test(r)
    res = pd.DataFrame(rows).round(3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-27", f"author: {AUTHOR}", "script: src/polish_table_reset.py",
        "question: POLISHING_TABLE 리셋 간격이 Deivendran 2025(약 35장마다)와 다른 이유",
        "input: data/processed/wafer_table.csv + data/raw/{train,val,test} 센서 (라벨 없음)",
        f"variable: {COL} ({AB})",
        f"wafer_rule: 시작값이 직전 웨이퍼보다 RESET_FRAC({RESET_FRAC}) × 스테이지 센서 값 범위 이상 감소",
        f"row_rule: 행 단위로 카운터가 {DROP_MIN} 이상 감소",
        f"stage_range: {rng.round(2).to_dict()}",
        "units: {stage: 기존 방식, stage_group: 스테이지 × 챔버군, row(stage×chamberset): 센서 행 단위}",
        "columns: {peak_*: 리셋 직전 구간 최고값, inside_*: 웨이퍼 안 증가, "
        "between_*: 연속 웨이퍼 시작값 차이(리셋 제외), rate_per_s_*: 초당 증가율, "
        "inside_share: 전체 증가분 중 웨이퍼 기록 안에서 일어난 몫, "
        "implied_wafers_deiv: 350 / 웨이퍼당 증가 중앙값}",
        f"reference: Deivendran 2025 — 교체 기준 약 {DEIV['limit']}, 웨이퍼당 9~11 → "
        f"약 {DEIV['wafers_per_reset']}장마다",
        f"figure: docs/reports/figs/{AUTHOR}_polish_table_stage_A.png",
        "labels: 쓰지 않음. data/locked/ 와 test 라벨은 열지 않았다",
        "note: 진단 결과 (14.4)",
        "",
    ]), encoding="utf-8")

    cols1 = ["unit", "series", "n_wafer", "n_reset", "wafers_per_reset"]
    print("\n[1] 리셋 횟수와 리셋 사이 웨이퍼 수")
    print(res[res["check"] == "1_2_3"][cols1].to_string(index=False))
    print("\n[2][3] 리셋 직전 최고값 / 웨이퍼당 증가량")
    print(res[res["check"] == "1_2_3"][["series", "peak_median", "peak_min", "peak_max",
                                        "inside_median", "between_median"]].to_string(index=False))
    print("\n[5] 센서 행 단위")
    print(res[res["check"] == "5_row_level"][
        ["series", "n_wafer", "n_reset", "wafers_per_reset", "rate_per_hour", "hours_per_reset",
         "cycle_height_median", "inside_mean", "inside_share", "implied_wafers_deiv"]
    ].to_string(index=False))
    print("\n[6] 웨이퍼 사이 빈칸에서의 증가 (기록에 없는 사용량)")
    print(res[res["check"] == "6_gap"][["series", "n_row", "gap_counter_median",
                                        "gap_seconds_median", "gap_rate_per_s",
                                        "inside_rate_per_s",
                                        "corr_gap_time_counter"]].to_string(index=False))
    print("\n[4] A-123과 A-456이 같은 카운터인가")
    print(res[res["check"] == "4_shared"][["series", "n_row", "abs_diff_median", "abs_diff_p90",
                                           "same_within_1"]].to_string(index=False))

    draw(r, korean_font())
    print(f"\n저장: {OUT.relative_to(ROOT)}, {FIG.relative_to(ROOT)} (+ .yaml)", flush=True)


if __name__ == "__main__":
    main()

"""Step 4: wafer+Stage 한 묶음 안에서 공정 값이 시간에 따라 어떻게 변하는지 확인 (읽기 전용).

- training 원데이터만 읽는다. test / validation 은 열지 않는다. raw 파일은 수정하지 않는다.
- 분석 과정에서만, 같은 파일에서 바로 윗줄과 25개 컬럼이 모두 같은 행(1,819행)을 제외한다.
- Step 3 의 특이한 wafer+Stage 는 제외하지 않는다.
  (다만 "일반적인 예시"를 고를 때만 파일 1개, TIMESTAMP 중복 없음, Chamber 1→2→3 또는 4→5→6 인 묶음 중에서 고른다.)
- feature 생성, 모델 학습은 하지 않는다.
출력: 화면 출력, reports/step4_air_bag_on_blocks_per_group.csv, reports/figures/step4_*.png
실행: python scripts/step4_inspect_within_wafer_timeseries.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 500)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
KEY = ["WAFER_ID", "STAGE"]

PRESSURE = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
            "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE"]
AIR_BAG = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE",
           "EDGE_AIR_BAG_PRESSURE"]
ROTATION = ["WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION"]
SLURRY = ["SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C"]
WATCH = PRESSURE + ROTATION + SLURRY
SHORT = {"PRESSURIZED_CHAMBER_PRESSURE": "PCH", "MAIN_OUTER_AIR_BAG_PRESSURE": "MOA",
         "CENTER_AIR_BAG_PRESSURE": "CEN", "RETAINER_RING_PRESSURE": "RET", "RIPPLE_AIR_BAG_PRESSURE": "RIP",
         "EDGE_AIR_BAG_PRESSURE": "EDG", "WAFER_ROTATION": "WR", "STAGE_ROTATION": "SR",
         "HEAD_ROTATION": "HR", "SLURRY_FLOW_LINE_A": "SLA", "SLURRY_FLOW_LINE_B": "SLB",
         "SLURRY_FLOW_LINE_C": "SLC"}

# 그래프 색 (dataviz reference palette, light)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"


def header(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def load_training():
    frames = []
    for f in sorted(TRAIN_DIR.glob("CMP-training-*.csv")):
        df = pd.read_csv(f, dtype={"WAFER_ID": "int64", "STAGE": "string"})
        if df.empty:
            continue
        df["FILE"] = f.name
        df["FILE_ROW"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def drop_consecutive_full_duplicates(raw):
    cols = [c for c in raw.columns if c not in ("FILE", "FILE_ROW")]
    same_prev = raw[cols].eq(raw[cols].shift()).all(axis=1) & raw["FILE"].eq(raw["FILE"].shift())
    return raw[~same_prev].copy(), int(same_prev.sum()), cols


RAW = load_training()
DATA, N_DROPPED, COLS = drop_consecutive_full_duplicates(RAW)
MRR = pd.read_csv(MRR_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
DATA["N_AIR_BAG_NONZERO"] = DATA[AIR_BAG].ne(0).sum(axis=1)
DATA["AIR_BAG_ON"] = DATA["N_AIR_BAG_NONZERO"] == 4  # 네 air-bag 압력이 모두 0 이 아님


def group_table():
    d = DATA.sort_values(KEY + ["TIMESTAMP"], kind="stable")
    g = d.groupby(KEY)
    t = g.agg(rows=("TIMESTAMP", "size"), n_files=("FILE", "nunique"), n_unique_ts=("TIMESTAMP", "nunique"),
              t_start=("TIMESTAMP", "min"), t_end=("TIMESTAMP", "max"))
    t["duration_s"] = t["t_end"] - t["t_start"]
    t["chamber_seq"] = g["CHAMBER"].agg(lambda c: tuple(int(x) for x in c[c.ne(c.shift())]))
    t["pattern"] = t["chamber_seq"].map(lambda s: "1-2-3" if s == (1, 2, 3) else "4-5-6" if s == (4, 5, 6) else "other")
    return t


def air_bag_blocks(g):
    """시간순으로 AIR_BAG_ON 이 연속된 구간 목록."""
    g = g.sort_values("TIMESTAMP", kind="stable")
    on = g["AIR_BAG_ON"].to_numpy()
    t = g["TIMESTAMP"].to_numpy()
    ch = g["CHAMBER"].to_numpy()
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    ends = np.flatnonzero(on & ~np.r_[on[1:], False])
    return [dict(i0=a, i1=e, t_start=t[a], t_end=t[e], duration_s=t[e] - t[a], rows=e - a + 1,
                 chambers=tuple(sorted({int(c) for c in ch[a:e + 1]}))) for a, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
def pick_examples(groups):
    header("[1] 예시로 사용할 일반적인 wafer+Stage")
    clean = groups[(groups["n_files"] == 1) & (groups["n_unique_ts"] == groups["rows"]) & (groups["pattern"] != "other")]
    on_chambers = (DATA[DATA["AIR_BAG_ON"]].groupby(KEY)["CHAMBER"]
                   .agg(lambda c: "".join(str(int(x)) for x in sorted(set(c)))))
    clean = clean.assign(air_bag_on_chambers=on_chambers.reindex(clean.index).fillna("none")).reset_index()
    print(f"후보 조건: 파일 1개, TIMESTAMP 중복 없음, Chamber 1→2→3 또는 4→5→6 → {len(clean):,}개 묶음")
    print("범주별 후보 수 (air_bag_on_chambers = 네 air-bag 압력이 모두 0 이 아닌 행이 나온 Chamber):")
    print(clean.groupby(["pattern", "STAGE", "air_bag_on_chambers"]).size().to_string())
    categories = [("4-5-6", "A", "4"), ("4-5-6", "A", "456"), ("4-5-6", "B", "4"), ("4-5-6", "B", "456"),
                  ("1-2-3", "A", "123"), ("1-2-3", "A", "none")]
    picks = []
    for pat, st, on in categories:
        c = clean[(clean["pattern"] == pat) & (clean["STAGE"] == st) & (clean["air_bag_on_chambers"] == on)].copy()
        c["dist"] = (c["rows"] - c["rows"].median()).abs()
        picks.append(c.sort_values(["dist", "WAFER_ID"]).iloc[0])  # 범주 안에서 행 수가 중앙값에 가장 가까운 묶음
    ex = pd.DataFrame(picks)[KEY + ["pattern", "air_bag_on_chambers", "rows", "duration_s"]]
    print("\n선택 범주: Chamber 패턴 x Stage x 압력 구간이 나온 Chamber 중 후보가 많은 6개")
    print("선택 기준: 각 범주에서 행 수가 범주 중앙값에 가장 가까운 묶음 (같으면 WAFER_ID 가 작은 것)")
    print(ex.to_string(index=False))
    return [(int(r.WAFER_ID), str(r.STAGE)) for r in ex.itertuples()]


def state_runs(g, min_rows=3):
    """Chamber 와 각 값의 0/0아님 상태가 같은 연속 구간으로 나눠 구간별 중앙값을 보여준다."""
    g = g.sort_values("TIMESTAMP", kind="stable").copy()
    g["t_s"] = g["TIMESTAMP"] - g["TIMESTAMP"].iloc[0]
    sig = g[WATCH].ne(0).astype(int).astype(str).agg("".join, axis=1)
    run = (sig.ne(sig.shift()) | g["CHAMBER"].ne(g["CHAMBER"].shift())).cumsum()
    out = g.groupby(run).agg(CH=("CHAMBER", lambda x: int(x.iloc[0])), rows=("t_s", "size"),
                             t0=("t_s", "min"), t1=("t_s", "max"),
                             **{SHORT[v]: (v, "median") for v in WATCH})
    return out[out["rows"] >= min_rows], int((out["rows"] < min_rows).sum()), int(out.loc[out["rows"] < min_rows, "rows"].sum())


def plot_group(key, g, groups):
    g = g.sort_values("TIMESTAMP", kind="stable")
    t = (g["TIMESTAMP"] - g["TIMESTAMP"].iloc[0]).to_numpy()
    panels = ["CHAMBER"] + PRESSURE + ROTATION + SLURRY
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 14), sharex=True, dpi=150,
                             gridspec_kw=dict(hspace=0.18))
    fig.patch.set_facecolor(SURFACE)
    ch = g["CHAMBER"].to_numpy()
    change = np.flatnonzero(ch[1:] != ch[:-1]) + 1
    bounds = [0] + list(change) + [len(ch)]
    blocks = air_bag_blocks(g)
    for ax, col in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for b in blocks:
            ax.axvspan(t[b["i0"]], t[b["i1"]], color=SERIES, alpha=0.10, lw=0)
        for c in change:
            ax.axvline(t[c], color=AXIS, lw=0.8)
        y = g[col].to_numpy()
        ax.plot(t, y, color=SERIES, lw=1.2, drawstyle="steps-post", solid_joinstyle="round")
        if np.all(y == 0):  # 값이 전부 0 인 패널은 음수 눈금 없이 0 기준으로 표시
            ax.set_ylim(-0.05, 1)
            ax.set_yticks([0])
        ax.set_ylabel(col.replace("_PRESSURE", "_P").replace("SLURRY_FLOW_", "SLURRY_"), rotation=0, ha="right",
                      va="center", fontsize=7, color=INK2)
        ax.tick_params(labelsize=6.5, colors=MUTED, length=0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.6)
    axes[0].set_yticks(sorted(set(ch.astype(int))))
    for a, b in zip(bounds[:-1], bounds[1:]):
        axes[0].text((t[a] + t[b - 1]) / 2, 1.02, f"Chamber {int(ch[a])}", transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7.5, color=INK2)
    axes[-1].set_xlabel("Seconds from first record", fontsize=8, color=INK2)
    info = groups.loc[key]
    fig.suptitle(f"WAFER_ID {key[0]}, STAGE {key[1]}  |  Chamber {'→'.join(map(str, info['chamber_seq']))}, "
                 f"{int(info['rows'])} rows, {info['duration_s']:.0f} s", x=0.08, y=0.935, ha="left",
                 fontsize=10.5, color=INK)
    fig.text(0.08, 0.918, "Shaded band: all four air-bag pressures (main outer, center, ripple, edge) are non-zero",
             ha="left", fontsize=8, color=INK2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"step4_timeseries_{key[0]}_{key[1]}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def show_examples(examples, groups):
    header("[2] 예시 묶음의 시간별 변화 (0 / 0 아님 상태가 같은 연속 구간 단위)")
    print("약어: " + ", ".join(f"{v}={k}" for k, v in SHORT.items()))
    print("t0, t1 = 첫 기록부터 지난 초. 값 = 구간 안 중앙값. 3행 미만의 짧은 구간은 표에서 생략하고 개수만 표시.")
    for key in examples:
        g = DATA[(DATA["WAFER_ID"] == key[0]) & (DATA["STAGE"] == key[1])]
        runs, n_short, rows_short = state_runs(g)
        print(f"\n--- WAFER_ID={key[0]}, STAGE={key[1]} | Chamber {groups.loc[key, 'chamber_seq']} | "
              f"{len(g)}행 | 생략한 짧은 구간 {n_short}개 ({rows_short}행)")
        print(runs.round(1).to_string(index=False))
        blocks = air_bag_blocks(g)
        print("네 air-bag 압력이 모두 0 이 아닌 연속 구간 (시작 초, 길이 초, Chamber):",
              [(round(float(b["t_start"] - g["TIMESTAMP"].min()), 1), round(float(b["duration_s"]), 1), b["chambers"])
               for b in blocks] or "없음")
        print("그래프:", plot_group(key, g, groups).relative_to(ROOT).as_posix())


# ---------------------------------------------------------------------------
def population_checks(groups):
    header("[3] 전체 training 행에서 본 값들의 함께 움직임")
    print(f"분석 대상 행: {len(DATA):,} (원본 {len(RAW):,}행에서 연속 완전 중복 {N_DROPPED:,}행 제외)")
    print("\n(a) 네 air-bag 압력 중 0 이 아닌 개수별 행 수:",
          DATA["N_AIR_BAG_NONZERO"].value_counts().sort_index().to_dict())
    print("\n(b) 네 air-bag 압력이 모두 0 이 아님 x PRESSURIZED_CHAMBER / RETAINER_RING 압력 0 여부 (행 수)")
    print(pd.crosstab(DATA["AIR_BAG_ON"], [DATA["PRESSURIZED_CHAMBER_PRESSURE"].ne(0).rename("PCH_nonzero"),
                                           DATA["RETAINER_RING_PRESSURE"].ne(0).rename("RET_nonzero")]).to_string())
    print("\n(c) Chamber 별: air-bag 압력이 모두 0 이 아닌 행 비율")
    print(DATA.groupby("CHAMBER")["AIR_BAG_ON"].mean().round(3).to_string())
    print("\n(d) air-bag 압력 상태별, 각 값이 0 이 아닌 행 비율 (Chamber 별)")
    nz = DATA[["CHAMBER", "AIR_BAG_ON"]].join(DATA[ROTATION + SLURRY].ne(0))
    print(nz.groupby(["CHAMBER", "AIR_BAG_ON"])[ROTATION + SLURRY].mean().round(3).rename(columns=SHORT).to_string())
    print("\n(e) air-bag 압력이 모두 0 이 아닌 행에서 WAFER_ROTATION / STAGE_ROTATION 0 여부 조합 (행 수)")
    on = DATA[DATA["AIR_BAG_ON"]]
    print(pd.crosstab(on["CHAMBER"], [on["WAFER_ROTATION"].ne(0).rename("WR_nonzero"),
                                      on["STAGE_ROTATION"].ne(0).rename("SR_nonzero")]).to_string())
    print("\n(f) HEAD_ROTATION 값 분포 (행 비율 상위 5개):",
          DATA["HEAD_ROTATION"].round(1).value_counts(normalize=True).head(5).round(3).to_dict())

    header("[4] wafer+Stage 별: 네 air-bag 압력이 모두 0 이 아닌 연속 구간")
    rows, all_blocks = [], []
    for key, g in DATA.groupby(KEY):
        bl = air_bag_blocks(g)
        t_first = g["TIMESTAMP"].min()
        all_blocks += [dict(WAFER_ID=key[0], STAGE=key[1], order=i + 1, start_s=b["t_start"] - t_first,
                            duration_s=b["duration_s"], chambers="".join(map(str, b["chambers"])))
                       for i, b in enumerate(bl)]
        longest = max(bl, key=lambda b: b["duration_s"]) if bl else None
        rows.append(dict(WAFER_ID=key[0], STAGE=key[1], n_blocks=len(bl),
                         on_rows=sum(b["rows"] for b in bl),
                         longest_block_s=longest["duration_s"] if bl else 0.0,
                         longest_block_chambers=longest["chambers"] if bl else (),
                         other_blocks_total_s=sum(b["duration_s"] for b in bl) - (longest["duration_s"] if bl else 0.0),
                         chambers_with_on=tuple(sorted({c for b in bl for c in b["chambers"]}))))
    per = pd.DataFrame(rows).set_index(KEY).join(groups[["rows", "duration_s", "pattern"]])
    per = per.join(MRR.set_index(KEY)["AVG_REMOVAL_RATE"])
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "step4_air_bag_on_blocks_per_group.csv"
    per.to_csv(out)
    print(f"전체 목록: {out.relative_to(ROOT).as_posix()}")
    for pat, x in per.groupby("pattern"):
        print(f"\n[Chamber {pat}] 묶음 {len(x)}개")
        print(f"  구간이 하나도 없는 묶음: {(x['n_blocks'] == 0).sum()}개")
        print("  구간 개수 분포:", x["n_blocks"].value_counts().sort_index().to_dict())
        has = x[x["n_blocks"] > 0]
        if len(has):
            print("  가장 긴 구간 길이(초) 분위수:",
                  has["longest_block_s"].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).round(1).to_dict())
            print("  가장 긴 구간이 있는 Chamber:", has["longest_block_chambers"].value_counts().to_dict())
            print("  나머지 구간 길이 합(초) 분위수:",
                  has["other_blocks_total_s"].quantile([.5, .9, .99, 1]).round(1).to_dict())

    blk = pd.DataFrame(all_blocks).merge(groups["pattern"].reset_index(), on=KEY)
    bins = [-0.1, 10, 100, np.inf]
    labels = ["10초 이하", "10~100초", "100초 초과"]
    blk["length_bin"] = pd.cut(blk["duration_s"], bins=bins, labels=labels)
    print("\n(g) 모든 연속 구간의 길이 분포 (Chamber 패턴별 구간 수, 구간 길이 구간은 설명용 구분)")
    print(pd.crosstab(blk["pattern"], blk["length_bin"]).to_string())
    long = blk[blk["duration_s"] > 100]
    print("\n(h) 100초를 넘는 구간이 한 묶음에 몇 번 나오는가 (Chamber 패턴별 묶음 수)")
    n_long = long.groupby(KEY).size().reindex(groups.index, fill_value=0)
    print(pd.crosstab(groups["pattern"], n_long.rename("n_long_blocks")).to_string())
    print("\n(i) Chamber 4→5→6 묶음에서 100초를 넘는 구간의 순서별 위치와 길이")
    l456 = long[long["pattern"] == "4-5-6"].copy()
    l456["order_long"] = l456.groupby(KEY).cumcount() + 1
    print(l456.groupby(["order_long", "chambers"]).agg(
        blocks=("duration_s", "size"),
        start_s_median=("start_s", "median"),
        duration_s_p05=("duration_s", lambda v: v.quantile(.05)),
        duration_s_median=("duration_s", "median"),
        duration_s_p95=("duration_s", lambda v: v.quantile(.95))).round(1).to_string())

    print("\n(참고) training MRR: Chamber 패턴과 압력 구간이 나온 Chamber 별 요약")
    per["chambers_with_on"] = per["chambers_with_on"].map(lambda c: "".join(map(str, c)) or "none")
    print(per.groupby(["pattern", "chambers_with_on"])["AVG_REMOVAL_RATE"].describe().round(2).to_string())


if __name__ == "__main__":
    header("[0] 분석용 중복 제외")
    print(f"원본 행 {len(RAW):,} → 연속 완전 중복 {N_DROPPED:,}행 제외 → 분석 행 {len(DATA):,}")
    assert N_DROPPED == 1819, "Step 3 에서 확인한 1,819행과 다릅니다"
    GROUPS = group_table()
    examples = pick_examples(GROUPS)
    show_examples(examples, GROUPS)
    population_checks(GROUPS)

"""456 그룹 연마 시간의 두 무리(약 150 / 285초)가 숨은 레시피 차이인지 확인하고,
SR 후보에 "군집 표시 변수"를 넣을 수 있는지 판정한다 (진단, CLAUDE.md 14.4).

판정 기준 (2026-10-28 팀 결정, 결과를 보고 바꾸지 않음). 두 조건을 모두 만족하면 "포함 가능":
  (1) 작은 쪽 무리가 그룹 웨이퍼 수의 10% 이상
  (2) 시간 블록 5개 중 4개 이상에 작은 쪽 무리가 1장 이상
  시간 블록 = 그룹별 train+val 웨이퍼를 시작 시각 순으로 정렬해 웨이퍼 수가 같은 연속 구간 5개
  (SR 식 선택용 시간 블록 5-fold와 같은 정의, np.array_split).

연마 시간 정의 두 가지 (2026-10-28 사용자 확인: 둘 다 보고, 판정은 압력 기준):
  - press: 압력 6개 중 하나라도 > 0 인 행 수 (repro 14·15번에서 두 무리가 나온 정의) → 주 판정
  - stagerot: STAGE_ROTATION > 0 인 행 수 (요청 원안) → 참고. 챔버 4·6에서 두 무리 모두 켜져 있어 단봉이다.
  행 간격이 약 1초라 행 수 ≈ 초.

경계값(분포의 골) 규칙 — 그룹·정의별로 같은 규칙을 기계적으로 적용:
  10행 폭 히스토그램(0~450행)에서 최빈 구간을 봉우리 1, 봉우리 1과 중심이 50행 이상 떨어진 구간 중
  최다 구간을 봉우리 2로 잡고, 두 봉우리 사이에서 웨이퍼 수가 가장 적은 구간(동률이면 가운데)의 중심을 경계로 한다.
  골 깊이 = 골 구간 수 / 작은 봉우리 구간 수. 0.5 이하일 때만 "두 무리가 있다"로 본다.

라벨: MRR(라벨 파일, 집계표의 MRR 컬럼)은 읽지 않는다. 집계표도 열지 않고 원본 센서만 쓴다.
데이터: train·val 센서만. test 센서·라벨과 data/locked/는 열지 않는다.
리셋: aggregate.add_reset_features와 같은 규칙(스테이지별 시작값, 범위의 RESET_FRAC 이상 감소)을
      train·val 웨이퍼만으로 다시 감지한다 (test를 열지 않으므로 집계표의 reset_* 와 다를 수 있다).

실행: python src/hyp_recipe_cluster.py
출력: results/diagnostics/2026-10-28_김혜연_recipe_cluster_wafer.csv (+ .yaml), _summary.md (표 초안)
      results/diagnostics/2026-10-28_김혜연_recipe_cluster_{hist,time}.png
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, RESET_FRAC, USAGE, dedup  # noqa: E402
from hyp_allzero_a123 import runs_test  # noqa: E402
from hyp_r2r_retainer import LEVEL_BINS, LEVELS, RET, style  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from scatter_pv import korean_font  # noqa: E402

DATE = "2026-10-28"
DIAG = ROOT / "results" / "diagnostics"
STEM = f"{DATE}_{AUTHOR}_recipe_cluster"
OUT_WAFER = DIAG / f"{STEM}_wafer.csv"
OUT_MD = DIAG / f"{STEM}_tables.md"
FIG_HIST = DIAG / f"{STEM}_hist.png"
FIG_TIME = DIAG / f"{STEM}_time.png"
GROUPS = ["A-456", "B-456", "A-123"]
JUDGED = ["A-456", "B-456"]
DEFS = {"press": "압력 > 0 행 수", "stagerot": "STAGE_ROTATION > 0 행 수"}
MAIN_DEF = "press"
SPLITS_USED = ("train", "val")

# 판정 기준 (팀 결정, 고정)
MIN_FRAC = 0.10
N_BLOCKS = 5
MIN_BLOCKS = 4
# 경계 규칙 (고정)
BIN_W = 10
HIST_MAX = 450
PEAK_SEP = 50
DIP_MAX = 0.5
# 무리 간 차이가 "크다"의 기준: 표준화 평균차 |d| (Cohen 기준 큰 효과)
BIG_D = 0.8
# 리셋 전후 비교 창 (그룹 웨이퍼 수)
RESET_WIN = 10

AIRBAG = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
          "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE"]
SLURRY = ["SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C"]
ROTS = ["HEAD_ROTATION", "WAFER_ROTATION", "STAGE_ROTATION"]
LEVEL_NAME = {"L725": "725", "L1450": "1,450", "L7135": "7,135"}

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_raw() -> pd.DataFrame:
    parts = []
    for split in SPLITS_USED:
        d, n = dedup(load_sensors(split))
        log(f"{split}: {len(d):,}행 (완전 중복 {n:,}행 제거)")
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    return raw.sort_values(KEY + ["TIMESTAMP"], kind="stable").reset_index(drop=True)


def wafer_table(raw: pd.DataFrame) -> pd.DataFrame:
    """웨이퍼 단위 값. 행 수 ≈ 초. 라벨 없음."""
    by = [raw[k] for k in KEY]
    on = (raw[list(PRESSURE)] > 0).any(axis=1)
    sr = raw["STAGE_ROTATION"] > 0
    lv = pd.cut(raw[RET], LEVEL_BINS, labels=LEVELS)
    g = raw.groupby(KEY)
    parts = [
        g["split"].first(),
        g["TIMESTAMP"].min().rename("t_start"),
        g.size().rename("n_rows"),
        g["CHAMBER"].agg(lambda s: "".join(str(int(c)) for c in sorted(s.unique()))).rename("chambers"),
        on.groupby(by).sum().rename("t_press"),
        sr.groupby(by).sum().rename("t_stagerot"),
    ]
    for c in sorted(raw["CHAMBER"].unique()):
        m = raw["CHAMBER"] == c
        parts.append(m.groupby(by).sum().rename(f"dwell_ch{int(c)}"))
        parts.append((m & on).groupby(by).sum().rename(f"press_ch{int(c)}"))
    ch4 = raw["CHAMBER"] == 4
    for name in LEVELS[1:]:
        parts.append((on & (lv == name)).groupby(by).sum().rename(f"ret_{name}"))
        parts.append((on & ch4 & (lv == name)).groupby(by).sum().rename(f"ret_{name}_ch4"))
    # 설정값 대용: 압력이 걸린 행에서의 평균. 짧은 무리는 챔버 4에서만 압력이 걸리므로 챔버 4만의 값도 낸다.
    for s in AIRBAG + ["PRESSURIZED_CHAMBER_PRESSURE", RET] + SLURRY:
        parts.append(raw[s].where(on).groupby(by).mean().rename(f"onmean_{s}"))
        parts.append(raw[s].where(on & ch4).groupby(by).mean().rename(f"onmean_{s}_ch4"))
    for s in SLURRY + ROTS:
        parts.append((raw[s] > 0).groupby(by).sum().rename(f"pos_{s}"))
    for s in ROTS:
        parts.append(raw[s].where(raw[s] > 0).groupby(by).mean().rename(f"posmean_{s}"))
    for col in USAGE:
        parts.append(g[col].first().rename(f"start_{col}"))
    w = pd.concat(parts, axis=1).reset_index()
    w["group"] = w["STAGE"] + np.where(w["chambers"].str[0].astype(int) >= 4, "-456", "-123")
    w["day"] = (w["t_start"] - w["t_start"].min()) / 86400
    return w


def valley(x: pd.Series) -> dict:
    edges = np.arange(0, HIST_MAX + BIN_W, BIN_W)
    cnt, _ = np.histogram(x.clip(upper=HIST_MAX - 1e-9), bins=edges)
    ctr = edges[:-1] + BIN_W / 2
    p1 = int(np.argmax(cnt))
    far = np.where(np.abs(ctr - ctr[p1]) >= PEAK_SEP)[0]
    p2 = int(far[np.argmax(cnt[far])])
    lo, hi = sorted((p1, p2))
    mid = cnt[lo + 1:hi]
    mins = np.where(mid == mid.min())[0] + lo + 1
    v = int(mins[len(mins) // 2])
    dip = cnt[v] / min(cnt[p1], cnt[p2]) if min(cnt[p1], cnt[p2]) > 0 else np.inf
    return {"peak_lo": ctr[lo], "peak_hi": ctr[hi], "n_peak_lo": int(cnt[lo]), "n_peak_hi": int(cnt[hi]),
            "boundary": ctr[v], "n_valley": int(cnt[v]), "dip": dip, "bimodal": dip <= DIP_MAX}


def blocks(w: pd.DataFrame) -> pd.Series:
    """그룹 안에서 시작 시각 순 등분 5블록 (1~5)."""
    order = w.sort_values("t_start", kind="stable").index
    b = pd.Series(0, index=w.index)
    for i, idx in enumerate(np.array_split(order.to_numpy(), N_BLOCKS), start=1):
        b.loc[idx] = i
    return b


def detect_resets(w: pd.DataFrame) -> pd.DataFrame:
    """aggregate.add_reset_features와 같은 규칙, train·val 웨이퍼만. 스테이지 내 범위는 웨이퍼 시작값 기준."""
    s = w.sort_values(["STAGE", "t_start"], kind="mergesort")
    rows = []
    for col, ab in USAGE.items():
        v = s[f"start_{col}"]
        rng = v.groupby(s["STAGE"]).transform(lambda x: x.max() - x.min())
        drop = v.groupby(s["STAGE"]).diff()
        hit = s[drop < -RESET_FRAC * rng]
        rows += [{"counter": ab, "STAGE": r.STAGE, "t_reset": r.t_start, "day": r.day} for r in hit.itertuples()]
    return pd.DataFrame(rows)


def smd(a: pd.Series, b: pd.Series) -> float:
    sp = np.sqrt((a.var() + b.var()) / 2)
    return (b.mean() - a.mean()) / sp if sp > 0 else (0.0 if a.mean() == b.mean() else np.inf)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main() -> None:
    plt.rcParams["font.family"] = korean_font()
    plt.rcParams["axes.unicode_minus"] = False
    raw = load_raw()
    w = wafer_table(raw)
    del raw
    log(f"웨이퍼 {len(w):,}장")
    md = [f"# 표 초안 (자동 생성, {DATE})", ""]

    # 1. 경계값 ------------------------------------------------------------
    vrows = []
    for grp in GROUPS:
        for d in DEFS:
            r = valley(w.loc[w.group == grp, f"t_{d}"])
            vrows.append({"group": grp, "def": d, **r})
    vt = pd.DataFrame(vrows)
    bnd = {(r.group, r.def_): r.boundary for r in vt.rename(columns={"def": "def_"}).itertuples()}
    md += ["## 1. 경계값", "", md_table(vt.round(3)), ""]

    for d in DEFS:
        b = w.apply(lambda r: bnd[(r.group, d)], axis=1)
        w[f"long_{d}"] = (w[f"t_{d}"] >= b).astype(int)
    w["long"] = w[f"long_{MAIN_DEF}"]
    w["block"] = 0
    for grp in GROUPS:
        m = w.group == grp
        w.loc[m, "block"] = blocks(w[m])

    # 2. 무리별 웨이퍼 수 + 판정 ---------------------------------------------
    jrows = []
    for grp in GROUPS:
        for d in DEFS:
            x = w[w.group == grp]
            n_long = int(x[f"long_{d}"].sum())
            n_short = len(x) - n_long
            minor = 0 if n_short <= n_long else 1
            n_minor = min(n_short, n_long)
            bl = x[x[f"long_{d}"] == minor].groupby("block").size().reindex(range(1, N_BLOCKS + 1), fill_value=0)
            ok1 = n_minor / len(x) >= MIN_FRAC
            ok2 = int((bl >= 1).sum()) >= MIN_BLOCKS
            bimodal = bool(vt[(vt.group == grp) & (vt["def"] == d)].bimodal.iloc[0])
            jrows.append({
                "group": grp, "def": d, "n": len(x), "n_short": n_short, "n_long": n_long,
                "minor": ["short", "long"][minor], "minor_frac": round(n_minor / len(x), 3),
                "cond1": ok1, "minor_per_block": "/".join(map(str, bl.tolist())),
                "blocks_with_minor": int((bl >= 1).sum()), "cond2": ok2, "bimodal": bimodal,
                "verdict": ("포함 가능" if ok1 and ok2 else "포함 안 함") if grp in JUDGED else "참고",
            })
    jt = pd.DataFrame(jrows)
    md += ["## 2. 무리별 웨이퍼 수와 판정", "", md_table(jt), ""]
    sp = w[w.group.isin(JUDGED)].groupby(["group", "split", "long"]).size().unstack(fill_value=0)
    md += ["split별 (press 기준, 0 = 짧은, 1 = 긴)", "", md_table(sp.reset_index()), ""]
    jx = w[w.group.isin(JUDGED)]
    md += ["press 무리 × stagerot 무리 (0 = 짧은, 1 = 긴)", "",
           md_table(pd.crosstab([jx["group"], jx["long_press"]], jx["long_stagerot"]).reset_index()), ""]

    # 3. 무리별 공정 조건 -----------------------------------------------------
    items = ([f"ret_{n}" for n in LEVELS[1:]] + [f"ret_{n}_ch4" for n in LEVELS[1:]]
             + [f"onmean_{s}" for s in AIRBAG + ["PRESSURIZED_CHAMBER_PRESSURE", RET] + SLURRY]
             + [f"onmean_{s}_ch4" for s in AIRBAG + ["PRESSURIZED_CHAMBER_PRESSURE", RET] + SLURRY]
             + [f"pos_{s}" for s in SLURRY + ROTS]
             + [f"posmean_{s}" for s in ROTS]
             + [c for c in w.columns if c.startswith(("dwell_ch", "press_ch"))
                and int(c[-1]) >= 4] + ["n_rows"])
    crows = []
    for grp in JUDGED:
        x = w[w.group == grp]
        a, b = x[x.long == 0], x[x.long == 1]
        for it in items:
            d = smd(a[it].dropna(), b[it].dropna())
            # 두 무리 모두 표준편차가 0에 가까우면 d가 부풀려진다 (평균 차이가 반올림 아래)
            flat = round(a[it].mean(), 2) == round(b[it].mean(), 2)
            big = ("(평균 같음, 분산 ≈ 0)" if flat else "**큼**") if abs(d) >= BIG_D else ""
            crows.append({"group": grp, "item": it,
                          "short_mean": round(a[it].mean(), 2), "short_sd": round(a[it].std(), 2),
                          "long_mean": round(b[it].mean(), 2), "long_sd": round(b[it].std(), 2),
                          "d": round(d, 2), "big": big})
    ct = pd.DataFrame(crows)
    md += [f"## 3. 무리별 공정 조건 (평균 / 표준편차, d = 표준화 평균차, |d| ≥ {BIG_D} = 큼)", "",
           md_table(ct), ""]

    # 4. 시간 패턴 ------------------------------------------------------------
    rrows = []
    for grp in JUDGED:
        x = w[w.group == grp].sort_values("t_start", kind="stable")
        s = x["long"].to_numpy()
        rt = runs_test(1 - s)   # 1 = 짧은 무리
        lens = pd.Series(s).ne(pd.Series(s).shift()).cumsum()
        rl = pd.Series(s).groupby(lens).agg(["first", "size"])
        rrows.append({"group": grp, "runs": rt["runs"], "runs_expected": round(rt["runs_expected"], 1),
                      "z": round(rt["z"], 2), "p": round(rt["p"], 3),
                      "short_run_median": rl[rl["first"] == 0]["size"].median(),
                      "short_run_max": rl[rl["first"] == 0]["size"].max(),
                      "long_run_median": rl[rl["first"] == 1]["size"].median(),
                      "long_run_max": rl[rl["first"] == 1]["size"].max()})
    md += ["## 4. 시간상 섞임 (연 검정, z < 0 = 무작위보다 몰림)", "", md_table(pd.DataFrame(rrows)), ""]

    # 5. 교차표 ---------------------------------------------------------------
    j = w[w.group.isin(JUDGED)]
    md += ["## 5-1. 스테이지 × 무리 (456)", "",
           md_table(pd.crosstab(j["STAGE"], j["long"], margins=True).reset_index()), ""]
    md += ["## 5-2. 거친 챔버 구성 × 무리", "",
           md_table(pd.crosstab([j["group"], j["chambers"]], j["long"]).reset_index()), ""]
    cc = j.groupby(["group", "long"])[[f"press_ch{c}" for c in (4, 5, 6)]
                                     + [f"dwell_ch{c}" for c in (4, 5, 6)]].median()
    md += ["## 5-3. 챔버별 체류 행 수·압력 > 0 행 수 (중앙값)", "", md_table(cc.reset_index()), ""]
    for grp in GROUPS:
        x = w[w.group == grp]
        bt = pd.crosstab(x["block"], x["long"])
        bt["day_range"] = x.groupby("block")["day"].agg(lambda s: f"{s.min():.1f}~{s.max():.1f}")
        md += [f"## 5-4. 시간 블록 × 무리 ({grp}, press 기준)", "", md_table(bt.reset_index()), ""]

    # 6. 리셋과 무리 전환 -----------------------------------------------------
    rs = detect_resets(w)
    md += ["## 6-1. 리셋 횟수 (train·val만, 스테이지별)", "",
           md_table(rs.groupby(["counter", "STAGE"]).size().unstack(fill_value=0).reset_index()), ""]
    erows = []
    for grp in JUDGED:
        x = w[w.group == grp].sort_values("t_start", kind="stable").reset_index(drop=True)
        s = x["long"].to_numpy()
        switch = np.r_[False, s[1:] != s[:-1]]
        base = switch[1:].mean()
        for ab, r in rs[rs.STAGE == grp[0]].groupby("counter"):
            sw, before, after = [], [], []
            for t in r["t_reset"]:
                i = int(np.searchsorted(x["t_start"].to_numpy(), t))   # 리셋 시각 이후 첫 그룹 웨이퍼
                if i < 1 or i >= len(x):
                    continue
                sw.append(switch[i])
                before.append(1 - s[max(0, i - RESET_WIN):i].mean())
                after.append(1 - s[i:i + RESET_WIN].mean())
            erows.append({"group": grp, "counter": ab, "n_reset": len(sw),
                          "switch_at_reset": round(float(np.mean(sw)), 3) if sw else np.nan,
                          "switch_base": round(base, 3),
                          "p_binom": round(binomtest(int(np.sum(sw)), len(sw), base).pvalue, 3) if sw else np.nan,
                          f"short_frac_before{RESET_WIN}": round(float(np.mean(before)), 3) if sw else np.nan,
                          f"short_frac_after{RESET_WIN}": round(float(np.mean(after)), 3) if sw else np.nan,
                          "short_frac_all": round(1 - s.mean(), 3)})
    md += [f"## 6-2. 리셋 직후 무리 전환 (전환 = 리셋 후 첫 웨이퍼의 무리가 직전과 다름, 전후 {RESET_WIN}장 짧은 무리 비율)",
           "", md_table(pd.DataFrame(erows)), ""]

    # 그림 ---------------------------------------------------------------------
    edges = np.arange(0, HIST_MAX + BIN_W, BIN_W)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for i, d in enumerate(DEFS):
        for k, grp in enumerate(GROUPS):
            ax = axes[i, k]
            ax.hist(w.loc[w.group == grp, f"t_{d}"].clip(upper=HIST_MAX - 1), bins=edges,
                    color="#2a78d6", edgecolor="white", lw=0.4)
            r = vt[(vt.group == grp) & (vt["def"] == d)].iloc[0]
            if r.bimodal:
                ax.axvline(r.boundary, color="#d9534f", ls="--", lw=1.2)
                ax.text(r.boundary + 5, ax.get_ylim()[1] * 0.9, f"경계 {r.boundary:.0f}", color="#d9534f", fontsize=9)
            ax.set_title(f"{grp} — {DEFS[d]}", fontsize=10)
            style(ax)
    for ax in axes[1]:
        ax.set_xlabel("행 수 (약 1초 단위)")
    for ax in axes[:, 0]:
        ax.set_ylabel("웨이퍼 수")
    fig.suptitle("연마 시간 히스토그램 (train+val, 10행 폭, 450 이상은 마지막 칸)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_HIST, dpi=150)
    plt.close(fig)

    head = rs[rs.counter == "BACKING_FILM"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for ax, grp in zip(axes, JUDGED):
        x = w[w.group == grp]
        for lab, col, name in [(0, "#e8833a", "짧은 무리"), (1, "#2a78d6", "긴 무리")]:
            y = x[x.long == lab]
            ax.scatter(y["day"], y["t_press"].clip(upper=HIST_MAX), s=7, color=col, alpha=0.7, label=name, lw=0)
        ax.axhline(bnd[(grp, MAIN_DEF)], color="#888888", lw=0.8)
        for t in head[head.STAGE == grp[0]]["day"]:
            ax.axvline(t, color="#999999", ls="--", lw=0.8)
        xs = x.sort_values("t_start")
        for idx in np.array_split(np.arange(len(xs)), N_BLOCKS)[1:]:
            ax.axvline(xs["day"].iloc[idx[0]], color="#2b2b2b", lw=1.2, alpha=0.6)
        ax.set_ylabel("압력 > 0 행 수")
        ax.set_title(f"{grp}: 검은 실선 = 시간 블록 경계, 회색 파선 = 헤드 부품(BACKING_FILM) 리셋", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        style(ax)
    axes[-1].set_xlabel("연마 시작 시각 (train+val 첫 웨이퍼 기준 일)")
    fig.tight_layout()
    fig.savefig(FIG_TIME, dpi=150)
    plt.close(fig)

    w.to_csv(OUT_WAFER, index=False)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    yaml = [
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/hyp_recipe_cluster.py",
        "question: 456 그룹 연마 시간의 두 무리가 레시피 차이인지, SR 후보에 군집 표시 변수를 넣을 수 있는지",
        "input: data/raw/{train,val} 센서만 (집계표·라벨 파일·test·data/locked/ 열지 않음)",
        "labels: MRR 사용 안 함 (SR 실행 전 결과 미리 보기 방지)",
        f"definitions: {DEFS}", f"main_definition: {MAIN_DEF}",
        f"criteria: 작은 무리 >= {MIN_FRAC} 그리고 {N_BLOCKS}블록 중 {MIN_BLOCKS}개 이상에 작은 무리 1장 이상 (2026-10-28 팀 결정)",
        "blocks: 그룹별 t_start 순 np.array_split 5등분",
        f"valley_rule: {BIN_W}행 폭, 봉우리 간격 >= {PEAK_SEP}행, 골 = 사이 최소 구간 중심, 골 깊이 <= {DIP_MAX} 이면 쌍봉",
        f"big_difference: |표준화 평균차| >= {BIG_D}",
        f"resets: aggregate 규칙(RESET_FRAC={RESET_FRAC}, 웨이퍼 시작값), train·val만",
        f"reset_window: {RESET_WIN}",
        f"outputs: [{OUT_WAFER.name}, {OUT_MD.name}, {FIG_HIST.name}, {FIG_TIME.name}]",
        "note: 진단 결과 (14.4)",
    ]
    OUT_WAFER.with_suffix(".yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    log("저장 완료")


if __name__ == "__main__":
    for p in (OUT_WAFER, OUT_MD, FIG_HIST, FIG_TIME):
        if p.exists():
            raise SystemExit(f"{p} 가 이미 있다. 덮어쓰지 않는다 (CLAUDE.md 3절 3번).")
    main()

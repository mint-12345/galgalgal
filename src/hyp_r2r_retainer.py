"""가설 1: 리테이너 링 압력(P1_on)이 run-to-run(R2R) 제어의 결과인가 (진단, CLAUDE.md 14.4).

CMP 전문가 자문을 대신해 데이터로 확인한다. 456 그룹(A-456, B-456)을 스테이지별로 본다.

P1_on = RETAINER_RING_PRESSURE를 "압력 6개 중 하나라도 > 0 이고 STAGE_ROTATION > 0"인 행에서만 평균
(`src/p1_on_val.py`와 같은 정의).

1. 분포: 연속인가, 몇 개의 설정값 근처에 몰리는가. 행 단위 원값과 웨이퍼 단위 P1_on 히스토그램.
   원값이 단계(약 725 / 1,450 / 7,135)로 나뉘므로 P1_on을 "단계별 체류 비율"과 "단계 안 평균"으로 나눠 본다.
2. 시간순: 계단식으로 바뀌는가. lag-1 자기상관과 "이동 중앙값이 설명하는 분산 비율"을 낸다.
3. P1_on(t)와 직전 웨이퍼 MRR의 부분상관 (통제: t_start + 소모품 6개 시작값).
   직전 MRR은 train·val 라벨이 있는 같은 그룹 웨이퍼만, t보다 먼저 시작한 것.
   비교용으로 직후 웨이퍼 MRR(가짜 대조)도 같은 방식으로 낸다. R2R이면 직전 쪽만 음의 신호가 나와야 한다.
   P1_on은 자기상관이 커서, 바로 앞 웨이퍼의 P1_on까지 통제한 변형을 함께 낸다
   (앞 웨이퍼의 압력이 같을 때, 앞 웨이퍼의 MRR이 다음 압력을 바꾸는가).
4. 반대 방향: MRR(t)와 P1_on(t−1)의 부분상관. P1_on(t−1)은 split과 무관하게 같은 그룹의 바로 앞 웨이퍼
   (센서만 쓰므로 test 웨이퍼도 앞 웨이퍼가 될 수 있다).

라벨: train은 wafer_table.csv의 MRR, val은 `data/raw/labels/`에서 읽어 집계표와 대조한 뒤 쓴다.
test 라벨과 `data/locked/`는 어떤 경로로도 읽지 않는다. 456 그룹에는 이상치·전체-0 센서 웨이퍼가 없다.

실행: python src/hyp_r2r_retainer.py
출력: results/diagnostics/2026-10-06_김혜연_r2r_levels.csv   분포·단계·계단성 (+ .yaml)
      results/diagnostics/2026-10-06_김혜연_r2r_partial.csv  부분상관 (+ .yaml)
      docs/reports/figs/김혜연_r2r_p1_on_hist.png, 김혜연_r2r_p1_on_time.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, SPLITS, load_labels, load_sensors  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from pv_check_followup import partial_corr  # noqa: E402
from pv_partial_consumables import START6  # noqa: E402
from scatter_pv import korean_font  # noqa: E402

DATE = "2026-10-06"
DIAG = ROOT / "results" / "diagnostics"
OUT_LV = DIAG / f"{DATE}_{AUTHOR}_r2r_levels.csv"
OUT_PC = DIAG / f"{DATE}_{AUTHOR}_r2r_partial.csv"
FIGS = ROOT / "docs" / "reports" / "figs"
FIG_HIST = FIGS / f"{AUTHOR}_r2r_p1_on_hist.png"
FIG_TIME = FIGS / f"{AUTHOR}_r2r_p1_on_time.png"
GROUPS = ["A-456", "B-456"]
RET = "RETAINER_RING_PRESSURE"
# 원값 단계 경계 (탐색에서 확인: 725·1,450·7,135 근처에 몰리고 그 사이는 드물다)
LEVEL_BINS = [-np.inf, 300, 1100, 3000, np.inf]
LEVELS = ["low", "L725", "L1450", "L7135"]
CONTROLS = ["t_start"] + START6
ROLL = 25   # 이동 중앙값 창 (웨이퍼 수)


def fisher_ci(r: float, n: int, k: int = 0) -> tuple[float, float]:
    """(부분)상관의 95% 신뢰구간. k = 통제 변수 개수."""
    if not np.isfinite(r) or n - 3 - k < 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(n - 3 - k)
    return tuple(float(np.tanh(z + s * 1.96 * se)) for s in (-1, 1))


def load_raw(keys: pd.DataFrame) -> pd.DataFrame:
    """세 split 센서를 읽어 완전 중복을 지우고 keys 웨이퍼만 남긴다. 라벨은 읽지 않는다."""
    frames = []
    for split in SPLITS:
        df, n_dup = dedup(load_sensors(split))
        frames.append(df.merge(keys, on=KEY))
        print(f"  {split}: 중복 {n_dup:,}행 제거", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    cols = list(PRESSURE) + ["STAGE_ROTATION"]
    raw[cols] = raw[cols].apply(pd.to_numeric, errors="raise")
    return raw


def wafer_values(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """웨이퍼별 P1_on, 단계별 체류 비율 frac_*, 단계 안 평균 lvl_*. 함께 on 행도 돌려준다."""
    on = (raw[list(PRESSURE)] > 0).any(axis=1) & (raw["STAGE_ROTATION"] > 0)
    r = raw.loc[on, KEY + ["group", RET]].copy()
    r["level"] = pd.cut(r[RET], LEVEL_BINS, labels=LEVELS)
    w = r.groupby(KEY)[RET].mean().rename("P1_on").to_frame()
    w = w.join(r.groupby(KEY)["level"].value_counts(normalize=True).unstack(fill_value=0)
               .reindex(columns=LEVELS, fill_value=0).add_prefix("frac_"))
    w = w.join(r.groupby(KEY + ["level"], observed=True)[RET].mean().unstack()
               .reindex(columns=LEVELS).add_prefix("lvl_"))
    return w.reset_index(), r


def add_neighbors(d: pd.DataFrame) -> pd.DataFrame:
    """한 그룹 안에서 시작 시각 순 이웃 값을 붙인다.

    - P1_on_prev / P1_on_next, MRR_prev_imm / MRR_next_imm: split 무관 바로 앞·뒤 웨이퍼 (test면 MRR은 NaN)
    - MRR_lag1~3, MRR_prev5: 라벨 있는 웨이퍼만 줄 세운 직전 1·2·3번째와 직전 5장 평균
    - MRR_lead1~3, MRR_next5: 같은 방식의 직후 (가짜 대조)
    """
    d = d.sort_values("t_start", kind="stable").reset_index(drop=True)
    for c in ("P1_on", "MRR"):
        suf = "" if c == "P1_on" else "_imm"
        d[f"{c}_prev{suf}"] = d[c].shift(1)
        d[f"{c}_next{suf}"] = d[c].shift(-1)
    lab = d[d["MRR"].notna()]
    if lab["t_start"].duplicated().any():
        raise ValueError("같은 그룹에 시작 시각이 같은 라벨 웨이퍼가 있다 (직전 정의가 모호)")
    m = lab["MRR"]
    lag = pd.DataFrame(index=lab.index)
    for k in (1, 2, 3):
        lag[f"MRR_lag{k}"] = m.shift(k)
        lag[f"MRR_lead{k}"] = m.shift(-k)
    lag["MRR_prev5"] = m.rolling(5).mean().shift(1)
    lag["MRR_next5"] = m[::-1].rolling(5).mean()[::-1].shift(-1)
    return d.join(lag)


def step_stats(x: pd.Series) -> dict:
    """시간순 계열의 계단성: lag-1 자기상관, 이동 중앙값이 설명하는 분산 비율."""
    x = x.dropna().reset_index(drop=True)
    smooth = x.rolling(ROLL, center=True, min_periods=ROLL // 2).median()
    return {"n": len(x), "lag1_autocorr": x.autocorr(1),
            "smooth_var_share": smooth.var() / x.var(),
            "resid_sd_over_sd": (x - smooth).std() / x.std()}


def style(ax) -> None:
    ax.grid(True, color="#e8e8e8", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors="#666666", labelsize=9)


def draw_hist(w: pd.DataFrame, r: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.subplots_adjust(top=0.86, hspace=0.45, wspace=0.18)
    for j, g in enumerate(GROUPS):
        ax = axes[0, j]
        v = r.loc[r["group"] == g, RET]
        ax.hist(v, bins=np.arange(0, 8600, 40), color="#2a78d6", lw=0)
        ax.set_yscale("log")
        ax.set_title(f"{g}  행 단위 원값 (압력·STAGE on 행 {len(v):,}개)", fontsize=10.5, color="#333333")
        ax.set_xlabel("RETAINER_RING_PRESSURE", fontsize=9.5, color="#555555")
        style(ax)
        ax = axes[1, j]
        d = w[w["group"] == g]
        ax.hist(d["P1_on"].dropna(), bins=60, color="#2a78d6", lw=0)
        ax.set_title(f"{g}  웨이퍼 단위 P1_on ({d['P1_on'].notna().sum():,}장)", fontsize=10.5, color="#333333")
        ax.set_xlabel("P1_on", fontsize=9.5, color="#555555")
        style(ax)
    axes[0, 0].set_ylabel("행 수 (로그)", fontsize=9.5, color="#555555")
    axes[1, 0].set_ylabel("웨이퍼 수", fontsize=9.5, color="#555555")
    fig.text(0.02, 0.975, "리테이너 링 압력 분포 — 원값은 세 단계, P1_on은 연속",
             fontsize=13, color="#222222", va="top")
    fig.text(0.02, 0.925, "456 그룹, train·val·test 센서 (라벨 미사용). 원값은 약 725 / 1,450 / 7,135 근처에 몰린다",
             fontsize=9.5, color="#777777", va="top")
    fig.savefig(FIG_HIST, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_time(w: pd.DataFrame, tab: pd.DataFrame) -> None:
    rows = [("P1_on", "P1_on"), ("frac_L7135", "7,135 단계 체류 비율"), ("lvl_L1450", "1,450 단계 안 평균")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 8.6), sharex=True)
    fig.subplots_adjust(top=0.88, hspace=0.28, wspace=0.16)
    t0 = tab["t_start"].min()
    for j, g in enumerate(GROUPS):
        d = w[w["group"] == g].sort_values("t_start")
        day = (d["t_start"] - t0) / 86400
        stage = g[0]
        resets = tab[(tab["STAGE"] == stage) & (tab["reset_BACKING_FILM_w"] == 1)]["t_start"]
        for i, (col, label) in enumerate(rows):
            ax = axes[i, j]
            ax.scatter(day, d[col], s=5, color="#9bbbe3", lw=0)
            ax.plot(day, d[col].rolling(ROLL, center=True, min_periods=ROLL // 2).median(),
                    color="#0b3c78", lw=1.4)
            for t in resets:
                ax.axvline((t - t0) / 86400, color="#d6872a", lw=0.9, ls="--")
            if j == 0:
                ax.set_ylabel(label, fontsize=9.5, color="#555555")
            if i == 0:
                ax.set_title(g, fontsize=11, color="#333333")
            style(ax)
        axes[2, j].set_xlabel("웨이퍼 시작 시각 (첫 웨이퍼 기준 일)", fontsize=9.5, color="#555555")
    fig.text(0.02, 0.975, "P1_on 시간순 — 점 = 웨이퍼, 선 = 이동 중앙값(25장), 주황 점선 = 헤드 부품 교체",
             fontsize=13, color="#222222", va="top")
    fig.text(0.02, 0.935, "헤드 부품 교체 = 같은 스테이지의 BACKING_FILM 리셋 (MEMBRANE·PSHEET와 같은 웨이퍼). "
             "train·val·test 센서, 라벨 미사용", fontsize=9.5, color="#777777", va="top")
    fig.savefig(FIG_TIME, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for p in (OUT_LV, OUT_PC, FIG_HIST, FIG_TIME):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    val = load_labels("val").rename(columns={"AVG_REMOVAL_RATE": "MRR_raw"})
    chk = tab[tab["split"] == "val"].merge(val, on=KEY)
    if len(chk) != len(val) or not np.allclose(chk["MRR"], chk["MRR_raw"]):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")
    if tab.loc[tab["split"] == "test", "MRR"].notna().any():
        raise ValueError("집계표에 test MRR이 있다")
    t456 = tab[tab["group"].isin(GROUPS)]
    assert (t456["is_outlier"] == 0).all() and (t456["n_allzero_sensors"] == 0).all()
    print(f"456 그룹 {len(t456):,}장 (라벨 {int(t456['MRR'].notna().sum()):,}장). 센서 읽는 중", flush=True)

    raw = load_raw(t456[KEY + ["group"]])
    w, r = wafer_values(raw)
    w = t456[KEY + ["group", "split", "t_start", "MRR"] + CONTROLS[1:]].merge(w, on=KEY, how="left")
    w = pd.concat([add_neighbors(d) for _, d in w.groupby("group")], ignore_index=True)

    # 1·2. 분포·단계·계단성
    lv = []
    for g in GROUPS:
        rg, wg = r[r["group"] == g], w[w["group"] == g]
        for name in LEVELS:
            v = rg.loc[rg["level"] == name, RET]
            lv.append({"check": "1_row_level", "group": g, "item": name, "n": len(v),
                       "share": len(v) / len(rg), "mean": v.mean(), "std": v.std()})
        for c in ["P1_on"] + [f"frac_{x}" for x in LEVELS[1:]] + [f"lvl_{x}" for x in LEVELS[1:]]:
            x = wg[c]
            lv.append({"check": "1_wafer", "group": g, "item": c, "n": int(x.notna().sum()),
                       "mean": x.mean(), "std": x.std(), "cv": x.std() / x.mean(),
                       "corr_with_P1_on": x.corr(wg["P1_on"])})
        for c in ("P1_on", "frac_L7135", "lvl_L1450", "lvl_L725", "MRR"):
            lv.append({"check": "2_step", "group": g, "item": c, **step_stats(wg[c])})
    lv = pd.DataFrame(lv).round(4)

    # 3·4. 부분상관
    pc = []

    def add(g, d, y, x, extra=()):
        cols = CONTROLS + list(extra)
        ok = d[[x, y] + cols].notna().all(axis=1)
        rr = d.loc[ok, x].corr(d.loc[ok, y])
        p = partial_corr(d[x], d[y], d[cols])
        lo, hi = fisher_ci(p, int(ok.sum()), len(cols))
        pc.append({"group": g, "y": y, "x": x, "extra_control": "+".join(extra) or "-",
                   "n": int(ok.sum()), "corr": rr, "partial_corr": p, "pc_ci_lo": lo, "pc_ci_hi": hi})

    for g in GROUPS:
        d = w[(w["group"] == g) & w["MRR"].notna()]   # 대상 = 라벨 있는 웨이퍼
        for x in ("MRR_lag1", "MRR_lag2", "MRR_lag3", "MRR_prev5",
                  "MRR_lead1", "MRR_lead2", "MRR_lead3", "MRR_next5", "MRR"):
            add(g, d, "P1_on", x)
        add(g, d, "P1_on", "MRR_prev_imm")
        add(g, d, "P1_on", "MRR_prev_imm", ("P1_on_prev",))
        add(g, d, "P1_on", "MRR_next_imm")
        add(g, d, "P1_on", "MRR_next_imm", ("P1_on_next",))
        add(g, d, "MRR", "P1_on_prev")
        add(g, d, "MRR", "P1_on_prev", ("P1_on",))
        add(g, d, "MRR", "P1_on_next")
        add(g, d, "MRR", "P1_on_next", ("P1_on",))
        add(g, d, "MRR", "P1_on")
    pc = pd.DataFrame(pc).round(4)

    font = korean_font()
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    FIGS.mkdir(parents=True, exist_ok=True)
    draw_hist(w, r)
    draw_time(w, tab)

    DIAG.mkdir(parents=True, exist_ok=True)
    lv.to_csv(OUT_LV, index=False)
    pc.to_csv(OUT_PC, index=False)
    common = [
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/hyp_r2r_retainer.py",
        "question: 가설 1 — 리테이너 링 압력(P1_on)이 run-to-run 제어의 결과인가",
        "input: data/processed/wafer_table.csv + data/raw/{train,val,test} 센서 + "
        "data/raw/labels/CMP-validation-removalrate.csv (집계표 val MRR과 대조 후 사용)",
        f"groups: {GROUPS}",
        "P1_on: RETAINER_RING_PRESSURE, 압력 6개 중 하나라도 >0 이고 STAGE_ROTATION >0 인 행만 평균",
        f"levels: {dict(zip(LEVELS, [str(b) for b in zip(LEVEL_BINS[:-1], LEVEL_BINS[1:])]))}  # (하한, 상한]",
        "locked: data/locked/ 와 test 라벨은 열지 않았다",
    ]
    OUT_LV.with_suffix(".yaml").write_text("\n".join(common + [
        "content: {1_row_level: on 행 원값의 단계별 비율·평균, 1_wafer: 웨이퍼 단위 P1_on·단계 체류 비율·"
        "단계 안 평균의 분포와 P1_on 상관, 2_step: 시간순 계단성}",
        "wafers: 456 그룹 전부 (세 split, 센서만). 2_step 의 MRR 은 라벨 있는 웨이퍼만",
        f"step: lag1_autocorr = 시작 시각 순 lag-1 자기상관, smooth_var_share = 이동 중앙값({ROLL}장, 중심) "
        "분산 / 원 분산, resid_sd_over_sd = 이동 중앙값을 뺀 잔차 표준편차 / 원 표준편차",
        f"figs: {FIG_HIST.relative_to(ROOT).as_posix()}, {FIG_TIME.relative_to(ROOT).as_posix()}",
        "note: 진단 결과 (14.4)", "",
    ]), encoding="utf-8")
    OUT_PC.with_suffix(".yaml").write_text("\n".join(common + [
        f"controls: {CONTROLS}  # + extra_control 열",
        "targets: 라벨 있는 456 웨이퍼 (train + val)",
        "neighbors: 같은 그룹, t_start 순. MRR_lag/lead = 라벨 있는 웨이퍼만 줄 세운 직전/직후, "
        "MRR_prev5/next5 = 직전/직후 5장 평균, *_prev_imm/*_next_imm 과 P1_on_prev/next = split 무관 바로 앞/뒤 웨이퍼",
        "metrics: Pearson. partial_corr = 통제 변수 선형 회귀 잔차끼리의 Pearson. "
        "pc_ci = Fisher z, se = 1/sqrt(n-3-통제 수)",
        "note: 진단 결과 (14.4). MRR 은 상관 계산에만 사용", "",
    ]), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n" + lv.to_string(index=False))
    print("\n" + pc.to_string(index=False))
    for p in (OUT_LV, OUT_PC, FIG_HIST, FIG_TIME):
        print(f"저장: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

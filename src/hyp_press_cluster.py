"""456 그룹의 압력 > 0 시간이 두 무리(약 150초 / 약 285초)로 나뉘는 현상 (진단, CLAUDE.md 14.4).

무리 표시는 `src/features.py`의 `press_long`(press_on_s >= 220초)을 `data/processed/features_S.csv`에서 읽는다.

1. 무리별 웨이퍼 수, 스테이지, split, 시작 시각 분포 (첫 웨이퍼 기준 일).
2. 무리 사이 센서 차이 (센서만, 세 split):
   챔버별 체류 시간·압력 > 0 시간 (빈칸 제외), 리테이너 링 단계별 시간 (hyp_r2r_retainer.py의 경계),
   챔버별 센서 13개의 "값 > 0 인 행 비율", 압력 > 0 행에서의 센서 평균, 챔버별 소모품 카운터 증가.
   웨이퍼 단위 값의 중앙값과 Mann–Whitney p.
   MRR(train·val 라벨)은 단순 중앙값과, t_start + 소모품 6개 시작값을 통제한 무리 계수(OLS)를 낸다.
3. 시간상 섞임: 그룹별 Wald–Wolfowitz 연 검정(hyp_allzero_a123.py), 5일 구간별 긴 무리 비율,
   A-456과 B-456의 동기: A 웨이퍼마다 시작 시각이 가장 가까운 B 웨이퍼의 무리와 같은 비율 (무작위 기대와 비교).

라벨: train은 집계표, val은 `data/raw/labels/`에서 읽어 집계표와 대조한 뒤 쓴다.
test 라벨과 `data/locked/`는 읽지 않는다. 456 그룹에는 이상치·전체-0 센서 웨이퍼가 없다.

실행: python src/hyp_press_cluster.py   (먼저 python src/features.py)
출력: results/diagnostics/2026-10-14_김혜연_press_cluster.csv (+ .yaml)
      docs/reports/figs/김혜연_press_cluster_time.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import FLOW, PRESSURE, ROTATION, dedup  # noqa: E402
from features import GAP_S, OUT as FEATURES  # noqa: E402
from hyp_allzero_a123 import runs_test  # noqa: E402
from hyp_r2r_retainer import LEVEL_BINS, LEVELS, RET, style  # noqa: E402
from load import KEY, ROOT, SPLITS, load_labels, load_sensors  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from pv_partial_consumables import START6  # noqa: E402
from scatter_pv import korean_font  # noqa: E402

DATE = "2026-10-14"
DIAG = ROOT / "results" / "diagnostics"
OUT = DIAG / f"{DATE}_{AUTHOR}_press_cluster.csv"
FIG = ROOT / "docs" / "reports" / "figs" / f"{AUTHOR}_press_cluster_time.png"
GROUPS = ["A-456", "B-456"]
CHAMBERS = [4, 5, 6]
SENSORS = list(PRESSURE) + list(FLOW) + list(ROTATION)
COUNTERS = ["USAGE_OF_POLISHING_TABLE", "USAGE_OF_BACKING_FILM", "USAGE_OF_DRESSER"]
BIN_DAYS = 5
MIN_BIN = 20    # 그림에서 이보다 웨이퍼가 적은 구간은 비율을 그리지 않는다
Y_MAX = 450
CLUSTER = {0: "short", 1: "long"}


def wafer_sensor_table(raw: pd.DataFrame) -> pd.DataFrame:
    """웨이퍼 단위 센서 요약 (넓은 표). 라벨 없음."""
    raw = raw.sort_values(KEY + ["TIMESTAMP"], kind="stable").reset_index(drop=True)
    dt = (-raw.groupby(KEY)["TIMESTAMP"].diff(-1)).fillna(0.0)
    dt = dt.where(dt <= GAP_S, 0.0)
    on = (raw[list(PRESSURE)] > 0).any(axis=1)
    lv = pd.cut(raw[RET], LEVEL_BINS, labels=LEVELS)
    by = [raw[k] for k in KEY]
    parts = [raw.groupby(KEY).size().rename("n_rows"),
             raw.groupby(KEY)["TIMESTAMP"].agg(lambda s: s.max() - s.min()).rename("t_span_s")]
    for c in CHAMBERS:
        m = raw["CHAMBER"] == c
        parts.append(dt.where(m, 0.0).groupby(by).sum().rename(f"dwell_ch{c}_s"))
        parts.append(dt.where(m & on, 0.0).groupby(by).sum().rename(f"press_on_ch{c}_s"))
        for s in SENSORS:
            parts.append((raw[s] > 0).astype(float).where(m).groupby(by).mean().rename(f"pos_{s}_ch{c}"))
        for u in COUNTERS:
            v = raw[u].where(m)
            parts.append((v.groupby(by).last() - v.groupby(by).first()).rename(f"inc_{u}_ch{c}"))
    for name in LEVELS[1:]:
        parts.append(dt.where(on & (lv == name), 0.0).groupby(by).sum().rename(f"ret_{name}_s"))
    for s in SENSORS:
        parts.append(raw[s].where(on).groupby(by).mean().rename(f"onmean_{s}"))
    return pd.concat(parts, axis=1).astype(float).reset_index()


def ols_flag(y: pd.Series, flag: pd.Series, ctrl: pd.DataFrame) -> tuple[float, float, float]:
    """y ~ 1 + flag + ctrl(z-score) 에서 flag 계수, 표준오차, 양측 p (정규 근사)."""
    c = ctrl.to_numpy(float)
    c = (c - c.mean(0)) / c.std(0)
    X = np.column_stack([np.ones(len(y)), flag.to_numpy(float), c])
    yv = y.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    s2 = resid @ resid / (len(yv) - X.shape[1])
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    return float(beta[1]), se, float(2 * norm.sf(abs(beta[1] / se)))


def draw(d: pd.DataFrame, tab: pd.DataFrame, t0: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.8]})
    fig.subplots_adjust(top=0.9, hspace=0.18)
    colors = {0: "#d6602a", 1: "#2a78d6"}
    for i, g in enumerate(GROUPS):
        ax = axes[i]
        dg = d[d["group"] == g]
        for k, lab in ((1, "긴 무리 (≥ 220초)"), (0, "짧은 무리 (< 220초)")):
            s = dg[dg["press_long"] == k]
            ax.scatter(s["day"], s["press_on_s"], s=6, color=colors[k], lw=0, alpha=0.7,
                       label=f"{lab} {len(s):,}장")
        resets = tab[(tab["STAGE"] == g[0]) & (tab["reset_BACKING_FILM_w"] == 1)]["t_start"]
        for t in resets:
            ax.axvline((t - t0) / 86400, color="#888888", lw=0.9, ls="--")
        ax.axhline(220, color="#555555", lw=0.8, ls=":")
        ax.set_ylim(0, Y_MAX)
        n_out = int((dg["press_on_s"] > Y_MAX).sum())
        if n_out:
            ax.text(0.995, 0.97, f"{Y_MAX}초 초과 {n_out}장은 축 밖", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color="#777777")
        ax.set_ylabel(f"{g}\n압력 > 0 시간 (초)", fontsize=9.5, color="#555555")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
        style(ax)
    ax = axes[2]
    for g, col in zip(GROUPS, ("#0b3c78", "#7a9a2a")):
        dg = d[d["group"] == g]
        b = (dg["day"] // BIN_DAYS) * BIN_DAYS + BIN_DAYS / 2
        share = dg.groupby(b)["press_long"].agg(["mean", "size"])
        share = share[share["size"] >= MIN_BIN]["mean"]
        ax.plot(share.index, share.values, marker="o", ms=4, color=col, lw=1.4, label=g)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"긴 무리 비율\n({BIN_DAYS}일 구간)", fontsize=9.5, color="#555555")
    ax.set_xlabel("웨이퍼 시작 시각 (첫 웨이퍼 기준 일)", fontsize=9.5, color="#555555")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=2)
    style(ax)
    fig.text(0.02, 0.975, "456 그룹 압력 > 0 시간의 두 무리 — 시간순", fontsize=13, color="#222222", va="top")
    fig.text(0.02, 0.945, "점 = 웨이퍼 (train·val·test 센서, 라벨 미사용). 점선(:) = 220초 경계, "
             "회색 파선 = 헤드 부품 교체(BACKING_FILM 리셋)", fontsize=9.5, color="#777777", va="top")
    fig.savefig(FIG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for p in (OUT, FIG):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    val = load_labels("val").rename(columns={"AVG_REMOVAL_RATE": "MRR_raw"})
    chk = tab[tab["split"] == "val"].merge(val, on=KEY)
    if len(chk) != len(val) or not np.allclose(chk["MRR"], chk["MRR_raw"]):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")
    if tab.loc[tab["split"] == "test", "MRR"].notna().any():
        raise ValueError("집계표에 test MRR이 있다")
    t0 = tab["t_start"].min()
    feat = pd.read_csv(FEATURES)
    d = tab[tab["group"].isin(GROUPS)][KEY + ["split", "group", "t_start", "MRR"] + START6].merge(feat, on=KEY)
    assert d["press_long"].notna().all()
    d["press_long"] = d["press_long"].astype(int)
    d["day"] = (d["t_start"] - t0) / 86400
    print(f"456 그룹 {len(d):,}장. 센서 읽는 중", flush=True)

    frames = []
    for split in SPLITS:
        df, _ = dedup(load_sensors(split))
        frames.append(df.merge(d[KEY], on=KEY))
    ws = wafer_sensor_table(pd.concat(frames, ignore_index=True))
    d = d.merge(ws, on=KEY)
    if len(d) != (tab["group"].isin(GROUPS)).sum():
        raise ValueError("센서 웨이퍼 수가 집계표와 다르다")

    rows = []

    def add(check, g, item, a, b, p=np.nan, **kw):
        rows.append({"check": check, "group": g, "item": item, "short": a, "long": b, "p": p, **kw})

    # 1. 수·split·시기
    for g in GROUPS:
        dg = d[d["group"] == g]
        s, lg = dg[dg["press_long"] == 0], dg[dg["press_long"] == 1]
        add("1_count", g, "n", len(s), len(lg))
        for sp in SPLITS:
            add("1_count", g, f"n_{sp}", int((s["split"] == sp).sum()), int((lg["split"] == sp).sum()))
        add("1_count", g, "share_long", np.nan, len(lg) / len(dg))
        for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
            add("1_day", g, f"day_q{int(q * 100):02d}", s["day"].quantile(q), lg["day"].quantile(q))
        add("1_day", g, "press_on_s_median", s["press_on_s"].median(), lg["press_on_s"].median())

    # 2. 센서 차이
    num = [c for c in ws.columns if c not in KEY]
    for g in GROUPS:
        dg = d[d["group"] == g]
        s, lg = dg[dg["press_long"] == 0], dg[dg["press_long"] == 1]
        for c in num:
            a, b = s[c].dropna(), lg[c].dropna()
            p = mannwhitneyu(a, b).pvalue if len(a) and len(b) and (a.nunique() > 1 or b.nunique() > 1) else np.nan
            add("2_sensor", g, c, a.median(), b.median(), p, mean_short=a.mean(), mean_long=b.mean())
        lab = dg[dg["MRR"].notna()]
        ls, ll = lab[lab["press_long"] == 0]["MRR"], lab[lab["press_long"] == 1]["MRR"]
        add("2_mrr", g, "MRR_median (train·val)", ls.median(), ll.median(), mannwhitneyu(ls, ll).pvalue,
            n_short=len(ls), n_long=len(ll))
        coef, se, p = ols_flag(lab["MRR"], lab["press_long"], lab[["t_start"] + START6])
        add("2_mrr", g, "MRR long − short, t_start + 소모품 6개 통제 (OLS)", np.nan, coef, p, se=se)

    # 3. 시간상 섞임
    for g in GROUPS:
        dg = d[d["group"] == g].sort_values("t_start")
        rt = runs_test(1 - dg["press_long"].to_numpy())      # n_allzero 열 = 짧은 무리 수
        add("3_runs", g, "runs (관측 / 무작위 기대)", rt["runs"], rt["runs_expected"], rt["p"], z=rt["z"])
        x = dg["press_long"].to_numpy()
        lens = np.diff(np.flatnonzero(np.r_[True, x[1:] != x[:-1], True]))
        first = x[0]
        rl = pd.Series(lens, index=[(first + i) % 2 for i in range(len(lens))])
        add("3_runs", g, "연 길이 중앙값 / 최대", rl[rl.index == 0].median(), rl[rl.index == 1].median(),
            max_short=int(rl[rl.index == 0].max()), max_long=int(rl[rl.index == 1].max()))
        b = (dg["day"] // BIN_DAYS).astype(int)
        for k, share in dg.groupby(b)["press_long"].agg(["mean", "size"]).iterrows():
            add("3_bin", g, f"day_{k * BIN_DAYS:02d}-{(k + 1) * BIN_DAYS:02d}", np.nan, share["mean"],
                n=int(share["size"]))
    a = d[d["group"] == "A-456"].sort_values("t_start")
    bb = d[d["group"] == "B-456"].sort_values("t_start")
    j = np.searchsorted(bb["t_start"].to_numpy(), a["t_start"].to_numpy())
    j = np.clip(j, 1, len(bb) - 1)
    tb = bb["t_start"].to_numpy()
    near = np.where(np.abs(tb[j - 1] - a["t_start"].to_numpy()) <= np.abs(tb[j] - a["t_start"].to_numpy()), j - 1, j)
    same = (a["press_long"].to_numpy() == bb["press_long"].to_numpy()[near])
    pa, pb = a["press_long"].mean(), bb["press_long"].mean()
    gap_min = np.abs(tb[near] - a["t_start"].to_numpy()) / 60
    add("3_sync", "A-456 vs B-456", "가장 가까운 B 웨이퍼와 무리 일치 비율 (관측 / 무작위 기대)",
        same.mean(), pa * pb + (1 - pa) * (1 - pb), n=len(a), gap_min_median=float(np.median(gap_min)))
    ss = (a["press_long"].to_numpy() == 0)
    add("3_sync", "A-456 vs B-456", "A가 짧은 무리일 때 가까운 B도 짧은 비율 / B 전체 짧은 비율",
        float((bb["press_long"].to_numpy()[near][ss] == 0).mean()), 1 - pb)

    res = pd.DataFrame(rows)
    num_cols = res.select_dtypes("number").columns
    res[num_cols] = res[num_cols].round(4)

    plt.rcParams.update({"font.family": korean_font(), "axes.unicode_minus": False})
    FIG.parent.mkdir(parents=True, exist_ok=True)
    draw(d, tab, t0)
    DIAG.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/hyp_press_cluster.py",
        "question: 456 그룹 압력 > 0 시간의 두 무리(약 150초 / 285초)는 무엇이 다른가",
        "input: data/processed/wafer_table.csv + data/processed/features_S.csv (press_long) + "
        "data/raw/{train,val,test} 센서 + data/raw/labels/CMP-validation-removalrate.csv (집계표 val MRR과 대조 후 사용)",
        f"groups: {GROUPS}",
        "cluster: press_long = 1 (long) if press_on_s >= 220 s, else 0 (short). src/features.py",
        f"time: 행 dt = 다음 기록까지 간격, dt > {GAP_S} 초 빈칸은 0. dwell/press_on 은 챔버별 dt 합",
        f"ret_levels: {dict(zip(LEVELS, [str(x) for x in zip(LEVEL_BINS[:-1], LEVEL_BINS[1:])]))}  # 압력 > 0 행만",
        "pos_{센서}_ch{c}: 그 챔버 행 중 값 > 0 비율. onmean_{센서}: 압력 > 0 행 평균. "
        "inc_{카운터}_ch{c}: 챔버 안 마지막 − 첫 값",
        "p: Mann–Whitney (2_sensor, 2_mrr 중앙값), OLS 정규 근사 (2_mrr 통제), 연 검정 정규 근사 (3_runs)",
        f"mrr_controls: {['t_start'] + START6} (z-score)",
        "sync: A-456 웨이퍼마다 시작 시각이 가장 가까운 B-456 웨이퍼. 기대 = pA*pB + (1-pA)*(1-pB)",
        f"fig: {FIG.relative_to(ROOT).as_posix()}",
        "locked: data/locked/ 와 test 라벨은 열지 않았다",
        "note: 진단 결과 (14.4). MRR 은 비교에만 사용", "",
    ]), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(res.to_string(index=False))
    print(f"저장: {OUT.relative_to(ROOT)}, {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

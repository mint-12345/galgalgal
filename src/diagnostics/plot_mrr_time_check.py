"""mrr_time_check (2026-11-03) 결과 그림 3개 (사후 진단, CLAUDE.md 14.4).

그림에 적는 숫자(b, CI, R², 통제 후 b, CV 비율, 평균 MRR)는 커밋된
results/diagnostics/2026-11-03_김혜연_mrr_time_check.csv에서 읽는다.
점(웨이퍼별 값)은 집계 csv에 없으므로 mrr_time_check.py와 같은 입력을 읽는다:
  data/processed/features_v1.csv (키·group·dur_POLISH_s1·ch56_step) + data/raw/labels/의 train·val 라벨.
점으로 다시 계산한 b·평균 MRR이 csv 값과 같은지 확인한 뒤 그린다. 웨이퍼 단위 파일은 저장하지 않는다.
test 센서·라벨과 data/locked/는 읽지 않는다. 난수를 쓰지 않는다.

그림
  1. mrr_time_scatter_ch4: 그룹별 MRR 대 챔버 4(A-123은 챔버 1) 체류시간, 로그-로그 축.
     1~99% 절삭 범위 밖 웨이퍼는 회색. 실선 = 절삭 후 OLS(b), 점선 = 기울기 -1(목표 두께 정지 가설).
  2. mrr_time_ch56_groups: A-456(참고로 B-456)의 ch56_step 0(짧은 무리) 대 1(긴 무리) MRR 분포.
  3. mrr_time_cv_ratio: 그룹 × 시간 정의별 CV(연마량)/CV(MRR), 1~99% 절삭. 1보다 작으면 연마량이 MRR보다 고르다.

실행: python src/diagnostics/plot_mrr_time_check.py
출력: results/figures/2026-11-04_김혜연_mrr_time_{scatter_ch4, ch56_groups, cv_ratio}.png (있으면 멈춤, 덮어쓰기 금지)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import FontProperties, findSystemFonts  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load import KEY, ROOT, load_labels  # noqa: E402

DATE = "2026-11-04"
AUTHOR = "김혜연"
IN_CSV = ROOT / "results" / "diagnostics" / "2026-11-03_김혜연_mrr_time_check.csv"
FEAT = ROOT / "data" / "processed" / "features_v1.csv"
FIGS = {k: ROOT / "results" / "figures" / f"{DATE}_{AUTHOR}_mrr_time_{k}.png"
        for k in ["scatter_ch4", "ch56_groups", "cv_ratio"]}
GROUPS = ["A-456", "B-456", "A-123"]
TIME_KO = {"dur_s1": "첫 챔버 체류\n(dur_POLISH_s1)", "n_press": "압력 > 0 행 수\n(웨이퍼 전체)",
           "dur_w": "웨이퍼 전체 길이\n(t_end - t_start)"}
COLORS = {"A-456": "#1f77b4", "B-456": "#d62728", "A-123": "#2ca02c"}
N_EXPECTED = 2401
TRIM_Q = 0.01

for p in [IN_CSV, FEAT]:
    assert "locked" not in str(p).lower(), p


def korean_font() -> str:
    want = ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "Noto Sans KR", "AppleGothic"]
    have = {FontProperties(fname=f).get_name() for f in findSystemFonts()}
    return next((n for n in want if n in have), plt.rcParams["font.family"][0])


plt.rcParams.update({"font.family": korean_font(), "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def stat(R, group, subset, tdef, trim, name, col="value"):
    x = R[(R["group"] == group) & (R["subset"] == subset) & (R["time_def"] == tdef)
          & (R["trim"] == trim) & (R["stat"] == name)]
    assert len(x) == 1, (group, subset, tdef, trim, name, len(x))
    return float(x.iloc[0][col])


def mean_mrr(R, group, subset):
    x = R[(R["group"] == group) & (R["subset"] == subset) & (R["section"] == "cv") & (R["trim"] == "none")
          & R["time_def"].isna() & (R["stat"] == "mean_mrr")]
    assert len(x) == 1, (group, subset)
    return float(x.iloc[0]["value"]), int(x.iloc[0]["n"])


def load_points():
    f = pd.read_csv(FEAT, usecols=KEY + ["group", "dur_POLISH_s1", "ch56_step"])
    assert len(f) == N_EXPECTED, len(f)
    lab = pd.concat([load_labels(s) for s in ("train", "val")], ignore_index=True)
    d = f.merge(lab.rename(columns={"AVG_REMOVAL_RATE": "MRR"}), on=KEY, how="left", validate="1:1")
    assert d["MRR"].notna().all() and (d["MRR"] <= 400).all()
    return d


def trimmed(s):
    s = s[s["dur_POLISH_s1"] > 0]
    lo, hi = s["dur_POLISH_s1"].quantile([TRIM_Q, 1 - TRIM_Q])
    return (s["dur_POLISH_s1"] >= lo) & (s["dur_POLISH_s1"] <= hi)


def fig_scatter(d, R, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, g in zip(axes, GROUPS):
        s = d[d["group"] == g]
        keep = trimmed(s)
        x, y = s["dur_POLISH_s1"], s["MRR"]
        lx, ly = np.log(x[keep]), np.log(y[keep])
        b = float(((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum())
        b_csv = stat(R, g, "전체", "dur_s1", "p01_p99", "b")
        assert abs(b - b_csv) < 1e-4 * max(1, abs(b_csv)), (g, b, b_csv)
        assert int(keep.sum()) == int(R[(R["group"] == g) & (R["subset"] == "전체") & (R["time_def"] == "dur_s1")
                                        & (R["trim"] == "p01_p99") & (R["stat"] == "b")]["n"].iloc[0])
        ax.scatter(x[~keep], y[~keep], s=10, marker="x", color="#aaaaaa", lw=0.8, label="1~99% 밖 (회귀에서 제외)")
        ax.scatter(x[keep], y[keep], s=7, alpha=0.35, color=COLORS[g], lw=0, label="회귀에 쓴 웨이퍼")
        xs = np.linspace(lx.min(), lx.max(), 50)
        a = ly.mean() - b * lx.mean()
        ax.plot(np.exp(xs), np.exp(a + b * xs), color="k", lw=1.5, label="OLS 기울기 b")
        ax.plot(np.exp(xs), np.exp(ly.mean() - 1.0 * (xs - lx.mean())), color="k", lw=1, ls="--",
                label="기울기 -1 (목표 두께 정지)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # 축은 회귀 범위에 맞춘다 (긴 꼬리 몇 장이 축을 늘려 점이 한 줄로 몰리는 것을 막음). 축 밖 장수는 글자로 적는다
        xl = (x[keep].min() / 1.08, x[keep].max() * 1.08)
        ax.set_xlim(*xl)
        n_out = int(((x < xl[0]) | (x > xl[1])).sum())
        yl = ax.get_ylim()
        for a_, (v0, v1) in ((ax.xaxis, xl), (ax.yaxis, yl)):   # 로그 축이지만 범위 안에 보통 숫자 눈금
            ticks = [t for t in matplotlib.ticker.MaxNLocator(5).tick_values(v0, v1) if v0 <= t <= v1]
            a_.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            a_.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            a_.set_minor_locator(matplotlib.ticker.NullLocator())
        lo, hi = stat(R, g, "전체", "dur_s1", "p01_p99", "b", "ci_lo"), stat(R, g, "전체", "dur_s1", "p01_p99", "b", "ci_hi")
        bc = stat(R, g, "전체", "dur_s1", "p01_p99", "b_ctrl")
        bcl, bch = (stat(R, g, "전체", "dur_s1", "p01_p99", "b_ctrl", c) for c in ["ci_lo", "ci_hi"])
        r2 = stat(R, g, "전체", "dur_s1", "p01_p99", "r2")
        txt = (f"b = {b_csv:.2f} [{lo:.2f}, {hi:.2f}], R² = {r2:.3f}\n"
               f"통제 후 b = {bc:.2f} [{bcl:.2f}, {bch:.2f}]\nn = {int(keep.sum())} (전체 {len(s)}, 축 밖 {n_out}장)")
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
        ch = "챔버 1" if g == "A-123" else "챔버 4"
        ax.set_title(f"{g}")
        ax.set_xlabel(f"{ch} 체류시간 dur_POLISH_s1 (초, 로그)")
        ax.set_ylabel("MRR (로그)")
        if g == "A-123":
            ax.legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle("MRR 대 첫 챔버 체류시간 (train+val, 이상치 제외). b = log(MRR)~log(T) 기울기, 1~99% 절삭", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def fig_ch56(d, R, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, g in zip(axes, ["A-456", "B-456"]):
        s = d[d["group"] == g]
        bins = np.linspace(s["MRR"].quantile(0.002), s["MRR"].quantile(0.998), 40)
        for step, sub, c in [(0, "ch56_step=0 (짧은 무리)", "#1f77b4"), (1, "ch56_step=1 (긴 무리)", "#ff7f0e")]:
            m = s.loc[s["ch56_step"] == step, "MRR"]
            mu, n = mean_mrr(R, g, sub)
            assert abs(m.mean() - mu) < 1e-3 and len(m) == n, (g, step)
            med_t = stat(R, g, sub, "n_press", "none", "median_time")
            ax.hist(m, bins=bins, density=True, alpha=0.45, color=c,
                    label=f"{sub}: 압력 행 중앙값 {med_t:.0f}, n = {n}, 평균 {mu:.2f}")
            ax.axvline(mu, ymax=0.72, color=c, lw=1.5)   # 범례 아래에서 멈춤
        ax.set_title(g + ("" if g == "A-456" else " (참고)"))
        ax.set_xlabel("MRR")
        ax.set_ylabel("밀도")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)   # 범례 자리 (위 0.72와 짝)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("챔버 5·6 시간이 다른 두 무리의 MRR 분포 (세로선 = 평균). 압력 > 0 시간이 약 2배여도 평균 MRR이 거의 같다",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def fig_cv(R, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    tds = list(TIME_KO)
    w = 0.26
    for i, g in enumerate(GROUPS):
        vals = [stat(R, g, "전체", td, "p01_p99", "cv_ratio_removal_to_mrr") for td in tds]
        xs = np.arange(len(tds)) + (i - 1) * w
        ax.bar(xs, vals, w, color=COLORS[g], label=g)
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.06, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1, color="k", lw=0.8, ls="--", label="1 = 연마량과 MRR의 상대 변동이 같음")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 60)
    ax.set_xticks(np.arange(len(tds)), [TIME_KO[t] for t in tds])
    ax.set_ylabel("CV(연마량 = MRR × T) / CV(MRR), 로그")
    ax.set_title("목표 두께에서 멈춘다면 이 비율이 1보다 훨씬 작아야 한다 (1~99% 절삭)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def main():
    for p in FIGS.values():
        if p.exists():
            raise SystemExit(f"{p} 가 이미 있다 (덮어쓰기 금지)")
    R = pd.read_csv(IN_CSV)
    d = load_points()
    FIGS["scatter_ch4"].parent.mkdir(parents=True, exist_ok=True)
    fig_scatter(d, R, FIGS["scatter_ch4"])
    fig_ch56(d, R, FIGS["ch56_groups"])
    fig_cv(R, FIGS["cv_ratio"])
    for p in FIGS.values():
        print(f"저장: {p}", flush=True)


if __name__ == "__main__":
    main()

"""A-123 전이학습(456 → A-123, PR #25) 결과 PDF 보고서.

숫자는 모두 PR #25 결과 csv에서 다시 계산한다. 모델을 다시 돌리지 않고, 결과 csv를 고치지 않는다.
features·라벨 파일·test·data/locked/는 읽지 않는다. A-123 MRR 변동계수는 t2_decomposition의
all_wafers 행(A-123 train+val 431장의 평균·표준편차)에서 계산한다.

검산 (스크립트 안 assert):
- T0·T1이 소스·입력과 무관하게 fold마다 같다
- 다시 계산한 보정 t-검정(cv_common.corrected_ttest)이 stats·judgment csv와 같다 (1e-9)
- 다시 계산한 ρ 분포가 rho csv와 같다
- GP 경고 진단의 건수가 주 결과 csv의 건수와 같다 (공개 2~16%)
- PDF에 쓰는 모든 글자가 내장 폰트에 있다 (□ 방지)

판정 규칙은 preregistration.md 5.5절과 5.7-16 그대로다 (8%, f_456 full, 평균 MSE 차이 < 0이고 p < 0.05가 S1·S2 둘 다).

폰트: Noto Sans KR 가변 폰트(Windows 기본)의 기본 인스턴스가 Thin이라, matplotlib에 딸린 fontTools로
Regular(400)·Bold(700) 고정 인스턴스를 임시 폴더에 만들어 쓴다. PDF에는 TrueType(Type 42) 부분집합으로 내장된다.

실행: python src/report_a123_transfer.py
      python src/report_a123_transfer.py --out-dir <폴더>   (재현 확인용. 같은 이름으로 그 폴더에 저장)
출력: results/reports/{OUT_STEM}.pdf (+ .yaml), results/figures/{OUT_STEM}_fig{1~3}_*.png,
      results/diagnostics/{DAY}_{AUTHOR}_a123_transfer_report_refcheck.csv (+ .yaml)
      이미 있으면 멈춘다 (덮어쓰기 금지).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cv  # noqa: E402
from a123_transfer import sha, write_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUTHOR = "김혜연"
DAY = "2026-11-04"
STEM = "2026-11-01_김혜연_FV1_a123_transfer"
OUT_STEM = f"{DAY}_{AUTHOR}_FV1_a123_transfer_report"
IN = {c: RESULTS / f"{STEM}_{c}.csv" for c in ["folds", "stats", "judgment", "rho"]}
IN_YAML = RESULTS / f"{STEM}_folds.yaml"
DIAG = {k: RESULTS / "diagnostics" / f"2026-11-01_김혜연_{k}.csv" for k in ["gp_warnings", "t2_decomposition"]}
RATIOS = [2, 4, 8, 16, 100]
MODELS = ["T0", "T1", "T2", "T3"]
SOURCES = ["S1", "S2"]
INPUTS = ["full", "usage"]
PRIMARY = 8
NF = 25
SUBSET_KO = {"all": "전체", "clean": "센서 정상", "zero": "전체-0"}
COLORS = {"T0": "#7f7f7f", "T1": "#1f77b4", "T2": "#2ca02c", "T3": "#d62728"}
SRC_COLORS = {"S1": "#d62728", "S2": "#9467bd"}

# 참고값 (사용자 전달, 대조용). 보고서 숫자는 이 값이 아니라 결과 파일에서 계산한다.
REFERENCE = [
    ("주 판정 p, S1", "0.409"), ("주 판정 p, S2", "0.484"),
    ("2차 판정 T3−T2, S1", "−33.2"), ("2차 판정 T3−T2, S2", "−94.6"),
    ("T1의 T0 대비 RMSE 개선, 8%", "9.5%"), ("T1의 T0 대비 RMSE 개선, 16%", "18.6%"),
    ("T1의 T0 대비 RMSE 개선, 100%", "37.3%"),
    ("ρ, S1 full, 비율 전체 범위", "0.00~0.13"), ("ρ, S1 full, 8% 평균", "0.07"),
    ("ρ, usage, 범위", "0.21~0.32"),
    ("f456–MRR 상관, full 전체", "0.08"), ("f456–MRR 상관, full 센서 정상", "0.16"),
    ("f456–MRR 상관, usage 전체", "0.44"),
    ("90% 구간 포함률, 공개 2~8%", "0.65~0.75"),
    ("T1 RMSE, 센서 정상 (8%)", "3.42"), ("T1 RMSE, 전체-0 (8%)", "3.70"),
    ("GP 경고 합계", "680"), ("GP 경고, 길이척도 상한", "506"), ("GP 경고, 잡음 하한", "141"),
    ("A-123 MRR 변동계수", "0.027"),
]

for p in [*IN.values(), *DIAG.values()]:
    assert "locked" not in str(p).lower(), p




def git_state():
    """a123_transfer.git_state와 같지만 git 출력을 UTF-8로 읽는다 (한글 파일명이 있으면 cp949 해독이 실패함)."""
    def run(*a):
        r = subprocess.run(["git", "-c", "core.quotepath=false", *a], cwd=ROOT, capture_output=True,
                           encoding="utf-8", errors="replace")
        return r.stdout.strip()
    dirty = [ln for ln in run("status", "--porcelain").splitlines() if ln]
    return {"commit": run("rev-parse", "HEAD") or "unknown", "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_files": len(dirty)}


# ---------------------------------------------------------------- 폰트
def static_fonts() -> dict:
    """Noto Sans KR 가변 폰트에서 Regular·Bold 고정 인스턴스를 만든다 (임시 폴더에 캐시)."""
    vf = Path("C:/Windows/Fonts/NotoSansKR-VF.ttf")
    if not vf.exists():
        hits = [f for f in fm.findSystemFonts() if Path(f).name.lower().startswith("notosanskr-vf")]
        if not hits:
            sys.exit("Noto Sans KR 가변 폰트(NotoSansKR-VF.ttf)가 없다. 설치 후 다시 실행")
        vf = Path(hits[0])
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    key = hashlib.sha256(vf.read_bytes()).hexdigest()[:12]
    cache = Path(tempfile.gettempdir()) / "drift_k_fonts"
    cache.mkdir(exist_ok=True)
    out = {}
    for name, w in [("regular", 400), ("bold", 700)]:
        p = cache / f"NotoSansKR-{name}-{key}-v2.ttf"
        if not p.exists():
            print(f"폰트 인스턴스 생성: {name} (처음 한 번, 수십 초)", flush=True)
            f = instantiateVariableFont(TTFont(vf), {"wght": w})
            style = name.capitalize()
            f["OS/2"].usWeightClass = w
            nt = f["name"]
            for nid in (16, 17, 25):
                nt.removeNames(nameID=nid)
            for nid, val in [(1, "Noto Sans KR"), (2, style), (3, f"NotoSansKR-{style}-static"),
                             (4, f"Noto Sans KR {style}"), (6, f"NotoSansKR-{style}")]:
                nt.setName(val, nid, 3, 1, 0x409)
            f.save(p)
        out[name] = p
    out["cmap"] = set(TTFont(out["regular"]).getBestCmap())
    out["source"] = vf
    return out


FONTS = static_fonts()
fm.fontManager.addfont(str(FONTS["regular"]))
FP = fm.FontProperties(fname=str(FONTS["regular"]))
FPB = fm.FontProperties(fname=str(FONTS["bold"]))
plt.rcParams.update({
    "font.family": FP.get_name(), "axes.unicode_minus": False, "pdf.fonttype": 42,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
})
USED_CHARS: set = set()


def T(s) -> str:
    """그림·표·본문에 쓰는 문자열을 모아 폰트에 글자가 있는지 마지막에 확인한다."""
    s = str(s)
    USED_CHARS.update(s)
    return s


# ---------------------------------------------------------------- 숫자 형식
def f(x, nd=3):
    return T(f"{x:.{nd}f}".replace("-", "−"))


def fpv(p):
    return T(f"{p:.1e}".replace("-", "−") if p < 1e-3 else f"{p:.3f}")


def pm(s, nd=3):
    return T(f"{f(s.mean(), nd)} ± {s.std(ddof=1):.{nd}f}")


# ---------------------------------------------------------------- 결과 읽기와 검산
def load():
    F = pd.read_csv(IN["folds"])
    S = pd.read_csv(IN["stats"])
    J = pd.read_csv(IN["judgment"])
    RH = pd.read_csv(IN["rho"])
    G = pd.read_csv(DIAG["gp_warnings"])
    D = pd.read_csv(DIAG["t2_decomposition"])
    Y = {}
    for line in IN_YAML.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\s*)([\w\-.]+): (.*)$", line)
        if m:
            Y[m.group(2)] = m.group(3).strip().strip('"')
    assert set(F["ratio"]) == set(RATIOS) and F.groupby(["ratio", "input", "source", "model", "subset"]).size().eq(NF).all()
    # T0·T1은 소스·입력과 무관
    for m in ["T0", "T1"]:
        g = F[F["model"] == m]
        assert g.groupby(["ratio", "r", "subset"])["mse"].nunique().max() == 1, m
    return F, S, J, RH, G, D, Y


def sel(F, ratio=None, model=None, source="S1", inp="full", subset="all"):
    g = F
    if ratio is not None:
        g = g[g["ratio"] == ratio]
    if model is not None:
        g = g[g["model"] == model]
    g = g[(g["source"] == source) & (g["input"] == inp) & (g["subset"] == subset)]
    return g.sort_values(["ratio", "r"])


def ttest(F, ratio, a, b, source, inp):
    A, B = sel(F, ratio, a, source, inp), sel(F, ratio, b, source, inp)
    assert len(A) == NF and (A["r"].to_numpy() == B["r"].to_numpy()).all()
    assert (A["n_eval"].to_numpy() == B["n_eval"].to_numpy()).all()
    d = A["mse"].to_numpy() - B["mse"].to_numpy()
    full = cv.corrected_ttest(d, A["n_train"].to_numpy(), A["n_eval"].to_numpy())
    rev = cv.corrected_ttest(d, A["n_public"].to_numpy(), A["n_eval"].to_numpy())
    full["nb_ratio"] = full.pop("ratio")  # 공개 비율 열과 이름이 겹치지 않게
    return {**full, "p_revealed": rev["p"], "wins": int((d < 0).sum()),
            "rmse_a": A["rmse"].mean(), "rmse_b": B["rmse"].mean()}


def all_tests(F, S, J):
    rows = []
    for ratio in RATIOS:
        for inp in INPUTS:
            for s in SOURCES:
                for cmp_ in ["T3-T1", "T3-T2", "T2-T1"]:
                    a, b = cmp_.split("-")
                    rows.append({"ratio": ratio, "input": inp, "source": s, "compare": cmp_,
                                 **ttest(F, ratio, a, b, s, inp)})
    R = pd.DataFrame(rows)
    # 저장된 stats csv와 같은지
    m = R.merge(S, on=["ratio", "input", "source", "compare"], suffixes=("", "_saved"))
    assert len(m) == len(R) == len(S)
    for c in ["mean_diff", "t", "p"]:
        assert np.allclose(m[c], m[f"{c}_saved"], rtol=0, atol=1e-9), c
    assert np.allclose(m["p_revealed"], m["p_revealed_saved"], rtol=0, atol=1e-9)
    # 판정 다시 내리기 (5.5) + judgment csv와 같은지
    verdict = {}
    for kind, cmp_ in [("주 판정", "T3-T1"), ("2차 판정", "T3-T2")]:
        x = R[(R["ratio"] == PRIMARY) & (R["input"] == "full") & (R["compare"] == cmp_)].set_index("source")
        passed = {s: bool(x.loc[s, "mean_diff"] < 0 and x.loc[s, "p"] < 0.05) for s in SOURCES}
        n = sum(passed.values())
        v = "효과 있음" if n == 2 else ("소스에 따라 갈림 (기본값 T1 유지)" if n == 1 else "효과 없음 (기본값 T1 유지)")
        j = J[J["kind"] == kind].iloc[0]
        for s in SOURCES:
            assert abs(j[f"{s}_mean_diff"] - x.loc[s, "mean_diff"]) < 1e-9 and abs(j[f"{s}_p"] - x.loc[s, "p"]) < 1e-9
            assert bool(j[f"{s}_passed"]) == passed[s]
        assert j["verdict"] == v, (j["verdict"], v)
        verdict[kind] = {"verdict": v, "passed": passed}
    return R, verdict


def rho_dist(F, RH):
    rows = []
    for inp in INPUTS:
        for s in SOURCES:
            for ratio in RATIOS:
                g = sel(F, ratio, "T3", s, inp)["rho"]
                rows.append({"input": inp, "source": s, "ratio": ratio, "mean": g.mean(), "std": g.std(ddof=1),
                             "min": g.min(), "max": g.max(), "n0": int((g <= 1e-12).sum()), "n2": int((g >= 2 - 1e-12).sum())})
    R = pd.DataFrame(rows)
    m = R.merge(RH, on=["input", "source", "ratio"], suffixes=("", "_saved"))
    assert len(m) == len(R) == len(RH)
    for c in ["mean", "std", "min", "max"]:
        assert np.allclose(m[c], m[f"{c}_saved"], atol=1e-12), c
    return R


def gp_warning_tables(F, G):
    t1 = F[(F["model"] == "T1") & (F["subset"] == "all")].drop_duplicates(["ratio", "r"])
    t3 = F[(F["model"] == "T3") & (F["subset"] == "all")]
    main = pd.DataFrame({
        "T1": t1.groupby("ratio")["gp_warn_t1"].sum(),
        "T1_folds": t1.groupby("ratio")["gp_warn_t1"].apply(lambda s: int((s > 0).sum())),
        "T3": t3.groupby("ratio")["gp_warn_t3"].sum(),
        "T3_units": t3.groupby("ratio")["gp_warn_t3"].apply(lambda s: int((s > 0).sum())),
    }).reindex(RATIOS)
    # 진단 파일의 건수가 주 결과와 같은지 (공개 2~16%)
    dg = G.groupby(["ratio", "model"])["n_warn"].sum().unstack()
    for ratio in [2, 4, 8, 16]:
        assert dg.loc[ratio, "T1"] == main.loc[ratio, "T1"] and dg.loc[ratio, "T3"] == main.loc[ratio, "T3"], ratio
    assert set(G["ratio"]) == {2, 4, 8, 16}
    assert G["n_warn"].sum() == G["n_convergence"].sum() and G["n_user"].sum() == 0 and G["n_runtime"].sum() == 0
    E = G.assign(msg=G["warn_messages"].str.split(" || ", regex=False)).explode("msg").dropna(subset=["msg"]).reset_index(drop=True)
    k = E["msg"].str.extract(r"parameter (\S+) is close to the specified (\w+) bound")
    assert k.notna().all().all()
    name = {"k1__k2__length_scale": "길이척도", "k2__noise_level": "잡음", "k1__k1__constant_value": "진폭"}
    E["kind"] = k[0].map(name) + " " + k[1].map({"upper": "상한", "lower": "하한"})
    assert E["kind"].notna().all() and len(E) == G["n_warn"].sum()
    kinds = pd.crosstab(E["kind"], E["ratio"]).reindex(columns=[2, 4, 8, 16], fill_value=0)
    kinds = kinds.loc[kinds.sum(axis=1).sort_values(ascending=False).index]
    return main, kinds


# ---------------------------------------------------------------- 그림
def xpos(ratios=RATIOS):
    return np.arange(len(ratios))


def fig_rmse(fig, rect, F):
    gs = fig.add_gridspec(1, 2, left=rect[0], bottom=rect[1], right=rect[0] + rect[2], top=rect[1] + rect[3], wspace=0.25)
    x = xpos()
    for i, s in enumerate(SOURCES):
        ax = fig.add_subplot(gs[0, i])
        for j, m in enumerate(MODELS):
            g = sel(F, None, m, s, "full").groupby("ratio")["rmse"]
            mu, se = g.mean().reindex(RATIOS), (g.std(ddof=1) / np.sqrt(g.count())).reindex(RATIOS)
            ax.errorbar(x + (j - 1.5) * 0.06, mu, yerr=se, marker="o", ms=3.5, lw=1.4, capsize=2.5,
                        color=COLORS[m], label=T(m))
        ax.set_xticks(x, [T(f"{r}") for r in RATIOS])
        ax.set_xlabel(T("공개 비율 (%)"))
        ax.set_ylabel(T("RMSE (평가 쪽, 25 fold 평균)"))
        ax.set_title(T(f"{s} ({'A-456' if s == 'S1' else 'A-456 + B-456'}), f456 입력 full"))
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        ax.set_ylim(0, None)
        if i == 1:
            ax.legend(ncol=2, frameon=False, loc="center right")


def fig_rho(fig, rect, RD):
    gs = fig.add_gridspec(1, 2, left=rect[0], bottom=rect[1], right=rect[0] + rect[2], top=rect[1] + rect[3], wspace=0.25)
    x = xpos()
    for i, inp in enumerate(INPUTS):
        ax = fig.add_subplot(gs[0, i])
        for j, s in enumerate(SOURCES):
            g = RD[(RD["input"] == inp) & (RD["source"] == s)].set_index("ratio").reindex(RATIOS)
            xx = x + (j - 0.5) * 0.18
            ax.vlines(xx, g["min"], g["max"], color=SRC_COLORS[s], lw=1.2, alpha=0.6)
            ax.plot(xx, g["mean"], "o", ms=4.5, color=SRC_COLORS[s], label=T(f"{s} 평균 (세로선 = 25 fold 최소~최대)"))
        ax.set_xticks(x, [T(f"{r}") for r in RATIOS])
        ax.set_xlabel(T("공개 비율 (%)"))
        ax.set_ylabel(T("ρ 추정값 (0~2로 제한)"))
        ax.set_title(T(f"f456 입력 {inp} ({'112개 피처' if inp == 'full' else '사용량 36개'})"))
        ax.set_ylim(-0.03, 0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        if i == 0:
            ax.legend(frameon=False, loc="upper right")


def fig_coverage(fig, rect, F):
    ax = fig.add_axes(rect)
    x = xpos()
    series = [("T1", "S1", "full", COLORS["T1"], "-", "T1 (A-123 단독 GP)"),
              ("T3", "S1", "full", SRC_COLORS["S1"], "-", "T3 S1 full"),
              ("T3", "S2", "full", SRC_COLORS["S2"], "-", "T3 S2 full"),
              ("T3", "S1", "usage", SRC_COLORS["S1"], "--", "T3 S1 usage"),
              ("T3", "S2", "usage", SRC_COLORS["S2"], "--", "T3 S2 usage")]
    for j, (m, s, inp, c, ls, lab) in enumerate(series):
        g = sel(F, None, m, s, inp).groupby("ratio")["coverage90"]
        mu, se = g.mean().reindex(RATIOS), (g.std(ddof=1) / np.sqrt(g.count())).reindex(RATIOS)
        ax.errorbar(x + (j - 2) * 0.05, mu, yerr=se, marker="o", ms=3.5, lw=1.3, ls=ls, capsize=2.5, color=c, label=T(lab))
    ax.axhline(0.9, color="k", lw=0.8, ls=":")
    ax.text(x[-1] + 0.25, 0.9, T("목표 0.90"), va="center", fontsize=7)
    ax.set_xticks(x, [T(f"{r}") for r in RATIOS])
    ax.set_xlim(-0.3, x[-1] + 0.6)
    ax.set_xlabel(T("공개 비율 (%)"))
    ax.set_ylabel(T("90% 예측구간 포함률 (전체 웨이퍼)"))
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, loc="lower right", ncol=2)


# ---------------------------------------------------------------- 쪽 배치
A4 = (8.27, 11.69)
ML, MR, MT, MB = 0.75, 0.75, 0.75, 0.8
TW = A4[0] - ML - MR


class Doc:
    def __init__(self):
        self.pages = []
        self.new_page()
        self._w = {}

    def new_page(self):
        self.fig = plt.figure(figsize=A4)
        self.pages.append(self.fig)
        self.y = A4[1] - MT

    def ensure(self, h):
        if self.y - h < MB:
            self.new_page()

    def space(self, h):
        self.y -= h

    # 글자 폭(인치) — 글자별 캐시
    def width(self, s, size, bold=False):
        tot = 0.0
        for ch in s:
            k = (ch, size, bold)
            if k not in self._w:
                if ch == " ":
                    w = self._char_w("a a", size, bold) - self._char_w("aa", size, bold)
                else:
                    w = self._char_w(ch, size, bold)
                self._w[k] = w
            tot += self._w[k]
        return tot

    def _char_w(self, s, size, bold):
        t = self.fig.text(0, 0, s, fontproperties=FPB if bold else FP, size=size)
        w = t.get_window_extent(self.fig.canvas.get_renderer()).width / self.fig.dpi
        t.remove()
        return w

    def wrap(self, s, width, size, bold=False):
        out = []
        for para in str(s).split("\n"):
            line = ""
            for word in para.split(" "):
                cand = word if not line else line + " " + word
                if self.width(cand, size, bold) <= width:
                    line = cand
                    continue
                if line:
                    out.append(line)
                line = word
                while self.width(line, size, bold) > width:  # 긴 단어는 글자 단위로 자른다
                    i = len(line)
                    while i > 1 and self.width(line[:i], size, bold) > width:
                        i -= 1
                    out.append(line[:i])
                    line = line[i:]
            out.append(line)
        return out

    def text_at(self, x, y, s, size=9, bold=False, color="k", ha="left", va="top"):
        self.fig.text(x / A4[0], y / A4[1], T(s), fontproperties=FPB if bold else FP, size=size,
                      color=color, ha=ha, va=va)

    def para(self, s, size=9, bold=False, indent=0.0, color="k", gap=0.06, lead=1.5):
        lines = self.wrap(s, TW - indent, size, bold)
        lh = size * lead / 72
        for ln in lines:
            self.ensure(lh)
            self.text_at(ML + indent, self.y, ln, size, bold, color)
            self.y -= lh
        self.y -= gap

    def bullets(self, items, size=9):
        for it in items:
            lines = self.wrap(it, TW - 0.2, size)
            lh = size * 1.5 / 72
            self.ensure(lh * min(len(lines), 2))
            self.text_at(ML + 0.05, self.y, "•", size)
            for ln in lines:
                self.ensure(lh)
                self.text_at(ML + 0.2, self.y, ln, size)
                self.y -= lh
            self.y -= 0.02
        self.y -= 0.05

    def h1(self, s):
        self.ensure(1.8)  # 제목만 쪽 끝에 남지 않게
        self.y -= 0.12
        self.text_at(ML, self.y, s, 13, bold=True, color="#1a3a5c")
        self.y -= 0.27
        self.fig.add_artist(plt.Line2D([ML / A4[0], (A4[0] - MR) / A4[0]], [self.y / A4[1]] * 2,
                                       color="#1a3a5c", lw=0.8))
        self.y -= 0.12

    def h2(self, s):
        self.ensure(1.0)
        self.y -= 0.05
        self.text_at(ML, self.y, s, 10.5, bold=True, color="#1a3a5c")
        self.y -= 0.24

    def table(self, header, rows, widths, size=7.6, caption=None, note=None, align=None, bold_rows=()):
        """widths: 열 너비 비율. 셀 글자는 칸 안에서 줄바꿈한다. 표는 쪽을 넘기지 않는다."""
        widths = np.asarray(widths, float)
        widths = widths / widths.sum() * TW
        align = align or (["left"] + ["right"] * (len(header) - 1))
        lh, pad = size * 1.35 / 72, 0.05

        def cells(row, bold):
            return [self.wrap(c, w - 2 * pad, size, bold) for c, w in zip(row, widths)]

        H = cells(header, True)
        B = [cells(r, i in bold_rows) for i, r in enumerate(rows)]
        hh = max(len(c) for c in H) * lh + 2 * pad
        bh = [max(len(c) for c in r) * lh + 2 * pad for r in B]
        cap_h = (len(self.wrap(caption, TW, 8.5, True)) * 8.5 * 1.45 / 72 + 0.04) if caption else 0
        self.ensure(cap_h + hh + sum(bh) + 0.15)
        if caption:
            self.para(caption, 8.5, bold=True, gap=0.02, lead=1.45)
        xs = ML + np.concatenate([[0], np.cumsum(widths)])
        fig = self.fig

        def hline(y, lw):
            fig.add_artist(plt.Line2D([ML / A4[0], (ML + TW) / A4[0]], [y / A4[1]] * 2, color="k", lw=lw))

        def draw_row(c_lines, h, bold, fill=None):
            if fill:
                fig.add_artist(Rectangle((ML / A4[0], (self.y - h) / A4[1]), TW / A4[0], h / A4[1],
                                         transform=fig.transFigure, color=fill, lw=0, zorder=0))
            for k, lines in enumerate(c_lines):
                for n, ln in enumerate(lines):
                    yy = self.y - pad - n * lh
                    if align[k] == "left":
                        self.text_at(xs[k] + pad, yy, ln, size, bold)
                    elif align[k] == "center":
                        self.text_at((xs[k] + xs[k + 1]) / 2, yy, ln, size, bold, ha="center")
                    else:
                        self.text_at(xs[k + 1] - pad, yy, ln, size, bold, ha="right")
            self.y -= h

        hline(self.y, 1.0)
        draw_row(H, hh, True, fill="#e8eef5")
        hline(self.y, 0.6)
        for i, (r, h) in enumerate(zip(B, bh)):
            draw_row(r, h, i in bold_rows, fill="#f6f6f6" if i % 2 else None)
        hline(self.y, 1.0)
        self.y -= 0.06
        if note:
            self.para(note, 7.4, color="#444444", gap=0.04, lead=1.4)
        self.y -= 0.08

    def figure(self, draw, h, caption, png=None, png_w=None):
        cap_lines = self.wrap(caption, TW, 7.8)
        need = h + len(cap_lines) * 7.8 * 1.45 / 72 + 0.15
        self.ensure(need)
        b = self.y - h
        rect = [(ML + 0.45) / A4[0], (b + 0.4) / A4[1], (TW - 0.6) / A4[0], (h - 0.55) / A4[1]]
        draw(self.fig, rect)
        self.y = b - 0.05
        self.para(caption, 7.8, color="#333333", gap=0.1, lead=1.45)
        if png is not None:
            w = png_w or TW
            pf = plt.figure(figsize=(w, h))
            draw(pf, [0.45 / w, 0.4 / h, (w - 0.6) / w, (h - 0.55) / h])
            pf.savefig(png, dpi=200, metadata={"Software": None})
            plt.close(pf)

    def kpis(self, items):
        """요약 핵심 숫자 상자: [(값, 설명), ...]"""
        n = len(items)
        gap = 0.12
        w = (TW - gap * (n - 1)) / n
        lines = [self.wrap(d, w - 0.16, 7.6) for _, d in items]
        h = 0.62 + max(len(x) for x in lines) * 7.6 * 1.4 / 72
        self.ensure(h + 0.1)
        for i, ((v, _), ls) in enumerate(zip(items, lines)):
            x = ML + i * (w + gap)
            self.fig.add_artist(Rectangle((x / A4[0], (self.y - h) / A4[1]), w / A4[0], h / A4[1],
                                          transform=self.fig.transFigure, facecolor="#f2f6fa",
                                          edgecolor="#9fb3c8", lw=0.6))
            self.text_at(x + 0.08, self.y - 0.1, v, 14, bold=True, color="#1a3a5c")
            for n_, ln in enumerate(ls):
                self.text_at(x + 0.08, self.y - 0.52 - n_ * 7.6 * 1.4 / 72, ln, 7.6, color="#333333")
        self.y -= h + 0.15

    def finish(self, path, title):
        n = len(self.pages)
        for i, fig in enumerate(self.pages, 1):
            fig.text(ML / A4[0], 0.42 / A4[1], T(title), fontproperties=FP, size=7, color="#666666")
            fig.text((A4[0] - MR) / A4[0], 0.42 / A4[1], T(f"{i} / {n}"), fontproperties=FP, size=7,
                     color="#666666", ha="right")
        missing = sorted(c for c in USED_CHARS if ord(c) >= 32 and ord(c) not in FONTS["cmap"])
        assert not missing, f"폰트에 없는 글자: {missing}"
        meta = {"Title": "A-123 전이학습 결과 보고서", "Author": AUTHOR, "Subject": "preregistration.md 5절",
                "Creator": "src/report_a123_transfer.py", "Producer": None, "CreationDate": None}
        with PdfPages(path, metadata=meta) as pdf:
            for fig in self.pages:
                pdf.savefig(fig)
                plt.close(fig)
        return n


# ---------------------------------------------------------------- 본문
def build(out_dir_reports, out_dir_figs, out_dir_diag):
    git = git_state()  # 출력 파일을 쓰기 전 상태
    F, S, J, RH, G, D, Y = load()
    R, verdict = all_tests(F, S, J)
    RD = rho_dist(F, RH)
    GW, GK = gp_warning_tables(F, G)

    def tst(ratio, inp, s, cmp_):
        return R[(R["ratio"] == ratio) & (R["input"] == inp) & (R["source"] == s) & (R["compare"] == cmp_)].iloc[0]

    def mrmse(ratio, m, s="S1", inp="full", subset="all", k="rmse"):
        return sel(F, ratio, m, s, inp, subset)[k].mean()

    def impr(ratio, m, s="S1", inp="full"):
        t0 = mrmse(ratio, "T0")
        return (t0 - mrmse(ratio, m, s, inp)) / t0 * 100

    AW = D[(D["row_type"] == "all_wafers")].set_index(["source", "input", "subset"])
    assert (AW["ratio"] == PRIMARY).all() and AW.loc[("S1", "full", "all"), "n"] == 431
    cv_mrr = AW.loc[("S1", "full", "all"), "y_sd"] / AW.loc[("S1", "full", "all"), "y_mean"]
    pub = F[(F["model"] == "T1") & (F["subset"] == "all") & (F["source"] == "S1") & (F["input"] == "full")]
    cov = {(m, s, inp): sel(F, None, m, s, inp).groupby("ratio")["coverage90"]
           for m, s, inp in [("T1", "S1", "full"), ("T3", "S1", "full"), ("T3", "S2", "full"),
                             ("T3", "S1", "usage"), ("T3", "S2", "usage")]}
    cov_lo = {k: v.mean() for k, v in cov.items()}
    cov28 = [cov_lo[k].loc[r] for k in cov_lo for r in [2, 4, 8]]

    figs = {k: out_dir_figs / f"{OUT_STEM}_{k}.png" for k in ["fig1_rmse", "fig2_rho", "fig3_coverage"]}
    pdf_path = out_dir_reports / f"{OUT_STEM}.pdf"
    refcheck = out_dir_diag / f"{DAY}_{AUTHOR}_a123_transfer_report_refcheck.csv"
    for p in [pdf_path, pdf_path.with_suffix(".yaml"), refcheck, refcheck.with_suffix(".yaml"), *figs.values()]:
        if p.exists():
            sys.exit(f"이미 있다 (덮어쓰기 금지): {p}")

    rho_s1_8 = RD[(RD["input"] == "full") & (RD["source"] == "S1") & (RD["ratio"] == PRIMARY)]["mean"].iloc[0]
    cov16 = [cov_lo[k].loc[16] for k in cov_lo]
    cov100 = [cov_lo[k].loc[100] for k in cov_lo]
    n_train8 = pub[pub["ratio"] == PRIMARY]["n_train"].mean()
    n_zero = int(AW.loc[("S1", "full", "zero"), "n"])
    repro = [float(x) for ln in DIAG["gp_warnings"].with_suffix(".yaml").read_text(encoding="utf-8").splitlines()
             if re.match(r"^\s+\d+: \{", ln) for x in re.findall(r": ([0-9.eE+-]+)", ln.split("{", 1)[1])]
    assert len(repro) == 32
    doc = Doc()
    p1, p2 = tst(PRIMARY, "full", "S1", "T3-T1"), tst(PRIMARY, "full", "S2", "T3-T1")
    q1, q2 = tst(PRIMARY, "full", "S1", "T3-T2"), tst(PRIMARY, "full", "S2", "T3-T2")

    # ------------------------------------------------ 표지·1. 요약
    doc.text_at(ML, doc.y, "A-123 전이학습 (456 → A-123) 결과 보고서", 17, bold=True, color="#1a3a5c")
    doc.y -= 0.38
    doc.para(f"{DAY} · {AUTHOR} · preregistration.md v0.2.2 5절 기준 · 결과 PR #25 (실행 커밋 "
             f"{Y.get('commit', '')[:7]}, 2026-11-01)", 8.5, color="#444444", gap=0.02)
    doc.para(f"입력: results/{STEM}_{{folds, stats, judgment, rho}}.csv, results/diagnostics/2026-11-01_김혜연_"
             "{gp_warnings, t2_decomposition}.csv. 숫자는 전부 이 파일에서 다시 계산했다.", 8, color="#444444")
    doc.space(0.05)
    doc.h1("1. 요약")
    doc.para("질문: 라벨이 전부 공개된 456 그룹(A-456, B-456)의 소스 모델을 A-123으로 옮기면, A-123 라벨이 "
             "일부(2~16%)만 공개된 조건에서 A-123 단독 모델보다 MRR 예측이 좋아지는가?")
    doc.para(f"결론: 주 판정(공개 8%, T3 대 T1)은 S1·S2 모두 통과하지 못해 \"{verdict['주 판정']['verdict']}\"이다. "
             "456에서 학습한 f456이 A-123 MRR을 거의 설명하지 못해(ρ ≈ 0), T3는 사실상 T1과 같은 모델이 된다.",
             bold=True)
    doc.kpis([
        (f"p = {p1['p']:.3f} / {p2['p']:.3f}",
         f"주 판정 p (S1 / S2). 8%에서 T3−T1 MSE 차이 {f(p1['mean_diff'], 2)} / {f(p2['mean_diff'], 2)} "
         "(양수 = T3가 나쁨)"),
        (f"{impr(PRIMARY, 'T1'):.1f}%",
         f"공개 8%에서 T1의 T0(공개 평균) 대비 RMSE 개선 ({mrmse(PRIMARY, 'T0'):.2f} → {mrmse(PRIMARY, 'T1'):.2f})"),
        (f"ρ = {rho_s1_8:.2f}",
         "공개 8%, S1 full의 ρ 평균 (25 fold). 소스 예측에 거의 무게를 두지 않음"),
        (f"r = {AW.loc[('S1', 'full', 'all'), 'pearson']:.2f}",
         f"f456(S1 full) 예측과 A-123 MRR의 상관 (431장, 사후 분석). usage 입력은 "
         f"{AW.loc[('S1', 'usage', 'all'), 'pearson']:.2f}"),
    ])
    doc.bullets([
        f"2차 판정(T3 대 T2)은 S1·S2 모두 T3가 크게 낮아 \"{verdict['2차 판정']['verdict']}\"이다. 다만 이는 "
        "T2(f456 + 상수)가 T0보다도 크게 나쁘기 때문이며(8% RMSE: T2 S1 "
        f"{mrmse(PRIMARY, 'T2'):.2f}, S2 {mrmse(PRIMARY, 'T2', 'S2'):.2f}), 소스 모델이 A-123에 쓸모 있다는 뜻은 아니다.",
        "보조 판정(전진 검증)과 보조 실험(무작위 공개, δ XGBoost, 라벨 피처)은 실행하지 않았다(8절). "
        "따라서 결론은 주 평가(folds_v1 고정주기 공개)에만 근거한다.",
        f"A-123 MRR의 변동이 작아(변동계수 {cv_mrr:.3f}) 개선 여지 자체가 작다(10절).",
    ], size=8.8)

    # ------------------------------------------------ 2. 실험 설계
    doc.h1("2. 실험 설계")
    doc.table(
        ["항목", "설정"],
        [["소스", "S1 = A-456만 / S2 = A-456 + B-456. 둘 다 수행. 456 라벨은 전부 공개(비대칭 가정)"],
         ["타깃", "A-123 train + val, 이상치 제외 431장"],
         ["소스 모델 f456", "XGBoost (3.2 고정 설정: n_estimators 400, max_depth 4, learning_rate 0.03, "
                           "subsample 0.8, colsample_bytree 0.8, min_child_weight 3, random_state 42, n_jobs 2). "
                           "소스별 456 전체로 1회 학습"],
         ["f456 입력 2종", "full = features_v1 피처 113개 중 since_POLISH_TABLE_w를 뺀 112개(리셋 누락으로 값 오류, #24) "
                          "/ usage = 소모품 사용량 36개 ({mean, std, slope} × 사용량 6개 × s1·s2). 판정은 full, usage는 보고만"],
         ["평가 분할", "folds_v1.csv 25 fold (5-fold × 5회 반복, r = 반복 × 5 + fold). 각 fold의 A-123 학습 쪽에서만 "
                      "라벨 공개, 평가 쪽으로 채점"],
         ["공개 비율·방식", "2 / 4 / 8 / 16 / 100%. 고정주기 공개: 학습 쪽(N장)을 시간순 정렬, n = 반올림(비율 × N), "
                          "j번째 위치 = 내림((j + r/25) × N/n)"],
         ["검정", "fold별 평가 쪽 MSE의 쌍별 차이로 Nadeau–Bengio 보정 t-검정 (자유도 24, 평가/학습 비율은 실제 "
                 "fold 크기, n_train = A-123 학습 쪽 전체). 공개 라벨 수 기준 p는 참고로만 병기"]],
        [1.3, 5.2], size=7.6, caption="표 2-1. 실험 설정", align=["left", "left"])
    doc.table(
        ["기호", "모델", "구성", "역할"],
        [["T0", "기준", "공개 라벨의 평균 (상수 예측)", "하한 기준선, 개선율의 분모"],
         ["T1", "A-123 단독", "공개 라벨만으로 GP 학습 (δ와 같은 입력·커널 규칙)", "기본값. 주 판정의 비교 대상"],
         ["T2", "소스 + 오프셋", "f456(x) + c, c만 공개 라벨로 추정", "소스를 그대로 옮긴 경우. 2차 판정의 비교 대상"],
         ["T3", "전이학습", "ρ · f456(x) + b + δ(x). ρ·b는 공개 라벨 최소제곱, ρ ∈ [0, 2]로 자른 뒤 b 재추정, "
                        "잔차에 δ GP (순차)", "검정 대상"]],
        [0.5, 1.0, 3.2, 1.8], size=7.6, caption="표 2-2. 모델 T0~T3", align=["left", "left", "left", "left"])
    rows = []
    for ratio in RATIOS:
        g = pub[pub["ratio"] == ratio]
        nd = g["delta_cols"].str.count(",") + 1
        rows.append([T(f"{ratio}"), T(f"{g['n_public'].mean():.1f} ({g['n_public'].min()}~{g['n_public'].max()})"),
                     T(f"{g['n_public_clean'].mean():.1f} ({g['n_public_clean'].min()}~{g['n_public_clean'].max()})"),
                     T(f"{nd.min()}~{nd.max()}"), T(f"{int(g['iso_kernel'].sum())} / {NF}"),
                     T(f"{int(g['p1_on_included'].sum())} / {NF}")])
    doc.table(["공개 비율(%)", "fold당 공개 장수: 전체 (최소~최대)", "센서 정상 (최소~최대)", "δ 입력 개수",
               "등방성 커널 fold", "P1_on 포함 fold"], rows, [1, 2, 1.7, 1, 1.1, 1.1], size=7.6,
              caption="표 2-3. fold당 실제 공개 장수와 δ 입력 (25 fold)")
    d8 = pub[pub["ratio"] == PRIMARY]["delta_cols"].value_counts()
    doc.h2("δ 입력 규칙과 GP 커널 (5.4, 5.7)")
    doc.bullets([
        "δ 후보와 순서: 연마 시작 시각(t_start) → mean_POLISH_TABLE_s1 → mean_DRESSER_s1 → mean_MEMBRANE_s1 → "
        "mean_BACKING_FILM_s1 → mean_PSHEET_s1 → mean_DRESSER_TABLE_s1 → P1_on·allzero(한 쌍, 2개로 셈). "
        "이미 들어간 변수와 |스피어만 상관| > 0.9이면 건너뛴다(상관은 fold 학습 쪽 피처 값, MRR 미사용).",
        "개수 상한 = max(2, 내림(공개 라벨 수 / 5)). P1_on은 공개 라벨 중 센서 정상이 10장 이상일 때만, 상한이 "
        "1개 남았는데 다음이 P1_on·allzero 쌍이면 멈춘다. T1과 T3에 똑같이 적용.",
        f"공개 8%의 실제 δ 입력: {d8.index[0].replace(',', ', ')} ({d8.iloc[0]}/{NF} fold). P1_on·allzero 쌍은 어느 fold에도 "
        "들어가지 않아 P1_on 포함/제외 비교(5.7-13)는 할 수 없다.",
        "GP 커널: 공개 10장 이상은 ConstantKernel × RBF(변수별 길이척도) + WhiteKernel, 10장 미만은 RBF 길이척도 1개(등방성). "
        "진폭 0.01~100, 길이척도 0.01~100(표준화 입력), 잡음 1e−3~10, normalize_y, 재시작 5회, "
        "random_state = 42 + r. 입력 표준화와 결측 대치(중앙값)는 fold 학습 쪽 전체 웨이퍼 기준.",
    ], size=8.3)

    # ------------------------------------------------ 3. 판정 결과
    doc.h1("3. 판정 결과 (공개 8%, f456 입력 full)")

    def jrow(kind, cmp_, s, x, passed):
        return [kind, cmp_.replace("-", " − "), s, T(f"{x['rmse_a']:.3f} / {x['rmse_b']:.3f}"), f(x["mean_diff"], 3),
                f(x["t"], 2), fpv(x["p"]), fpv(x["p_revealed"]), T(f"{x['wins']}/{NF}"), "통과" if passed else "불통과"]

    hdr = ["판정", "비교", "소스", "RMSE (앞 / 뒤 모델)", "평균 MSE 차이 (MRR²)", "t", "p", "p (참고: 공개 수 기준)",
           "앞 모델이 낮은 fold", "기준 통과"]
    rows = [jrow("주 판정", "T3-T1", "S1", p1, verdict["주 판정"]["passed"]["S1"]),
            jrow("주 판정", "T3-T1", "S2", p2, verdict["주 판정"]["passed"]["S2"]),
            jrow("2차 판정", "T3-T2", "S1", q1, verdict["2차 판정"]["passed"]["S1"]),
            jrow("2차 판정", "T3-T2", "S2", q2, verdict["2차 판정"]["passed"]["S2"])]
    doc.table(hdr, rows, [0.8, 0.75, 0.5, 1.1, 1.0, 0.6, 0.75, 0.95, 0.8, 0.7], size=7.4,
              caption="표 3-1. 주 판정과 2차 판정. 차이 = 앞 모델 MSE − 뒤 모델 MSE (단위 MRR², 25 fold 평균). 음수 = 앞 모델이 낮음",
              note=f"기준(5.5, 5.7-16): 평균 MSE 차이 < 0이고 양측 p < 0.05를 S1·S2가 모두 만족할 때만 \"효과 있음\". "
                   f"p는 n_train = A-123 학습 쪽 전체(fold당 평균 {n_train8:.1f}장)로 계산한 판정용 값. 공개 수 기준 p는 판정에 쓰지 않는다.",
              align=["left", "left", "left", "right", "right", "right", "right", "right", "right", "center"])
    doc.table(["판정", "S1", "S2", "최종 판정"],
              [["주 판정 (T3 대 T1)", "불통과" if not verdict["주 판정"]["passed"]["S1"] else "통과",
                "불통과" if not verdict["주 판정"]["passed"]["S2"] else "통과", verdict["주 판정"]["verdict"]],
               ["2차 판정 (T3 대 T2)", "통과" if verdict["2차 판정"]["passed"]["S1"] else "불통과",
                "통과" if verdict["2차 판정"]["passed"]["S2"] else "불통과", verdict["2차 판정"]["verdict"]],
               ["보조 판정 (전진 검증 5단계, 8%)", "미실행", "미실행", "미실행 (8절)"]],
              [2.2, 0.8, 0.8, 2.7], size=7.8, caption="표 3-2. 최종 판정",
              align=["left", "center", "center", "left"], bold_rows=(0,))
    rows = []
    for ratio in RATIOS:
        r_ = [T(f"{ratio}")]
        for cmp_ in ["T3-T1", "T2-T1"]:
            for s in SOURCES:
                x = tst(ratio, "full", s, cmp_)
                r_.append(T(f"{f(x['mean_diff'], 2)} (p {fpv(x['p'])})"))
        rows.append(r_)
    doc.table(["공개 비율(%)", "T3 − T1, S1", "T3 − T1, S2", "T2 − T1, S1", "T2 − T1, S2"], rows, [0.9, 1.5, 1.5, 1.5, 1.5],
              size=7.6, caption="표 3-3. 참고: 비율별 평균 MSE 차이 (MRR², f456 full). 8% 외에는 경향만 보고(5.5)",
              note="T2 − T1은 2차 판정과 별개로 보고만 한다. 100%에서 S1의 T3 − T1은 p < 0.05지만 양수(T3가 나쁨)다.",
              bold_rows=(2,))

    # ------------------------------------------------ 4. 공개 비율별 결과
    doc.h1("4. 공개 비율별 결과")
    doc.figure(lambda fg, rc: fig_rmse(fg, rc, F), 3.0,
               "그림 1. 공개 비율 대 평가 쪽 RMSE (f456 입력 full). 점 = 25 fold 평균, 오차 막대 = 표준오차 "
               "(표준편차/√25). T0·T1은 소스와 무관해 두 칸이 같다. 가로축은 비율 순서(간격 균등).",
               png=figs["fig1_rmse"])
    rows = []
    for ratio in RATIOS:
        for m in MODELS:
            r_ = [T(f"{ratio}") if m == "T0" else "", m]
            for s in SOURCES:
                g = sel(F, ratio, m, s, "full")
                r_ += [pm(g["rmse"]), pm(g["mae"])]
            rows.append(r_)
    doc.table(["공개 비율(%)", "모델", "S1 RMSE", "S1 MAE", "S2 RMSE", "S2 MAE"], rows, [0.9, 0.5, 1.3, 1.3, 1.3, 1.3],
              size=7.3, caption="표 4-1. 비율별 RMSE·MAE (평균 ± 표준편차, 25 fold, f456 full, 전체 웨이퍼)",
              align=["left", "left", "right", "right", "right", "right"])
    rows = []
    for ratio in RATIOS:
        rows.append([T(f"{ratio}"), f(mrmse(ratio, "T0"), 3), f(mrmse(ratio, "T1"), 3), T(f"{impr(ratio, 'T1'):.1f}%".replace("-", "−")),
                     T(f"{impr(ratio, 'T3', 'S1'):.1f}%".replace("-", "−")), T(f"{impr(ratio, 'T3', 'S2'):.1f}%".replace("-", "−")),
                     T(f"{impr(ratio, 'T3', 'S1', 'usage'):.1f}%".replace("-", "−")), T(f"{impr(ratio, 'T3', 'S2', 'usage'):.1f}%".replace("-", "−"))])
    doc.table(["공개 비율(%)", "T0 RMSE", "T1 RMSE", "T1 개선율", "T3 S1 full", "T3 S2 full", "T3 S1 usage", "T3 S2 usage"],
              rows, [0.9, 0.8, 0.8, 0.8, 0.85, 0.85, 0.9, 0.9], size=7.5,
              caption="표 4-2. T0 대비 RMSE 개선율 = (T0 RMSE − 모델 RMSE) / T0 RMSE × 100 (각 RMSE는 25 fold 평균)",
              note="음수 = T0보다 나쁨. 4%에서는 T1도 T0보다 나쁘다. 4%는 공개 14장으로, 변수별 길이척도 커널이 처음 쓰이는 비율이다(2%는 등방성, 표 2-3).")
    rows = []
    for ratio in RATIOS:
        r_ = [T(f"{ratio}")]
        for s in SOURCES:
            for m in ["T2", "T3"]:
                r_.append(pm(sel(F, ratio, m, s, "usage")["rmse"]))
        rows.append(r_)
    doc.table(["공개 비율(%)", "T2 S1", "T3 S1", "T2 S2", "T3 S2"], rows, [0.9, 1.3, 1.3, 1.3, 1.3], size=7.5,
              caption="표 4-3. 참고: f456 입력 usage의 RMSE (평균 ± 표준편차, 보고만). T0·T1은 표 4-1과 같다")

    # ------------------------------------------------ 5. 왜 전이가 안 되는가
    doc.h1("5. 왜 전이가 안 되는가")
    rho8 = RD.set_index(["input", "source", "ratio"])
    doc.para("T3 = ρ · f456 + b + δ에서 ρ가 0에 가까우면 소스 항이 사라지고 T3는 b + δ, 즉 T1과 같은 구조가 된다. "
             f"f456 full의 ρ 평균은 S1 {rho8.loc[('full', 'S1'), 'mean'].min():.2f}~"
             f"{rho8.loc[('full', 'S1'), 'mean'].max():.2f}, S2 {rho8.loc[('full', 'S2'), 'mean'].min():.2f}~"
             f"{rho8.loc[('full', 'S2'), 'mean'].max():.2f}이고, 공개 8%에서 ρ = 0에 붙은 fold가 S1 "
             f"{int(rho8.loc[('full', 'S1', 8), 'n0'])}개, S2 {int(rho8.loc[('full', 'S2', 8), 'n0'])}개다. "
             "T3가 T1보다 약간 나쁜 것은 ρ·b를 추정하느라 공개 라벨의 정보를 일부 쓰기 때문으로 보인다.")
    doc.figure(lambda fg, rc: fig_rho(fg, rc, RD), 2.7,
               "그림 2. T3의 ρ 추정값 분포 (25 fold). 점 = 평균, 세로선 = 최소~최대. ρ는 0~2로 제한(2에 붙은 fold는 없음).",
               png=figs["fig2_rho"])
    rows = []
    for ratio in RATIOS:
        r_ = [T(f"{ratio}")]
        for inp in INPUTS:
            for s in SOURCES:
                x = rho8.loc[(inp, s, ratio)]
                r_.append(T(f"{x['mean']:.3f} [{x['min']:.3f}, {x['max']:.3f}] · {int(x['n0'])}"))
        rows.append(r_)
    doc.table(["공개 비율(%)", "full S1", "full S2", "usage S1", "usage S2"], rows, [0.9, 1.6, 1.6, 1.6, 1.6], size=7.4,
              caption="표 5-1. ρ 평균 [최소, 최대] · ρ = 0에 붙은 fold 수 (25 fold)")
    rows = []
    for s in SOURCES:
        for inp in INPUTS:
            r_ = [s, inp]
            for sub in ["all", "clean", "zero"]:
                x = AW.loc[(s, inp, sub)]
                r_.append(T(f"{f(x['pearson'], 3)} ({int(x['n'])}장)"))
            r_.append(f(AW.loc[(s, inp, "all"), "spearman"], 3))
            r_.append(T(f"{AW.loc[(s, inp, 'clean'), 'ideal_offset']:.1f} / {AW.loc[(s, inp, 'zero'), 'ideal_offset']:.1f}"))
            rows.append(r_)
    doc.table(["소스", "f456 입력", "피어슨 r: 전체", "센서 정상", "센서 전체-0", "스피어만: 전체",
               "필요 오프셋 (정상 / 전체-0)"], rows, [0.5, 0.7, 1.15, 1.1, 1.1, 0.95, 1.4], size=7.4,
              caption="표 5-2. f456 예측과 A-123 MRR의 상관 [사후 분석, 판정에 쓰지 않음]",
              note="출처: results/diagnostics/2026-11-01_김혜연_t2_decomposition.csv의 all_wafers 행(A-123 431장, f456는 fold와 "
                   "무관). 필요 오프셋 = 부분집합별 평균(MRR − f456). 이 표는 판정(PR #25) 이후 원인 확인용으로 만든 것이다.",
              align=["left", "left", "right", "right", "right", "right", "right"])
    a_full, a_use = AW.loc[("S1", "full")], AW.loc[("S1", "usage")]
    doc.bullets([
        f"f456 full과 A-123 MRR의 상관은 전체 {a_full.loc['all', 'pearson']:.2f}, 센서 정상 "
        f"{a_full.loc['clean', 'pearson']:.2f}로 낮다. 사용량만 쓴 usage는 {a_use.loc['all', 'pearson']:.2f}로 높지만, "
        f"공개 8%에서 T3 − T1은 S1 {f(tst(8, 'usage', 'S1', 'T3-T1')['mean_diff'], 2)} (p "
        f"{fpv(tst(8, 'usage', 'S1', 'T3-T1')['p'])}), S2 {f(tst(8, 'usage', 'S2', 'T3-T1')['mean_diff'], 2)} (p "
        f"{fpv(tst(8, 'usage', 'S2', 'T3-T1')['p'])})로 차이가 없다. 사용량 정보는 T1의 δ 입력"
        "(t_start, 사용량 s1 평균)에 이미 들어 있기 때문으로 보인다.",
        f"T2가 크게 나쁜 이유: 센서 정상과 전체-0 웨이퍼에서 필요한 오프셋이 다르다(S1 full "
        f"{a_full.loc['clean', 'ideal_offset']:.1f} 대 {a_full.loc['zero', 'ideal_offset']:.1f}). f456은 456에 없는 "
        "전체-0 패턴의 입력에서 다른 값을 내므로 상수 c 하나로는 맞출 수 없다. T3는 ρ를 0 가까이 줄여 이 문제를 피한다.",
        f"S1 대 S2 (관찰만, 5.0 해석 제한): S2 full은 상관이 {AW.loc[('S2', 'full', 'all'), 'pearson']:.2f}로 S1보다 낮고, "
        f"8% ρ 평균 {rho8.loc[('full', 'S2', 8), 'mean']:.2f}(S1 {rho8.loc[('full', 'S1', 8), 'mean']:.2f}), T2 RMSE "
        f"{mrmse(8, 'T2', 'S2'):.2f}(S1 {mrmse(8, 'T2'):.2f})다. 같은 스테이지(A)만 쓴 소스가 A-123과 조금 더 비슷하지만, "
        "카운터가 장비 전체에서 공유되므로(5.0) \"같은 부품 공유\"의 근거로 쓰지 않는다. T3 기준으로는 두 소스 모두 T1과 차이가 없다.",
    ], size=8.3)

    # ------------------------------------------------ 6. 불확실성
    doc.h1("6. 불확실성: 90% 예측구간 포함률")
    doc.figure(lambda fg, rc: fig_coverage(fg, rc, F), 2.9,
               "그림 3. T1·T3의 90% 예측구간 포함률 대 공개 비율 (전체 웨이퍼). 점 = 25 fold 평균, 오차 막대 = 표준오차. "
               "구간은 WhiteKernel 잡음을 포함한 관측 예측구간이며 f456·ρ·b의 불확실성은 무시한다(5.7-10).",
               png=figs["fig3_coverage"], png_w=5.5)
    rows = []
    for ratio in RATIOS:
        rows.append([T(f"{ratio}")] + [pm(cov[k].get_group(ratio), 3) for k in cov])
    doc.table(["공개 비율(%)", "T1", "T3 S1 full", "T3 S2 full", "T3 S1 usage", "T3 S2 usage"], rows,
              [0.9, 1.2, 1.2, 1.2, 1.2, 1.2], size=7.5,
              caption="표 6-1. 90% 예측구간 포함률 (평균 ± 표준편차, 25 fold, 목표 0.90)",
              note=f"공개 2~8%에서 포함률은 {min(cov28):.3f}~{max(cov28):.3f}로 목표 0.90보다 낮다(구간이 너무 좁음). "
                   f"16%에서 {min(cov16):.3f}~{max(cov16):.3f}, 100%에서 {min(cov100):.3f}~{max(cov100):.3f}가 된다. "
                   "공개 라벨이 적을 때 GP 잡음이 하한(1e−3) 가까이 추정되는 "
                   "fold가 많은 것(9절)과 같은 방향이다.")

    # ------------------------------------------------ 7. 웨이퍼 유형별 오차
    doc.h1("7. 웨이퍼 유형별 오차 (공개 8%, f456 full)")
    rows = []
    for sub in ["all", "clean", "zero"]:
        r_ = [SUBSET_KO[sub], T(f"{sel(F, 8, 'T1', subset=sub)['n'].mean():.1f}")]
        for m, s in [("T0", "S1"), ("T1", "S1"), ("T2", "S1"), ("T3", "S1"), ("T2", "S2"), ("T3", "S2")]:
            r_.append(pm(sel(F, 8, m, s, "full", sub)["rmse"]))
        rows.append(r_)
    doc.table(["웨이퍼", "fold당 평가 장수", "T0", "T1", "T2 S1", "T3 S1", "T2 S2", "T3 S2"], rows,
              [0.8, 0.8, 1.05, 1.05, 1.05, 1.05, 1.1, 1.05], size=7.4,
              caption="표 7-1. 웨이퍼 유형별 RMSE (평균 ± 표준편차, 25 fold)",
              note="센서 정상 = 전체-0 센서가 없는 웨이퍼, 전체-0 = 압력·유량·회전 센서 중 하나 이상이 웨이퍼 전체에서 0. "
                   "분해는 보고만 한다(5.7-16, 검정 대상은 전체 웨이퍼).")
    rows = []
    for ratio in RATIOS:
        r_ = [T(f"{ratio}")]
        for sub in ["clean", "zero"]:
            for m, s in [("T1", "S1"), ("T3", "S1"), ("T3", "S2")]:
                r_.append(f(mrmse(ratio, m, s, "full", sub), 3))
        rows.append(r_)
    doc.table(["공개 비율(%)", "정상: T1", "정상: T3 S1", "정상: T3 S2", "전체-0: T1", "전체-0: T3 S1", "전체-0: T3 S2"],
              rows, [0.9, 1, 1, 1, 1, 1, 1], size=7.5, caption="표 7-2. 비율별 웨이퍼 유형 RMSE (25 fold 평균, f456 full)",
              bold_rows=(2,))
    doc.para(f"공개 8%에서 T1 RMSE는 센서 정상 {mrmse(8, 'T1', subset='clean'):.2f}, 전체-0 "
             f"{mrmse(8, 'T1', subset='zero'):.2f}다. T3는 두 유형 모두 T1보다 약간 높아, 어느 유형에서도 전이 이득이 없다.",
             size=8.5)

    # ------------------------------------------------ 8. 보조 실험
    doc.h1("8. 보조 판정과 보조 실험")
    doc.table(["항목", "사전 등록", "상태"],
              [["보조 판정: 시간순 전진 검증 5단계 (각 단계 8% 공개, T3가 T1보다 낮은 단계 수)", "5.2, 5.5", "미실행"],
               ["무작위 공개 (fold마다 시드 42 + r)", "5.3", "미실행"],
               ["δ를 XGBoost로 바꾼 T1·T3", "5.4 보조", "미실행"],
               ["라벨 피처 보조 실험 (T1·T3, 2~100%, 경향만)", "5.6", "미실행"]],
              [4.2, 0.9, 0.8], size=7.8, caption="표 8-1. 보조 판정·보조 실험 실행 여부",
              align=["left", "left", "center"])
    doc.bullets([
        "결과 파일이 없어 수치를 보고하지 않는다. 5.7-17의 일정상 전진 검증·무작위 공개는 11/1 이후 2차 면담 전, "
        "δ XGBoost·라벨 피처 실험은 2차 면담 이후에 한다.",
        "5.2는 주 평가와 보조 평가를 결과표에 나란히 보고하도록 정했다. 보조 평가가 없으므로, 주 판정 결과가 "
        "\"이웃 시점 정보 덕분\"인지 확인하지 못한 상태다. 다만 주 판정이 이미 \"효과 없음\"이라 기본값(T1)은 바뀌지 않는다.",
    ], size=8.3)

    # ------------------------------------------------ 9. 실행 신뢰도
    doc.h1("9. 실행 신뢰도")
    rows = []
    for ratio in RATIOS:
        x = GW.loc[ratio]
        rows.append([T(f"{ratio}"), T(f"{int(x['T1'])}"), T(f"{int(x['T1_folds'])} / {NF}"), T(f"{int(x['T3'])}"),
                     T(f"{int(x['T3_units'])} / {NF * 4}")])
    rows.append(["합계", T(f"{int(GW['T1'].sum())}"), "", T(f"{int(GW['T3'].sum())}"), ""])
    doc.table(["공개 비율(%)", "T1 경고 건수", "T1 경고 fold", "T3 경고 건수", "T3 경고 칸 (fold × 소스 × 입력)"], rows,
              [0.9, 1, 1, 1, 1.6], size=7.5,
              caption="표 9-1. GP 최적화 경고 건수 (주 결과 csv의 gp_warn_t1·gp_warn_t3)",
              note="T1은 fold당 1회 학습(비율당 25회), T3는 fold × 소스 2 × 입력 2마다 1회(비율당 100회). "
                   "경고가 나도 결과는 그대로 썼다(5.7-15).", bold_rows=(len(rows) - 1,))
    rows = [[k] + [T(f"{int(v)}") for v in GK.loc[k]] + [T(f"{int(GK.loc[k].sum())}"),
                                                          T(f"{GK.loc[k].sum() / GK.values.sum() * 100:.1f}%")]
            for k in GK.index]
    rows.append(["합계"] + [T(f"{int(v)}") for v in GK.sum()] + [T(f"{int(GK.values.sum())}"), "100%"])
    doc.table(["경고 종류 (하이퍼파라미터가 붙은 경계)", "2%", "4%", "8%", "16%", "합계", "비율"], rows,
              [2.2, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7], size=7.5,
              caption="표 9-2. GP 경고 종류 (공개 2~16%, 경고 메시지 기준)",
              note="출처: results/diagnostics/2026-11-01_김혜연_gp_warnings.csv (같은 설정으로 다시 학습해 메시지를 기록, "
                   f"주 결과와 지표 최대 차이 {max(repro):.0e}, 건수 일치). 모두 sklearn ConvergenceWarning(값이 탐색 경계 근처)이며 "
                   "최적화 실패(lbfgs 비수렴)는 없다. 공개 100%의 경고(T1 "
                   f"{int(GW.loc[100, 'T1'])}건, T3 {int(GW.loc[100, 'T3'])}건)는 종류를 기록하지 않았다. 길이척도 상한(100)은 "
                   "그 δ 입력이 사실상 쓰이지 않는다는 뜻이고, 잡음 하한은 공개 라벨이 적을 때 GP가 라벨을 그대로 통과하려 한다는 뜻이다.",
              bold_rows=(len(rows) - 1,))
    env = {k: Y.get(k, "?") for k in ["python", "numpy", "pandas", "scipy", "scikit-learn", "xgboost"]}
    doc.table(["항목", "값"],
              [["시드", "f456 random_state = 42 / GP random_state = 42 + r (r = 반복 × 5 + fold), 재시작 5회도 이 시드에서 시작. "
                       "고정주기 공개는 시드 없음(5.3 공식)"],
               ["패키지", " · ".join(f"{k} {v}" for k, v in env.items()) + " (xgboost 버전이 다르면 코드가 멈춤, 5.7-11)"],
               ["f456 스레드", "n_jobs = 2 고정 (subsample < 1이면 스레드 수에 따라 결과가 달라짐, #23)"],
               ["실행", f"커밋 {Y.get('commit', '?')[:7]} (main), {Y.get('run_started', '?')} 시작, {Y.get('runtime_sec', '?')}초. "
                       f"CPU {Y.get('cpu', '?')}, {Y.get('os', '?')}"],
               ["실행 명령", "python src/a123_transfer.py → 주 결과 4종 / python src/diagnostics/gp_warnings_a123.py → 경고 종류 / "
                           "python src/diagnostics/t2_decomposition_a123.py → 상관·오프셋 / python src/report_a123_transfer.py → 이 보고서"],
               ["입력 해시 (sha256 앞 16자리)", " · ".join(f"{k.removesuffix('.csv')} {Y.get(k, '?')}" for k in
                                           ["features_v1.csv", "folds_v1.csv", "reveal_masks_v1.csv", "wafer_table.csv"])
                + " (주 결과 yaml)"]],
              [1.4, 5.1], size=7.4, caption="표 9-3. 시드·환경·실행 명령", align=["left", "left"])

    # ------------------------------------------------ 10. 한계
    doc.h1("10. 한계")
    doc.bullets([
        f"A-123 MRR의 변동이 작다: 431장 평균 {AW.loc[('S1', 'full', 'all'), 'y_mean']:.1f}, 표준편차 "
        f"{AW.loc[('S1', 'full', 'all'), 'y_sd']:.2f}, 변동계수 {cv_mrr:.3f}. 가장 좋은 T1(100% 공개)도 RMSE "
        f"{mrmse(100, 'T1'):.2f}로 T0 대비 {impr(100, 'T1'):.1f}% 개선에 그치며, 8%에서는 RMSE 차이가 "
        f"{mrmse(8, 'T0') - mrmse(8, 'T1'):.2f}다. "
        "작은 차이를 가려내기 어려운 조건이다.",
        "데이터가 PHM 2016 한 개이고 A-123은 한 장비의 한 챔버군이다. 다른 장비·레시피로 일반화할 수 없다.",
        "압력·회전·유량의 단위와 공정 레시피가 공개되지 않아, f456이 A-123에서 맞지 않는 이유(레시피 차이, 센서 결측 패턴)를 "
        f"물리적으로 확인할 수 없다. A-123 431장 중 {n_zero}장이 센서 전체-0 웨이퍼다.",
        "90% 예측구간은 δ GP의 잡음만 반영하고 f456·ρ·b의 추정 불확실성을 무시한다(5.7-10). 공개 2~16%의 포함률이 목표보다 낮다.",
        "GP 경고가 많다(공개 2~16%에서 680건, 전체 비율 합계 "
        f"{int(GW['T1'].sum() + GW['T3'].sum())}건). 대부분 길이척도가 상한에 붙은 경우로, 그 입력이 쓰이지 않는다는 뜻이라 "
        "결과를 무효로 만들지는 않지만 하이퍼파라미터 추정이 불안정함을 보여 준다.",
        "보조 판정(전진 검증)과 보조 실험이 미실행이라 결론은 주 평가에만 근거한다(8절). 5절의 상관·오프셋 분석은 사후 분석이다.",
    ], size=8.5)

    title = "A-123 전이학습 결과 보고서 · " + DAY + " · " + AUTHOR
    n_pages = doc.finish(pdf_path, title)

    # ------------------------------------------------ 참고값 대조표
    rho_s1 = RD.query("input == 'full' and source == 'S1'")
    use = RD.query("input == 'usage'")
    computed = {
        "주 판정 p, S1": f"{p1['p']:.3f}", "주 판정 p, S2": f"{p2['p']:.3f}",
        "2차 판정 T3−T2, S1": f"{q1['mean_diff']:.1f} (MSE 차이, MRR²)",
        "2차 판정 T3−T2, S2": f"{q2['mean_diff']:.1f} (MSE 차이, MRR²)",
        "T1의 T0 대비 RMSE 개선, 8%": f"{impr(8, 'T1'):.1f}%", "T1의 T0 대비 RMSE 개선, 16%": f"{impr(16, 'T1'):.1f}%",
        "T1의 T0 대비 RMSE 개선, 100%": f"{impr(100, 'T1'):.1f}%",
        "ρ, S1 full, 비율 전체 범위": f"비율별 평균 {rho_s1['mean'].min():.3f}~{rho_s1['mean'].max():.3f}, "
                                  f"fold 값 {rho_s1['min'].min():.3f}~{rho_s1['max'].max():.3f}",
        "ρ, S1 full, 8% 평균": f"{rho_s1.set_index('ratio').loc[8, 'mean']:.3f}",
        "ρ, usage, 범위": f"비율별 평균 S1 {use[use.source == 'S1']['mean'].min():.3f}~{use[use.source == 'S1']['mean'].max():.3f}, "
                         f"S2 {use[use.source == 'S2']['mean'].min():.3f}~{use[use.source == 'S2']['mean'].max():.3f}",
        "f456–MRR 상관, full 전체": f"{a_full.loc['all', 'pearson']:.3f} (S1, 431장)",
        "f456–MRR 상관, full 센서 정상": f"{a_full.loc['clean', 'pearson']:.3f} (S1, 166장)",
        "f456–MRR 상관, usage 전체": f"{a_use.loc['all', 'pearson']:.3f} (S1, 431장)",
        "90% 구간 포함률, 공개 2~8%": f"{min(cov28):.3f}~{max(cov28):.3f} (T1·T3 5계열 평균)",
        "T1 RMSE, 센서 정상 (8%)": f"{mrmse(8, 'T1', subset='clean'):.3f}",
        "T1 RMSE, 전체-0 (8%)": f"{mrmse(8, 'T1', subset='zero'):.3f}",
        "GP 경고 합계": f"{int(GK.values.sum())} (공개 2~16%만, 100% 포함 시 {int(GW['T1'].sum() + GW['T3'].sum())})",
        "GP 경고, 길이척도 상한": f"{int(GK.loc['길이척도 상한'].sum())}",
        "GP 경고, 잡음 하한": f"{int(GK.loc['잡음 하한'].sum())}",
        "A-123 MRR 변동계수": f"{cv_mrr:.4f}",
    }
    RC = pd.DataFrame([{"item": k, "reference": v, "recomputed": computed[k]} for k, v in REFERENCE])
    RC.to_csv(refcheck, index=False, encoding="utf-8")
    inputs = [*IN.values(), IN_YAML, *DIAG.values()]
    common = {"author": AUTHOR, "git": git,
              "inputs_sha256_16": {p.name: sha(p) for p in inputs},
              "font": f"{FONTS['source'].name} → wght 400·700 고정 인스턴스 (fontTools), PDF Type 42 내장",
              "command": "python src/report_a123_transfer.py",
              "matplotlib": matplotlib.__version__}
    write_yaml(refcheck.with_suffix(".yaml"), {
        "this_file": refcheck.name, "experiment": "a123_transfer 보고서 참고값 대조 (14.4)",
        "note": "reference = 작성 요청에 적힌 대조용 값, recomputed = 결과 파일에서 다시 계산한 값. 보고서는 recomputed를 쓴다",
        **common})
    write_yaml(pdf_path.with_suffix(".yaml"), {
        "this_file": pdf_path.name, "experiment": "a123_transfer 결과 PDF 보고서 (preregistration 5절)",
        "pages": n_pages, "figures": [p.name for p in figs.values()], "refcheck": refcheck.name,
        "not_run": ["전진 검증 보조 판정", "무작위 공개", "δ XGBoost", "라벨 피처 보조 실험(5.6)"],
        "checks": "T0·T1 소스 무관 / t-검정·판정 = stats·judgment csv (1e-9) / ρ = rho csv / GP 경고 건수 진단 = 주 결과 / "
                  "모든 글자 폰트 포함",
        **common})
    return pdf_path, n_pages, RC, figs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="재현 확인용. 지정하면 reports·figures·diagnostics 대신 이 폴더 하나에 저장")
    args = ap.parse_args(argv)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        dirs = (args.out_dir,) * 3
    else:
        dirs = (RESULTS / "reports", RESULTS / "figures", RESULTS / "diagnostics")
    pdf, n, RC, figs = build(*dirs)
    print(f"저장: {pdf} ({n}쪽)")
    for p in figs.values():
        print(f"      {p}")
    print("\n참고값 대조")
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(RC.to_string(index=False))


if __name__ == "__main__":
    main()

"""mrr_time_check(PR #28) 결과 PDF 보고서 (사후 진단, CLAUDE.md 14.4).

입력 (모두 커밋된 파일, 읽기만):
  results/diagnostics/2026-11-03_김혜연_mrr_time_check.csv (+ .yaml)  ← 표의 숫자는 전부 여기서 다시 읽는다
  results/reports/2026-11-04_김혜연_mrr_time_check.md                  ← 숫자 대조용 (다르면 고치지 않고 출력)
  results/figures/2026-11-04_김혜연_mrr_time_{scatter_ch4, ch56_groups, cv_ratio}.png  ← 그대로 싣는다
웨이퍼 단위 데이터, 라벨 파일, test, data/locked/는 읽지 않는다.

그림 글씨 크기: 산점도(3칸)와 분포(2칸) PNG는 쪽 너비에 그대로 넣으면 글씨가 작아져서, PNG의 흰 세로띠를 찾아
칸별로 잘라 2열 또는 1열로 배치한다. 잘라 쓸 뿐 원본 PNG는 바꾸지 않는다.

배치·폰트는 src/report_a123_transfer.py(PR #29)와 같다: Noto Sans KR 가변 폰트에서 fontTools로 Regular·Bold 고정
인스턴스를 임시 폴더에 만들어 PDF에 Type 42로 내장하고, 모든 글자가 폰트에 있는지 확인한다. 이 브랜치에는
그 파일이 없어서 해당 부분(git_state, 폰트, 숫자 형식, Doc)을 옮겨 왔다.

실행: python src/diagnostics/report_mrr_time_check.py
      python src/diagnostics/report_mrr_time_check.py --out-dir <폴더>   (재현 확인용)
출력: results/reports/2026-11-04_김혜연_mrr_time_check_report.pdf (+ .yaml). 이미 있으면 멈춘다 (덮어쓰기 금지).
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = "김혜연"
DAY = "2026-11-04"
REPORT_TITLE = "MRR 대 연마 시간: 종점검출 점검"
IN_CSV = ROOT / "results" / "diagnostics" / "2026-11-03_김혜연_mrr_time_check.csv"
IN_YAML = IN_CSV.with_suffix(".yaml")
IN_MD = ROOT / "results" / "reports" / "2026-11-04_김혜연_mrr_time_check.md"
FIGS = {k: ROOT / "results" / "figures" / f"2026-11-04_김혜연_mrr_time_{k}.png"
        for k in ["scatter_ch4", "ch56_groups", "cv_ratio"]}
OUT_NAME = f"{DAY}_{AUTHOR}_mrr_time_check_report.pdf"
GROUPS = ["A-123", "A-456", "B-456"]
TIME_KO = {"dur_s1": "첫 챔버 체류", "n_press": "압력 > 0 행 수", "dur_w": "웨이퍼 전체 길이"}
CH = {"A-123": "챔버 1", "A-456": "챔버 4", "B-456": "챔버 4"}

for p in [IN_CSV, IN_MD, *FIGS.values()]:
    assert "locked" not in str(p).lower(), p


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


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
            return [self.wrap(c, w - 2 * pad - 0.05, size, bold) for c, w in zip(row, widths)]  # 여유 0.05인치

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
        meta = {"Title": REPORT_TITLE, "Author": AUTHOR, "Subject": "사후 진단 분석 (PR #28), 사전 확정 문서의 판정에는 쓰지 않음",
                "Creator": "src/diagnostics/report_mrr_time_check.py", "Producer": None, "CreationDate": None}
        with PdfPages(path, metadata=meta) as pdf:
            for fig in self.pages:
                pdf.savefig(fig)
                plt.close(fig)
        return n




# ---------------------------------------------------------------- 결과 읽기
def stat(R, group, subset, tdef, trim, name, col="value"):
    x = R[(R["group"] == group) & (R["subset"] == subset) & (R["time_def"] == tdef)
          & (R["trim"] == trim) & (R["stat"] == name)]
    assert len(x) == 1, (group, subset, tdef, trim, name, len(x))
    return float(x.iloc[0][col])


def nrow(R, group, subset, tdef, trim):
    x = R[(R["group"] == group) & (R["subset"] == subset) & (R["time_def"] == tdef) & (R["trim"] == trim)
          & (R["stat"] == "b")]
    assert len(x) == 1
    return int(x.iloc[0]["n"]), int(x.iloc[0]["n_zero_time"])


def mean_mrr(R, group, subset):
    x = R[(R["group"] == group) & (R["subset"] == subset) & (R["section"] == "cv") & (R["trim"] == "none")
          & R["time_def"].isna() & (R["stat"] == "mean_mrr")]
    assert len(x) == 1
    return float(x.iloc[0]["value"]), int(x.iloc[0]["n"])


def m2(x, nd=2):
    return T(f"{x:.{nd}f}".replace("-", "−"))


def bci(R, g, sub, td, trim, name="b", nd=2):
    return T(f"{m2(stat(R, g, sub, td, trim, name), nd)} [{m2(stat(R, g, sub, td, trim, name, 'ci_lo'), nd)}, "
             f"{m2(stat(R, g, sub, td, trim, name, 'ci_hi'), nd)}]")


def md_check(R, md):
    """md 요약에 적힌 숫자 문자열이 csv에서 다시 만든 값과 같은지. 다르면 목록으로 돌려준다(고치지 않음)."""
    items = []
    for g in ["A-456", "B-456", "A-123"]:
        items += [
            (f"{g} b [CI] (첫 챔버, 절삭)", bci(R, g, "전체", "dur_s1", "p01_p99")),
            (f"{g} R² (첫 챔버, 절삭)", f"{stat(R, g, '전체', 'dur_s1', 'p01_p99', 'r2'):.3f}"),
            (f"{g} 통제 후 b [CI]", bci(R, g, "전체", "dur_s1", "p01_p99", "b_ctrl")),
            (f"{g} CV 비율 (첫 챔버, 절삭)", f"{stat(R, g, '전체', 'dur_s1', 'p01_p99', 'cv_ratio_removal_to_mrr'):.2f}"),
            (f"{g} 절삭 전 b", m2(stat(R, g, "전체", "dur_s1", "none", "b"))),
            (f"{g} 스피어만", m2(stat(R, g, "전체", "dur_s1", "p01_p99", "spearman"))),
            (f"{g} CV 비율 (압력 행)", f"{stat(R, g, '전체', 'n_press', 'p01_p99', 'cv_ratio_removal_to_mrr'):.2f}"),
            (f"{g} CV 비율 (전체 길이)", f"{stat(R, g, '전체', 'dur_w', 'p01_p99', 'cv_ratio_removal_to_mrr'):.2f}"),
        ]
    for g in ["A-456", "B-456"]:
        for sub in ["ch56_step=0 (짧은 무리)", "ch56_step=1 (긴 무리)"]:
            items.append((f"{g} {sub} 평균 MRR", f"{mean_mrr(R, g, sub)[0]:.2f}"))
            items.append((f"{g} {sub} 첫 챔버 b", m2(stat(R, g, sub, "dur_s1", "p01_p99", "b"))))
        items.append((f"{g} 압력 행 b [CI]", bci(R, g, "전체", "n_press", "p01_p99", nd=3)))
    for sub in ["센서 정상", "전체-0"]:
        items.append((f"A-123 {sub} b [CI]", bci(R, "A-123", sub, "dur_s1", "p01_p99")))
        items.append((f"A-123 {sub} 통제 후 b [CI]", bci(R, "A-123", sub, "dur_s1", "p01_p99", "b_ctrl")))
    return [(k, v, v in md) for k, v in items]


# ---------------------------------------------------------------- 그림 칸 나누기
def split_panels(img, n, top_frac=0.08):
    """흰 세로띠(모든 행이 흰색인 열 구간) 중 가장 넓은 n-1개의 가운데에서 자른다. 위쪽 top_frac(전체 제목)은 뺀다."""
    top = int(img.shape[0] * top_frac)
    body = img[top:, :, :3]
    ink = (body < 0.96).any(axis=2).any(axis=0)
    runs, start = [], None
    for x, v in enumerate(ink):
        if not v and start is None:
            start = x
        if v and start is not None:
            runs.append((start, x))
            start = None
    W = img.shape[1]
    inner = [(a, b) for a, b in runs if a > W * 0.1 and b < W * 0.9]
    cuts = sorted((a + b) // 2 for a, b in sorted(inner, key=lambda r: r[0] - r[1])[:n - 1])
    assert len(cuts) == n - 1, (n, inner)
    edges = [0, *cuts, W]
    return [body[:, edges[i]:edges[i + 1]] for i in range(n)]


def place_images(doc, imgs, width, caption, cols=2, dpi=160):
    """잘라 낸 칸들을 cols열로 배치. width는 한 칸의 인치 너비."""
    gap = 0.15
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    for row in rows:
        h = max(im.shape[0] / im.shape[1] * width for im in row)
        doc.ensure(h + 0.1)
        x0 = ML + (TW - (len(row) * width + (len(row) - 1) * gap)) / 2
        for k, im in enumerate(row):
            hh = im.shape[0] / im.shape[1] * width
            ax = doc.fig.add_axes([(x0 + k * (width + gap)) / A4[0], (doc.y - hh) / A4[1], width / A4[0], hh / A4[1]])
            ax.imshow(im, interpolation="lanczos")
            ax.set_axis_off()
        doc.y -= h + 0.06
    doc.para(caption, 7.8, color="#333333", gap=0.12, lead=1.45)


def git_out(*a):
    r = subprocess.run(["git", "-c", "core.quotepath=false", *a], cwd=ROOT, capture_output=True,
                       encoding="utf-8", errors="replace")
    return r.stdout.strip()


# ---------------------------------------------------------------- 본문
def build(out_dir):
    git = git_state()
    R = pd.read_csv(IN_CSV)
    md = IN_MD.read_text(encoding="utf-8")
    pdf_path = out_dir / OUT_NAME
    for p in [pdf_path, pdf_path.with_suffix(".yaml")]:
        if p.exists():
            sys.exit(f"이미 있다 (덮어쓰기 금지): {p}")
    MC = md_check(R, md)
    run_commit = git_out("log", "--diff-filter=A", "--format=%h", "--", str(IN_CSV.relative_to(ROOT)).replace("\\", "/"))
    s = {g: {k: stat(R, g, "전체", "dur_s1", "p01_p99", k) for k in
             ["b", "r2", "b_ctrl", "cv_ratio_removal_to_mrr", "spearman", "cv_mrr", "cv_time", "median_time",
              "r2_ctrl_only", "r2_ctrl_full"]} for g in GROUPS}
    mA0, nA0 = mean_mrr(R, "A-456", "ch56_step=0 (짧은 무리)")
    mA1, nA1 = mean_mrr(R, "A-456", "ch56_step=1 (긴 무리)")
    pA0 = stat(R, "A-456", "ch56_step=0 (짧은 무리)", "n_press", "none", "median_time")
    pA1 = stat(R, "A-456", "ch56_step=1 (긴 무리)", "n_press", "none", "median_time")

    doc = Doc()
    # ------------------------------------------------ 머리
    doc.fig.add_artist(Rectangle((ML / A4[0], (doc.y - 0.34) / A4[1]), TW / A4[0], 0.34 / A4[1],
                                 transform=doc.fig.transFigure, facecolor="#fff4e5", edgecolor="#e0a050", lw=0.8))
    doc.text_at(ML + 0.12, doc.y - 0.09, "사후 진단 분석, 사전 확정 문서의 판정에는 쓰지 않음", 10, bold=True, color="#8a4b00")
    doc.y -= 0.5
    doc.text_at(ML, doc.y, REPORT_TITLE, 17, bold=True, color="#1a3a5c")
    doc.y -= 0.38
    doc.para(f"{DAY} · {AUTHOR} · PR #28 (diag/mrr-time) · 결과 실행 커밋 {run_commit} (2026-11-03) · "
             "SR 결과를 본 뒤 설계한 탐색 분석. 1~99% 절삭과 통제 회귀는 첫 실행 결과를 본 뒤 추가했다.",
             8.3, color="#444444", gap=0.02)
    doc.para(f"숫자는 모두 {IN_CSV.relative_to(ROOT).as_posix()}에서 다시 읽었다. 대상: train+val, 이상치 4장 제외 2,401장 "
             "(A-456 983, B-456 987, A-123 431). test 센서·라벨과 data/locked/는 쓰지 않았다.", 8, color="#444444")

    doc.h1("1. 요약")
    doc.para("질문: 라벨 MRR은 고정 시간으로 나눈 값인가, 아니면 장비가 목표 두께에서 멈추는 종점검출 때문에 연마 시간이 "
             "웨이퍼마다 바뀌는가? 그래서 R2R 시뮬레이션에서 \"제거량 = MRR × 연마 시간\"을 쓸 수 있는가?")
    doc.para("종합 판정: 중간. 456에서는 연마 시간이 MRR과 관련된 증거가 있지만 종점검출로 보기에는 기울기와 변동 비율이 맞지 않고, "
             "A-123에서는 그런 관련이 없다.", bold=True)
    doc.kpis([
        (f"b = {m2(s['B-456']['b'])}", f"B-456: MRR 대 챔버 4 체류시간 로그 기울기, R² {s['B-456']['r2']:.3f} "
                                       f"(A-456은 {m2(s['A-456']['b'])}, R² {s['A-456']['r2']:.3f})"),
        (f"{mA0:.2f} / {mA1:.2f}", f"A-456 평균 MRR: 챔버 5·6 시간이 짧은 무리 / 긴 무리 (압력 행 중앙값 {pA0:.0f} 대 {pA1:.0f})"),
        (f"b = {m2(s['A-123']['b_ctrl'])}", f"A-123: 통제 후 기울기, 0을 포함 (통제 전 {m2(s['A-123']['b'])})"),
        (f"{s['B-456']['cv_ratio_removal_to_mrr']:.2f}", "B-456: CV(연마량)/CV(MRR). 종점검출이면 1보다 훨씬 작아야 함 "
                                                       f"(A-456 {s['A-456']['cv_ratio_removal_to_mrr']:.2f}, A-123 "
                                                       f"{s['A-123']['cv_ratio_removal_to_mrr']:.2f})"),
    ])
    doc.bullets([
        f"A-123 (챔버 1 기준): 통제 후 시간 효과 없음, 종점검출 패턴 아님. 통제 후 b = {bci(R, 'A-123', '전체', 'dur_s1', 'p01_p99', 'b_ctrl')}.",
        f"A-456·B-456 (챔버 4 기준): 챔버 4 체류시간과 MRR이 강한 음의 관계(b = {m2(s['A-456']['b'])} / {m2(s['B-456']['b'])})지만, "
        "−1보다 가파르고 연마량 변동이 충분히 작지 않다. 챔버 5·6 시간이 달라도 MRR이 같다. 챔버 4 시간만 MRR과 관련된다.",
        "R2R: 456은 \"연마 시간을 바꿔도 MRR은 그대로\" 가정과 충돌할 가능성이 있고 A-123은 상대적으로 낫다. 팀 논의가 필요하다(6절).",
    ], size=8.6)

    doc.h1("2. 왜 확인하나")
    doc.bullets([
        "R2R 시뮬레이션은 \"제거량 = MRR × 연마 시간\"으로 두고, 연마 시간을 조정해 목표 제거량을 맞춘다. 이때 \"연마 시간을 "
        "바꿔도 MRR은 그대로\"라고 가정한다.",
        "장비가 종점검출로 연마 시간을 정한다면, 연마 시간은 MRR에 반응한 결과다(MRR이 낮은 웨이퍼는 오래, 높은 웨이퍼는 짧게 "
        "연마). 그러면 기록된 시간과 MRR 사이에 음의 관계가 생기고, 시간을 자유롭게 바꿀 수 있는 입력으로 볼 수 없어 위 가정이 흔들린다.",
        "제거량(두께) 자료가 데이터에 없으므로, MRR과 연마 시간의 관계만으로 어느 쪽에 가까운지 간접적으로 본다.",
    ], size=8.6)

    doc.h1("3. 방법")
    doc.table(["항목", "내용"], [
        ["시간 정의 T", "① 첫 챔버 체류 dur_POLISH_s1 (456 = 챔버 4, A-123 = 챔버 1, 초) ② 압력 6개 중 하나라도 > 0인 행 수 "
                       "(웨이퍼 전체, 챔버 5·6 포함) ③ 웨이퍼 전체 길이 t_end − t_start (초)"],
        ["기울기 b", "log(MRR) = a + b·log(T)의 OLS 기울기와 95% CI, R². 연마량 = MRR × T가 일정하면 b = −1"],
        ["통제 후 b", "같은 회귀에 t_start와 소모품 카운터 시작값 6개(z-score)를 더한 뒤의 b (사후 추가). 시간 흐름·소모품 상태와 "
                     "겹치는 몫을 뺀다"],
        ["절삭", "부분집단·시간 정의마다 T의 1~99 백분위수 밖 웨이퍼를 뺀다(사후 추가). 456의 긴 꼬리(B-456 최대 5,283초)가 OLS를 "
                "좌우해서 넣었다. 본문 표는 절삭 결과이고 절삭 전 b를 함께 적는다"],
        ["CV 비율", "CV(연마량 = MRR × T) / CV(MRR). CV = 표본 표준편차 / 평균"],
        ["부분집단", "그룹 3개. 456은 ch56_step 0 / 1(챔버 5·6 시간이 짧은 / 긴 무리), A-123은 센서 정상 / 전체-0"],
    ], [1.1, 5.4], size=7.6, caption="표 3-1. 비교 방법", align=["left", "left"])
    doc.table(["가설", "기울기 b", "CV(연마량)/CV(MRR)", "뜻"], [
        ["고정 시간으로 나눈 값 (시간과 무관)", "≈ 0", "≈ 1 이상", "시간이 MRR과 상관없이 정해진다"],
        ["종점검출 (목표 두께에서 멈춤)", "≈ −1", "≪ 1 (연마량 변동이 작음)", "시간이 MRR에 반응해 정해진다"],
    ], [2.0, 0.8, 1.6, 2.1], size=7.8, caption="표 3-2. 판단 기준", align=["left", "center", "center", "left"])

    doc.h1("4. 결과")
    rows = []
    for g in GROUPS:
        n, _ = nrow(R, g, "전체", "dur_s1", "p01_p99")
        rows.append([g, CH[g], T(f"{n}"), T(f"{s[g]['median_time']:.1f}"), m2(s[g]["spearman"]),
                     bci(R, g, "전체", "dur_s1", "p01_p99"), T(f"{s[g]['r2']:.3f}"),
                     bci(R, g, "전체", "dur_s1", "p01_p99", "b_ctrl"), T(f"{s[g]['cv_ratio_removal_to_mrr']:.2f}"),
                     m2(stat(R, g, "전체", "dur_s1", "none", "b"))])
    doc.table(["그룹", "기준 챔버", "n", "T 중앙값(초)", "스피어만", "b [95% CI]", "R²", "통제 후 b [95% CI]", "CV 비율",
               "절삭 전 b"], rows, [0.55, 0.65, 0.4, 0.75, 0.6, 1.25, 0.5, 1.25, 0.55, 0.6], size=7.3,
              caption="표 4-1. 그룹별 결과 (T = 첫 챔버 체류시간, 1~99% 절삭). A-123은 챔버 1, 456은 챔버 4 기준",
              note=f"통제 변수만의 R²: A-123 {s['A-123']['r2_ctrl_only']:.2f}, A-456 {s['A-456']['r2_ctrl_only']:.2f}, "
                   f"B-456 {s['B-456']['r2_ctrl_only']:.2f} → 시간을 더하면 {s['A-123']['r2_ctrl_full']:.2f}, "
                   f"{s['A-456']['r2_ctrl_full']:.2f}, {s['B-456']['r2_ctrl_full']:.2f}. CI는 웨이퍼 독립을 가정한 고전 OLS 값이다.",
              align=["left", "left", "right", "right", "right", "right", "right", "right", "right", "right"])
    rows = []
    for g in GROUPS:
        r_ = [g]
        for td in ["n_press", "dur_w"]:
            n, nz = nrow(R, g, "전체", td, "p01_p99")
            r_ += [bci(R, g, "전체", td, "p01_p99", nd=3), T(f"{stat(R, g, '전체', td, 'p01_p99', 'r2'):.3f}"),
                   T(f"{stat(R, g, '전체', td, 'p01_p99', 'cv_ratio_removal_to_mrr'):.2f}"), T(f"{n}")]
        rows.append(r_)
    doc.table(["그룹", "압력 행: b [95% CI]", "R²", "CV 비율", "n", "전체 길이: b [95% CI]", "R²", "CV 비율", "n"], rows,
              [0.55, 1.35, 0.45, 0.55, 0.4, 1.35, 0.45, 0.55, 0.4], size=7.3,
              caption="표 4-2. 다른 시간 정의 (1~99% 절삭)",
              note=f"A-123은 압력 > 0 행이 웨이퍼 전체에서 0인 웨이퍼 {nrow(R, 'A-123', '전체', 'n_press', 'none')[1]}장이 "
                   "압력 행 정의에서 빠졌다(센서 전체-0 웨이퍼).")
    rows = []
    for g in ["A-456", "B-456"]:
        for sub, lab in [("ch56_step=0 (짧은 무리)", "짧은 무리"), ("ch56_step=1 (긴 무리)", "긴 무리")]:
            mu, n = mean_mrr(R, g, sub)
            rows.append([g, lab, T(f"{n}"), T(f"{stat(R, g, sub, 'n_press', 'none', 'median_time'):.0f}"), T(f"{mu:.2f}"),
                         m2(stat(R, g, sub, "dur_s1", "p01_p99", "b")),
                         T(f"{stat(R, g, sub, 'dur_s1', 'p01_p99', 'cv_ratio_removal_to_mrr'):.2f}")])
    doc.table(["그룹", "챔버 5·6", "n", "압력 > 0 행 중앙값", "평균 MRR", "첫 챔버 b (절삭)", "CV 비율 (첫 챔버)"], rows,
              [0.6, 0.8, 0.5, 1.2, 0.9, 1.2, 1.2], size=7.5, caption="표 4-3. 챔버 5·6 시간이 다른 두 무리 (456)")
    rows = []
    for sub in ["전체", "센서 정상", "전체-0"]:
        n, _ = nrow(R, "A-123", sub, "dur_s1", "p01_p99")
        rows.append([sub, T(f"{n}"), bci(R, "A-123", sub, "dur_s1", "p01_p99"),
                     T(f"{stat(R, 'A-123', sub, 'dur_s1', 'p01_p99', 'r2'):.3f}"),
                     bci(R, "A-123", sub, "dur_s1", "p01_p99", "b_ctrl"),
                     T(f"{stat(R, 'A-123', sub, 'dur_s1', 'p01_p99', 'cv_ratio_removal_to_mrr'):.2f}")])
    doc.table(["A-123 부분집단", "n", "통제 전 b [95% CI]", "R²", "통제 후 b [95% CI]", "CV 비율"], rows,
              [1.1, 0.5, 1.5, 0.6, 1.5, 0.8], size=7.5, caption="표 4-4. A-123 부분집단 (챔버 1 체류시간, 1~99% 절삭)")

    sc = split_panels(mpimg.imread(FIGS["scatter_ch4"]), 3)
    place_images(doc, sc, 3.25, "그림 1. MRR 대 첫 챔버 체류시간(로그-로그; 위 왼쪽 A-456, 위 오른쪽 B-456, 아래 A-123). 실선은 "
                 "절삭 후 OLS 기울기 b, 점선은 종점검출일 때의 기울기 −1이다. 456은 음의 기울기가 뚜렷하지만 −1보다 가파르고, "
                 "A-123은 기울기가 완만하다. (원본: " + FIGS["scatter_ch4"].name + ")")
    ch = split_panels(mpimg.imread(FIGS["ch56_groups"]), 2, top_frac=0.1)
    place_images(doc, ch, 3.25, "그림 2. 챔버 5·6 시간이 다른 두 무리의 MRR 분포(왼쪽 A-456, 오른쪽 B-456 참고, 세로선 = 평균). "
                 "압력 시간이 약 2배 차이 나도 평균 MRR이 거의 같다. (원본: " + FIGS["ch56_groups"].name + ")")
    cvimg = mpimg.imread(FIGS["cv_ratio"])
    place_images(doc, [cvimg[:, :, :3]], 6.2, "그림 3. 그룹 × 시간 정의별 CV(연마량)/CV(MRR), 1~99% 절삭, 로그 축. "
                 "1 아래면 연마량이 MRR보다 고르다. 456의 첫 챔버 정의만 1보다 조금 작고, 나머지는 1 이상이다. (원본: "
                 + FIGS["cv_ratio"].name + ")", cols=1)

    doc.h1("5. 판정과 근거: 왜 '중간'인가")
    doc.h2("A-123: 통제 후 시간 효과 없음, 종점검출 패턴 아님")
    doc.bullets([
        f"통제 전 b = {bci(R, 'A-123', '전체', 'dur_s1', 'p01_p99')}로 0은 아니지만, 시작 시각·소모품 상태를 통제하면 "
        f"{bci(R, 'A-123', '전체', 'dur_s1', 'p01_p99', 'b_ctrl')}로 0을 포함하고 R²도 늘지 않는다"
        f"({s['A-123']['r2_ctrl_only']:.2f} → {s['A-123']['r2_ctrl_full']:.2f}).",
        f"CV(연마량)/CV(MRR) = {s['A-123']['cv_ratio_removal_to_mrr']:.2f}로, 연마량이 MRR보다 더 흩어진다. 목표 두께에서 멈추는 패턴이 없다.",
    ], size=8.5)
    doc.h2("456: 방향은 종점검출과 맞지만 그대로 성립하지는 않음")
    doc.bullets([
        f"챔버 4 체류시간과 MRR이 강한 음의 관계다: B-456 b = {m2(s['B-456']['b'])}, R² {s['B-456']['r2']:.3f} / "
        f"A-456 b = {m2(s['A-456']['b'])}, R² {s['A-456']['r2']:.3f}.",
        "하지만 b가 −1보다 가파르다. 그러면 시간이 길수록 연마량(MRR × T)이 오히려 줄어든다. CV 비율도 "
        f"{s['A-456']['cv_ratio_removal_to_mrr']:.2f} / {s['B-456']['cv_ratio_removal_to_mrr']:.2f}로, 종점검출에서 기대하는 "
        "\"훨씬 작음\"이 아니다.",
        f"통제 후 b는 {m2(s['A-456']['b_ctrl'])} / {m2(s['B-456']['b_ctrl'])}로 줄어든다. 기울기의 상당 부분이 시간 흐름·소모품 "
        f"상태와 겹친다. 챔버 4 시간의 CV가 {s['A-456']['cv_time']:.3f} / {s['B-456']['cv_time']:.3f}로 작아, 작은 시간 차이에 큰 "
        "MRR 차이가 대응한다.",
        f"챔버 5·6 시간이 달라도 MRR이 같다(A-456 평균 {mA0:.2f} 대 {mA1:.2f}). 웨이퍼 전체 압력 시간의 b는 "
        f"{bci(R, 'A-456', '전체', 'n_press', 'p01_p99', nd=3)}로 0을 포함한다. 챔버 4 시간만 MRR과 관련된다.",
    ], size=8.5)
    doc.para("종합: 456에서는 시간과 MRR의 관련이 있으나 종점검출 기준(b ≈ −1, 연마량 일정)에 맞지 않고, A-123에서는 관련이 없다. "
             "\"고정 시간으로 나눈 값\"과 \"종점검출\" 어느 쪽으로도 이 데이터만으로 확정할 수 없어 '중간'으로 판정한다.", size=8.6,
             bold=True)

    doc.h1("6. R2R에 주는 의미 (팀 논의 필요)")
    doc.h2("456 (A-456, B-456)")
    doc.bullets([
        "챔버 4 체류시간과 MRR이 강한 음의 관계라, 연마 시간이 MRR에 반응해 정해졌을 가능성이 있다.",
        "그렇다면 \"연마 시간을 자유롭게 바꿔도 MRR은 그대로\"라는 시뮬레이션 가정과 충돌할 수 있다. 시간을 바꿨을 때 MRR이 "
        "어떻게 반응할지 이 관측 자료로는 알 수 없다.",
        "456에서 챔버 4 시간이 MRR에 반응해 정해졌다면 조정 변수로 쓰기 어렵다. 이 해석이 맞는지 팀 논의가 필요하다. "
        "가정을 쓴다면 \"시간–MRR 독립\"을 명시하고 b를 바꿔 보는 민감도 분석을 함께 두는 방안을 논의할 수 있다.",
    ], size=8.5)
    doc.h2("A-123")
    doc.bullets([
        "통제 후 시간–MRR 관계가 없고 종점검출 패턴도 없어, 위 가정을 쓰기에 456보다 상대적으로 낫다.",
        f"다만 가정이 맞다는 증거가 아니라 반대 증거가 없다는 뜻이다. A-123은 MRR 변동이 작아(CV "
        f"{s['A-123']['cv_mrr']:.3f}) 관계를 찾아낼 힘이 약하다.",
    ], size=8.5)
    doc.para("열린 질문: R2R 시뮬레이션의 대상 그룹을 무엇으로 할지(456, A-123, 둘 다)는 아직 정하지 않았다. 이 결과는 그 선택에 "
             "참고할 근거 중 하나이며, 팀 논의로 정한다.", size=8.6, bold=True)

    doc.h1("7. 한계")
    doc.bullets([
        "레이블 MRR의 정의가 공개되지 않았다. MRR이 챔버 4 시간 기준으로 정의됐을 가능성(예: 챔버 4 제거량 ÷ 챔버 4 시간)과 "
        "\"챔버 5·6이 제거에 기여하지 않는다\"를 이 데이터로 구분할 수 없다. 표 4-3의 결과는 둘 다와 맞는다.",
        f"A-123은 압력 > 0 행이 전부 0인 웨이퍼({nrow(R, 'A-123', '전체', 'n_press', 'none')[1]}장, 센서 전체-0)가 압력 행 "
        "정의의 계산에서 빠졌다.",
        "압력·회전 단위와 공정 레시피가 공개되지 않았다. 456 두 무리(ch56_step)가 왜 생기는지 모른다.",
        "제거량(두께) 자료가 없어 종점검출 여부를 직접 확인할 수 없다. 연마량은 MRR × T로 추정했다.",
        "절삭과 통제 회귀는 첫 실행 결과를 본 뒤 추가했다. 절삭 전에는 456의 b가 0에 가깝다"
        f"({m2(stat(R, 'A-456', '전체', 'dur_s1', 'none', 'b'))} / {m2(stat(R, 'B-456', '전체', 'dur_s1', 'none', 'b'))}). "
        "고전 OLS CI는 웨이퍼 독립을 가정해 실제보다 좁을 수 있다. 데이터는 PHM 2016 한 개, 장비 하나다.",
    ], size=8.5)

    doc.h1("8. 재현 정보")
    ymd = IN_YAML.read_text(encoding="utf-8")
    env = (f"python {platform.python_version()} · numpy {np.__version__} · pandas {pd.__version__} · "
           f"matplotlib {matplotlib.__version__}")
    doc.table(["항목", "값"], [
        ["결과 실행", f"src/diagnostics/mrr_time_check.py, 커밋 {run_commit} (2026-11-03). 2026-11-04에 다시 실행해 같은 csv가 "
                     "나오는 것을 확인(줄바꿈 문자만 다름). 난수 사용 없음(seed 42 기록)"],
        ["입력 해시 (sha256 앞 16자리)", T(" · ".join(f"{p.name} {sha(p)}" for p in [IN_CSV, IN_YAML, IN_MD, *FIGS.values()]))],
        ["결과 입력 파일", T(re.search(r"^input: (.*)$", ymd, re.M).group(1))],
        ["패키지", T(f"결과 yaml에 패키지 버전 기록이 없다. 이 보고서 생성 환경: {env}")],
        ["실행 명령", "python src/diagnostics/mrr_time_check.py (결과 csv) / python src/diagnostics/plot_mrr_time_check.py (그림 3개) "
                     "/ python src/diagnostics/report_mrr_time_check.py (이 PDF). 모두 출력이 있으면 멈춘다"],
        ["이전 버전과의 관계", "10/14 src/hyp_mrr_time.py(결과 2026-10-14_김혜연_mrr_time_beta.csv)는 MRR이 실제 연마 시간으로 나눈 값인지 "
                            "고정 시간으로 나눈 값인지(β)를 센서 정상 웨이퍼만으로 본 앞선 확인이다. 이번 점검은 질문을 종점검출로 바꾸고 "
                            "A-123 전체-0 웨이퍼, 챔버 5·6 무리, 절삭, 통제 회귀를 더한 후속이다. 10/14 파일은 그대로 둔다"],
        ["요약 문서", IN_MD.relative_to(ROOT).as_posix()],
    ], [1.5, 5.0], size=7.4, caption="표 8-1. 재현 정보", align=["left", "left"])

    n_pages = doc.finish(pdf_path, f"{REPORT_TITLE} · {DAY} · {AUTHOR} · 사후 진단")
    pdf_path.with_suffix(".yaml").write_text("\n".join([
        f"this_file: {pdf_path.name}",
        "purpose: mrr_time_check(PR #28) 결과 PDF. 사후 진단 분석, 사전 확정 문서의 판정에는 쓰지 않음",
        f"author: {AUTHOR}",
        f"pages: {n_pages}",
        f"git_commit: {git['commit']}",
        f"git_branch: {git['branch']}",
        f"git_uncommitted_files: {git['uncommitted_files']}",
        "inputs_sha256_16:",
        *[f"  {p.name}: {sha(p)}" for p in [IN_CSV, IN_YAML, IN_MD, *FIGS.values()]],
        f"environment: \"{env}\"",
        f"font: {FONTS['source'].name} → wght 400·700 고정 인스턴스 (fontTools), PDF Type 42 내장",
        "command: python src/diagnostics/report_mrr_time_check.py",
        f"md_check: md 요약 숫자 {sum(ok for *_, ok in MC)}/{len(MC)}개가 csv 재계산값과 같음",
        "",
    ]), encoding="utf-8")
    return pdf_path, n_pages, MC


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None, help="재현 확인용 출력 폴더")
    args = ap.parse_args(argv)
    out = args.out_dir or ROOT / "results" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    pdf, n, MC = build(out)
    print(f"저장: {pdf} ({n}쪽)")
    bad = [(k, v) for k, v, ok in MC if not ok]
    print(f"md 대조: {len(MC) - len(bad)}/{len(MC)}개 일치" + ("" if not bad else " / 다른 항목:"))
    for k, v in bad:
        print(f"  {k}: csv 재계산 '{v}'가 md에 없음")


if __name__ == "__main__":
    main()

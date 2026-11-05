"""공동 SR 바깥 5-fold 검증 포스터 그림 (계획 docs/plans/2026-11-05_joint_sr_outer5_plan.md 7절).

입력: results/2026-11-05_전준혁_FV1_sr_joint_outer5*.csv (sr_joint_outer5.py evaluate 결과), 2026-11-04 선택·계수 결과.
출력: results/reports/poster_joint_sr/fig{1..4}_*.{png,pdf}, summary_panel.{png,pdf}. PNG 300 dpi. 그림 글자는 영어.
그림의 Fold 1~5 = folds_v1 반복 0의 fold 0~4.

실행: python src/sr_joint_figs_outer5.py [fig1 fig2 fig3 fig4 summary]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_joint as sj  # noqa: E402

OUT = Path(os.environ.get("SR_FIG_OUT", sj.ROOT / "results" / "reports" / "poster_joint_sr"))   # 초안은 다른 폴더로
BASE = Path(os.environ.get("SR_FIG_BASE", sj.RESULTS / "2026-11-05_전준혁_FV1_sr_joint_outer5"))   # 초안 점검용 대체 입력

JOINT, LR = "#2a78d6", "#eb6834"            # 공동 SR / 선형회귀 (2026-11-04 그림과 같은 색)
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
TRAIN_FILL, HELD_FILL = "#dbe6f5", "#374151"
TABLES = {"A-456": "Table A-456", "B-456": "Table B-456"}

LABEL = {
    "auc_PCHAMBER_s1": "Chamber pressure ∫dt",
    "auc_MAIN_OUTER_s1": "Main air-bag pressure ∫dt",
    "auc_SLURRY_C_s1": "Slurry flow C ∫dt",
    "auc_RETAINER_s1": "Retainer pressure ∫dt",
    "P1_on": "Retainer pressure (polishing mean)",
    "frac_hi": "High-retainer-pressure fraction",
    "dur_POLISH_s1": "Polishing time",
    "std_POLISH_TABLE_s1": "Pad-table usage (SD)",
    "std_MEMBRANE_s1": "Membrane usage (SD)",
    "std_PSHEET_s1": "Pressurized-sheet usage (SD)",
    "mean_DRESSER_TABLE_s2": "Dresser-table usage (mean, s2)",
    "mean_WAFER_ROT_s1": "Wafer rotation (mean)",
    "since_MEMBRANE_w": "Since membrane change",
    "since_BACKING_FILM_w": "Since backing-film change",
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / f"{name}.{ext}"
        assert not p.exists(), f"이미 있음 (덮어쓰지 않음): {p}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("저장:", name)


def load():
    met = pd.read_csv(f"{BASE}.csv", encoding="utf-8-sig")
    oof = pd.read_csv(f"{BASE}_oof.csv", encoding="utf-8-sig")
    summ = pd.read_csv(f"{BASE}_summary.csv", encoding="utf-8-sig")
    struct = pd.read_csv(f"{BASE}_structure.csv", encoding="utf-8-sig", dtype={"fold": str})
    return met, oof, summ, struct


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig = plt.figure(figsize=(7.2, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    ax.text(0.12, 3.12, r"Model:  MRR$_g$ = a$_g$ · f(x) + b$_g$", fontsize=11, color=INK, va="center", weight="bold")
    ax.text(3.05, 3.12, "f is shared by both tables;  a$_g$, b$_g$ are fitted per table (A-456, B-456)",
            fontsize=8.5, color=MUTED, va="center")

    # 왼쪽: 5개 fold
    x0, w, h, gap = 0.62, 2.35, 0.30, 0.13
    ax.text(x0 + w / 2, 2.70, "Train+Val wafers split into 5 folds", ha="center", fontsize=9, color=INK, weight="bold")
    for i in range(5):
        y = 2.25 - i * (h + gap)
        ax.text(x0 - 0.08, y + h / 2, f"Fold {i + 1}", ha="right", va="center", fontsize=8.5, color=INK)
        for j in range(5):
            held = j == i
            ax.add_patch(Rectangle((x0 + j * w / 5, y), w / 5 - 0.025, h, facecolor=HELD_FILL if held else TRAIN_FILL,
                                   edgecolor="none"))
            if held:
                ax.text(x0 + j * w / 5 + w / 10 - 0.0125, y + h / 2, "20%", ha="center", va="center", fontsize=7.5,
                        color="white", weight="bold")
    yb = 2.25 - 4 * (h + gap) - 0.30
    ax.add_patch(Rectangle((x0, yb), 0.16, 0.13, facecolor=TRAIN_FILL, edgecolor="none"))
    ax.text(x0 + 0.21, yb + 0.065, "80% search + fit", va="center", fontsize=7.5, color=INK)
    ax.add_patch(Rectangle((x0 + 1.32, yb), 0.16, 0.13, facecolor=HELD_FILL, edgecolor="none"))
    ax.text(x0 + 1.53, yb + 0.065, "20% held out", va="center", fontsize=7.5, color=INK)

    # 오른쪽: 한 fold 안의 단계
    steps = [
        ("Joint SR search on the 80%", "PySR template, 14 candidate variables, 10 seeds × 500 iterations"),
        ("Select f by inner 4-fold CV", "1-standard-error rule on the 80% only"),
        ("Refit f constants and a$_g$, b$_g$", "on the 80%"),
        ("Predict MRR of the held-out 20%", "held-out labels used only for scoring"),
        ("Compare with linear regression", "per table, same variables as the selected f"),
    ]
    bx, bw, bh, bgap = 3.55, 3.50, 0.37, 0.11
    ax.text(bx + bw / 2, 2.70, "In each fold (equation re-discovered from scratch)", ha="center", fontsize=9, color=INK,
            weight="bold")
    for i, (t1, t2) in enumerate(steps):
        y = 2.20 - i * (bh + bgap)
        ax.add_patch(FancyBboxPatch((bx, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor="#f8fafc", edgecolor=JOINT if i < 4 else LR, linewidth=1.0))
        ax.text(bx + 0.13, y + bh / 2, str(i + 1), ha="center", va="center", fontsize=9, color=INK, weight="bold")
        ax.text(bx + 0.28, y + bh * 0.67, t1, va="center", fontsize=8.5, color=INK, weight="bold")
        ax.text(bx + 0.28, y + bh * 0.28, t2, va="center", fontsize=7.3, color=MUTED)
        if i < 4:
            ax.annotate("", xy=(bx + bw / 2, y - bgap - 0.012), xytext=(bx + bw / 2, y + 0.012),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=9))
    # 연결 화살표와 반복 표시
    ax.annotate("", xy=(bx - 0.06, 1.55), xytext=(x0 + w + 0.08, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=10))
    ax.text((x0 + w + bx) / 2, 1.66, "× 5", ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(0.12, 0.10, "Each wafer is predicted exactly once (out-of-fold).  A-456: 940 wafers,  B-456: 951 wafers "
            "(Train+Val only; no test labels).", fontsize=7.8, color=MUTED, va="center")
    save(fig, "fig1_validation_flow")


# ---------------------------------------------------------------- Fig 2
def fig2():
    met, oof, summ, _ = load()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.38})
    models = [("joint_sr", "Joint SR", JOINT), ("lr_group_samevars", "Linear regression (same variables)", LR)]
    bw = 0.36
    ymax = 0
    for gi_, g in enumerate(TABLES):
        for mi, (m, lab, col) in enumerate(models):
            s = summ[(summ["group"] == g) & (summ["model"] == m)].iloc[0]
            x = gi_ + (mi - 0.5) * (bw + 0.04)
            a1.bar(x, s["rmse_oof"], width=bw, color=col, label=lab if gi_ == 0 else None, zorder=2)
            f = met[(met["group"] == g) & (met["model"] == m)]["rmse"].to_numpy()
            jit = np.linspace(-0.09, 0.09, len(f))
            a1.scatter(x + jit, f, s=13, facecolor="white", edgecolor=INK, linewidth=0.7, zorder=3,
                       label="Each fold" if (gi_ == 0 and mi == 0) else None)
            a1.text(x, s["rmse_oof"] / 2, f"{s['rmse_oof']:.2f}", ha="center", va="center", fontsize=8.5, color="white",
                    weight="bold", zorder=4)
            ymax = max(ymax, f.max(), s["rmse_oof"])
    a1.set_xticks(range(len(TABLES)), list(TABLES.values()))
    a1.set_ylim(0, np.ceil(ymax + 1.2))
    a1.set_ylabel("RMSE of MRR (lower is better)")
    a1.set_title("Out-of-fold RMSE (5 folds pooled)", fontsize=9.5, color=INK, loc="left", weight="bold")
    a1.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    a1.set_axisbelow(True)
    h, lb = a1.get_legend_handles_labels()
    order = [lb.index(x) for x in ("Joint SR", "Linear regression (same variables)", "Each fold")]
    a1.legend([h[i] for i in order], [lb[i] for i in order], loc="upper left", fontsize=7.5, handlelength=1.2,
              borderaxespad=0.1)
    a1.tick_params(axis="x", length=0)

    # 오른쪽: fold별 차이
    lim = 0
    for gi_, g in enumerate(TABLES):
        j = met[(met["group"] == g) & (met["model"] == "joint_sr")].sort_values("fold")
        l_ = met[(met["group"] == g) & (met["model"] == "lr_group_samevars")].sort_values("fold")
        diff = j["rmse"].to_numpy() - l_["rmse"].to_numpy()
        lim = max(lim, np.abs(diff).max())
        xs = gi_ + np.linspace(-0.2, 0.2, len(diff))
        a2.scatter(xs, diff, s=26, color=[JOINT if v < 0 else LR for v in diff], zorder=3, edgecolor="white", linewidth=0.6)
        for x, v, k in zip(xs, diff, j["fold"]):
            a2.text(x, v + (0.06 if v >= 0 else -0.06) * max(lim, 1), f"{int(k) + 1}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.5, color=MUTED)
        s = summ[(summ["group"] == g) & (summ["model"] == "diff_joint_minus_lr")].iloc[0]
        a2.hlines(diff.mean(), gi_ - 0.3, gi_ + 0.3, color=INK, lw=1.2, zorder=2)
        a2.text(gi_, -0.03, f"Joint SR better\nin {int(s['joint_better_folds'])}/5 folds", transform=a2.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.3, color=INK)
    lim = max(0.5, np.ceil(lim * 1.35 * 4) / 4)
    a2.set_ylim(-lim, lim)
    a2.axhline(0, color=INK, lw=0.9, zorder=1)
    a2.axhspan(-lim, 0, color=JOINT, alpha=0.06, zorder=0)
    a2.axhspan(0, lim, color=LR, alpha=0.06, zorder=0)
    a2.text(1.02, 0.25, "Joint SR\nbetter", transform=a2.transAxes, fontsize=7.3, color=JOINT, va="center", weight="bold")
    a2.text(1.02, 0.75, "Linear reg.\nbetter", transform=a2.transAxes, fontsize=7.3, color=LR, va="center", weight="bold")
    a2.set_xticks(range(len(TABLES)), list(TABLES.values()))
    a2.tick_params(axis="x", length=0, pad=26)
    a2.set_xlim(-0.6, 1.6)
    a2.set_ylabel("RMSE difference (Joint SR − LR)")
    a2.set_title("Per-fold difference", fontsize=9.5, color=INK, loc="left", weight="bold")
    a2.text(0.03, 0.97, "dot = one fold (number = fold), line = mean", transform=a2.transAxes, fontsize=6.8,
            color=MUTED, va="top", ha="left")
    n = {g: int(summ[(summ["group"] == g) & (summ["model"] == "joint_sr")]["n_oof"].iloc[0]) for g in TABLES}
    fig.text(0.0, -0.06, f"Held-out wafers: A-456 n = {n['A-456']}, B-456 n = {n['B-456']} (each predicted once). "
             "MRR units as in the PHM 2016 data.", fontsize=7, color=MUTED)
    save(fig, "fig2_joint_vs_lr")


# ---------------------------------------------------------------- Fig 3
def fig3():
    _, _, _, st = load()
    st = st.copy()
    rows = ["full"] + [str(k) for k in range(5)]
    st = st.set_index("fold").loc[rows]
    names = [c[4:] for c in st.columns if c.startswith("use_")]
    used = st[[f"use_{c}" for c in names]].to_numpy()
    fold_used = used[1:]
    order = [i for i in np.argsort(-fold_used.sum(0) - 0.1 * used[0], kind="stable") if used[:, i].any()]
    cols = [names[i] for i in order]
    M = used[:, order]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    nr, nc = M.shape
    for r in range(nr):
        for c in range(nc):
            on = M[r, c] == 1
            ax.add_patch(Rectangle((c + 0.06, nr - 1 - r + 0.08), 0.88, 0.84,
                                   facecolor=(JOINT if r > 0 else INK) if on else "#f3f4f6", edgecolor="none"))
    ax.axhline(nr - 1, color="white", lw=4)
    ax.set_xlim(0, nc + 3.4)
    ax.set_ylim(-0.1, nr + 0.05)
    ax.axis("off")
    lab_rows = ["Full-data f (Nov 4)"] + [f"Fold {k + 1}" for k in range(5)]
    for r, lab in enumerate(lab_rows):
        y = nr - 1 - r + 0.5
        ax.text(-0.12, y, lab, ha="right", va="center", fontsize=8.5, color=INK, weight="bold" if r == 0 else None)
        s = st.iloc[r]
        ax.text(nc + 0.85, y, f"{int(s['complexity'])}", ha="center", va="center", fontsize=8.5, color=INK)
        ax.text(nc + 2.45, y, "yes" if bool(s["S_star"]) else "no", ha="center", va="center", fontsize=8.5,
                color=INK if bool(s["S_star"]) else MUTED, weight="bold" if bool(s["S_star"]) else None)
    ax.text(nc + 0.85, nr + 0.05, "Complexity", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(nc + 2.45, nr + 0.05, "Retainer × log(∫)\nstructure", ha="center", va="bottom", fontsize=7.5,
            color=MUTED)
    for c, nm in enumerate(cols):
        ax.text(c + 0.5, -0.08, LABEL.get(nm, nm), rotation=40, ha="right", va="top", fontsize=7.5, color=INK,
                rotation_mode="anchor")
        k = int(fold_used[:, order[c]].sum())
        ax.text(c + 0.5, nr + 0.05, f"{k}/5", ha="center", va="bottom", fontsize=7.5, color=INK if k >= 4 else MUTED,
                weight="bold" if k >= 4 else None)
    ax.text(-0.12, nr + 0.05, "used in folds:", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    save(fig, "fig3_equation_reproducibility")


# ---------------------------------------------------------------- Fig 4
def fig4():
    _, oof, summ, _ = load()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2), sharex="col", sharey="col",
                             gridspec_kw={"hspace": 0.16, "wspace": 0.28})
    models = [("joint_sr", "Joint SR", JOINT), ("lr_group_samevars", "Linear regression (same variables)", LR)]
    for ci, g in enumerate(TABLES):
        sub = oof[oof["group"] == g]
        lo = np.floor(sub["mrr_true"].min() - 3)
        hi = np.ceil(sub["mrr_true"].max() + 3)
        for ri, (m, lab, col) in enumerate(models):
            ax = axes[ri, ci]
            o = sub[sub["model"] == m]
            inside = o["mrr_pred"].between(lo, hi)
            ax.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls="--", zorder=1)
            ax.scatter(o.loc[inside, "mrr_true"], o.loc[inside, "mrr_pred"], s=7, color=col, alpha=0.45, linewidth=0, zorder=2)
            out = o[~inside]
            for _, w in out.iterrows():
                yv = hi - 0.6 if w["mrr_pred"] > hi else lo + 0.6
                ax.scatter(w["mrr_true"], yv, marker="^" if w["mrr_pred"] > hi else "v", s=28, color=col, zorder=3,
                           edgecolor=INK, linewidth=0.5)
            if len(out):
                ex = out.loc[(out["mrr_pred"] - out["mrr_true"]).abs().idxmax()]
                ax.text(0.97, 0.04, f"{len(out)} beyond axis (triangles)\nlargest: measured {ex['mrr_true']:.0f}, "
                        f"predicted {ex['mrr_pred']:.0f}", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.8, color=INK)
            s = summ[(summ["group"] == g) & (summ["model"] == m)].iloc[0]
            ax.text(0.04, 0.96, f"{lab}\nOOF RMSE = {s['rmse_oof']:.2f}\nn = {int(s['n_oof'])}", transform=ax.transAxes,
                    va="top", fontsize=7.6, color=INK)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(TABLES[g], fontsize=10, color=INK, weight="bold")
            if ri == 1:
                ax.set_xlabel("Measured MRR")
            if ci == 0:
                ax.set_ylabel("Predicted MRR (out-of-fold)")
    fig.text(0.5, 0.025, "Dashed line: perfect prediction (y = x)", ha="center", fontsize=7.3, color=MUTED)
    save(fig, "fig4_measured_vs_predicted")


# ---------------------------------------------------------------- 요약 패널
def summary():
    _, _, summ, st = load()
    sel = pd.read_csv(sj.RESULTS / "2026-11-04_전준혁_FV1_sr_joint_coefs.csv", encoding="utf-8-sig")
    conc = json.loads((sj.ROOT / "results" / "reports" / "poster_joint_sr" / "conclusion.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    # 왼쪽 상자: 전체 데이터 최종 식
    ax.add_patch(FancyBboxPatch((0.08, 1.30), 3.45, 2.98, boxstyle="round,pad=0,rounding_size=0.06", facecolor="#f8fafc",
                                edgecolor=MUTED, lw=0.8))
    ax.text(0.22, 4.10, "Discovered joint equation", fontsize=10, weight="bold", color=INK, va="top")
    ax.text(0.22, 3.85, "Searched on all Train+Val wafers (Nov 4); this exact f is\nnot what the 5-fold check on the right tests",
            fontsize=7.5, color=MUTED, va="top", linespacing=1.3)
    ax.text(0.22, 3.42, r"MRR$_g$ = a$_g$ · f + b$_g$", fontsize=10.5, color=INK, va="top")
    ax.text(0.22, 3.12, r"f = (ln P$_{ret}$ − ∫P$_{ret}$) · ln[ ∫Q$_{C}$ · (∫P$_{ch}$ + 0.09) · ∫P$_{bag}$ ]",
            fontsize=9.2, color=INK, va="top")
    legend = ("P$_{ret}$: retainer-ring pressure (polishing mean);  ∫: time integral in chamber 4\n"
              "Q$_C$: slurry flow C;  P$_{ch}$: chamber pressure;  P$_{bag}$: main air-bag pressure\n"
              "(variables scaled by their median)")
    ax.text(0.22, 2.74, legend, fontsize=6.9, color=MUTED, va="top", linespacing=1.4)
    co = {r["group"]: {"a": r["a_raw"], "b": r["b_raw"]} for _, r in sel.iterrows() if r["group"] in TABLES}
    assert all(abs(co[g][k] - conc["full_coefs"][g][k]) < 1e-3 for g in TABLES for k in "ab"), "conclusion.json 계수 불일치"
    ax.text(0.22, 2.02, "Table", fontsize=8, color=MUTED, va="top")
    ax.text(1.35, 2.02, "a$_g$ (slope)", fontsize=8, color=MUTED, va="top")
    ax.text(2.45, 2.02, "b$_g$ (offset)", fontsize=8, color=MUTED, va="top")
    for i, g in enumerate(TABLES):
        y = 1.78 - i * 0.22
        ax.text(0.22, y, g, fontsize=8.5, color=INK, va="top", weight="bold")
        ax.text(1.35, y, f"{co[g]['a']:.1f}", fontsize=8.5, color=INK, va="top")
        ax.text(2.45, y, f"{co[g]['b']:.1f}", fontsize=8.5, color=INK, va="top")
    # 오른쪽 상자: 5-fold 결과
    ax.add_patch(FancyBboxPatch((3.67, 1.30), 3.45, 2.98, boxstyle="round,pad=0,rounding_size=0.06", facecolor="#f8fafc",
                                edgecolor=MUTED, lw=0.8))
    ax.text(3.81, 4.10, "5-fold re-discovery check", fontsize=10, weight="bold", color=INK, va="top")
    ax.text(3.81, 3.85, "Equation searched again in each fold without its held-out 20%", fontsize=7.5, color=MUTED, va="top")
    ax.text(3.81, 3.50, "Out-of-fold RMSE", fontsize=8, color=MUTED, va="top")
    ax.text(5.25, 3.50, "Joint SR", fontsize=8, color=JOINT, va="top", weight="bold")
    ax.text(6.20, 3.50, "Lin. reg.", fontsize=8, color=LR, va="top", weight="bold")
    for i, g in enumerate(TABLES):
        y = 3.24 - i * 0.24
        j = summ[(summ["group"] == g) & (summ["model"] == "joint_sr")].iloc[0]
        l_ = summ[(summ["group"] == g) & (summ["model"] == "lr_group_samevars")].iloc[0]
        d = summ[(summ["group"] == g) & (summ["model"] == "diff_joint_minus_lr")].iloc[0]
        ax.text(3.81, y, g, fontsize=8.5, color=INK, va="top", weight="bold")
        ax.text(5.25, y, f"{j['rmse_oof']:.2f}", fontsize=8.5, color=INK, va="top")
        ax.text(6.20, y, f"{l_['rmse_oof']:.2f}", fontsize=8.5, color=INK, va="top")
        ax.text(3.81, 2.62 - i * 0.22, f"{g}: Joint SR better in {int(d['joint_better_folds'])}/5 folds",
                fontsize=7.8, color=INK, va="top")
    ax.text(3.81, 2.08, conc["structure_line"], fontsize=7.8, color=INK, va="top", wrap=True)
    ax.text(3.81, 1.62, conc["identity_line"], fontsize=7.0, color=MUTED, va="top")
    # 결론
    ax.text(0.10, 1.12, "Conclusion", fontsize=10, weight="bold", color=INK, va="top")
    ax.text(0.10, 0.86, conc["conclusion_en"], fontsize=8.4, color=INK, va="top", linespacing=1.4)
    ax.text(0.10, 0.12, "Post-hoc follow-up on Train+Val only (no test labels). Candidate variables were pre-selected on all "
            "Train+Val data, so this is not an external validation.", fontsize=6.6, color=MUTED, va="bottom")
    save(fig, "summary_panel")


def main():
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "summary"]
    for w in which:
        {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "summary": summary}[w]()


if __name__ == "__main__":
    main()

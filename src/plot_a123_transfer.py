"""A-123 전이학습 주 평가(PR #25) 결과 정리와 그림.

숫자는 결과 csv 4개(folds·stats·judgment·rho)에서만 계산한다. 모델을 다시 돌리지 않고, 결과 csv를 고치지 않는다.
features·라벨 파일·test·data/locked/는 읽지 않는다.

정리 (preregistration.md 5.5절 보고 항목):
1. 비율 × 모델 RMSE·MAE·MSE 25 fold 평균 ± 표준편차 (full, usage 따로)
2. T0 대비 RMSE 개선폭 (%)
3. T2 대 T1 (보고만, 8%)
4. T1·T3 90% 예측구간 커버리지
5. ρ 분포 (평균, 범위, 경계 0·2에 붙은 fold 수)
6. 8% 웨이퍼 분해 (전체 / 센서 정상 / 전체-0)
7. 8% P1_on 포함/제외 fold별 T3−T1 (5.7-13)
8. GP 경고 건수 (csv에는 건수만 있음)
9. 8% δ 입력 변수 빈도
표준편차는 ddof=1, 표준오차 = 표준편차 / √(fold 수).

실행: python src/plot_a123_transfer.py
출력: results/reports/{STEM}_summary.md, results/figures/{STEM}_fig{1~4}_*.png (있으면 멈춤, 덮어쓰기 금지)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import FontProperties, findSystemFonts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STEM = "2026-11-01_김혜연_FV1_a123_transfer"
IN = {c: RESULTS / f"{STEM}_{c}.csv" for c in ["folds", "stats", "judgment", "rho"]}
OUT_MD = RESULTS / "reports" / f"{STEM}_summary.md"
FIGS = {k: RESULTS / "figures" / f"{STEM}_{k}.png"
        for k in ["fig1_rmse", "fig2_t3_t1_diff", "fig3_rho", "fig4_coverage"]}
RATIOS = [2, 4, 8, 16, 100]
MODELS = ["T0", "T1", "T2", "T3"]
SOURCES = ["S1", "S2"]
PRIMARY = 8
SUBSET_KO = {"all": "전체", "clean": "센서 정상", "zero": "전체-0"}
COLORS = {"T0": "#888888", "T1": "#1f77b4", "T2": "#2ca02c", "T3": "#d62728"}

for p in IN.values():
    assert "locked" not in str(p).lower(), p


def korean_font() -> str:
    want = ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic"]
    have = {FontProperties(fname=f).get_name() for f in findSystemFonts()}
    return next((n for n in want if n in have), plt.rcParams["font.family"][0])


def pm(s, nd=2):
    return f"{s.mean():.{nd}f} ± {s.std(ddof=1):.{nd}f}"


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fp(p):
    return f"{p:.2g}" if p < 1e-3 else f"{p:.3f}"


# ---------------------------------------------------------------- 표
def t_metrics(A, inp):
    rows = []
    for ratio in RATIOS:
        for m in MODELS:
            row = {"비율(%)": ratio, "모델": m}
            for s in SOURCES:
                g = A[(A["ratio"] == ratio) & (A["model"] == m) & (A["source"] == s) & (A["input"] == inp)]
                assert len(g) == 25, (ratio, m, s, inp, len(g))
                for k in ["rmse", "mae", "mse"]:
                    row[f"{s} {k.upper()}"] = pm(g[k], 3 if k != "mse" else 2)
            rows.append(row)
    return pd.DataFrame(rows)


def mean_rmse(A, ratio, m, s, inp):
    return A[(A["ratio"] == ratio) & (A["model"] == m) & (A["source"] == s) & (A["input"] == inp)]["rmse"].mean()


def t_improve(A):
    rows = []
    for ratio in RATIOS:
        row = {"비율(%)": ratio}
        t0 = mean_rmse(A, ratio, "T0", "S1", "full")
        row["T0 RMSE"] = f"{t0:.3f}"
        row["T1"] = f"{(t0 - mean_rmse(A, ratio, 'T1', 'S1', 'full')) / t0 * 100:.1f}"
        for inp in ["full", "usage"]:
            for s in SOURCES:
                for m in ["T2", "T3"]:
                    row[f"{m} {s} {inp}"] = f"{(t0 - mean_rmse(A, ratio, m, s, inp)) / t0 * 100:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def t_stats(S, compare, ratios):
    g = S[(S["compare"] == compare) & (S["ratio"].isin(ratios))].sort_values(["ratio", "input", "source"])
    return pd.DataFrame({"비율(%)": g["ratio"], "입력": g["input"], "소스": g["source"],
                         "평균 MSE 차이": g["mean_diff"].map("{:.3f}".format),
                         "차이 SD": g["sd_diff"].map("{:.3f}".format),
                         "앞 모델이 낮은 fold": g["wins"].astype(str) + "/" + g["J"].astype(str),
                         "p (full)": g["p"].map(fp), "p (revealed)": g["p_revealed"].map(fp)})


def t_coverage(A):
    rows = []
    for ratio in RATIOS:
        g = A[A["ratio"] == ratio]
        row = {"비율(%)": ratio, "T1": pm(g[(g["model"] == "T1") & (g["source"] == "S1") & (g["input"] == "full")]["coverage90"], 3)}
        for inp in ["full", "usage"]:
            for s in SOURCES:
                row[f"T3 {s} {inp}"] = pm(g[(g["model"] == "T3") & (g["source"] == s) & (g["input"] == inp)]["coverage90"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def t_rho(A, RH):
    T3 = A[A["model"] == "T3"]
    rows = []
    for (ratio, s, inp), g in T3.groupby(["ratio", "source", "input"]):
        ref = RH[(RH["ratio"] == ratio) & (RH["source"] == s) & (RH["input"] == inp)].iloc[0]
        assert np.isclose(ref["mean"], g["rho"].mean()) and np.isclose(ref["min"], g["rho"].min())
        rows.append({"비율(%)": ratio, "소스": s, "입력": inp, "평균": f"{g['rho'].mean():.3f}",
                     "SD": f"{g['rho'].std(ddof=1):.3f}",
                     "범위": f"{g['rho'].min():.3f} ~ {g['rho'].max():.3f}",
                     "ρ=0 fold": int((g["rho"] <= 0).sum()), "ρ=2 fold": int((g["rho"] >= 2).sum())})
    return pd.DataFrame(rows)


def t_subset(R):
    g8 = R[(R["ratio"] == PRIMARY) & (R["input"] == "full")]
    rows = []
    for sub in ["all", "clean", "zero"]:
        row = {"웨이퍼": SUBSET_KO[sub]}
        x = g8[(g8["subset"] == sub) & (g8["source"] == "S1") & (g8["model"] == "T1")]
        row["fold 수"] = len(x)
        row["fold당 평가 장수"] = f"{x['n'].mean():.1f}"
        row["T1"] = pm(x["rmse"], 3)
        for s in SOURCES:
            row[f"T3 {s}"] = pm(g8[(g8["subset"] == sub) & (g8["source"] == s) & (g8["model"] == "T3")]["rmse"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def diff_t3_t1(A, ratio, s, inp):
    g = A[(A["ratio"] == ratio) & (A["source"] == s) & (A["input"] == inp)]
    P = g.pivot(index="r", columns="model", values="mse")
    meta = g.drop_duplicates("r").set_index("r").loc[P.index]
    return (P["T3"] - P["T1"]), meta


def t_p1on(A):
    rows = []
    for inp in ["full", "usage"]:
        for s in SOURCES:
            d, meta = diff_t3_t1(A, PRIMARY, s, inp)
            for inc in [1, 0]:
                x = d[meta["p1_on_included"] == inc]
                rows.append({"입력": inp, "소스": s, "P1_on": "포함" if inc else "제외", "fold 수": len(x),
                             "T3−T1 MSE 차이 평균": f"{x.mean():.3f}" if len(x) else "—",
                             "SD": f"{x.std(ddof=1):.3f}" if len(x) > 1 else "—",
                             "T3가 낮은 fold": f"{int((x < 0).sum())}/{len(x)}"})
    return pd.DataFrame(rows)


def t_warn(A):
    W = A.drop_duplicates(["ratio", "r", "source", "input"])
    rows = []
    for ratio in RATIOS:
        g = W[W["ratio"] == ratio]
        t1 = g.drop_duplicates("r")["gp_warn_t1"]
        row = {"비율(%)": ratio, "T1 경고 건수": int(t1.sum()), "T1 경고 fold": f"{int((t1 > 0).sum())}/25"}
        for inp in ["full", "usage"]:
            for s in SOURCES:
                x = g[(g["source"] == s) & (g["input"] == inp)]["gp_warn_t3"]
                row[f"T3 {s} {inp}"] = f"{int(x.sum())} ({int((x > 0).sum())}/25)"
        rows.append(row)
    return pd.DataFrame(rows)


def t_delta(A):
    d = A[(A["ratio"] == PRIMARY)].drop_duplicates("r")
    lists = d["delta_cols"].str.split(",")
    freq = pd.Series([c for L in lists for c in L]).value_counts()
    t_var = pd.DataFrame({"변수": freq.index, "들어간 fold 수": [f"{v}/25" for v in freq.values]})
    combo = d["delta_cols"].value_counts()
    t_combo = pd.DataFrame({"δ 입력 조합": combo.index.str.replace(",", ", "), "fold 수": combo.values})
    return t_var, t_combo, d


def t_reveal(A):
    d = A.drop_duplicates(["ratio", "r"])
    rows = []
    for ratio, g in d.groupby("ratio"):
        rows.append({"비율(%)": ratio,
                     "전체 (최소~최대)": f"{g['n_public'].mean():.1f} ({g['n_public'].min()}~{g['n_public'].max()})",
                     "센서 정상 (최소~최대)": f"{g['n_public_clean'].mean():.1f} ({g['n_public_clean'].min()}~{g['n_public_clean'].max()})",
                     "δ 입력 개수 (최소~최대)": f"{g['delta_cols'].str.count(',').min() + 1}~{g['delta_cols'].str.count(',').max() + 1}",
                     "등방성 커널 fold": int(g["iso_kernel"].sum())})
    return pd.DataFrame(rows)


def t_judgment(J):
    rows = []
    for _, j in J.iterrows():
        for s in SOURCES:
            rows.append({"판정": f"{j['kind']} ({j['compare']})", "소스": s,
                         "평균 MSE 차이": f"{j[f'{s}_mean_diff']:.3f}", "p (full)": fp(j[f"{s}_p"]),
                         "p (revealed)": fp(j[f"{s}_p_revealed"]), "통과": "✓" if j[f"{s}_passed"] else "✗",
                         "최종 판정": j["verdict"] if s == "S1" else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그림
def fig_rmse(A, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s in zip(axes, SOURCES):
        for m in MODELS:
            g = A[(A["source"] == s) & (A["input"] == "full") & (A["model"] == m)].groupby("ratio")["rmse"]
            mu, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
            ax.plot(mu.index, mu.values, "o-", color=COLORS[m], label=m)
            ax.fill_between(mu.index, mu - se, mu + se, color=COLORS[m], alpha=0.2)
        ax.set_xscale("log")
        ax.set_xticks(RATIOS)
        ax.set_xticklabels([str(r) for r in RATIOS])
        ax.set_xlabel("공개 비율 (%)")
        ax.set_title(f"{s} (입력 full)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("평가 쪽 RMSE (25 fold 평균 ± 표준오차)")
    axes[0].legend(frameon=False)
    fig.suptitle("A-123 전이학습: 공개 비율별 RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_diff(A, path):
    data = [diff_t3_t1(A, PRIMARY, s, "full")[0].values for s in SOURCES]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(data, tick_labels=SOURCES, showfliers=False)  # fold 점을 따로 찍으므로 이상치 표시는 끈다
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.linspace(-0.08, 0.08, len(d)), d, s=10, color="#d62728", alpha=0.6)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("T3 - T1 MSE 차이 (fold별, 음수 = T3가 낮음)")
    ax.set_title(f"공개 {PRIMARY}%, 입력 full, 25 fold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_rho(A, path):
    T3 = A[(A["model"] == "T3") & (A["input"] == "full")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s in zip(axes, SOURCES):
        ax.boxplot([T3[(T3["source"] == s) & (T3["ratio"] == r)]["rho"].values for r in RATIOS],
                   tick_labels=[str(r) for r in RATIOS])
        for y in (0, 2):
            ax.axhline(y, color="k", ls=":", lw=1)
        ax.set_xlabel("공개 비율 (%)")
        ax.set_title(f"{s} (입력 full)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("ρ (0~2로 제한)")
    fig.suptitle("T3의 ρ 추정값 분포 (25 fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_coverage(A, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    lines = [("T1", "S1", "T1", COLORS["T1"], "o-"), ("T3", "S1", "T3 S1", COLORS["T3"], "s-"),
             ("T3", "S2", "T3 S2", "#ff7f0e", "^-")]
    for m, s, lab, col, mk in lines:
        g = A[(A["model"] == m) & (A["source"] == s) & (A["input"] == "full")].groupby("ratio")["coverage90"].mean()
        ax.plot(g.index, g.values, mk, color=col, label=lab)
    ax.axhline(0.90, color="k", ls="--", lw=1, label="목표 0.90")
    ax.set_xscale("log")
    ax.set_xticks(RATIOS)
    ax.set_xticklabels([str(r) for r in RATIOS])
    ax.set_xlabel("공개 비율 (%)")
    ax.set_ylabel("90% 예측구간 커버리지 (25 fold 평균)")
    ax.set_title("T1·T3 커버리지 (입력 full, 전체 웨이퍼)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 실행
def main():
    outs = [OUT_MD, *FIGS.values()]
    exist = [p.name for p in outs if p.exists()]
    if exist:
        sys.exit(f"이미 있음 (덮어쓰기 금지): {exist}")
    for p in outs:
        p.parent.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(IN["folds"])
    S = pd.read_csv(IN["stats"])
    J = pd.read_csv(IN["judgment"])
    RH = pd.read_csv(IN["rho"])
    A = R[R["subset"] == "all"]
    assert len(A) == len(RATIOS) * 25 * len(SOURCES) * 2 * len(MODELS)
    # T0·T1은 소스·입력과 무관 (같은 fold 값이 반복 저장됨)
    for m in ["T0", "T1"]:
        assert A[A["model"] == m].groupby(["ratio", "r"])["mse"].nunique().max() == 1

    plt.rcParams["font.family"] = korean_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig_rmse(A, FIGS["fig1_rmse"])
    fig_diff(A, FIGS["fig2_t3_t1_diff"])
    fig_rho(A, FIGS["fig3_rho"])
    fig_coverage(A, FIGS["fig4_coverage"])

    t_var, t_combo, d8 = t_delta(A)
    n_clean10 = int((d8["n_public_clean"] >= 10).sum())
    max_singles = int(d8["delta_cols"].str.count(",").max() + 1)
    fig_rel = {k: f"../figures/{p.name}" for k, p in FIGS.items()}
    md = f"""# A-123 전이학습 주 평가 정리 (FV1)

- 입력: `results/{IN['folds'].name}`, `{IN['stats'].name}`, `{IN['judgment'].name}`, `{IN['rho'].name}` (PR #25)
- 실행 커밋 9947b4c, xgboost 3.2.0, f_456 n_jobs 2. 숫자는 모두 위 csv에서 계산 (`src/plot_a123_transfer.py`)
- 범위: folds_v1 25 fold, 고정주기 공개, 소스 S1(A-456)·S2(A-456+B-456), f_456 입력 full·usage, T0~T3.
  전진 검증(보조 판정), 무작위 공개, δ XGBoost, 라벨 피처 실험은 하지 않았다.
- 표기: 평균 ± 표준편차(25 fold, ddof=1). T0·T1은 소스·입력과 무관해 S1·S2 칸 값이 같다.

## 0. 판정 (preregistration.md 5.5절, 공개 8%, 입력 full)

{md_table(t_judgment(J))}

- **주 판정 (T3 대 T1): 효과 없음 (기본값 T1 유지).** S1·S2 모두 평균 MSE 차이 ≥ 0, p ≥ 0.05.
- **2차 판정 (T3 대 T2): 효과 있음.** S1·S2 모두 평균 차이 < 0, p < 0.05 (δ GP의 추가 효과).
- p (revealed)는 병기만 하고 판정에 쓰지 않는다 (5.5).
- 보조 판정(전진 검증)은 아직 실행하지 않았다 (5.7-17 일정상 11/1 이후).
- usage 입력은 보고만 한다 (판정 대상 아님). S1 대 S2 차이는 "같은 스테이지 데이터가 더 비슷한가"라는 관찰로만 본다 (5.0 해석 제한).
- 한계 (5.5): A-123은 MRR 변동이 작다 (변동계수 0.027). T0 대비 개선폭은 2절.
- 90% 예측구간은 WhiteKernel 잡음만 포함하고 f_456·ρ·b의 불확실성은 무시한다 (5.7-10).

### fold당 실제 공개 장수 (5.3)

{md_table(t_reveal(A))}

### 참고: 비율별 T3 대 T1 (2·4·16·100%는 경향만, 검정은 참고 — 5.5)

{md_table(t_stats(S, "T3-T1", RATIOS))}

## 1. 비율 × 모델 RMSE·MAE·MSE (전체 웨이퍼)

### 입력 full

{md_table(t_metrics(A, "full"))}

### 입력 usage (보고만)

{md_table(t_metrics(A, "usage"))}

## 2. T0 대비 RMSE 개선폭 (%)

(T0 RMSE − 모델 RMSE) / T0 RMSE × 100, 각 RMSE는 25 fold 평균. 음수 = T0보다 큼.

{md_table(t_improve(A))}

## 3. T2 대 T1 (보고만, 공개 8%)

평균 MSE 차이 = T2 − T1. "앞 모델이 낮은 fold" = T2가 낮은 fold 수.

{md_table(t_stats(S, "T2-T1", [PRIMARY]))}

## 4. 90% 예측구간 커버리지 (목표 0.90, 전체 웨이퍼)

{md_table(t_coverage(A))}

## 5. ρ 분포 (T3, 25 fold)

ρ는 0~2로 제한 (5.4). "ρ=0 fold"·"ρ=2 fold"는 경계에 붙은 fold 수.

{md_table(t_rho(A, RH))}

## 6. 웨이퍼 분해 (공개 8%, 입력 full, RMSE)

분해 결과는 보고만 (5.7-16). 전체-0 = A-123 전체-0 센서 웨이퍼 (`allzero`).

{md_table(t_subset(R))}

## 7. P1_on 포함/제외 fold (공개 8%, 5.7-13)

T3−T1 MSE 차이 (fold별, 음수 = T3가 낮음). δ 입력은 소스·입력과 무관하게 fold마다 같다.

{md_table(t_p1on(A))}

- 8%에서 센서 정상 공개가 10장 이상인 fold는 {n_clean10}/25이지만, 단일 변수가 {max_singles}개 들어간 뒤 상한 5개 중
  자리가 {5 - max_singles}개만 남아 `P1_on`·`allzero` 쌍은 어느 fold에도 들어가지 않았다 (5.7-8). 포함/제외 비교는 할 수 없다.

## 8. GP 경고

T1은 fold당 1회 학습 (비율당 25회). T3은 fold × 소스 × 입력마다 1회 (칸마다 25회). 값 = 경고 건수 (경고가 난 fold 수/25).

{md_table(t_warn(A))}

- 경고 종류(메시지)와 어느 하이퍼파라미터가 어느 쪽 경계에 붙었는지는 **결과 csv에 저장돼 있지 않아 이 정리에서 셀 수 없다.**
  csv에는 `gp_warn_t1`·`gp_warn_t3`(ConvergenceWarning·UserWarning·RuntimeWarning 합계)만 있다.
- 경고가 나도 결과는 그대로 쓴다 (5.7-15).

## 9. δ 입력 (공개 8%, fold별 실제 입력)

공개 28장 → 상한 max(2, 내림(28/5)) = 5개.

{md_table(t_var)}

{md_table(t_combo)}

## 그림

- 그림 1: 공개 비율 대 RMSE (T0~T3, 25 fold 표준오차 띠, 입력 full) ![]({fig_rel['fig1_rmse']})
- 그림 2: 8% fold별 T3−T1 MSE 차이 (입력 full) ![]({fig_rel['fig2_t3_t1_diff']})
- 그림 3: ρ 분포 (입력 full, 점선 0·2) ![]({fig_rel['fig3_rho']})
- 그림 4: T1·T3 커버리지 (입력 full, 기준선 0.90) ![]({fig_rel['fig4_coverage']})
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print("저장:", OUT_MD.relative_to(ROOT), "/", ", ".join(p.name for p in FIGS.values()), flush=True)


if __name__ == "__main__":
    main()

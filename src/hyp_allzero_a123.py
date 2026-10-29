"""가설 3: A-123 전체-0 센서 웨이퍼는 어떤 웨이퍼인가 (진단, CLAUDE.md 14.4).

1. 시간축 위치: 특정 기간에 몰려 있는가, 정상 웨이퍼 사이에 섞여 있는가.
   A-123 웨이퍼를 세 split 합쳐 시작 시각 순으로 세우고 Wald–Wolfowitz 연(run) 검정을 한다.
   연 수가 무작위 기대보다 적으면 몰려 있다는 뜻이다. 5일 구간별 비율도 낸다.
2. 정상 웨이퍼와 비교: MRR(train·val 라벨, 이상치 제외), 연마 시간, 소모품 사용량,
   직전 웨이퍼와의 시작 시각 간격(같은 A-123 / 같은 스테이지 A 전체). 중앙값·사분위와 Mann–Whitney p.
   MRR은 시간 추세가 크므로 t_start를 통제한 차이(선형 회귀 계수)도 낸다.
3. 어떤 센서가 0인지: 12개 센서(압력 6 + 슬러리 3 + 회전 3, 5.4절)의 웨이퍼 전체-0 묶음별 빈도.
   원본 센서에서 직접 계산하고, 12개 합이 집계표 `n_allzero_sensors`와 같은지 검산한다.

라벨: train은 집계표, val은 `data/raw/labels/`에서 읽어 집계표와 대조한 뒤 쓴다. test 웨이퍼는 센서만 쓴다.
test 라벨과 `data/locked/`는 어떤 경로로도 읽지 않는다.

실행: python src/hyp_allzero_a123.py
출력: results/diagnostics/2026-10-06_김혜연_allzero_a123_compare.csv   비교·시간 위치 (+ .yaml)
      results/diagnostics/2026-10-06_김혜연_allzero_a123_patterns.csv  센서 묶음 빈도 (+ .yaml)
      docs/reports/figs/김혜연_allzero_a123_time.png
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
from aggregate import ALLZERO_SENSORS, FLOW, OUTLIER_MRR, PRESSURE, ROTATION  # noqa: E402
from hyp_r2r_retainer import DATE, load_raw, style  # noqa: E402
from load import KEY, ROOT, load_labels  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from pv_partial_consumables import START6  # noqa: E402
from scatter_pv import korean_font  # noqa: E402

DIAG = ROOT / "results" / "diagnostics"
OUT = DIAG / f"{DATE}_{AUTHOR}_allzero_a123_compare.csv"
OUT_PAT = DIAG / f"{DATE}_{AUTHOR}_allzero_a123_patterns.csv"
FIG = ROOT / "docs" / "reports" / "figs" / f"{AUTHOR}_allzero_a123_time.png"
GROUP = "A-123"
ABBR = {**PRESSURE, **FLOW, **ROTATION}
BIN_DAYS = 5


def runs_test(x: np.ndarray) -> dict:
    """0/1 계열의 Wald–Wolfowitz 연 검정 (정규 근사). z < 0 이면 무작위보다 몰려 있다."""
    n1, n2 = int(x.sum()), int((1 - x).sum())
    n = n1 + n2
    runs = 1 + int((x[1:] != x[:-1]).sum())
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    return {"n": n, "n_allzero": n1, "runs": runs, "runs_expected": mu, "z": z, "p": 2 * norm.sf(abs(z))}


def ols_effect(y: pd.Series, flag: pd.Series, ctrl: pd.Series) -> tuple[float, float]:
    """y ~ 1 + flag + ctrl 에서 flag 계수와 표준오차."""
    X = np.column_stack([np.ones(len(y)), flag.to_numpy(float), ctrl.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)
    resid = y.to_numpy(float) - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return float(beta[1]), float(se[1])


def main() -> None:
    for p in (OUT, OUT_PAT, FIG):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    val = load_labels("val").rename(columns={"AVG_REMOVAL_RATE": "MRR_raw"})
    chk = tab[tab["split"] == "val"].merge(val, on=KEY)
    if len(chk) != len(val) or not np.allclose(chk["MRR"], chk["MRR_raw"]):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")

    # 직전 웨이퍼 간격: 같은 A-123 / 같은 스테이지 A 전체 (세 split, 시작 시각 순)
    sa = tab[tab["STAGE"] == "A"].sort_values("t_start", kind="stable").copy()
    sa["gap_prev_stageA"] = sa["t_start"].diff()
    a = sa[sa["group"] == GROUP].copy()
    a["gap_prev_A123"] = a["t_start"].diff()
    a["dur_total"] = a["t_end"] - a["t_start"]
    a["allzero"] = (a["n_allzero_sensors"] > 0).astype(int)
    print(f"{GROUP} {len(a):,}장 중 전체-0 센서 웨이퍼 {a['allzero'].sum():,}장. 센서 읽는 중", flush=True)

    # 3. 센서별 전체-0 (원본에서 직접)
    raw = load_raw(a[KEY])
    raw[ALLZERO_SENSORS] = raw[ALLZERO_SENSORS].apply(pd.to_numeric, errors="raise")
    z = (raw[KEY + ALLZERO_SENSORS].set_index(KEY) == 0).groupby(KEY).all()
    z.columns = [ABBR[c] for c in z.columns]
    a = a.merge(z.astype(int).reset_index(), on=KEY, how="left")
    if not (a[list(z.columns)].sum(axis=1) == a["n_allzero_sensors"]).all():
        raise ValueError("원본에서 센 전체-0 센서 수가 집계표 n_allzero_sensors와 다르다")
    print("검산: 12개 센서 전체-0 합 = n_allzero_sensors (전 웨이퍼 일치)", flush=True)
    a["pattern"] = a[list(z.columns)].apply(lambda r: "+".join(c for c in z.columns if r[c]) or "(없음)", axis=1)
    pat = (a[a["allzero"] == 1].groupby("pattern")
           .agg(n=("pattern", "size"), n_sensors=("n_allzero_sensors", "first"),
                n_train=("split", lambda s: int((s == "train").sum())),
                n_val=("split", lambda s: int((s == "val").sum())),
                n_test=("split", lambda s: int((s == "test").sum())),
                first_day=("t_start", "min"), last_day=("t_start", "max"))
           .sort_values("n", ascending=False).reset_index())
    t0 = tab["t_start"].min()
    pat[["first_day", "last_day"]] = (pat[["first_day", "last_day"]] - t0) / 86400
    pat["share"] = pat["n"] / pat["n"].sum()
    per_sensor = a.loc[a["allzero"] == 1, list(z.columns)].sum()
    pat = pd.concat([pat, pd.DataFrame({"pattern": [f"[센서별] {c}" for c in per_sensor.index],
                                        "n": per_sensor.to_numpy(),
                                        "share": per_sensor.to_numpy() / a["allzero"].sum()})],
                    ignore_index=True).round(4)

    # 1. 시간 위치
    a["day"] = (a["t_start"] - t0) / 86400
    rows = [{"check": "1_time", "item": "runs_test", **runs_test(a["allzero"].to_numpy())}]
    a["bin"] = (a["day"] // BIN_DAYS * BIN_DAYS).astype(int)
    for b, d in a.groupby("bin"):
        rows.append({"check": "1_time", "item": f"day_{b:02d}-{b + BIN_DAYS:02d}", "n": len(d),
                     "n_allzero": int(d["allzero"].sum()), "share_allzero": d["allzero"].mean()})
    for split, d in a.groupby("split"):
        rows.append({"check": "1_time", "item": f"split_{split}", "n": len(d),
                     "n_allzero": int(d["allzero"].sum()), "share_allzero": d["allzero"].mean()})

    # 2. 정상 대 전체-0 비교
    a["is_outlier_mrr"] = a["MRR"] > OUTLIER_MRR
    a["MRR_ok"] = a["MRR"].where(~a["is_outlier_mrr"])
    compare = ["MRR_ok", "dur_total", "dur_POLISH_s1", "dur_POLISH_s2", "n_rows",
               *START6, "delta_POLISH_TABLE_w", "gap_prev_A123", "gap_prev_stageA"]
    for c in compare:
        n0, n1 = a.loc[a["allzero"] == 0, c].dropna(), a.loc[a["allzero"] == 1, c].dropna()
        row = {"check": "2_compare", "item": c, "n_normal": len(n0), "n_allzero": len(n1),
               "p": mannwhitneyu(n0, n1).pvalue}
        for tag, s in (("normal", n0), ("allzero", n1)):
            row.update({f"{tag}_median": s.median(), f"{tag}_q25": s.quantile(0.25),
                        f"{tag}_q75": s.quantile(0.75), f"{tag}_mean": s.mean(), f"{tag}_std": s.std()})
        rows.append(row)
    m = a[a["MRR_ok"].notna()]
    coef, se = ols_effect(m["MRR_ok"], m["allzero"], (m["t_start"] - t0) / 86400)
    rows.append({"check": "2_compare", "item": "MRR_ok_allzero_effect_ctrl_t_start", "n": len(m),
                 "effect": coef, "effect_se": se, "p": 2 * norm.sf(abs(coef / se))})
    o = a[a["is_outlier_mrr"]]
    rows.append({"check": "2_compare", "item": "outliers_MRR_gt_400", "n": len(o),
                 "n_allzero": int(o["allzero"].sum())})
    res = pd.DataFrame(rows).round(4)

    # 그림
    font = korean_font()
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1, 1.4]})
    fig.subplots_adjust(top=0.88, hspace=0.22)
    colors = {0: "#2a78d6", 1: "#d6872a"}
    labels = {0: "정상", 1: "전체-0 센서"}
    ax = axes[0]
    rng = np.random.default_rng(42)
    for f in (0, 1):
        d = a[a["allzero"] == f]
        ax.scatter(d["day"], f + rng.uniform(-0.25, 0.25, len(d)), s=7, color=colors[f], lw=0)
    ax.set_yticks([0, 1], [labels[0], labels[1]])
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("A-123 웨이퍼 시작 시각 (세 split, 점 = 웨이퍼, 세로 흔들림은 겹침 방지)",
                 fontsize=10.5, color="#333333", loc="left")
    style(ax)
    ax = axes[1]
    share = a["allzero"].rolling(25, center=True, min_periods=12).mean()
    ax.plot(a["day"], share, color="#d6872a", lw=1.5)
    ax.axhline(a["allzero"].mean(), color="#999999", lw=0.9, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("전체-0 비율", fontsize=9.5, color="#555555")
    ax.set_title("전체-0 센서 웨이퍼 비율 (A-123 25장 이동 평균, 점선 = 전체 평균)", fontsize=10.5,
                 color="#333333", loc="left")
    style(ax)
    ax = axes[2]
    for f in (0, 1):
        d = m[m["allzero"] == f]
        ax.scatter(d["day"], d["MRR_ok"], s=12, color=colors[f], lw=0, alpha=0.85, label=labels[f])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_ylabel("MRR", fontsize=9.5, color="#555555")
    ax.set_xlabel("웨이퍼 시작 시각 (첫 웨이퍼 기준 일)", fontsize=9.5, color="#555555")
    ax.set_title("MRR (train·val 라벨, 이상치 4장 제외)", fontsize=10.5, color="#333333", loc="left")
    style(ax)
    fig.text(0.02, 0.975, "A-123 전체-0 센서 웨이퍼의 시간축 위치", fontsize=13, color="#222222", va="top")
    rt = rows[0]
    fig.text(0.02, 0.935, f"연 검정: 연 {rt['runs']}개 / 무작위 기대 {rt['runs_expected']:.0f}개, z = {rt['z']:.2f}",
             fontsize=9.5, color="#777777", va="top")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    DIAG.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    pat.to_csv(OUT_PAT, index=False)
    common = [
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/hyp_allzero_a123.py",
        "question: 가설 3 — A-123 전체-0 센서 웨이퍼의 성격",
        "input: data/processed/wafer_table.csv + data/raw/{train,val,test} 센서 (A-123) + "
        "data/raw/labels/CMP-validation-removalrate.csv (집계표 val MRR과 대조 후 사용)",
        f"wafers: A-123 세 split {len(a)}장 (전체-0 {int(a['allzero'].sum())}장). MRR 비교는 train·val, MRR > 400 제외",
        "allzero: 12개 센서(압력 6 + 슬러리 3 + 회전 3) 중 하나 이상이 웨이퍼 모든 행에서 0 (5.4절). "
        "원본에서 직접 계산, n_allzero_sensors 와 전 웨이퍼 일치",
        "locked: data/locked/ 와 test 라벨은 열지 않았다",
    ]
    OUT.with_suffix(".yaml").write_text("\n".join(common + [
        "content: {1_time: 연 검정(Wald–Wolfowitz, 정규 근사)·5일 구간별·split별 비율, "
        "2_compare: 정상 대 전체-0 중앙값·사분위·평균 + Mann–Whitney p, MRR 은 t_start 통제 회귀 계수도}",
        "gaps: gap_prev_A123 = 직전 A-123 웨이퍼와의 시작 시각 차(초), gap_prev_stageA = 직전 스테이지 A 웨이퍼"
        "(A-123·A-456 무관)와의 차",
        "dur: dur_total = t_end − t_start, dur_POLISH_s1/s2 = 집계표 R 피처 (ID 22~23)",
        f"fig: {FIG.relative_to(ROOT).as_posix()}",
        "note: 진단 결과 (14.4)", "",
    ]), encoding="utf-8")
    OUT_PAT.with_suffix(".yaml").write_text("\n".join(common + [
        "content: 전체-0 센서 묶음(pattern)별 웨이퍼 수, split별 수, 첫·마지막 등장일. "
        "아래 '[센서별]' 행은 센서 하나하나가 전체-0 웨이퍼 중 몇 장에서 0인지",
        "note: 진단 결과 (14.4)", "",
    ]), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n" + res.to_string(index=False))
    print("\n" + pat.to_string(index=False))
    print("\n[chambers]\n" + pd.crosstab(a["chambers"], a["allzero"]).to_string())
    for p in (OUT, OUT_PAT, FIG):
        print(f"저장: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

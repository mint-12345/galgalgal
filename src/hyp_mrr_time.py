"""MRR(AVG_REMOVAL_RATE)은 무엇으로 나눈 값인가 (진단, CLAUDE.md 14.4).

가설 1: 실제 연마 시간으로 나눈 값 → MRR이 연마 시간과 무관 (log-log 기울기 β ≈ 0)
가설 2: 고정된 공정 시간으로 나눈 값 → MRR이 연마 시간에 비례 (β ≈ 1)

1. 웨이퍼별 연마 시간(초) 5가지 정의 × 빈칸 포함/제외 = 10가지.
   행 i의 시간 = 다음 기록까지의 간격 dt_i = t_(i+1) − t_i (마지막 행은 0). 정의는 조건을 만족하는 행의 dt 합이다.
     a_total     웨이퍼 전체 기록 (포함 값 = t_end − t_start)
     b_chamber   주 연마 챔버 체류 (456 = 챔버 4, 123 = 챔버 1)
     c_press     압력 6개 중 하나라도 > 0
     d_press_rot c 이고 STAGE_ROTATION > 0
     e_ret_high  RETAINER_RING_PRESSURE > 3,000 (약 7,135 단계, hyp_r2r_retainer.py의 경계)
   빈칸 = dt > GAP_S(2초, 명목 기록 간격 1초의 2배). 제외 값은 빈칸 간격을 0으로 센다.
   탐색 확인 (train 정상 웨이퍼): dt는 0.33~1.33초에 몰리고 3초 근처에 두 번째 봉우리가 있다(샘플 누락).
   첫 챔버 → 둘째 챔버 전환마다 약 17초 빈칸이 웨이퍼당 1개 있다. 이 빈칸은 앞 행(첫 챔버)에 붙는다.
2. 그룹별 회귀 log(MRR) = α + β·log(T) + γ·t_start + δ·소모품 6개 시작값(start_{U}_w).
   β의 95% 신뢰구간은 웨이퍼 쌍 부트스트랩 N_BOOT회 백분위수. 그룹마다 같은 재표본을 10가지 정의에 쓴다.
   참고로 통제 없는 β(log-log 단순회귀)도 같은 방식으로 낸다. 판정에는 쓰지 않는다.
3. 웨이퍼 간 변동계수(CV): 각 정의의 CV와 MRR CV. 판별력 규칙(결과 보기 전에 고정):
   CV_T / CV_MRR < POWER_RATIO(0.25)이면 "판별력 없음". 가설 2라면 log T의 변동이 log MRR에 그대로 실리는데,
   그 몫이 MRR 변동의 1/4 미만(분산 1/16 미만)이면 다른 요인에 묻힌다.
4. 판정 규칙(결과 보기 전에 고정, 통제 있는 β의 CI로만):
   CI가 0을 포함하고 1을 포함하지 않음 → 가설 1 지지 / 1을 포함하고 0을 포함하지 않음 → 가설 2 지지 /
   둘 다 포함 또는 둘 다 제외 → 판단 불가.
5. MRR 대 연마 시간 산점도(로그 축), 기울기 0과 1 기준선(두 변수의 기하평균 점을 지남).

대상: train·val 라벨 웨이퍼. 이상치(MRR > 400), 전체-0 센서 웨이퍼, flag_incomplete_chamber, flag_long_trace는 뺀다
(456 그룹에는 이상치·전체-0이 없고, 플래그 웨이퍼는 train 6장). T = 0인 웨이퍼는 그 정의에서만 빠진다.
라벨: train은 집계표, val은 `data/raw/labels/`에서 읽어 집계표와 대조한 뒤 쓴다.
test 센서·라벨과 `data/locked/`는 어떤 경로로도 읽지 않는다.

실행: python src/hyp_mrr_time.py
출력: results/diagnostics/2026-10-14_김혜연_mrr_time_beta.csv   β·CI·CV·판정 (+ .yaml)
      results/diagnostics/2026-10-14_김혜연_polish_time_wafer.csv 웨이퍼별 연마 시간 10가지 (라벨 없음, + .yaml)
      docs/reports/figs/김혜연_mrr_time_scatter_gapout.png, 김혜연_mrr_time_scatter_gapin.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, dedup  # noqa: E402
from hyp_r2r_retainer import style  # noqa: E402
from load import KEY, ROOT, load_labels, load_sensors  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from pv_partial_consumables import START6  # noqa: E402
from scatter_pv import korean_font  # noqa: E402

DATE = "2026-10-14"
SEED = 42
DIAG = ROOT / "results" / "diagnostics"
OUT = DIAG / f"{DATE}_{AUTHOR}_mrr_time_beta.csv"
OUT_W = DIAG / f"{DATE}_{AUTHOR}_polish_time_wafer.csv"
FIGS = ROOT / "docs" / "reports" / "figs"
FIG = {"out": FIGS / f"{AUTHOR}_mrr_time_scatter_gapout.png",
       "in": FIGS / f"{AUTHOR}_mrr_time_scatter_gapin.png"}
GROUPS = ["A-456", "B-456", "A-123"]
MAIN_CHAMBER = {"456": 4, "123": 1}
RET = "RETAINER_RING_PRESSURE"
RET_HIGH = 3000
GAP_S = 2.0
N_BOOT = 2000
POWER_RATIO = 0.25
CONTROLS = ["t_start"] + START6
DEFS = {"a_total": "웨이퍼 전체 기록", "b_chamber": "주 연마 챔버(4 / 1) 체류",
        "c_press": "압력 > 0", "d_press_rot": "압력 > 0 이고 STAGE 회전 on",
        "e_ret_high": "리테이너 링 고압 단계(> 3,000)"}
GAPS = {"in": "빈칸 포함", "out": "빈칸 제외"}


def polish_times(raw: pd.DataFrame) -> pd.DataFrame:
    """웨이퍼별 연마 시간 10가지 (T_{정의}_{in|out}, 초)."""
    raw = raw.sort_values(KEY + ["TIMESTAMP"], kind="stable").reset_index(drop=True)
    dt = (-raw.groupby(KEY)["TIMESTAMP"].diff(-1)).fillna(0.0)
    gap = dt > GAP_S
    press = (raw[list(PRESSURE)] > 0).any(axis=1)
    main = raw["group"].str[-3:].map(MAIN_CHAMBER)
    masks = {"a_total": pd.Series(True, index=raw.index),
             "b_chamber": raw["CHAMBER"] == main,
             "c_press": press,
             "d_press_rot": press & (raw["STAGE_ROTATION"] > 0),
             "e_ret_high": raw[RET] > RET_HIGH}
    cols = {}
    for name, m in masks.items():
        cols[f"T_{name}_in"] = dt.where(m, 0.0)
        cols[f"T_{name}_out"] = dt.where(m & ~gap, 0.0)
    cols["n_gap"] = gap.astype(int)
    cols["gap_time"] = dt.where(gap, 0.0)
    out = pd.DataFrame(cols).astype(float)
    return pd.concat([raw[KEY], out], axis=1).groupby(KEY).sum().reset_index()


def ols_beta(y: np.ndarray, X: np.ndarray) -> float:
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def beta_ci(y: np.ndarray, X: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """β(X의 1번 열)와 쌍 부트스트랩 백분위수 95% CI."""
    bs = np.array([ols_beta(y[i], X[i]) for i in idx])
    return ols_beta(y, X), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def verdict(lo: float, hi: float) -> str:
    has0, has1 = lo <= 0 <= hi, lo <= 1 <= hi
    if has0 and not has1:
        return "가설 1 지지"
    if has1 and not has0:
        return "가설 2 지지"
    return "판단 불가"


def draw(d: pd.DataFrame, gap: str) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(17, 10.4))
    fig.subplots_adjust(top=0.89, hspace=0.42, wspace=0.28)
    for i, g in enumerate(GROUPS):
        dg = d[d["group"] == g]
        for j, name in enumerate(DEFS):
            ax = axes[i, j]
            col = f"T_{name}_{gap}"
            s = dg[dg[col] > 0]
            ax.scatter(s[col], s["MRR"], s=5, color="#2a78d6", alpha=0.45, lw=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if len(s) > 1:
                x0, y0 = np.exp(np.log(s[col]).mean()), np.exp(np.log(s["MRR"]).mean())
                xs = np.geomspace(s[col].min(), s[col].max(), 50)
                ax.plot(xs, np.full_like(xs, y0), color="#555555", lw=1.2, ls="--", label="기울기 0 (가설 1)")
                ax.plot(xs, y0 * xs / x0, color="#d6602a", lw=1.2, label="기울기 1 (가설 2)")
                ax.set_ylim(s["MRR"].min() / 1.03, s["MRR"].max() * 1.03)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(subs=(1, 1.5, 2, 3, 5, 7)))
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_locator(NullLocator())
            ax.set_title(f"{g} · {name} (n={len(s):,})", fontsize=9.5, color="#333333")
            if i == 2:
                ax.set_xlabel(f"{DEFS[name]} (초, 로그)", fontsize=8.5, color="#555555")
            if j == 0:
                ax.set_ylabel("MRR (로그)", fontsize=9, color="#555555")
            style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.text(0.02, 0.975, f"MRR 대 연마 시간 ({GAPS[gap]}) — 점선 = 기울기 0, 주황 = 기울기 1",
             fontsize=13, color="#222222", va="top")
    fig.text(0.02, 0.94, "train·val 라벨 웨이퍼 (이상치·전체-0 센서·불완전 챔버·긴 트레이스 제외). "
             "기준선은 두 변수의 기하평균 점을 지난다. 통제 변수 없는 원자료",
             fontsize=9.5, color="#777777", va="top")
    fig.savefig(FIG[gap], dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for p in (OUT, OUT_W, *FIG.values()):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    val = load_labels("val").rename(columns={"AVG_REMOVAL_RATE": "MRR_raw"})
    chk = tab[tab["split"] == "val"].merge(val, on=KEY)
    if len(chk) != len(val) or not np.allclose(chk["MRR"], chk["MRR_raw"]):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")
    lab = tab[tab["split"].isin(["train", "val"])]
    keep = ((lab["is_outlier"] == 0) & (lab["n_allzero_sensors"] == 0)
            & (lab["flag_incomplete_chamber"] == 0) & (lab["flag_long_trace"] == 0))
    print(f"라벨 웨이퍼 {len(lab):,}장 중 대상 {int(keep.sum()):,}장", flush=True)
    lab = lab[keep]

    frames = []
    for split in ("train", "val"):
        df, n_dup = dedup(load_sensors(split))
        frames.append(df.merge(lab[KEY + ["group"]], on=KEY))
        print(f"  {split}: 중복 {n_dup:,}행 제거", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    times = polish_times(raw)
    d = lab[KEY + ["split", "group", "t_start", "t_end", "MRR"] + START6].merge(times, on=KEY, how="inner")
    if len(d) != len(lab):
        raise ValueError(f"센서가 있는 웨이퍼 {len(d)}장 (기대 {len(lab)}장)")
    # 검산: 빈칸 포함 전체 기록 = t_end − t_start
    err = (d["T_a_total_in"] - (d["t_end"] - d["t_start"])).abs().max()
    if err > 1e-6:
        raise ValueError(f"T_a_total_in 과 t_end − t_start 가 다르다 (최대 {err})")

    rows = []
    for g in GROUPS:
        dg = d[d["group"] == g].reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        idx_all = rng.integers(0, len(dg), size=(N_BOOT, len(dg)))
        cv_mrr = dg["MRR"].std() / dg["MRR"].mean()
        ctrl = dg[CONTROLS].to_numpy(float)
        ctrl = (ctrl - ctrl.mean(0)) / ctrl.std(0)
        for name in DEFS:
            for gap in GAPS:
                col = f"T_{name}_{gap}"
                ok = (dg[col] > 0).to_numpy()
                t = dg.loc[ok, col].to_numpy(float)
                y = np.log(dg.loc[ok, "MRR"].to_numpy(float))
                one = np.ones((len(t), 1))
                X = np.column_stack([one, np.log(t), ctrl[ok]])
                X0 = np.column_stack([one, np.log(t)])
                # 부트스트랩 재표본은 그룹 전체 기준. T = 0 웨이퍼가 빠진 정의는 남은 웨이퍼로 번호를 다시 매긴다
                pos = np.full(len(dg), -1)
                pos[ok] = np.arange(ok.sum())
                idx = [pos[i][pos[i] >= 0] for i in idx_all] if not ok.all() else idx_all
                b, lo, hi = beta_ci(y, X, idx)
                b0, lo0, hi0 = beta_ci(y, X0, idx)
                cv_t = t.std(ddof=1) / t.mean()
                ratio = cv_t / cv_mrr
                power = ratio >= POWER_RATIO
                rows.append({
                    "group": g, "definition": name, "gap": gap, "n": int(ok.sum()), "n_zero_time": int((~ok).sum()),
                    "time_median_s": float(np.median(t)), "cv_time": cv_t, "cv_mrr": cv_mrr, "cv_ratio": ratio,
                    "power": "있음" if power else "없음",
                    "beta": b, "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo, "verdict": verdict(lo, hi),
                    "beta_nocontrol": b0, "ci_lo_nocontrol": lo0, "ci_hi_nocontrol": hi0})
        print(f"  {g}: {len(dg):,}장 완료", flush=True)
    res = pd.DataFrame(rows)

    font = korean_font()
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    FIGS.mkdir(parents=True, exist_ok=True)
    for gap in GAPS:
        draw(d, gap)

    DIAG.mkdir(parents=True, exist_ok=True)
    num = res.select_dtypes("number").columns
    res[num] = res[num].round(4)
    res.to_csv(OUT, index=False)
    tcols = [c for c in times.columns if c not in KEY]
    d[KEY + ["split", "group"] + tcols].round(3).to_csv(OUT_W, index=False)

    common = [
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/hyp_mrr_time.py",
        "question: MRR 이 실제 연마 시간으로 나눈 값(가설 1, β≈0)인가 고정 공정 시간으로 나눈 값(가설 2, β≈1)인가",
        "input: data/processed/wafer_table.csv + data/raw/{train,val} 센서 + "
        "data/raw/labels/CMP-validation-removalrate.csv (집계표 val MRR과 대조 후 사용)",
        "wafers: train·val 라벨 웨이퍼 중 is_outlier=0, n_allzero_sensors=0, flag_incomplete_chamber=0, flag_long_trace=0",
        "locked: data/locked/, test 센서·라벨은 열지 않았다",
        "time: 행 i 의 시간 = 다음 기록까지 dt (마지막 행 0). 정의별로 조건 만족 행의 dt 합",
        f"definitions: {DEFS}",
        f"main_chamber: {MAIN_CHAMBER}", f"ret_high: {RET} > {RET_HIGH}",
        f"gap: dt > {GAP_S} 초. in = 빈칸 포함, out = 빈칸 간격을 0으로",
    ]
    OUT.with_suffix(".yaml").write_text("\n".join(common + [
        "model: log(MRR) = a + beta*log(T) + t_start + start_{U}_w 6개 (통제 변수는 z-score). OLS",
        f"controls: {CONTROLS}",
        f"ci: 웨이퍼 쌍 부트스트랩 {N_BOOT}회 백분위수 95%, seed {SEED}, 그룹마다 같은 재표본을 모든 정의에 사용",
        "nocontrol: beta_nocontrol = log-log 단순회귀 (참고용, 판정에 쓰지 않음)",
        f"power: cv_time / cv_mrr >= {POWER_RATIO} 이면 있음 (결과 보기 전 고정)",
        "verdict: CI가 0 포함·1 미포함 = 가설 1 지지, 1 포함·0 미포함 = 가설 2 지지, 그 밖 = 판단 불가 (결과 보기 전 고정)",
        "n_zero_time: 그 정의의 T 가 0 이라 로그를 못 써서 빠진 웨이퍼 수",
        f"figs: {FIG['out'].relative_to(ROOT).as_posix()}, {FIG['in'].relative_to(ROOT).as_posix()}",
        "note: 진단 결과 (14.4). MRR 은 회귀 계산에만 사용", "",
    ]), encoding="utf-8")
    OUT_W.with_suffix(".yaml").write_text("\n".join(common + [
        "content: 웨이퍼별 T_{정의}_{in|out} (초), n_gap (빈칸 수), gap_time (빈칸 합, 초). 라벨 없음",
        "note: 진단 결과 (14.4)", "",
    ]), encoding="utf-8")

    pd.set_option("display.width", 250)
    print("\n" + res.drop(columns=["ci_width"]).to_string(index=False))
    pw = res[res["power"] == "있음"]
    vs = sorted(pw["verdict"].unique())
    print(f"\n판별력 있는 조합 {len(pw)}개의 판정: {pw['verdict'].value_counts().to_dict()}"
          f" → {'일치' if len(vs) == 1 else '불일치'}")
    for p in (OUT, OUT_W, *FIG.values()):
        print(f"저장: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""MRR 변동을 "연마 시간 효과"(목표 두께에 닿으면 멈춤)로 얼마나 설명하는가 (탐색 진단, CLAUDE.md 14.4).

판정용이 아니라 다음 연구 방향을 정하기 위한 확인이다. SR 결과를 본 뒤 설계한 탐색 분석이다.

장비가 목표 두께에서 멈추면 연마량(MRR × T)이 웨이퍼마다 거의 같고, MRR은 시간에 반비례한다.
  → 연마량 CV ≪ MRR CV, log(MRR) = a + b·log(T)의 b ≈ −1.
시간과 무관하면 b ≈ 0이고, 연마량 CV는 MRR CV와 시간 CV를 합친 만큼 커진다.

시간 정의 (T, 초 또는 행):
  dur_s1   dur_POLISH_s1: 첫 챔버(4 / 1) 체류 = 구간 마지막 − 첫 TIMESTAMP (features_v1.csv 그대로)
  n_press  압력 6개 중 하나라도 > 0인 행 수, 웨이퍼 전체 (원본 센서, 완전 중복 제거 후. ch56_step과 같은 정의)
  dur_w    웨이퍼 전체 길이 = t_end − t_start (wafer_table.csv)

계산 (부분집단마다):
  1. MRR과 T의 스피어만·피어슨 상관 (95% CI = Fisher z, 정규근사)
  2. CV(표본 표준편차 / 평균): MRR, T, 연마량 = MRR × T. 비율 = CV(연마량) / CV(MRR)
  3. log(MRR) = a + b·log(T) OLS. b의 95% CI = 고전적 표준오차 × t(n−2) 분위수, R²
  4. 456 그룹은 ch56_step(0 = 약 150초 무리, 1 = 약 285초 무리)별로 2·3을 따로
  5. (참고) 3의 잔차와 소모품 카운터(start_·delta_ 6종)·리셋 후 경과량(since_ 5종)의 스피어만 상관, |ρ| 상위 10개.
     since_POLISH_TABLE_w는 리셋 누락 오류가 있는 열이라(CLAUDE.md 5.5) 계산은 하되 표시한다.

부분집단: 그룹 3개(A-456, B-456, A-123) 전체, A-123 센서 정상 / 전체-0, 456 그룹 ch56_step 0 / 1.
사후 추가 (첫 실행 결과를 본 뒤 추가, 탐색용):
  - trim = p01_p99: 부분집단·시간 정의마다 T의 1~99 백분위수 밖 웨이퍼를 뺀 결과. 456 그룹 시간 분포의 꼬리
    (예: B-456 dur_POLISH_s1 최대 5,283초, A-456 전체 길이 최대 29,311초)가 CV·피어슨·OLS를 지배해서 추가했다.
  - loglog_ctrl: log(MRR) = a + b·log(T) + t_start + start_{U}_w 6개(z-score). 시간 흐름·소모품 상태와 겹치는 몫을
    빼고도 시간 기울기가 남는지 본다 (2026-10-14 mrr_time_beta와 같은 통제 변수). r2_ctrl_only = 통제만의 R².
민감도: 그룹 전체에서 flag_long_trace·flag_incomplete_chamber 웨이퍼를 뺀 행도 낸다(dur_w가 긴 트레이스에 크게 흔들림).
T ≤ 0인 웨이퍼는 그 정의에서만 빠진다 (n_zero_time).

대상: train+val, 이상치 4장(MRR > 400) 제외 = features_v1.csv 2,401행.
라벨: data/raw/labels/의 train·val 파일만 읽는다. test 센서·라벨과 data/locked/는 어떤 경로로도 읽지 않는다.

실행: python src/diagnostics/mrr_time_check.py
CI는 고전 OLS 표준오차라 웨이퍼 간 독립을 가정한다 (같은 WAFER_ID의 A·B 짝, 시간 자기상관은 반영 안 함).

출력: results/diagnostics/2026-11-03_김혜연_mrr_time_check.csv (+ .yaml). 이미 있으면 멈춘다 (덮어쓰기 금지).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate import PRESSURE, dedup  # noqa: E402
from load import KEY, ROOT, load_labels, load_sensors  # noqa: E402

DATE = "2026-11-03"
AUTHOR = "김혜연"
SEED = 42                       # 난수를 쓰지 않지만 규칙대로 기록
PROC = ROOT / "data" / "processed"
FEAT = PROC / "features_v1.csv"
TABLE = PROC / "wafer_table.csv"
OUT = ROOT / "results" / "diagnostics" / f"{DATE}_{AUTHOR}_mrr_time_check.csv"
SPLITS = ("train", "val")
N_EXPECTED = 2401
OUTLIER = 400
TIME_DEFS = {
    "dur_s1": "dur_POLISH_s1 (첫 챔버 체류, 초)",
    "n_press": "압력 > 0 행 수 (웨이퍼 전체)",
    "dur_w": "웨이퍼 전체 길이 t_end − t_start (초)",
}
USAGE6 = ["BACKING_FILM", "DRESSER", "POLISH_TABLE", "DRESSER_TABLE", "MEMBRANE", "PSHEET"]
COUNTERS = ([f"start_{u}_w" for u in USAGE6] + [f"delta_{u}_w" for u in USAGE6]
            + [f"since_{u}_w" for u in USAGE6 if u != "DRESSER_TABLE"])
KNOWN_BAD = {"since_POLISH_TABLE_w"}    # 리셋 누락 오류 (PR #23, #24)
TOP = 10
# 아래 둘은 첫 실행 결과를 본 뒤 추가했다 (시간 분포 꼬리가 CV·OLS를 지배해서)
TRIMS = ("none", "p01_p99")
TRIM_Q = 0.01
CONTROLS = ["t_start"] + [f"start_{u}_w" for u in USAGE6]


def t975(df: int) -> float:
    """t 분포 0.975 분위수 (Cornish-Fisher 전개, df > 100에서 오차 < 1e-4)."""
    z = 1.959963985
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return math.tanh(z - 1.959963985 * se), math.tanh(z + 1.959963985 * se)


def spearman(x: pd.Series, y: pd.Series) -> float:
    """순위(평균 순위) 변환 후 피어슨. scipy 없이 pandas만 쓴다."""
    return float(x.rank().corr(y.rank()))


def cv(x: pd.Series) -> float:
    return float(x.std(ddof=1) / x.mean())


def loglog(mrr: np.ndarray, t: np.ndarray) -> dict:
    """log(MRR) = a + b·log(T) OLS. b, 95% CI, R², 잔차."""
    x, y = np.log(t), np.log(mrr)
    n = len(x)
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc**2).sum())
    a = float(y.mean() - b * x.mean())
    resid = y - a - b * x
    s2 = (resid**2).sum() / (n - 2)
    se = math.sqrt(s2 / (xc**2).sum())
    q = t975(n - 2)
    r2 = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
    return {"b": b, "lo": b - q * se, "hi": b + q * se, "se": se, "r2": float(r2), "resid": resid}


def press_rows(keys: pd.DataFrame) -> pd.DataFrame:
    """원본 센서(train·val, 완전 중복 제거)로 웨이퍼별 압력 > 0 행 수."""
    parts = []
    for split in SPLITS:
        d, _ = dedup(load_sensors(split))
        parts.append(d[KEY + list(PRESSURE)])
    raw = pd.concat(parts, ignore_index=True)
    n = (raw[list(PRESSURE)] > 0).any(axis=1).groupby([raw[k] for k in KEY]).sum().rename("n_press").reset_index()
    out = keys.merge(n, on=KEY, how="left", validate="1:1")
    if out["n_press"].isna().any():
        raise SystemExit("원본 센서에 없는 키가 있다")
    return out


def build() -> pd.DataFrame:
    f = pd.read_csv(FEAT, usecols=KEY + ["group", "dur_POLISH_s1", "ch56_step", "allzero"])
    if len(f) != N_EXPECTED:
        raise SystemExit(f"features_v1 {len(f)}행 (기대 {N_EXPECTED})")
    tab = pd.read_csv(TABLE, usecols=KEY + ["split", "t_start", "t_end", "flag_long_trace",
                                            "flag_incomplete_chamber"] + COUNTERS)   # 집계표의 MRR 열은 읽지 않음
    lab = pd.concat([load_labels(s) for s in SPLITS], ignore_index=True).rename(columns={"AVG_REMOVAL_RATE": "MRR"})
    d = (f.merge(tab, on=KEY, how="left", validate="1:1")
          .merge(lab, on=KEY, how="left", validate="1:1"))
    if d["split"].isna().any() or not d["split"].isin(SPLITS).all() or d["MRR"].isna().any():
        raise SystemExit("features_v1 키가 집계표·라벨의 train·val 행과 맞지 않는다")
    if (d["MRR"] > OUTLIER).any():
        raise SystemExit("이상치가 남아 있다")
    d = d.merge(press_rows(d[KEY]), on=KEY, how="left", validate="1:1")
    d["dur_s1"] = d["dur_POLISH_s1"]
    d["dur_w"] = d["t_end"] - d["t_start"]
    # ch56_step 재확인: features_v1의 경계 정의와 같아야 한다
    bound = d["group"].map({"A-456": 245, "B-456": 255})
    chk = np.where(bound.isna(), np.nan, (d["n_press"] >= bound).astype(float))
    if not np.array_equal(np.nan_to_num(chk, nan=-1), d["ch56_step"].fillna(-1).to_numpy()):
        raise SystemExit("압력 > 0 행 수로 다시 만든 ch56_step이 features_v1과 다르다")
    d["flagged"] = ((d["flag_long_trace"] == 1) | (d["flag_incomplete_chamber"] == 1)).astype(int)
    return d


def subsets(d: pd.DataFrame):
    for g in ("A-456", "B-456", "A-123"):
        dg = d[d["group"] == g]
        yield g, "전체", dg
        if dg["flagged"].any():
            yield g, "전체(플래그 제외)", dg[dg["flagged"] == 0]
        if g == "A-123":
            yield g, "센서 정상", dg[dg["allzero"] == 0]
            yield g, "전체-0", dg[dg["allzero"] == 1]
        else:
            yield g, "ch56_step=0 (짧은 무리)", dg[dg["ch56_step"] == 0]
            yield g, "ch56_step=1 (긴 무리)", dg[dg["ch56_step"] == 1]


def row(section, g, sub, trim, tdef, n, nz, stat, value, lo=np.nan, hi=np.nan, variable="", note=""):
    return {"section": section, "group": g, "subset": sub, "trim": trim, "time_def": tdef, "variable": variable,
            "n": n, "n_zero_time": nz, "stat": stat, "value": value, "ci_lo": lo, "ci_hi": hi, "note": note}


def ctrl_fit(sk: pd.DataFrame, tdef: str) -> dict:
    """(사후 추가) log(MRR) = a + b·log(T) + t_start + start_{U}_w 6개(z-score). b의 95% CI, R², 통제만의 R²."""
    y = np.log(sk["MRR"].to_numpy(float))
    z = sk[CONTROLS].to_numpy(float)
    z = (z - z.mean(0)) / z.std(0)
    out = {}
    for name, X in (("ctrl_only", z), ("full", np.column_stack([np.log(sk[tdef].to_numpy(float)), z]))):
        X1 = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ beta
        out[f"r2_{name}"] = float(1 - (resid**2).sum() / ((y - y.mean())**2).sum())
        if name == "full":
            dof = len(y) - X1.shape[1]
            cov = (resid**2).sum() / dof * np.linalg.pinv(X1.T @ X1)
            se, q = math.sqrt(cov[1, 1]), t975(dof)
            out.update(b=float(beta[1]), lo=float(beta[1] - q * se), hi=float(beta[1] + q * se))
    return out


def analyze(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, sub, s in subsets(d):
        rows.append(row("cv", g, sub, "none", "", len(s), 0, "cv_mrr", cv(s["MRR"])))
        rows.append(row("cv", g, sub, "none", "", len(s), 0, "mean_mrr", float(s["MRR"].mean())))
        for tdef in TIME_DEFS:
            ok = s[tdef] > 0
            nz = int((~ok).sum())
            for trim in TRIMS:
                sk = s[ok]
                if trim != "none":
                    lo, hi = sk[tdef].quantile([TRIM_Q, 1 - TRIM_Q])
                    sk = sk[(sk[tdef] >= lo) & (sk[tdef] <= hi)]
                n = len(sk)
                if n < 10:
                    rows.append(row("skip", g, sub, trim, tdef, n, nz, "n<10", np.nan))
                    continue
                mrr, t = sk["MRR"], sk[tdef]
                for meth, r in (("spearman", spearman(mrr, t)), ("pearson", float(mrr.corr(t)))):
                    rows.append(row("corr", g, sub, trim, tdef, n, nz, meth, r, *fisher_ci(r, n)))
                removal = mrr * t
                cm, ct, cr = cv(mrr), cv(t), cv(removal)
                rows += [row("cv", g, sub, trim, tdef, n, nz, "median_time", float(t.median())),
                         row("cv", g, sub, trim, tdef, n, nz, "cv_mrr", cm),
                         row("cv", g, sub, trim, tdef, n, nz, "cv_time", ct),
                         row("cv", g, sub, trim, tdef, n, nz, "cv_removal", cr),
                         row("cv", g, sub, trim, tdef, n, nz, "cv_ratio_removal_to_mrr", cr / cm)]
                if t.nunique() < 3:
                    rows.append(row("loglog", g, sub, trim, tdef, n, nz, "b", np.nan, note="시간 값 종류 < 3"))
                    continue
                fit = loglog(mrr.to_numpy(float), t.to_numpy(float))
                cf = ctrl_fit(sk, tdef)
                post = "사후 추가 (결과를 본 뒤)"
                rows += [row("loglog", g, sub, trim, tdef, n, nz, "b", fit["b"], fit["lo"], fit["hi"]),
                         row("loglog", g, sub, trim, tdef, n, nz, "r2", fit["r2"]),
                         row("loglog_ctrl", g, sub, trim, tdef, n, nz, "b_ctrl", cf["b"], cf["lo"], cf["hi"], note=post),
                         row("loglog_ctrl", g, sub, trim, tdef, n, nz, "r2_ctrl_only", cf["r2_ctrl_only"], note=post),
                         row("loglog_ctrl", g, sub, trim, tdef, n, nz, "r2_ctrl_full", cf["r2_full"], note=post)]
                if sub != "전체":
                    continue
                # 5. 잔차와 소모품 카운터 (참고)
                res = pd.Series(fit["resid"], index=sk.index)
                cors = []
                for c in COUNTERS:
                    m = sk[c].notna()
                    if m.sum() < 10 or sk.loc[m, c].nunique() < 2:
                        continue
                    cors.append((c, spearman(res[m], sk.loc[m, c]), int(m.sum())))
                cors.sort(key=lambda x: -abs(x[1]))
                for rank, (c, r, nn) in enumerate(cors[:TOP], 1):
                    rows.append(row("resid_corr", g, sub, trim, tdef, nn, nz, f"spearman_rank{rank}", r,
                                    *fisher_ci(r, nn), variable=c,
                                    note="리셋 누락 오류 열 (CLAUDE.md 5.5)" if c in KNOWN_BAD else ""))
    return pd.DataFrame(rows)


def main() -> None:
    for p in (OUT, OUT.with_suffix(".yaml")):
        if p.exists():
            raise SystemExit(f"{p} 가 이미 있다 (덮어쓰기 금지)")
    d = build()
    print(f"웨이퍼 {len(d)}장: " + ", ".join(f"{g} {n}" for g, n in d["group"].value_counts().items()), flush=True)
    res = analyze(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, float_format="%.6g", encoding="utf-8")
    OUT.with_suffix(".yaml").write_text("\n".join([
        f"date: {DATE}",
        f"author: {AUTHOR}",
        "script: src/diagnostics/mrr_time_check.py",
        "purpose: 탐색 분석 (판정용 아님). SR 결과를 본 뒤 설계. 다음 연구 방향을 정하기 위한 확인",
        "question: MRR 변동이 연마 시간 효과(목표 두께에서 멈춤)로 얼마나 설명되는가",
        "input: data/processed/features_v1.csv (키·group·dur_POLISH_s1·ch56_step·allzero) + data/processed/wafer_table.csv "
        "(t_start·t_end·플래그·start_/delta_/since_ 카운터, MRR 열은 읽지 않음) + data/raw/{train,val} 센서 + "
        "data/raw/labels/CMP-{training,validation}-removalrate.csv",
        "locked: data/locked/, test 센서·라벨은 열지 않았다",
        f"wafers: train+val, 이상치 4장(MRR > {OUTLIER}) 제외 = features_v1 {N_EXPECTED}행",
        f"time_defs: {TIME_DEFS}",
        "n_press: 압력 6개 중 하나라도 > 0인 행 수, 완전 중복 행 제거 후 (ch56_step과 같은 정의, 재계산해 일치 확인)",
        "removal: 추정 연마량 = MRR × T (T의 단위 그대로)",
        "cv: 표본 표준편차(ddof=1) / 평균. cv_ratio_removal_to_mrr = CV(연마량) / CV(MRR)",
        "loglog: log(MRR) = a + b*log(T), OLS. CI = 고전 표준오차 × t(n-2) 0.975 분위수 (Cornish-Fisher 근사). "
        "b≈-1 연마량 일정, b≈0 시간과 무관",
        "corr_ci: Fisher z 정규근사 95%",
        "subsets: 그룹 전체 / 전체(플래그 제외: flag_long_trace·flag_incomplete_chamber) / A-123 센서 정상·전체-0 / "
        "456 ch56_step 0·1",
        f"resid_corr: 그룹 '전체' 회귀 잔차와 카운터 {len(COUNTERS)}열의 스피어만 상관, |rho| 상위 {TOP}. "
        "since_ NaN(첫 리셋 이전)은 그 열에서만 뺌. since_POLISH_TABLE_w는 리셋 누락 오류 열 (note 표시). 참고용",
        f"post_hoc: 첫 실행 결과를 본 뒤 추가. trim=p01_p99 (부분집단·정의마다 T의 {TRIM_Q:.0%}~{1-TRIM_Q:.0%} 백분위수 밖 제외), "
        f"loglog_ctrl (통제 변수 {CONTROLS}, z-score, 2026-10-14 mrr_time_beta와 같은 통제)",
        "ci_caveat: 고전 OLS CI는 웨이퍼 독립 가정. 같은 WAFER_ID의 A·B 짝·시간 자기상관 미반영 → CI가 좁게 나올 수 있음",
        "n_zero_time: T <= 0이라 그 정의에서 빠진 웨이퍼 수",
        f"seed: {SEED} (난수 사용 없음)",
        "note: 진단 결과 (14.4). 판정·모델 입력에 쓰지 않는다", "",
    ]), encoding="utf-8")
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()

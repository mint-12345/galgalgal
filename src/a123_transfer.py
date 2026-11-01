"""A-123 전이학습 (preregistration.md 5절) 주 평가 구현.

범위 (5.7-17 "필수"): folds_v1 25 fold 주 평가, 고정주기 공개, 공개 2/4/8/16/100%,
T0~T3, 소스 S1(A-456)·S2(A-456+B-456), f_456 full·usage, 8% 주·2차 판정, 90% 커버리지, ρ 분포.
(전진 검증·무작위 공개·δ XGBoost·라벨 피처 실험은 이 파일 범위 밖)

규칙 대응 (절 번호는 preregistration.md):
- 분할: 같은 repeat 안에서 eval = fold==f, train = fold!=f, r = repeat*5 + fold (5.7-1)
- 공개: data/processed/reveal_masks_v1.csv (design=folds, method=fixed) (5.3, 5.7-7)
- T0 공개 라벨 평균 / T1 GP 단독 / T2 f_456 + c / T3 ρ·f_456 + b + δGP (5.4)
- ρ·b 최소제곱 → ρ를 [0, 2]로 자르면 b만 다시 추정 → 잔차에 δ GP (5.7-9)
- f_456 (5.4, 5.7-11): XGBoost 고정 설정(3.2), n_jobs=2 고정, 소스별 456 전체로 1회 학습, 결측은 NaN 그대로 (#27).
  입력 full = features_v1 피처에서 EXCLUDE_FULL을 뺀 112개, usage = 소모품 카운터 36개
- δ 입력: #26 ①~⑤, 5.7-8 / 커널·GP 설정: #28, 5.4, 5.7-14
- GP 경고는 숨기지 않고 fold·모델별로 센다 (5.7-15)
- 보정 t-검정 (5.5, 5.7-16): cv_common.corrected_ttest, n_train = A-123 학습 쪽 전체(full)로 판정.
  공개 라벨 수(revealed) 기준은 cv_common.ttest_ntrain_sensitivity로 병기만 한다
- 판정 (5.5): 공개 8%, f_456 full, S1·S2 둘 다 (평균 차이 < 0, p < 0.05)일 때만 "효과 있음" (주·2차 동일)
- 결과 파일: results/{날짜}_{이름}_FV1_a123_transfer_{내용}.csv + 같은 이름 .yaml (CLAUDE.md 12·14절)
- test 데이터(data/locked 등)는 읽지 않는다 (1절)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
import xgboost  # 없으면 여기서 멈춘다. 다른 모델로 대체하지 않음 (3.2 고정 설정)
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
P_FEATURES = DATA / "features_v1.csv"
P_FAMILIES = DATA / "feature_families_v1.csv"
P_FOLDS = DATA / "folds_v1.csv"
P_MASKS = DATA / "reveal_masks_v1.csv"
P_TABLE = DATA / "wafer_table.csv"
RESULTS = ROOT / "results"
AUTHOR = "김혜연"
TAG = "FV1"
FEATURESET = "features_v1"
TARGET_GROUP = "A-123"
KEY = ["WAFER_ID", "STAGE"]

# f_456 full 입력에서 빼는 피처 (5.4): 리셋 누락으로 값이 틀림 (#24). 수정본은 features_v2에서.
EXCLUDE_FULL = ["since_POLISH_TABLE_w"]
N_INPUT = {"full": 112, "usage": 36}

RATIOS = [2, 4, 8, 16, 100]
PRIMARY_RATIO = 8
PRIMARY_MODE = "full"          # usage는 보고만 (5.5)
SOURCES = {"S1": ["A-456"], "S2": ["A-456", "B-456"]}
MODES = ["full", "usage"]
USAGE_FAMILY = "소모품 카운터"
DELTA_SINGLES = ["t_start", "mean_POLISH_TABLE_s1", "mean_DRESSER_s1", "mean_MEMBRANE_s1",
                 "mean_BACKING_FILM_s1", "mean_PSHEET_s1", "mean_DRESSER_TABLE_s1"]
DELTA_PAIR = ["P1_on", "allzero"]
CORR_SKIP = 0.9
P1_MIN_CLEAN = 10
ISO_BELOW = 10
ALPHA = 0.05
Z90 = stats.norm.ppf(0.95)
SEED = 42
# xgboost는 subsample < 1이면 스레드 수에 따라 결과가 달라진다 (#23). 3.2 설정에 n_jobs만 고정해 덧붙인다.
XGB_N_JOBS = 2
XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, random_state=SEED, n_jobs=XGB_N_JOBS)
SEED_RULE = "f_456 random_state=42 / GP random_state=42+r (r = repeat*5+fold, 재시작 5회도 이 시드에서 시작)"
OUT_CONTENTS = ["folds", "stats", "judgment", "rho"]

for p in [P_FEATURES, P_FAMILIES, P_FOLDS, P_MASKS, P_TABLE]:
    assert "locked" not in str(p).lower() and "test" not in p.name.lower(), f"test 경로 금지: {p}"


# ---------------------------------------------------------------- 데이터
def input_columns(feat_columns):
    """f_456 입력 (5.4): full = 계열표 순서의 features_v1 피처 − EXCLUDE_FULL, usage = 소모품 카운터 계열."""
    fam = pd.read_csv(P_FAMILIES)
    full = [c for c in fam["feature"] if c in feat_columns and c not in EXCLUDE_FULL]
    usage = [c for c in fam.loc[fam["family"] == USAGE_FAMILY, "feature"] if c in feat_columns]
    cols = {"full": full, "usage": usage}
    for m, n in N_INPUT.items():
        assert len(cols[m]) == n, f"f_456 {m} 입력 {len(cols[m])}개 (기대 {n})"
    assert not set(EXCLUDE_FULL) & set(usage), "usage에 제외 피처가 있음"
    return cols


def load():
    feat = pd.read_csv(P_FEATURES)
    table = pd.read_csv(P_TABLE, usecols=KEY + ["MRR", "t_start"])
    assert "t_start" not in feat.columns and "MRR" not in feat.columns
    data = feat.merge(table, on=KEY, how="left", validate="one_to_one")
    assert data["MRR"].notna().all(), "MRR 결측 행 있음"
    cols = input_columns(feat.columns)
    for c in DELTA_SINGLES[1:] + DELTA_PAIR:
        assert c in data.columns, f"δ 후보 열 없음: {c}"
    folds = pd.read_csv(P_FOLDS)
    masks = pd.read_csv(P_MASKS)
    masks = masks[(masks["design"] == "folds") & (masks["method"] == "fixed")]
    a = data[data["group"] == TARGET_GROUP].reset_index(drop=True)
    assert len(a) == 431, f"A-123 행 수 {len(a)} (문서 431)"
    fa = folds.merge(a[KEY], on=KEY, how="inner")
    assert len(fa) == 431 * 5, f"A-123 fold 행 수 {len(fa)}"
    return data, a, fa, masks, cols


def fit_sources(data, cols):
    models = {}
    for s, groups in SOURCES.items():
        src = data[data["group"].isin(groups)]
        for m in MODES:
            mdl = XGBRegressor(**XGB_PARAMS)
            mdl.fit(src[cols[m]].to_numpy(float), src["MRR"].to_numpy(float))  # NaN 그대로 (#27)
            models[(s, m)] = mdl
    return models


def split(a, fa, masks, r, ratio):
    rep, f = divmod(r, 5)
    fr = fa[fa["repeat"] == rep]
    tr = a.merge(fr.loc[fr["fold"] != f, KEY], on=KEY)
    ev = a.merge(fr.loc[fr["fold"] == f, KEY], on=KEY)
    ktr, kev = set(map(tuple, tr[KEY].values)), set(map(tuple, ev[KEY].values))
    assert ktr.isdisjoint(kev), f"누수: r={r}"
    assert len(ktr) + len(kev) == 431
    mk = masks[(masks["ratio"] == ratio) & (masks["r"] == r)][KEY].drop_duplicates()
    assert len(mk) > 0, f"공개 마스크 없음: r={r}, ratio={ratio}"
    assert set(map(tuple, mk.values)) <= ktr, f"공개 웨이퍼가 학습 쪽 밖: r={r}"
    pub = tr.merge(mk, on=KEY)
    return tr, ev, pub


# ---------------------------------------------------------------- δ 입력 (#26)
def _corr_hi(tr, x, y):
    rho = tr[[x, y]].apply(pd.to_numeric).corr(method="spearman").iloc[0, 1]  # 쌍별 결측 제외
    return bool(np.isfinite(rho) and abs(rho) > CORR_SKIP)


def select_delta(tr, pub):
    """상관은 A-123 학습 쪽 전체(tr)로, MRR 미사용. P1_on 조건만 공개 라벨(pub) 기준."""
    cap = max(2, len(pub) // 5)
    clean_pub = int((pub["allzero"] == 0).sum())
    sel = []
    for c in DELTA_SINGLES:
        if len(sel) >= cap:
            return sel
        if tr[c].notna().sum() < 2 or any(_corr_hi(tr, s, c) for s in sel):
            continue
        sel.append(c)
    if clean_pub >= P1_MIN_CLEAN and cap - len(sel) >= 2:      # 쌍은 2개로 셈, 1자리 남으면 멈춤 (5.7-8)
        if not any(_corr_hi(tr, s, c) for s in sel for c in DELTA_PAIR):
            sel += DELTA_PAIR
    return sel


def delta_matrices(tr, pub, ev, cols):
    """결측 채우기·표준화 통계는 A-123 학습 쪽 전체(tr) 기준 (#26 ⑤, 5.4)."""
    T = tr[cols].apply(pd.to_numeric).to_numpy(float)
    med = np.nanmedian(T, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    T = np.where(np.isnan(T), med, T)
    mu, sd = T.mean(0), T.std(0)
    sd[sd == 0] = 1.0

    def z(df):
        X = df[cols].apply(pd.to_numeric).to_numpy(float)
        return (np.where(np.isnan(X), med, X) - mu) / sd
    return z(pub), z(ev)


# ---------------------------------------------------------------- GP (#28)
def fit_gp(Xp, y, Xe, n_pub, seed):
    d = Xp.shape[1]
    ls = np.ones(d) if n_pub >= ISO_BELOW else 1.0
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(ls, (1e-2, 1e2))
              + WhiteKernel(1.0, (1e-3, 10.0)))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=5, random_state=seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gp.fit(Xp, y)
    n_warn = sum(issubclass(x.category, (ConvergenceWarning, UserWarning, RuntimeWarning)) for x in w)
    mean, std = gp.predict(Xe, return_std=True)   # WhiteKernel 잡음 포함 (5.7-10)
    return mean, std, n_warn


# ---------------------------------------------------------------- 한 fold
def run_fold(tr, ev, pub, models, r):
    y = pub["MRR"].to_numpy(float)
    ye = ev["MRR"].to_numpy(float)
    zero = ev["allzero"].to_numpy() == 1
    cols = select_delta(tr, pub)
    Xp, Xe = delta_matrices(tr, pub, ev, cols)
    seed = SEED + r
    t1, t1_sd, w1 = fit_gp(Xp, y, Xe, len(pub), seed)          # T1·T0는 소스와 무관
    base = dict(n_public=len(pub), n_public_clean=int((pub["allzero"] == 0).sum()),
                n_train=len(tr), n_eval=len(ev), delta_cols=",".join(cols),
                p1_on_included=int("P1_on" in cols), iso_kernel=int(len(pub) < ISO_BELOW))
    out = []
    for (s, m), mdl in models.items():
        cols_src = MODEL_COLS[m]
        fp = mdl.predict(pub[cols_src].to_numpy(float))
        fe = mdl.predict(ev[cols_src].to_numpy(float))
        t2 = fe + float(np.mean(y - fp))
        lr = LinearRegression().fit(fp.reshape(-1, 1), y)
        rho = float(np.clip(lr.coef_[0], 0.0, 2.0))
        b = float(np.mean(y) - rho * np.mean(fp))     # 자르지 않았으면 최소제곱 절편과 같음
        dm, dsd, w3 = fit_gp(Xp, y - (rho * fp + b), Xe, len(pub), seed)
        t3 = rho * fe + b + dm
        preds = {"T0": (np.full(len(ev), y.mean()), None), "T1": (t1, t1_sd),
                 "T2": (t2, None), "T3": (t3, dsd)}
        for name, (p, sd) in preds.items():
            for sub, msk in [("all", np.ones(len(ev), bool)), ("clean", ~zero), ("zero", zero)]:
                if msk.sum() == 0:
                    continue
                e = ye[msk] - p[msk]
                rec = dict(source=s, input=m, model=name, subset=sub, n=int(msk.sum()),
                           mse=float(np.mean(e ** 2)), rmse=float(np.sqrt(np.mean(e ** 2))),
                           mae=float(np.mean(np.abs(e))), rho=rho, b=b,
                           gp_warn_t1=w1, gp_warn_t3=w3, **base)
                if sd is not None:
                    rec["coverage90"] = float(np.mean(np.abs(e) <= Z90 * sd[msk]))
                out.append(rec)
    return out


# ---------------------------------------------------------------- 검정 (5.5, 5.7-16)
def summarize(R):
    """fold별 MSE 차이를 cv_common.corrected_ttest로 검정 (n_train = 학습 쪽 전체).
    revealed 기준(n_train = 공개 라벨 수)은 ttest_ntrain_sensitivity로 병기만 한다."""
    A = R[R["subset"] == "all"]
    rows = []
    for (ratio, s, m), g in A.groupby(["ratio", "source", "input"]):
        P = g.pivot(index="r", columns="model", values="mse").sort_index()
        sz = g.drop_duplicates("r").set_index("r").loc[P.index]
        n_tr, n_ev, n_pub = (sz[c].to_numpy(float) for c in ["n_train", "n_eval", "n_public"])
        for a, b, kind in [("T3", "T1", "주 판정"), ("T3", "T2", "2차 판정"), ("T2", "T1", "보고만")]:
            d = (P[a] - P[b]).to_numpy()
            full = cv.corrected_ttest(d, n_tr, n_ev)
            rev = cv.ttest_ntrain_sensitivity(d, n_ev, n_tr, n_pub, alpha=ALPHA).set_index("n_train_basis").loc["revealed"]
            judged = ratio == PRIMARY_RATIO and m == PRIMARY_MODE and kind != "보고만"
            rows.append(dict(ratio=ratio, source=s, input=m, compare=f"{a}-{b}", kind=kind,
                             primary=int(judged), J=full["n_folds"], mean_diff=full["mean_diff"],
                             sd_diff=full["sd_diff"], wins=int((d < 0).sum()),
                             se=full["se"], t=full["t"], df=full["df"], p=full["p"], nb_ratio=full["ratio"],
                             se_revealed=float(rev["se"]), t_revealed=float(rev["t"]), p_revealed=float(rev["p"]),
                             nb_ratio_revealed=float(rev["ratio"]), significant_revealed=bool(rev["significant"]),
                             passed=bool(full["p"] < ALPHA and full["mean_diff"] < 0) if judged else None,
                             n_eval_mean=float(n_ev.mean()), n_train_mean=float(n_tr.mean()),
                             n_public_mean=float(n_pub.mean())))
    return pd.DataFrame(rows)


def judge(S):
    """판정 (5.5): 8%, f_456 full, S1·S2 둘 다 (평균 차이 < 0, p < 0.05)일 때만 '효과 있음'.
    하나만 통과하면 '소스에 따라 갈림', 둘 다 아니면 '효과 없음'. 두 경우 모두 기본값 T1 유지.
    p는 full 기준. revealed 기준 p는 병기만 하고 판정에 쓰지 않는다."""
    out = []
    for compare, kind in [("T3-T1", "주 판정"), ("T3-T2", "2차 판정")]:
        g = S[(S["primary"] == 1) & (S["compare"] == compare)].set_index("source")
        assert set(g.index) == set(SOURCES), f"판정 대상 소스 누락: {list(g.index)}"
        rec = dict(kind=kind, compare=compare, ratio=PRIMARY_RATIO, source="+".join(SOURCES), input=PRIMARY_MODE)
        for s_ in SOURCES:
            rec.update({f"{s_}_mean_diff": float(g.loc[s_, "mean_diff"]), f"{s_}_p": float(g.loc[s_, "p"]),
                        f"{s_}_passed": bool(g.loc[s_, "passed"]), f"{s_}_p_revealed": float(g.loc[s_, "p_revealed"])})
        n_pass = int(sum(rec[f"{s_}_passed"] for s_ in SOURCES))
        rec["verdict"] = {2: "효과 있음", 1: "소스에 따라 갈림 (기본값 T1 유지)", 0: "효과 없음 (기본값 T1 유지)"}[n_pass]
        out.append(rec)
    return pd.DataFrame(out)


def rho_table(R):
    g = R[(R["subset"] == "all") & (R["model"] == "T3")].groupby(["ratio", "source", "input"])["rho"]
    return g.agg(["mean", "std", "min", "max"]).reset_index()


# ---------------------------------------------------------------- 저장
def tag_columns(df):
    """모든 결과 csv 공통 열 (CLAUDE.md 12절): featureset, group, input, source를 앞에 둔다."""
    df = df.copy()
    df.insert(0, "featureset", FEATURESET)
    df.insert(1, "group", TARGET_GROUP)
    for i, c in enumerate(["input", "source"], start=2):
        assert c in df.columns, f"{c} 열 없음"
        df.insert(i, c, df.pop(c))
    return df


def out_paths(day):
    """results/{날짜}_{이름}_FV1_a123_transfer_{내용}.csv. 하나라도 있으면 _v2, _v3 …로 (덮어쓰기 금지)."""
    RESULTS.mkdir(exist_ok=True)
    base = f"{day:%Y-%m-%d}_{AUTHOR}_{TAG}_a123_transfer"
    stem, k = base, 1
    while any((RESULTS / f"{stem}_{c}{ext}").exists() for c in OUT_CONTENTS for ext in (".csv", ".yaml")):
        k += 1
        stem = f"{base}_v{k}"
    return {c: RESULTS / f"{stem}_{c}.csv" for c in OUT_CONTENTS}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def git_state():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""
    dirty = [ln for ln in run("status", "--porcelain").splitlines() if ln]
    return {"commit": run("rev-parse", "HEAD") or "unknown", "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_files": len(dirty)}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "cpu": cpu_name(), "cpu_count": os.cpu_count(), "os": platform.platform()}


def write_yaml(path, info):
    """한 단계 중첩까지. 값은 JSON 표기(YAML로도 유효)로 써서 특수문자를 안전하게 둔다."""
    lines = []
    for k, v in info.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"  {kk}: {json.dumps(vv, ensure_ascii=False)}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 실행 전 연결 점검 (성능 계산 없음)
def check(data, a, fa, masks, cols):
    """실제 데이터 연결 점검. 모델 학습·예측·오차 계산을 하지 않고 개수만 출력한다."""
    print("행 수:", len(data), data["group"].value_counts().to_dict(), "/ A-123", len(a), "/ fold 행", len(fa))
    print("f_456 입력 개수:", {m: len(c) for m, c in cols.items()}, "/ 제외:", EXCLUDE_FULL,
          "/ full에 제외 피처 없음:", not set(EXCLUDE_FULL) & set(cols["full"]))
    for s, groups in SOURCES.items():
        src = data[data["group"].isin(groups)]
        for m in MODES:
            X = src[cols[m]]
            print(f"  {s}·{m}: 학습 행 {len(src)}, 입력 열 {X.shape[1]}, MRR 빈 값 {int(src['MRR'].isna().sum())}, "
                  f"입력 빈 값 있는 열 {int(X.isna().any().sum())}")
    need = DELTA_SINGLES + DELTA_PAIR + KEY + ["MRR"]
    print("A-123 δ 후보·키 열 빈 값:", {c: int(a[c].isna().sum()) for c in need if a[c].isna().any()} or "없음")
    n_pub = {}
    for ratio in RATIOS:
        k = [len(split(a, fa, masks, r, ratio)[2]) for r in range(25)]
        n_pub[ratio] = (min(k), max(k))
    print("fold당 공개 장수(최소, 최대):", n_pub)
    print("연결 점검 끝: 모델 학습·성능 계산 안 함")


# ---------------------------------------------------------------- 실행
MODEL_COLS: dict = {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="실제 데이터 연결 점검만 (학습·성능 계산 없음)")
    ap.add_argument("--quick", action="store_true", help="r 0~1, 공개 8%만. 결과 저장·숫자 출력 없음")
    args = ap.parse_args(argv)
    t0 = time.time()
    started = dt.datetime.now()
    git = git_state()
    data, a, fa, masks, cols = load()
    if args.check:
        check(data, a, fa, masks, cols)
        return
    MODEL_COLS.clear()
    MODEL_COLS.update(cols)
    models = fit_sources(data, cols)
    rs = [0, 1] if args.quick else list(range(25))
    ratios = [PRIMARY_RATIO] if args.quick else RATIOS
    rec = []
    for ratio in ratios:
        for r in rs:
            tr, ev, pub = split(a, fa, masks, r, ratio)
            for row in run_fold(tr, ev, pub, models, r):
                row.update(ratio=ratio, r=r, repeat=r // 5, fold=r % 5)
                rec.append(row)
        print(f"공개 {ratio}% 완료 ({time.time() - t0:.0f}초)", flush=True)
    R = pd.DataFrame(rec)

    # 자체 점검 (숫자 출력 없음)
    assert set(R["model"]) == {"T0", "T1", "T2", "T3"}
    assert R.groupby(["ratio", "r"])["delta_cols"].nunique().max() == 1
    assert R["delta_cols"].str.startswith("t_start").all()
    dc = R.drop_duplicates(["ratio", "r"])
    assert ((dc["delta_cols"].str.contains("P1_on")) == (dc["delta_cols"].str.contains("allzero"))).all()
    assert R.loc[R["model"].isin(["T1", "T3"]), "coverage90"].notna().all()
    for _, x in dc.iterrows():
        assert x["delta_cols"].count(",") + 1 <= max(2, x["n_public"] // 5)
    print("PASS: 누수 없음, 공개 마스크 일치, δ 규칙(t_start 첫째·상한·P1_on 쌍), T0~T3·커버리지 생성", flush=True)
    if args.quick:
        print(dc[["ratio", "r", "n_public", "n_public_clean", "delta_cols", "iso_kernel"]].to_string(index=False))
        print("quick 끝: 결과 저장 안 함")
        return

    S = summarize(R)
    J = judge(S)
    RH = rho_table(R)
    paths = out_paths(started.date())
    tables = {"folds": R, "stats": S, "judgment": J, "rho": RH}
    for c, df in tables.items():
        tag_columns(df).to_csv(paths[c], index=False, encoding="utf-8")
    W = R[R["subset"] == "all"].drop_duplicates(["ratio", "r", "source", "input"])
    info = {
        "experiment": "a123_transfer (preregistration 5절 주 평가)",
        "author": AUTHOR, "tag": TAG, "featureset": FEATURESET, "group": TARGET_GROUP,
        "run_started": f"{started:%Y-%m-%d %H:%M:%S}", "run_finished": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
        "runtime_sec": round(time.time() - t0),
        "git": git,
        "environment": environment(),
        "ratios": RATIOS, "folds": "folds_v1 25 (r = repeat*5+fold)", "reveal": "reveal_masks_v1 fixed",
        "sources": SOURCES,
        "f456": {"model": "XGBRegressor (3.2 고정 설정)", "params": XGB_PARAMS, "n_jobs": XGB_N_JOBS,
                 "train": "소스별 456 전체 1회 학습, 결측 NaN 그대로",
                 "inputs": {m: len(c) for m, c in cols.items()}, "exclude_full": EXCLUDE_FULL},
        "seed_rule": SEED_RULE,
        "gp": "Constant(1e-2~1e2) x RBF(1e-2~1e2; 공개<10 등방성) + White(1e-3~10), restarts 5, normalize_y",
        "gp_warnings": {"t1_folds_with_warning": int((W.groupby(["ratio", "r"])["gp_warn_t1"].first() > 0).sum()),
                        "t3_fold_source_input_with_warning": int((W["gp_warn_t3"] > 0).sum())},
        "ttest": "cv_common.corrected_ttest (Nadeau-Bengio), n_train = A-123 학습 쪽 전체로 판정. "
                 "revealed(n_train = 공개 라벨 수)는 cv_common.ttest_ntrain_sensitivity로 병기, 판정에 안 씀",
        "judgment_rule": "8%, f_456 full, S1·S2 둘 다 (평균 차이 < 0, p < 0.05)일 때만 효과 있음 / 하나만 통과면 "
                         "소스에 따라 갈림 / 둘 다 아니면 효과 없음. 갈림·없음이면 기본값 T1. 주 판정 T3-T1, "
                         "2차 판정 T3-T2 같은 규칙. usage는 보고만",
        "inputs_sha256_16": {p.name: sha(p) for p in [P_FEATURES, P_FAMILIES, P_FOLDS, P_MASKS, P_TABLE]},
        "outputs": {c: p.name for c, p in paths.items()},
    }
    for c, p in paths.items():
        write_yaml(p.with_suffix(".yaml"), {"this_file": p.name, **info})
    print("저장: " + " / ".join(p.name for p in paths.values()) + " (+ 같은 이름 .yaml, 숫자·판정은 파일에서 확인)")


if __name__ == "__main__":
    main()

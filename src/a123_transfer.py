"""A-123 전이학습 (preregistration.md v0.2.1, 5절) 주 평가 구현.

범위 (5.7-17 "필수"): folds_v1 25 fold 주 평가, 고정주기 공개, 공개 2/4/8/16/100%,
T0~T3, 소스 S1(A-456)·S2(A-456+B-456), f_456 full·usage, 8% 주·2차 판정(S1·S2 합산), 90% 커버리지, ρ 분포.
(전진 검증·무작위 공개·δ XGBoost·라벨 피처 실험은 이 파일 범위 밖)

규칙 대응 (절 번호는 preregistration.md):
- 분할: 같은 repeat 안에서 eval = fold==f, train = fold!=f, r = repeat*5 + fold (5.7-1)
- 공개: data/processed/reveal_masks_v1.csv (design=folds, method=fixed) (5.3, 5.7-7)
- T0 공개 라벨 평균 / T1 GP 단독 / T2 f_456 + c / T3 ρ·f_456 + b + δGP (5.4)
- ρ·b 최소제곱 → ρ를 [0, 2]로 자르면 b만 다시 추정 → 잔차에 δ GP (5.7-9)
- f_456: XGBoost 고정 설정(3.2), 456 전체로 1회 학습, 결측은 NaN 그대로 (#27)
- δ 입력: #26 ①~⑤, 5.7-8 / 커널·GP 설정: #28, 5.4, 5.7-14
- GP 경고는 숨기지 않고 fold·모델별로 센다 (5.7-15)
- 보정 t-검정: Nadeau & Bengio(2003), 평가/학습 비율은 실제 fold 크기 (5.7-16)
- test 데이터(data/locked 등)는 읽지 않는다 (1절)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import platform
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
P_FEATURES = DATA / "features_v1.csv"
P_FAMILIES = DATA / "feature_families_v1.csv"
P_FOLDS = DATA / "folds_v1.csv"
P_MASKS = DATA / "reveal_masks_v1.csv"
P_TABLE = DATA / "wafer_table.csv"
RESULTS = ROOT / "results"
AUTHOR = "성수빈"
KEY = ["WAFER_ID", "STAGE"]

RATIOS = [2, 4, 8, 16, 100]
PRIMARY_RATIO = 8
PRIMARY_MODE = "full"          # usage는 보고만 (6절, 2026-11-01 결정)
SOURCES = {"S1": ["A-456"], "S2": ["A-456", "B-456"]}
MODES = ["full", "usage"]
USAGE_FAMILY = "소모품 카운터"
DELTA_SINGLES = ["t_start", "mean_POLISH_TABLE_s1", "mean_DRESSER_s1", "mean_MEMBRANE_s1",
                 "mean_BACKING_FILM_s1", "mean_PSHEET_s1", "mean_DRESSER_TABLE_s1"]
DELTA_PAIR = ["P1_on", "allzero"]
CORR_SKIP = 0.9
P1_MIN_CLEAN = 10
ISO_BELOW = 10
Z90 = stats.norm.ppf(0.95)
XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, random_state=42, n_jobs=2)

for p in [P_FEATURES, P_FAMILIES, P_FOLDS, P_MASKS, P_TABLE]:
    assert "locked" not in str(p).lower() and "test" not in p.name.lower(), f"test 경로 금지: {p}"


# ---------------------------------------------------------------- 데이터
def load():
    feat = pd.read_csv(P_FEATURES)
    table = pd.read_csv(P_TABLE, usecols=KEY + ["MRR", "t_start"])
    assert "t_start" not in feat.columns and "MRR" not in feat.columns
    data = feat.merge(table, on=KEY, how="left", validate="one_to_one")
    assert data["MRR"].notna().all(), "MRR 결측 행 있음"
    fam = pd.read_csv(P_FAMILIES)
    feat_cols = [c for c in fam["feature"] if c in feat.columns]
    usage_cols = [c for c in fam.loc[fam["family"] == USAGE_FAMILY, "feature"] if c in feat.columns]
    assert len(feat_cols) >= 100, f"피처 수 이상: {len(feat_cols)}"
    assert usage_cols, "usage 피처 0개"
    for c in DELTA_SINGLES[1:] + DELTA_PAIR:
        assert c in data.columns, f"δ 후보 열 없음: {c}"
    folds = pd.read_csv(P_FOLDS)
    masks = pd.read_csv(P_MASKS)
    masks = masks[(masks["design"] == "folds") & (masks["method"] == "fixed")]
    a = data[data["group"] == "A-123"].reset_index(drop=True)
    assert len(a) == 431, f"A-123 행 수 {len(a)} (문서 431)"
    fa = folds.merge(a[KEY], on=KEY, how="inner")
    assert len(fa) == 431 * 5, f"A-123 fold 행 수 {len(fa)}"
    return data, a, fa, masks, {"full": feat_cols, "usage": usage_cols}


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
    seed = 42 + r
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
                rec = dict(source=s, mode=m, model=name, subset=sub, n=int(msk.sum()),
                           mse=float(np.mean(e ** 2)), rmse=float(np.sqrt(np.mean(e ** 2))),
                           mae=float(np.mean(np.abs(e))), rho=rho, b=b,
                           gp_warn_t1=w1, gp_warn_t3=w3, **base)
                if sd is not None:
                    rec["coverage90"] = float(np.mean(np.abs(e) <= Z90 * sd[msk]))
                out.append(rec)
    return out


# ---------------------------------------------------------------- 검정 (5.7-16)
def nb_test(d, n_te, n_tr):
    J = len(d)
    var = np.var(d, ddof=1) * (1 / J + n_te / n_tr)
    if not var > 0:
        return np.nan, np.nan
    t = np.mean(d) / np.sqrt(var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def summarize(R):
    A = R[R["subset"] == "all"]
    rows = []
    for (ratio, s, m), g in A.groupby(["ratio", "source", "mode"]):
        P = g.pivot(index="r", columns="model", values="mse")
        sz = g.drop_duplicates("r")
        n_te, n_tr = sz["n_eval"].mean(), sz["n_train"].mean()
        for a, b, kind in [("T3", "T1", "주 판정"), ("T3", "T2", "2차 판정"), ("T2", "T1", "보고만")]:
            d = (P[a] - P[b]).to_numpy()
            t, p = nb_test(d, n_te, n_tr)
            judged = ratio == PRIMARY_RATIO and m == PRIMARY_MODE and kind != "보고만"
            rows.append(dict(ratio=ratio, source=s, mode=m, compare=f"{a}-{b}", kind=kind,
                             primary=int(judged), J=len(d), mean_diff=float(d.mean()),
                             wins=int((d < 0).sum()), t=t, p=p,
                             passed=bool(p < 0.05 and d.mean() < 0) if judged else None,
                             n_eval_mean=n_te, n_train_mean=n_tr))
    return pd.DataFrame(rows)


def judge(S):
    """합산 판정 (5.5, 2026-11-01 결정): 8%, f_456 full, S1·S2 둘 다 통과해야 '효과 있음'.
    하나만 통과하면 '소스에 따라 갈림', 둘 다 아니면 '효과 없음'. 갈림·없음이면 기본값 T1 유지.
    usage 버전과 다른 비율은 보고만."""
    out = []
    for compare, kind in [("T3-T1", "주 판정"), ("T3-T2", "2차 판정")]:
        g = S[(S["primary"] == 1) & (S["compare"] == compare)].set_index("source")
        assert set(g.index) == set(SOURCES), f"판정 대상 소스 누락: {list(g.index)}"
        n_pass = int(sum(bool(g.loc[s_, "passed"]) for s_ in SOURCES))
        verdict = {2: "효과 있음", 1: "소스에 따라 갈림 (기본값 T1 유지)", 0: "효과 없음 (기본값 T1 유지)"}[n_pass]
        out.append(dict(kind=kind, compare=compare, ratio=PRIMARY_RATIO, mode=PRIMARY_MODE,
                        S1_passed=bool(g.loc["S1", "passed"]), S2_passed=bool(g.loc["S2", "passed"]),
                        verdict=verdict))
    return pd.DataFrame(out)


def rho_table(R):
    g = R[(R["subset"] == "all") & (R["model"] == "T3")].groupby(["ratio", "source", "mode"])["rho"]
    return g.agg(["mean", "std", "min", "max"]).reset_index()


# ---------------------------------------------------------------- 저장
def out_paths():
    RESULTS.mkdir(exist_ok=True)
    stem = f"{dt.date.today():%Y-%m-%d}_{AUTHOR}_a123_transfer"
    k = 1
    while (RESULTS / f"{stem}.csv").exists():
        k += 1
        stem = f"{dt.date.today():%Y-%m-%d}_{AUTHOR}_a123_transfer_v{k}"
    return RESULTS / stem


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def write_yaml(path, info):
    lines = []
    for k, v in info.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"  {kk}: {vv}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 실행
MODEL_COLS: dict = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="r 0~1, 공개 8%만. 결과 저장·숫자 출력 없음")
    args = ap.parse_args()
    t0 = time.time()
    data, a, fa, masks, cols = load()
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

    S, RH = summarize(R), rho_table(R)
    base = out_paths()
    R.to_csv(base.with_suffix(".csv"), index=False)
    S.to_csv(base.parent / f"{base.name}_stats.csv", index=False)
    J = judge(S)
    J.to_csv(base.parent / f"{base.name}_judgment.csv", index=False)
    RH.to_csv(base.parent / f"{base.name}_rho.csv", index=False)
    W = R[R["subset"] == "all"].drop_duplicates(["ratio", "r", "source", "mode"])
    write_yaml(base.with_suffix(".yaml"), {
        "experiment": "a123_transfer (preregistration 5절 주 평가)",
        "author": AUTHOR, "date": f"{dt.datetime.now():%Y-%m-%d %H:%M}",
        "runtime_sec": round(time.time() - t0),
        "ratios": RATIOS, "folds": "folds_v1 25 (r = repeat*5+fold)", "reveal": "reveal_masks_v1 fixed",
        "sources": SOURCES, "f456_modes": {m: len(c) for m, c in cols.items()}, "xgb_params": XGB_PARAMS,
        "gp": "Constant(1e-2~1e2) x RBF(1e-2~1e2; 공개<10 등방성) + White(1e-3~10), restarts 5, seed 42+r, normalize_y",
        "gp_warnings": {"t1_folds_with_warning": int((W.groupby(["ratio", "r"])["gp_warn_t1"].first() > 0).sum()),
                        "t3_fold_source_mode_with_warning": int((W["gp_warn_t3"] > 0).sum())},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "inputs_sha256_16": {p.name: sha(p) for p in [P_FEATURES, P_FAMILIES, P_FOLDS, P_MASKS, P_TABLE]},
        "judgment_rule": "8%, f_456 full, S1·S2 모두 (평균 차이<0, 보정 p<0.05)이면 효과 있음 / 하나만 통과면 소스에 따라 갈림 / 둘 다 아니면 효과 없음. 갈림·없음이면 기본값 T1. 주 판정 T3-T1, 2차 판정 T3-T2 같은 규칙. usage는 보고만",
        "outputs": {"folds": base.with_suffix(".csv").name, "stats": f"{base.name}_stats.csv",
                    "judgment": f"{base.name}_judgment.csv", "rho": f"{base.name}_rho.csv"},
    })
    print(f"저장: {base.name}.csv / _stats.csv / _judgment.csv / _rho.csv / .yaml (숫자·판정은 파일에서 확인)")


if __name__ == "__main__":
    main()

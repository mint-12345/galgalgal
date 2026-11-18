"""PCA/PLS 차원 축소 + 그리드 서치 (SR 입력용, A-456·B-456 공통 축). 사후 탐색, 정식 판정을 바꾸지 않는다.

계획: docs/plans/2026-11-16_김혜연_dimred_plan.md (이 파일의 절 번호는 그 계획의 절 번호)

- 입력: data/processed/features_v1.csv, feature_families_v1.csv, folds_v1.csv, wafer_table.csv의 MRR,
  results/diagnostics/2026-11-01_박종화_sr_candidates.csv (주 집합 정의와 (b) 기준선 변수).
  split ∈ {train, val}, is_outlier = 0, A-456·B-456만. test 라벨과 data/locked/는 읽지 않는다.
- 핵심 함수: fit_transform (박종화님이 fold마다 그대로 호출하는 함수, 사용 예시는 그 docstring).

실행:
  python src/explore/dimred.py --selftest        누수·재현 자체 점검만
  python src/explore/dimred.py --check           주 집합 웨이퍼 수와 (b) 재현만 (10절 순서 1)
  python src/explore/dimred.py                   전체 (재현 확인 → 그리드·선택·비교 → loadings·PLS1 점검)
  python src/explore/dimred.py --out-dir DIR     결과를 DIR에 쓴다 (같은 seed 두 번 실행 비교용)
출력 (11절): results/{날짜}_김혜연_FV1_dimred_{grid,selection,compare}.csv,
             results/diagnostics/{날짜}_김혜연_dimred_{loadings,pls1_check}.csv, 각각 같은 이름의 .yaml.
             같은 이름이 있으면 덮어쓰지 않고 _v2, _v3 …를 붙인다.
"""
import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cv_common import corrected_ttest  # noqa: E402

PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CANDIDATES = RESULTS / "diagnostics" / "2026-11-01_박종화_sr_candidates.csv"
PLAN = "docs/plans/2026-11-16_김혜연_dimred_plan.md"
AUTHOR = "김혜연"
FEATURESET = "features_v1"
SEED = 42
KEYS = ["WAFER_ID", "STAGE"]
GROUPS = ["A-456", "B-456"]
METHODS = ["PCA", "PLS"]
KS = [1, 2, 3, 5, 8, 10]
N_INNER = 5
N_JOBS = -1
EXCLUDED_FEATURES = {"since_POLISH_TABLE_w"}   # sr_candidates.py와 같음 (4.8 11번)
MAX_NAN_FRAC = 0.05
MIN_ABS_MEDIAN = 1e-9
CONST_STD = 1e-12
EXPECTED_POOL = 106
EXPECTED_ALL = {"A-456": 983, "B-456": 987}
EXPECTED_MAIN = {"A-456": 940, "B-456": 951}
EXPECTED_LR = {"A-456": 4.450518416882203, "B-456": 5.066029725383169}   # sr_folds.yaml rmse_lr
LR_TOL = 1e-6
SR_FINAL_REF = {"A-456": 4.240, "B-456": 4.177}                          # 참고값, 낙관적
PLS1_FEATURES = ["dur_POLISH_s1", "auc_PCHAMBER_s1", "auc_MAIN_OUTER_s1", "auc_SLURRY_C_s1"]
PLS1_THRESHOLD = 0.9
N_TOP_LOADINGS = 15


# ---------------------------------------------------------------- 데이터
def feature_pool():
    """1.2절 피처 풀. 라벨을 쓰지 않는다 (features_v1의 값만). 두 테이블 목록이 같아야 한다."""
    feats = pd.read_csv(PROC / "features_v1.csv")
    fam = pd.read_csv(PROC / "feature_families_v1.csv")
    cand = [f for f in fam.loc[fam["sr_candidate"] == 1, "feature"] if f not in EXCLUDED_FEATURES]
    pools = []
    for g in GROUPS:
        x = feats[feats["group"] == g]
        pools.append([f for f in cand
                      if x[f].isna().mean() <= MAX_NAN_FRAC and x[f].abs().median() >= MIN_ABS_MEDIAN])
    assert pools[0] == pools[1], "두 테이블의 피처 풀이 다르다"
    assert len(pools[0]) == EXPECTED_POOL, f"피처 풀이 {len(pools[0])}개다"
    return pools[0]


def sr_candidates():
    """{테이블: 정식 SR 후보 10개}."""
    c = pd.read_csv(CANDIDATES)
    out = {g: c[(c["group"] == g) & (c["selected"] == 1)].sort_values("rank")["feature"].tolist() for g in GROUPS}
    assert all(len(v) == 10 for v in out.values())
    return out


def load_data():
    """A-456·B-456의 train+val(이상치 제외) 웨이퍼 표 + MRR + 주 집합 표시(in_main)."""
    feats = pd.read_csv(PROC / "features_v1.csv")
    table = pd.read_csv(PROC / "wafer_table.csv", usecols=KEYS + ["split", "MRR", "is_outlier"])
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    data = feats[feats["group"].isin(GROUPS)].merge(table, on=KEYS, validate="one_to_one")
    assert data["MRR"].notna().all()
    for g in GROUPS:
        assert (data["group"] == g).sum() == EXPECTED_ALL[g], f"{g} 웨이퍼 수가 다르다"
    cands = sr_candidates()
    data = data.copy()
    data["in_main"] = False
    for g in GROUPS:
        m = data["group"] == g
        data.loc[m, "in_main"] = data.loc[m, cands[g]].notna().all(axis=1)
    return data.reset_index(drop=True)


def outer_folds(data):
    """[(r, repeat, fold, 학습 마스크, 평가 마스크)] 25개. folds_v1 그대로 (2절)."""
    f = pd.read_csv(PROC / "folds_v1.csv")
    out = []
    for repeat in range(5):
        a = data[KEYS].merge(f[f["repeat"] == repeat][KEYS + ["fold"]], on=KEYS, how="left", validate="one_to_one")
        assert a["fold"].notna().all(), "fold가 없는 웨이퍼가 있다"
        fold = a["fold"].to_numpy(int)
        for k in range(5):
            ev = fold == k
            out.append((repeat * 5 + k, repeat, k, ~ev, ev))
    return out


def assert_disjoint(tr, ev):
    """학습·평가 쪽 키 겹침 0, WAFER_ID 겹침 0."""
    kt = set(map(tuple, tr[KEYS].to_numpy()))
    ke = set(map(tuple, ev[KEYS].to_numpy()))
    assert not kt & ke, "학습·평가 쪽 키가 겹친다"
    assert not set(tr["WAFER_ID"]) & set(ev["WAFER_ID"]), "학습·평가 쪽 WAFER_ID가 겹친다"


# ---------------------------------------------------------------- 핵심 변환 (3절)
def fit_transform(train_df, apply_dfs, method, k, seed=SEED, features=None):
    """3절 전처리와 공통 성분을 train_df로만 맞추고, 같은 변환을 apply_dfs에 적용한다.

    train_df : 학습 쪽 웨이퍼 표. 열 = group(A-456·B-456 둘 다 있어야 함), MRR, 피처.
    apply_dfs: 변환만 적용할 표의 리스트 (평가 쪽 등). MRR 열이 있어도 함수 시작 시 지운 복사본을 쓴다.
    method   : "PCA" 또는 "PLS".  k: 성분 수.  seed: 난수 시드 (PCA svd_solver="full"이라 실제로는 결정적).
    features : 쓸 피처 목록. None이면 feature_pool()의 106개.

    처리 (모두 train_df만으로 학습):
      1. 결측 대치: 테이블별 학습 쪽 중앙값
      2. 상수 열 제거: 어느 한 테이블이라도 학습 쪽 표준편차(ddof=0) < 1e-12
      3. X 테이블별 표준화 (적용 쪽은 자기 테이블의 학습 쪽 평균·표준편차로)
      4. y 테이블별 중심화 (PLS만 사용)
      5. A+B 합친 학습 쪽으로 성분 하나 학습 (PCA / PLSRegression(scale=False))
      6. 성분마다 |가중치|가 가장 큰 피처의 가중치가 양수가 되도록 부호 고정
    반환: (Z_train, [Z_apply ...], model_info)
      Z_*: 입력과 같은 index의 DataFrame, 열 = "{method}1" … "{method}k" (성분 점수)
      model_info: dict (features, dropped_const, weights(DataFrame 피처 × 성분), x_var_ratio, 통계값, 대치 칸 수)

    사용 예시 (folds_v1 25-fold 각각에서):
        from explore.dimred import fit_transform, load_data, outer_folds
        data = load_data(); main = data[data["in_main"]].reset_index(drop=True)
        for r, repeat, fold, tr, ev in outer_folds(main):
            Ztr, (Zev,), info = fit_transform(main[tr], [main[ev]], "PLS", 2)
            # Ztr·Zev를 SR 입력으로, MRR은 main[tr]["MRR"]로 학습하고 main[ev]["MRR"]로 채점한다.
            # 테이블 구분: main[tr]["group"]. Zev를 만들 때 평가 쪽 MRR은 쓰이지 않는다.
    """
    if method not in METHODS:
        raise ValueError(method)
    features = list(feature_pool() if features is None else features)
    train = train_df.copy()
    applies = [d.drop(columns=["MRR"], errors="ignore").copy() for d in apply_dfs]
    for a in applies:
        assert "MRR" not in a.columns, "적용 쪽에 MRR이 남아 있다"
    assert set(train["group"]) == set(GROUPS), "학습 쪽에 두 테이블이 모두 있어야 한다"
    for a in applies:
        assert set(a["group"]) <= set(GROUPS), "적용 쪽에 모르는 테이블이 있다"
    assert train["MRR"].notna().all()

    stats = {}
    n_imp_train = 0
    Xtr = train[features].astype(float)
    for g in GROUPS:
        m = (train["group"] == g).to_numpy()
        med = Xtr[m].median()
        n_imp_train += int(Xtr[m].isna().sum().sum())
        Xtr.loc[m] = Xtr[m].fillna(med)
        stats[g] = {"median": med}
    assert not Xtr.isna().any().any(), "학습 쪽 중앙값이 없는 열이 있다 (테이블 전체 결측)"
    stds = {g: Xtr[(train["group"] == g).to_numpy()].std(ddof=0) for g in GROUPS}
    const = sorted(set().union(*[set(s.index[s < CONST_STD]) for s in stds.values()]))
    used = [f for f in features if f not in const]
    Xtr = Xtr[used]
    y = train["MRR"].to_numpy(float).copy()
    for g in GROUPS:
        m = (train["group"] == g).to_numpy()
        stats[g]["mean"] = Xtr[m].mean()
        stats[g]["std"] = stds[g][used]
        stats[g]["y_mean"] = float(y[m].mean())
        Xtr.loc[m] = (Xtr[m] - stats[g]["mean"]) / stats[g]["std"]
        y[m] -= stats[g]["y_mean"]
    Xs = Xtr.to_numpy(float)

    if method == "PCA":
        model = PCA(n_components=k, svd_solver="full", random_state=seed).fit(Xs)
        W = model.components_.T
        x_var_ratio = model.explained_variance_ratio_
    else:
        model = PLSRegression(n_components=k, scale=False).fit(Xs, y)
        W = model.x_rotations_
        T, P = model.x_scores_, model.x_loadings_
        x_var_ratio = (T ** 2).sum(0) * (P ** 2).sum(0) / (Xs ** 2).sum()
    sign = np.sign(W[np.abs(W).argmax(axis=0), np.arange(k)])
    sign[sign == 0] = 1.0
    names = [f"{method}{j + 1}" for j in range(k)]

    def scores(X):
        return pd.DataFrame(model.transform(X) * sign, columns=names)

    Z_train = scores(Xs).set_index(train.index)
    Z_apply, n_imp_apply = [], 0
    for a in applies:
        Xa = a[used].astype(float)
        for g in GROUPS:
            m = (a["group"] == g).to_numpy()
            if not m.any():
                continue
            n_imp_apply += int(Xa[m].isna().sum().sum())
            Xa.loc[m] = (Xa[m].fillna(stats[g]["median"][used]) - stats[g]["mean"]) / stats[g]["std"]
        assert not Xa.isna().any().any()
        Z_apply.append(scores(Xa.to_numpy(float)).set_index(a.index))

    info = {
        "method": method, "k": k, "seed": seed, "features": used, "dropped_const": const,
        "weights": pd.DataFrame(W * sign, index=used, columns=names), "x_var_ratio": np.asarray(x_var_ratio),
        "stats": stats, "n_imputed_train": n_imp_train, "n_imputed_apply": n_imp_apply,
    }
    return Z_train, Z_apply, info


def per_table_lr(Ztr, ytr, gtr, Zev, gev):
    """5절 평가 모형: 테이블별 선형회귀 MRR_g = a_g·Z + b_g. 평가 쪽 예측을 돌려준다."""
    pred = np.full(len(Zev), np.nan)
    for g in GROUPS:
        mt, me = gtr == g, gev == g
        pred[me] = LinearRegression().fit(Ztr[mt], ytr[mt]).predict(Zev[me])
    return pred


def table_errors(y, pred, groups, n_train_by_group):
    out = {}
    for g in GROUPS:
        m = groups == g
        e = pred[m] - y[m]
        out[g] = {"mse": float(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e))),
                  "rmse": float(np.sqrt(np.mean(e ** 2))), "n_train": int(n_train_by_group[g]), "n_eval": int(m.sum())}
    return out


def dimred_errors(tr_df, ev_df, method, k):
    """tr_df로 변환·선형회귀를 맞추고 ev_df 테이블별 오차."""
    Ztr, (Zev,), info = fit_transform(tr_df, [ev_df], method, k)
    gtr, gev = tr_df["group"].to_numpy(), ev_df["group"].to_numpy()
    pred = per_table_lr(Ztr.to_numpy(), tr_df["MRR"].to_numpy(float), gtr, Zev.to_numpy(), gev)
    ntr = {g: int((gtr == g).sum()) for g in GROUPS}
    return table_errors(ev_df["MRR"].to_numpy(float), pred, gev, ntr), info


# ---------------------------------------------------------------- 6절 선택
def select(inner):
    """inner: 설정별 안쪽 지표 표 (method, k, mean, se). (1-SE 선택 행, 최저 행, 한계값)."""
    best = inner.loc[inner["mean"].idxmin()]
    limit = best["mean"] + best["se"]
    ok = inner[inner["mean"] <= limit]
    kmin = ok["k"].min()
    pick = ok[ok["k"] == kmin].sort_values("mean").iloc[0]
    return pick, best, float(limit)


# ---------------------------------------------------------------- 7절 기준선
def baseline_errors(tr_df, ev_df, model, cands, pool):
    gtr, gev = tr_df["group"].to_numpy(), ev_df["group"].to_numpy()
    ntr = {g: int((gtr == g).sum()) for g in GROUPS}
    pred = np.full(len(ev_df), np.nan)
    for g in GROUPS:
        a, b = tr_df[gtr == g], ev_df[gev == g]
        ya = a["MRR"].to_numpy(float)
        if model == "mean":
            pred[gev == g] = ya.mean()
            continue
        cols = cands[g] if model == "lr_sr10" else pool
        med = a[cols].median()
        Xa, Xb = a[cols].fillna(med).to_numpy(float), b[cols].fillna(med).to_numpy(float)
        if model == "lr_sr10":
            # 원 단위 그대로면 설계행렬 조건수가 약 1e8(A-456)이라 해가 수치적으로 흔들린다.
            # 정식 sr_select처럼 |중앙값|으로 나눈다 (여기서는 학습 쪽 |중앙값|). 선형회귀라 수학적으로는 같은 모형.
            s = np.abs(np.median(Xa, axis=0))
            Xa, Xb = Xa / s, Xb / s
            m = LinearRegression()
        else:
            m = RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=N_JOBS)
        pred[gev == g] = m.fit(Xa, ya).predict(Xb)
    return table_errors(ev_df["MRR"].to_numpy(float), pred, gev, ntr)


def frame_of(set_name, data):
    return data[data["in_main"]].reset_index(drop=True) if set_name == "main" else data.copy()


def check_reproduction(data, cands, pool):
    """10절 순서 1: 주 집합 웨이퍼 수와 (b) 선형회귀 RMSE."""
    main = frame_of("main", data)
    counts = main["group"].value_counts().to_dict()
    print("주 집합 웨이퍼:", counts, flush=True)
    rm = {g: [] for g in GROUPS}
    for r, repeat, fold, tr, ev in outer_folds(main):
        e = baseline_errors(main[tr], main[ev], "lr_sr10", cands, pool)
        for g in GROUPS:
            rm[g].append(e[g]["rmse"])
    lr = {g: float(np.mean(v)) for g, v in rm.items()}
    print("(b) lr_sr10 바깥 25-fold RMSE:", {g: round(v, 6) for g, v in lr.items()}, flush=True)
    ok = all(counts[g] == EXPECTED_MAIN[g] for g in GROUPS) and all(abs(lr[g] - EXPECTED_LR[g]) < LR_TOL for g in GROUPS)
    return ok, counts, lr


# ---------------------------------------------------------------- 실행
def run_set(set_name, data, cands, pool):
    df = frame_of(set_name, data)
    grid_rows, sel_rows, cmp_rows = [], [], []
    for r, repeat, fold, tr, ev in outer_folds(df):
        t0 = time.perf_counter()
        tr_df, ev_df = df[tr].reset_index(drop=True), df[ev].reset_index(drop=True)
        assert_disjoint(tr_df, ev_df)
        assert set(tr_df["group"]) == set(GROUPS)
        # 안쪽 5-fold (2절)
        skf = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + repeat)
        metric = {(mth, k): [] for mth in METHODS for k in KS}
        for i, (itr, iev) in enumerate(skf.split(tr_df, tr_df["group"], groups=tr_df["WAFER_ID"])):
            a, b = tr_df.iloc[itr], tr_df.iloc[iev]
            assert_disjoint(a, b)
            for mth in METHODS:
                for k in KS:
                    e, _ = dimred_errors(a, b, mth, k)
                    metric[(mth, k)].append(np.mean([e[g]["rmse"] for g in GROUPS]))
                    for g in GROUPS:
                        grid_rows.append({"set": set_name, "outer_r": r, "repeat": repeat, "outer_fold": fold,
                                          "inner_fold": i, "method": mth, "k": k, "group": g,
                                          "model": f"{mth}{k}_lr", "featureset": FEATURESET, "fold": r, **e[g]})
        inner = pd.DataFrame([{"method": m_, "k": k_, "mean": float(np.mean(v)),
                               "se": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for (m_, k_), v in metric.items()])
        pick, best, limit = select(inner)
        e_sel, info_sel = dimred_errors(tr_df, ev_df, pick["method"], int(pick["k"]))
        e_min, _ = dimred_errors(tr_df, ev_df, best["method"], int(best["k"]))
        for g in GROUPS:
            sel_rows.append({
                "set": set_name, "outer_r": r, "repeat": repeat, "outer_fold": fold, "group": g,
                "model": "dimred_select_lr", "featureset": FEATURESET, "fold": r,
                "method": pick["method"], "k": int(pick["k"]), "inner_mean": pick["mean"], "inner_se": pick["se"],
                "inner_limit": limit, "method_min": best["method"], "k_min": int(best["k"]),
                "inner_mean_min": best["mean"], "inner_se_min": best["se"], **e_sel[g],
                "rmse_outer_min": e_min[g]["rmse"], "mse_outer_min": e_min[g]["mse"],
                "n_dropped_const": len(info_sel["dropped_const"]),
                "n_imputed_train": info_sel["n_imputed_train"], "n_imputed_apply": info_sel["n_imputed_apply"],
            })
        # 7절 비교
        models = {"dimred_select_lr": e_sel}
        for bm in ("lr_sr10", "rf_pool", "mean"):
            models[bm] = baseline_errors(tr_df, ev_df, bm, cands, pool)
        for name, e in models.items():
            for g in GROUPS:
                cmp_rows.append({"row_type": "fold", "set": set_name, "group": g, "model": name,
                                 "featureset": FEATURESET, "fold": r, **e[g]})
        print(f"[{set_name}] r={r:2d} 선택 {pick['method']}{int(pick['k'])} (최저 {best['method']}{int(best['k'])}) "
              f"A {e_sel['A-456']['rmse']:.3f} B {e_sel['B-456']['rmse']:.3f} {time.perf_counter() - t0:.1f}s",
              flush=True)
    return pd.DataFrame(grid_rows), pd.DataFrame(sel_rows), pd.DataFrame(cmp_rows)


def summarize(cmp_folds):
    rows = []
    for (s, g), sub in cmp_folds.groupby(["set", "group"], sort=False):
        piv = sub.pivot(index="fold", columns="model", values="mse")
        size = sub[sub["model"] == "lr_sr10"].set_index("fold")
        for name, m in sub.groupby("model", sort=False):
            row = {"row_type": "summary", "set": s, "group": g, "model": name, "featureset": FEATURESET,
                   "fold": "mean25", "mse": m["mse"].mean(), "mae": m["mae"].mean(), "rmse": m["rmse"].mean(),
                   "rmse_se": m["rmse"].std(ddof=1) / np.sqrt(len(m)), "n_train": m["n_train"].mean(),
                   "n_eval": m["n_eval"].mean(), "n_folds": len(m)}
            if name == "dimred_select_lr":
                t = corrected_ttest((piv["dimred_select_lr"] - piv["lr_sr10"]).to_numpy(),
                                    size["n_train"].to_numpy(), size["n_eval"].to_numpy())
                row.update({"test": "corrected_ttest mse(a) - mse(b)", **{f"t_{k}": v for k, v in t.items()}})
            rows.append(row)
        if s == "main":
            rows.append({"row_type": "reference", "set": s, "group": g, "model": "sr_final_optimistic",
                         "featureset": FEATURESET, "fold": "mean25", "rmse": SR_FINAL_REF[g],
                         "note": "정식 SR 최종 식, 식 선택에 쓴 데이터로 잰 낙관적 값 (검정 제외)"})
    return pd.DataFrame(rows)


def recommend(sel):
    """6절 5번: 주 집합 바깥 fold에서 가장 많이 선택된 (method, k). 동률 → 작은 k → 안쪽 평균 RMSE 평균."""
    one = sel[(sel["set"] == "main") & (sel["group"] == GROUPS[0])]
    cnt = one.groupby(["method", "k"]).agg(n=("outer_r", "size"), inner=("inner_mean", "mean")).reset_index()
    cnt = cnt.sort_values(["n", "k", "inner"], ascending=[False, True, True])
    top = cnt.iloc[0]
    return top["method"], int(top["k"]), int(top["n"]), cnt


def loadings_and_check(data, method, k):
    main = frame_of("main", data)
    Z, _, info = fit_transform(main, [], method, k)
    W = info["weights"]
    # 표준화된 X (상관 계산용): fit_transform과 같은 통계값
    Xs = main[info["features"]].astype(float).copy()
    for g in GROUPS:
        m = (main["group"] == g).to_numpy()
        st = info["stats"][g]
        Xs.loc[m] = (Xs[m].fillna(st["median"][info["features"]]) - st["mean"]) / st["std"]
    fam = pd.read_csv(PROC / "feature_families_v1.csv").set_index("feature")["family"]
    rows = []
    for j, c in enumerate(W.columns):
        top = W[c].abs().sort_values(ascending=False).index[:N_TOP_LOADINGS]
        for rank, f in enumerate(top, 1):
            row = {"method": method, "k": k, "component": c, "x_var_ratio": float(info["x_var_ratio"][j]),
                   "rank": rank, "feature": f, "family": fam.get(f, ""), "weight": float(W.loc[f, c]),
                   "corr_pooled": float(np.corrcoef(Xs[f], Z[c])[0, 1])}
            for g in GROUPS:
                m = (main["group"] == g).to_numpy()
                row[f"corr_{g}"] = float(np.corrcoef(Xs[f][m], Z[c][m])[0, 1])
            rows.append(row)
    load = pd.DataFrame(rows)

    # 9절: PLS 1번 성분 대 네 피처 합 축 (테이블별 스피어만). 추천이 PCA면 PCA 1번 성분도 참고로.
    chk = []
    targets = [("PLS", k, True)] + ([(method, k, False)] if method != "PLS" else [])
    for mth, kk, judged in targets:
        Zc, _, inf = fit_transform(main, [], mth, kk)
        c1 = f"{mth}1"
        for g in GROUPS:
            m = (main["group"] == g).to_numpy()
            st = inf["stats"][g]
            z4 = (main.loc[m, PLS1_FEATURES].astype(float) - st["mean"][PLS1_FEATURES]) / st["std"][PLS1_FEATURES]
            score = Zc[c1][m]
            rho = float(pd.Series(score.to_numpy()).corr(pd.Series(z4.sum(axis=1).to_numpy()), method="spearman"))
            row = {"method": mth, "k": kk, "component": c1, "group": g, "judged": judged, "n": int(m.sum()),
                   "target": "+".join(PLS1_FEATURES), "spearman_sum_axis": rho, "abs_rho": abs(rho),
                   "threshold": PLS1_THRESHOLD, "prediction_met": (abs(rho) >= PLS1_THRESHOLD) if judged else None,
                   "spearman_mrr": float(pd.Series(score.to_numpy()).corr(
                       pd.Series(main.loc[m, "MRR"].to_numpy()), method="spearman"))}
            for f in PLS1_FEATURES:
                row[f"spearman_{f}"] = float(pd.Series(score.to_numpy()).corr(
                    pd.Series(main.loc[m, f].to_numpy()), method="spearman"))
            chk.append(row)
    return load, pd.DataFrame(chk)


# ---------------------------------------------------------------- 저장
def git_hash():
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/explore/dimred.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return h, bool(dirty)


def free_path(path):
    if not path.exists() and not path.with_suffix(".yaml").exists():
        return path
    i = 2
    while True:
        p = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not p.exists() and not p.with_suffix(".yaml").exists():
            return p
        i += 1


def write(df, path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    lines = []
    for k, v in meta.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"  {a}: {b}" for a, b in v.items()]
        else:
            lines.append(f"{k}: {v}")
    path.with_suffix(".yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("저장:", path, flush=True)


def selftest(data, pool):
    """누수·결정성 점검: (1) 적용 쪽 MRR은 결과에 영향이 없다 (2) 적용 쪽 피처를 바꿔도 학습 쪽 변환은 같다
    (3) 같은 입력이면 결과가 같다 (4) 적용 쪽의 순서·구성과 무관하게 같은 웨이퍼는 같은 점수."""
    main = frame_of("main", data)
    r, repeat, fold, tr, ev = outer_folds(main)[0]
    a, b = main[tr], main[ev]
    for mth in METHODS:
        Z1, (E1,), _ = fit_transform(a, [b], mth, 3)
        b2 = b.copy(); b2["MRR"] = np.random.default_rng(0).normal(size=len(b2)) * 1e3
        Z2, (E2,), _ = fit_transform(a, [b2], mth, 3)
        assert np.array_equal(Z1.to_numpy(), Z2.to_numpy()) and np.array_equal(E1.to_numpy(), E2.to_numpy()), "(1)"
        b3 = b.copy(); b3[pool] = b3[pool] * 7.0 + 3.0
        Z3, _, _ = fit_transform(a, [b3], mth, 3)
        assert np.array_equal(Z1.to_numpy(), Z3.to_numpy()), "(2)"
        Z4, (E4,), _ = fit_transform(a, [b], mth, 3)
        assert np.array_equal(E1.to_numpy(), E4.to_numpy()), "(3)"
        half = b.iloc[::2]
        _, (E5,), _ = fit_transform(a, [half.iloc[::-1]], mth, 3)
        assert np.allclose(E5.loc[half.index].to_numpy(), E1.loc[half.index].to_numpy(), atol=1e-10), "(4)"
        a_noy = a.drop(columns=["MRR"])
        try:
            fit_transform(a_noy, [b], mth, 3)
            raise AssertionError("MRR 없는 학습 쪽이 통과됐다")
        except KeyError:
            pass
    print("selftest 통과: 적용 쪽 MRR 무영향, 적용 쪽 피처 무영향, 결정성, 행 순서 무관", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pool = feature_pool()
    cands = sr_candidates()
    data = load_data()
    print(f"피처 풀 {len(pool)}개", flush=True)
    if args.selftest:
        selftest(data, pool)
        return
    ok, counts, lr = check_reproduction(data, cands, pool)
    if not ok:
        sys.exit("재현 실패: 주 집합 웨이퍼 수 또는 (b) RMSE가 다르다. 멈춘다.")
    print("재현 확인 통과", flush=True)
    if args.check:
        return

    commit, dirty = git_hash()
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    grids, sels, cmps = [], [], []
    for s in ("main", "aux"):
        g_, s_, c_ = run_set(s, data, cands, pool)
        grids.append(g_); sels.append(s_); cmps.append(c_)
    grid, sel, cmp_folds = pd.concat(grids), pd.concat(sels), pd.concat(cmps)
    summary = summarize(cmp_folds)
    compare = pd.concat([summary, cmp_folds], ignore_index=True)
    method, k, n_sel, counts_tbl = recommend(sel)
    load, chk = loadings_and_check(data, method, k)
    seconds = round(time.perf_counter() - t0, 1)

    stamp = started[:10]
    top = Path(args.out_dir) if args.out_dir else RESULTS
    diag = Path(args.out_dir) if args.out_dir else RESULTS / "diagnostics"
    n_sets = {s: {g: int((frame_of(s, data)["group"] == g).sum()) for g in GROUPS} for s in ("main", "aux")}
    base = {
        "date": stamp, "author": AUTHOR, "script": "src/explore/dimred.py", "plan": PLAN,
        "code_commit": commit, "code_dirty_at_run": dirty, "run_started": started, "run_seconds": seconds,
        "nature": "사후 탐색 (preregistration 사전확정 아님, 정식 판정을 바꾸지 않음)",
        "input": "data/processed/features_v1.csv + feature_families_v1.csv + folds_v1.csv + wafer_table.csv(MRR) + "
                 + CANDIDATES.relative_to(ROOT).as_posix(),
        "featureset": FEATURESET, "groups": GROUPS, "feature_pool_n": len(pool),
        "wafer_sets": {"main": f"A-456 {n_sets['main']['A-456']} / B-456 {n_sets['main']['B-456']} (정식 SR 후보 10개 결측 없음)",
                       "aux": f"A-456 {n_sets['aux']['A-456']} / B-456 {n_sets['aux']['B-456']} (전체, 학습 fold 중앙값 대치)"},
        "outer": "folds_v1 25-fold (r = repeat*5 + fold)",
        "inner": f"StratifiedGroupKFold({N_INNER}, shuffle=True, random_state=42+repeat), groups=WAFER_ID, 층화=group",
        "grid": f"method {METHODS} x k {KS}", "seed": SEED, "n_jobs": N_JOBS,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
        "repro_check": f"main {counts}, lr_sr10 {({g: round(v, 6) for g, v in lr.items()})}",
        "recommended": f"{method} k={k} ({n_sel}/25 바깥 fold)",
    }
    write(grid, free_path(top / f"{stamp}_{AUTHOR}_FV1_dimred_grid.csv"),
          {**base, "content": "집합 x 바깥 fold x 안쪽 fold x (method, k) x 테이블 안쪽 평가 오차 (5절 테이블별 선형회귀)"})
    write(sel, free_path(top / f"{stamp}_{AUTHOR}_FV1_dimred_selection.csv"),
          {**base, "content": "바깥 fold별 1-SE 선택 설정과 바깥 오차. method_min/k_min/rmse_outer_min은 1-SE 미적용 최저 설정 (참고용)",
           "selection_counts_main": "; ".join(f"{r.method}{r.k}: {r.n}" for r in counts_tbl.itertuples())})
    write(compare, free_path(top / f"{stamp}_{AUTHOR}_FV1_dimred_compare.csv"),
          {**base, "content": "7절 비교. row_type=summary(25-fold 평균, (a) 대 (b) 보정 t-검정), fold(fold별), reference(SR 최종 식 낙관적 참고값)"})
    write(load, free_path(diag / f"{stamp}_{AUTHOR}_dimred_loadings.csv"),
          {**base, "content": f"추천 설정({method} k={k})을 주 집합 전체로 한 번 맞춘 성분별 가중치 상위 {N_TOP_LOADINGS}개",
           "use": "해석 전용. 성능 평가에 쓰지 않는다 (전체 데이터로 맞춤)"})
    write(chk, free_path(diag / f"{stamp}_{AUTHOR}_dimred_pls1_check.csv"),
          {**base, "content": "9절 예상 확인: PLS 1번 성분 대 네 피처 테이블별 표준화 합 축의 테이블별 스피어만 상관",
           "use": "해석 전용 (주 집합 전체로 맞춤)", "threshold": PLS1_THRESHOLD})
    print(f"끝: {seconds}s, 추천 {method} k={k} ({n_sel}/25)", flush=True)


if __name__ == "__main__":
    main()

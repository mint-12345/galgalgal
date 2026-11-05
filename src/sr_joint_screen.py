"""공동 SR 1단계: 정식 SR 파레토 식을 두 테이블 공통 형태 f(x)로 두고 걸러 보기 (사후 탐색, 판정에 쓰지 않음).

계획: docs/plans/2026-11-04_joint_sr_plan.md (커밋 e37876d 이후 실행). 정식 SR 판정은 바꾸지 않는다.

모형: MRR_g = a_g · f(x) + b_g   (g = A-456, B-456)
  - f와 f 안의 상수는 두 테이블 공통, a_g·b_g만 테이블별.
  - 모든 상수는 각 fold 학습 쪽(두 그룹 합침)에서 비선형 최소제곱으로 한꺼번에 맞춘다.
    시작값: f 안 상수 = PySR 상수, a_g·b_g = 그 상수로 계산한 f에 그룹별 최소제곱. `^` 지수는 −3~3 (`sr_select.refit`과 같음).
  - 공통 형태 후보 = 정식 SR 파레토 식 281개(서로 다른 문자열 246개). PySR은 다시 돌리지 않는다.
척도: 변수 ÷ |중앙값|. 주 = 두 그룹 train+val 합침, 민감도 = 그룹별(정식 SR 방식).
웨이퍼: 정식 SR과 같은 집합(A-456 940 / B-456 951). fold = folds_v1 25개, 공유 WAFER_ID는 같은 쪽에 있다(계획 부록 C).
기준선 (같은 웨이퍼·같은 fold, 그룹별 평가):
  (가) lr_group  : 그룹별 선형회귀, 그 식의 변수, 절편·기울기 모두 그룹별 ← 핵심 비교
  (나) lr_shared : 공통 기울기 + 그룹별 절편 선형회귀 (두 그룹 합쳐 적합)
  (다) sr_final  : 정식 SR 그룹별 최종 식, 그룹 척도로 fold마다 상수 재적합 (낙관적)
  (라) rf, mean  : `sr_select`와 같은 RF(F-free 열, max_features=1.0), 학습 쪽 평균
  (다)·lr(후보 10개)·rf는 정식 SR `sr_folds.csv`와 같아야 한다 → 라벨·웨이퍼 정렬 검산.
라벨: `load.load_labels("train"/"val")`. test MRR과 data/locked/는 읽지 않는다.

실행: python src/sr_joint_screen.py
출력: results/{날짜}_김혜연_FV1_sr_joint_screen_folds.csv (상위 식·기준선의 fold별 지표)
      results/{날짜}_김혜연_FV1_sr_joint_screen_rank.csv  (식 281개 × 척도 2개 순위)
      results/{날짜}_김혜연_FV1_sr_joint_screen_tests.csv (상위 1개 대 기준선 보정 t-검정, 참고)
      + 같은 이름 .yaml. 이미 있으면 멈춘다.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_select as ss  # noqa: E402
from sr_cross_group import check_folds  # noqa: E402
from cv_common import corrected_ttest  # noqa: E402
from load import load_labels  # noqa: E402
from sr_run import CANDIDATES, GROUPS, MAX_ABS_EXPONENT, PROC, RESULTS  # noqa: E402

AUTHOR = "김혜연"
KEYS = ["WAFER_ID", "STAGE"]
SEED = 42
FEATURESET = "features_v1"
PARETO = RESULTS / "2026-11-01_박종화_FV1_sr_pareto.csv"
SELECTION = RESULTS / "diagnostics" / "2026-11-01_박종화_sr_selection.csv"
OFFICIAL_FOLDS = RESULTS / "2026-11-01_박종화_FV1_sr_folds.csv"
EXCLUDED = {"since_POLISH_TABLE_w"}
MAX_NFEV = 200          # sr_select.refit과 같은 반복 상한
MAX_NFEV_CHECK = 2000   # 상위 식 수렴 민감도
N_TOP = 5
N_JOBS = -1


# ---------------------------------------------------------------- 데이터
def candidates() -> dict:
    c = pd.read_csv(CANDIDATES)
    c = c[c["selected"] == 1]
    return {g: c[c["group"] == g].sort_values("rank")["feature"].tolist() for g in GROUPS}


def load_data(cands: dict) -> tuple[pd.DataFrame, list[str]]:
    """두 그룹 train+val(이상치 제외) 전체. 라벨은 load_labels로만 읽는다."""
    union = sorted((set(cands["A-456"]) | set(cands["B-456"])) - EXCLUDED)
    assert len(union) <= 14, f"후보 변수 {len(union)}개 (계획 상한 14개)"
    feats = pd.read_csv(PROC / "features_v1.csv")
    table = pd.read_csv(PROC / "wafer_table.csv", usecols=KEYS + ["split", "is_outlier"])
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    labels = pd.concat([load_labels("train"), load_labels("val")], ignore_index=True)
    labels = labels.rename(columns={"AVG_REMOVAL_RATE": "MRR"})[KEYS + ["MRR"]]
    d = (feats[feats["group"].isin(GROUPS)].merge(table, on=KEYS, validate="one_to_one")
         .merge(labels, on=KEYS, how="left", validate="one_to_one"))
    assert d["MRR"].notna().all(), "라벨이 없는 train/val 웨이퍼가 있다"
    return d, union


# ---------------------------------------------------------------- 공동 적합
def joint_refit(node, theta0, X, y, gB, max_nfev=MAX_NFEV):
    """(f 상수, [a_A, b_A, a_B, b_B], 수렴 여부). gB = B-456 행 표시."""
    ids = sorted(ss.const_ids(node))
    powers = ss.exponent_ids(node)
    theta = np.array(theta0, dtype=float)
    f0 = ss.predict(node, theta, X, len(y))
    ab0 = []
    for m in (~gB, gB):
        ok = m & np.isfinite(f0)
        if ok.sum() > 2 and np.ptp(f0[ok]) > 0:
            a, b = np.polyfit(f0[ok], y[ok], 1)
        else:
            a, b = 1.0, float(y[m].mean())
        ab0 += [a, b]
    low = np.array([-MAX_ABS_EXPONENT if i in powers else -np.inf for i in ids] + [-np.inf] * 4)
    high = np.array([MAX_ABS_EXPONENT if i in powers else np.inf for i in ids] + [np.inf] * 4)
    k = len(ids)

    def residual(values):
        trial = theta.copy()
        trial[ids] = values[:k]
        aA, bA, aB, bB = values[k:]
        f = ss.predict(node, trial, X, len(y))
        r = np.where(gB, aB * f + bB, aA * f + bA) - y
        r[np.isnan(r)] = 1e6
        return r

    x0 = np.clip(np.r_[theta[ids], ab0], low, high)
    res = least_squares(residual, x0, bounds=(low, high), max_nfev=max_nfev)
    theta[ids] = res.x[:k]
    return theta, res.x[k:], bool(res.status > 0)


def joint_predict(node, theta, ab, X, gB):
    f = ss.predict(node, theta, X, len(gB))
    return np.where(gB, ab[2] * f + ab[3], ab[0] * f + ab[1])


def metrics(group, model, r, err, n_train):
    return {"group": group, "model": model, "featureset": FEATURESET, "fold": r,
            "mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "n_train": int(n_train), "n_eval": int(err.size)}


def eval_joint(text, X, y, gB, masks, max_nfev=MAX_NFEV):
    """식 하나의 fold별 그룹 지표. 평가 쪽 예측이 계산되지 않으면 그 fold는 invalid."""
    eq = ss.parse_equation(text)
    rows, n_nonconv, n_invalid = [], 0, 0
    for r in sorted(masks):
        ev = masks[r]
        th, ab, conv = joint_refit(eq.node, eq.theta, ss._subset(X, ~ev), y[~ev], gB[~ev], max_nfev)
        n_nonconv += not conv
        p = joint_predict(eq.node, th, ab, ss._subset(X, ev), gB[ev])
        if np.isnan(p).any():
            n_invalid += 1
        for g, m in (("A-456", ~gB), ("B-456", gB)):
            e = p[m[ev]] - y[ev & m]
            n_tr = int((~ev & m).sum())
            row = metrics(g, "joint", r, e, n_tr) if np.isfinite(e).all() else \
                {"group": g, "model": "joint", "fold": r, "mse": np.inf, "rmse": np.inf, "n_train": n_tr,
                 "n_eval": int(e.size)}
            row.update({"a": ab[2] if g == "B-456" else ab[0], "b": ab[3] if g == "B-456" else ab[1],
                        "converged": conv})
            rows.append(row)
    return rows, n_nonconv, n_invalid


def lr_rows(text, X, y, gB, masks):
    """(가) 그룹별 선형회귀, (나) 공통 기울기 + 그룹별 절편. 변수는 그 식의 변수."""
    vars_ = sorted(ss.variables(ss.parse_equation(text).node))
    M = np.column_stack([X[v] for v in vars_]) if vars_ else np.zeros((len(y), 0))
    rows = []
    for r in sorted(masks):
        ev = masks[r]
        pg = np.empty(ev.sum())
        for m in (~gB, gB):
            tr, e = ~ev & m, ev & m
            fit = LinearRegression().fit(M[tr], y[tr]) if vars_ else None
            pg[m[ev]] = fit.predict(M[e]) if vars_ else y[tr].mean()
        D = np.column_stack([M, gB.astype(float)])
        ps = LinearRegression().fit(D[~ev], y[~ev]).predict(D[ev])
        for g, m in (("A-456", ~gB), ("B-456", gB)):
            n_tr = int((~ev & m).sum())
            rows.append(metrics(g, "lr_group", r, pg[m[ev]] - y[ev & m], n_tr))
            rows.append(metrics(g, "lr_shared", r, ps[m[ev]] - y[ev & m], n_tr))
    return rows


# ---------------------------------------------------------------- 기준선 (다)·(라)
def official_rows(d, cands, masks_g):
    """그룹별 정식 SR 최종 식(그룹 척도 재적합), 후보 10개 선형회귀, RF, 학습 쪽 평균."""
    sel = pd.read_csv(SELECTION, encoding="utf-8-sig")
    sel = sel[sel["rule"] == "1se"].set_index("group")
    rf_cols = ss.rf_columns()
    rows = []
    for g in GROUPS:
        t = d[d["group"] == g].reset_index(drop=True)
        names = cands[g]
        scale = t[names].abs().median()
        t = t[t[names].notna().all(axis=1)].reset_index(drop=True)
        t[names] = t[names] / scale   # sr_run.load_frame과 같이 후보 열만 척도 적용 (원척도 LR은 조건수 ~1e8)
        Xg = {n: t[n].to_numpy(float) for n in names}
        y = t["MRR"].to_numpy(float)
        eq = ss.parse_equation(sel.loc[g, "equation"])
        M = t[names].to_numpy(float)
        R = t[rf_cols].to_numpy(float)
        for r, ev in sorted(ss.fold_masks(t[KEYS]).items()):
            assert (ev == masks_g[g][r]).all(), f"{g} fold {r}: 웨이퍼 순서 불일치"
            n_tr = int((~ev).sum())
            th = ss.refit(eq.node, eq.theta, ss._subset(Xg, ~ev), y[~ev])
            preds = {
                "sr_final": ss.predict(eq.node, th, ss._subset(Xg, ev), int(ev.sum())),
                "lr_cand10": LinearRegression().fit(M[~ev], y[~ev]).predict(M[ev]),
                "rf": RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=N_JOBS)
                .fit(R[~ev], y[~ev]).predict(R[ev]),
                "mean": np.full(ev.sum(), y[~ev].mean()),
            }
            rows += [metrics(g, k, r, p - y[ev], n_tr) for k, p in preds.items()]
    return rows


# ---------------------------------------------------------------- 요약
def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    s = folds.groupby(["scale", "rank_key", "group", "model"], sort=False)["rmse"]
    return s.agg(rmse_mean="mean", rmse_se=lambda v: v.std(ddof=1) / np.sqrt(len(v))).reset_index()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=PROC.parents[1], text=True).strip()


def main() -> None:
    t0 = time.perf_counter()
    today = date.today().isoformat()
    stem = f"{today}_{AUTHOR}_FV1_sr_joint_screen"
    out_f, out_r, out_t = (RESULTS / f"{stem}_{k}.csv" for k in ("folds", "rank", "tests"))
    for p in (out_f, out_r, out_t, out_f.with_suffix(".yaml"), out_r.with_suffix(".yaml"), out_t.with_suffix(".yaml")):
        if p.exists():
            sys.exit(f"이미 있어서 멈춘다: {p}")

    cands = candidates()
    d, union = load_data(cands)
    # 정식 SR 웨이퍼 = 그룹별 후보 10개가 모두 있는 웨이퍼. 14개 합집합 기준과 같다(계획 부록 A).
    own = np.zeros(len(d), bool)
    for g in GROUPS:
        m = (d["group"] == g).to_numpy()
        own |= m & d[cands[g]].notna().all(axis=1).to_numpy()
    assert (own == d[union].notna().all(axis=1).to_numpy()).all()
    w = d[own].sort_values("group", kind="stable").reset_index(drop=True)   # A-456 먼저, 그룹 안 순서 유지
    n = w["group"].value_counts().to_dict()
    assert n == {"A-456": 940, "B-456": 951}, n
    gB = (w["group"] == "B-456").to_numpy()
    y = w["MRR"].to_numpy(float)

    masks = ss.fold_masks(w[KEYS])
    shared = set(w.loc[~gB, "WAFER_ID"]) & set(w.loc[gB, "WAFER_ID"])
    check_folds(w, masks)   # 두 그룹 합친 표: 공유 WAFER_ID가 학습·평가 쪽에 동시에 있으면 멈춘다
    masks_g = {g: {r: ev[(gB if g == "B-456" else ~gB)] for r, ev in masks.items()} for g in GROUPS}

    pooled = d[union].abs().median()
    Xs = {"pooled": {v: (w[v] / pooled[v]).to_numpy(float) for v in union}}
    gs = {g: d.loc[d["group"] == g, union].abs().median() for g in GROUPS}
    Xs["group"] = {v: np.where(gB, w[v] / gs["B-456"][v], w[v] / gs["A-456"][v]).astype(float) for v in union}

    pareto = pd.read_csv(PARETO)
    assert len(pareto) == 281 and pareto["exponent_ok"].all()
    texts = pareto["equation"].drop_duplicates().tolist()
    print(f"준비 {time.perf_counter() - t0:.0f}초: 식 {len(texts)}개(281행), 웨이퍼 {n}, 공유 WAFER_ID {len(shared)}", flush=True)

    # 1) 식 246개 × 척도 2개 공동 적합
    rank_rows, fold_store = [], {}
    for sc in ("pooled", "group"):
        t1 = time.perf_counter()
        out = Parallel(n_jobs=N_JOBS)(delayed(eval_joint)(t, Xs[sc], y, gB, masks) for t in texts)
        for text, (rows, nonconv, invalid) in zip(texts, out):
            f = pd.DataFrame(rows)
            gm = f.groupby("group")["rmse"].mean()
            rank_rows.append({"scale": sc, "equation": text, "rmse_A": gm["A-456"], "rmse_B": gm["B-456"],
                              "rmse_avg": (gm["A-456"] + gm["B-456"]) / 2, "n_fold_invalid": invalid,
                              "n_fold_nonconverged": nonconv})
            fold_store[(sc, text)] = f
        print(f"  척도 {sc}: {time.perf_counter() - t1:.0f}초", flush=True)
    rank = pd.DataFrame(rank_rows)
    meta = pareto.drop_duplicates("equation").set_index("equation")
    n_seeds = pareto.groupby("equation").apply(lambda p: ", ".join(f"{g}/{s}" for g, s in zip(p.group, p.seed)),
                                               include_groups=False)
    rank["source"] = rank["equation"].map(n_seeds)
    rank["complexity"] = rank["equation"].map(meta["complexity"])
    rank["rank"] = rank.groupby("scale")["rmse_avg"].rank(method="first").astype(int)
    rank = rank.sort_values(["scale", "rank"]).reset_index(drop=True)

    # 2) 상위 식 + 같은 변수 (가)·(나), 수렴 민감도
    folds = []
    top = {sc: rank[rank["scale"] == sc].head(N_TOP)["equation"].tolist() for sc in ("pooled", "group")}
    nfev_check = []
    for sc in ("pooled", "group"):
        for i, text in enumerate(top[sc], 1):
            key = f"top{i}"
            folds.append(fold_store[(sc, text)].assign(scale=sc, rank_key=key, equation=text))
            folds.append(pd.DataFrame(lr_rows(text, Xs[sc], y, gB, masks)).assign(scale=sc, rank_key=key,
                                                                                   equation=text))
            rows, nonconv, _ = eval_joint(text, Xs[sc], y, gB, masks, MAX_NFEV_CHECK)
            big = pd.DataFrame(rows).groupby("group")["rmse"].mean()
            base = fold_store[(sc, text)].groupby("group")["rmse"].mean()
            nfev_check.append({"scale": sc, "rank_key": key, "n_nonconv_2000": nonconv,
                               "max_abs_rmse_diff": float((big - base).abs().max())})
    off = pd.DataFrame(official_rows(d, cands, masks_g)).assign(scale="group", rank_key="official", equation="")
    folds = pd.concat(folds + [off], ignore_index=True)

    # 검산: (다)·lr·rf가 정식 SR fold 결과와 같은가
    ref = pd.read_csv(OFFICIAL_FOLDS, encoding="utf-8-sig")
    repro = {}
    for mine, theirs in (("sr_final", "sr_final"), ("lr_cand10", "lr"), ("rf", "rf")):
        a = off[off.model == mine].sort_values(["group", "fold"])["rmse"].to_numpy()
        b = ref[ref.model == theirs].sort_values(["group", "fold"])["rmse"].to_numpy()
        repro[mine] = float(np.abs(a - b).max())
    assert max(repro.values()) < 1e-6, f"정식 SR fold 결과 재현 실패: {repro}"   # sr_cross_group과 같은 기준

    # 3) 보정 t-검정 (참고): 상위 1개 공동 모형 대 (가), 그룹별, MSE 쌍 차이
    tests = []
    for sc in ("pooled", "group"):
        for g in GROUPS:
            f = folds[(folds.scale == sc) & (folds.rank_key == "top1") & (folds.group == g)]
            j = f[f.model == "joint"].set_index("fold").sort_index()
            for base in ("lr_group", "lr_shared"):
                b = f[f.model == base].set_index("fold").sort_index()
                t = corrected_ttest((j["mse"] - b["mse"]).to_numpy(), b["n_train"], b["n_eval"])
                tests.append({"scale": sc, "group": g, "vs": base, **t})
            sr = off[(off.group == g) & (off.model == "sr_final")].set_index("fold").sort_index()
            t = corrected_ttest((j["mse"] - sr["mse"]).to_numpy(), sr["n_train"], sr["n_eval"])
            tests.append({"scale": sc, "group": g, "vs": "sr_final", **t})
    tests = pd.DataFrame(tests)
    elapsed = time.perf_counter() - t0

    folds.round(8).to_csv(out_f, index=False, encoding="utf-8-sig")
    rank.round(8).to_csv(out_r, index=False, encoding="utf-8-sig")
    summ = summarize(folds)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 120):
        print(rank.groupby("scale").head(N_TOP).drop(columns=["source"]).round(4).to_string(index=False))
        print(summ.round(4).to_string(index=False))
        print(tests[["scale", "group", "vs", "mean_diff", "t", "p"]].round(5).to_string(index=False))
        print(pd.DataFrame(nfev_check).to_string(index=False))
    print("재현:", repro, f"실행 {elapsed:.0f}초")

    import joblib, scipy, sklearn
    head = {
        "date": today, "author": AUTHOR, "script": "src/sr_joint_screen.py",
        "nature": "사후 탐색(계획 커밋 이후 실행), 선택 낙관성 있음, 판정에 쓰지 않음. 정식 SR 판정은 바꾸지 않음",
        "plan": "docs/plans/2026-11-04_joint_sr_plan.md (커밋 e37876d), 1단계",
        "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src")),
        "featureset": FEATURESET, "seed": SEED,
        "model": "MRR_g = a_g * f(x) + b_g. f와 f 안 상수는 공통, a_g·b_g는 그룹별. 모든 상수를 fold 학습 쪽에서 함께 재적합",
        "refit": f"scipy least_squares(trf), max_nfev {MAX_NFEV}, 지수 −3~3, 시작값 = PySR 상수 + 그룹별 최소제곱 a·b",
        "scale_main": "pooled = 두 그룹 train+val 합친 |중앙값|", "scale_sensitivity": "group = 그룹별 |중앙값|",
        "wafers": {k: int(v) for k, v in n.items()}, "n_shared_wafer_id": len(shared), "n_equations": f"281행, 서로 다른 식 {len(texts)}개",
        "variables": union, "repro_max_abs_rmse_diff_vs_official": repro, "nfev_check": nfev_check,
        "inputs_sha256": {p.name: sha(p) for p in (PROC / "features_v1.csv", PROC / "wafer_table.csv",
                                                   PROC / "folds_v1.csv", PARETO, SELECTION, CANDIDATES)},
        "labels": "load_labels('train'), load_labels('val'). test MRR·data/locked/ 사용 안 함",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
                     "pysr": "사용 안 함"},
        "threads": {"joblib_n_jobs": N_JOBS, "rf_n_jobs": N_JOBS, "cpu_count": os.cpu_count(),
                    **{k: os.environ.get(k, "unset") for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                "MKL_NUM_THREADS")}},
        "runtime_sec": round(elapsed),
    }

    def to_yaml(obj):   # 값은 JSON 흐름 표기(YAML의 부분집합)로 쓴다. .venv에 PyYAML이 없다
        return "".join(f"{k}: {json.dumps(v, ensure_ascii=False)}\n" for k, v in obj.items())

    for p, extra in ((out_f, {"content": "상위 5개 식(척도별)의 공동 모형·(가) lr_group·(나) lr_shared fold 지표, "
                                         "rank_key=official: (다) sr_final·lr_cand10·(라) rf·mean"}),
                     (out_r, {"content": "식별 공동 모형 fold 평균 RMSE와 순위(두 그룹 평균)"}),
                     (out_t, {"content": "상위 1개 공동 모형 대 (가)·(나)·(다) 보정 t-검정(MSE 쌍 차이, 공동 − 기준선). 참고용"})):
        p.with_suffix(".yaml").write_text(to_yaml({**head, **extra}), encoding="utf-8")
    tests.round(6).to_csv(out_t, index=False, encoding="utf-8-sig")
    print("저장:", out_f.name, out_r.name)


if __name__ == "__main__":
    main()

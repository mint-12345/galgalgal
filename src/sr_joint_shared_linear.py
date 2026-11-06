"""공동 SR 사후 보조 분석 (계획 docs/plans/2026-11-06_joint_sr_shared_linear_plan.md).

#2 baseline : 전체 데이터 최종 f와 단순 기준선(단일 적분, 세 적분 조합, 연마 시간)의 테이블별 상관. 라벨 없음
              (참고로 MRR 표본 내 상관도 적는다).
#5 shared   : 공유 선형 함수 MRR = a_g·(wᵀz) + b_g를 2026-11-05와 같은 바깥 5-fold에서 학습 쪽만으로 적합하고
              공동 SR(저장된 OOF)·같은 변수 테이블별 LR과 비교한다.

사후 보조 분석이다. 기존 판정·결과·그림은 바꾸지 않는다. test 라벨과 data/locked/는 쓰지 않는다. PySR 탐색 없음.

실행: python src/sr_joint_shared_linear.py baseline | shared
"""
from __future__ import annotations

import hashlib
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_joint as sj  # noqa: E402
import sr_joint_outer5 as so  # noqa: E402
import sr_select as ss  # noqa: E402

STAMP = "2026-11-06"
PLAN = "docs/plans/2026-11-06_joint_sr_shared_linear_plan.md"
SEED = 42
N_BOOT = 2000
N_RANDOM_STARTS = 20
TOL, MAX_ITER = 1e-12, 5000
UNSTABLE = 1e-6
GROUPS = ["A-456", "B-456"]
NATURE = ("사후 보조 분석(2026-11-05 결과와 2026-11-06 사후 해석을 본 뒤 요청). 기존 판정·결과·그림 변경 없음. "
          "후보 14개는 전체 train+val 라벨로 선정된 것이라 외부 검증이 아님. test 라벨·data/locked/ 사용 안 함")
OUT5 = sj.RESULTS / f"{so.STAMP}_{sj.AUTHOR}_FV1_sr_joint_outer5"


# ================================================================ 공유 선형 모형
def _ab_given_s(s, y, gi):
    """테이블별 y ~ a_g·s + b_g (OLS)."""
    a, b = np.zeros(2), np.zeros(2)
    for g in (0, 1):
        m = gi == g
        sv, yv = s[m], y[m]
        vs = np.var(sv)
        a[g] = 0.0 if vs <= 0 else np.mean((sv - sv.mean()) * (yv - yv.mean())) / vs
        b[g] = yv.mean() - a[g] * sv.mean()
    return a, b


def _sse(Z, y, gi, w, a, b):
    r = y - (a[gi] * (Z @ w) + b[gi])
    return float(r @ r)


def _normalize(w, a):
    nw = np.linalg.norm(w)
    w, a = w / nw, a * nw
    if a[0] < 0:                       # 식별성: ‖w‖ = 1, a_A > 0
        w, a = -w, -a
    return w, a


def svd_start(Z, y, gi):
    """테이블별 OLS 기울기 행렬 [β_A β_B]의 rank-1 근사 첫 왼쪽 특이벡터."""
    B = np.column_stack([LinearRegression().fit(Z[gi == g], y[gi == g]).coef_ for g in (0, 1)])
    u, _, _ = np.linalg.svd(B, full_matrices=False)
    return u[:, 0]


def fit_shared_linear(Z, y, gi, w0):
    """교대 최소제곱. 반환: w(‖w‖=1), a(a_A>0), b, sse, 반복 수, 기울기 노름."""
    w = np.asarray(w0, float) / np.linalg.norm(w0)
    a, b = _ab_given_s(Z @ w, y, gi)
    sse = _sse(Z, y, gi, w, a, b)
    it = 0
    for it in range(1, MAX_ITER + 1):
        if np.allclose(a, 0):
            raise RuntimeError("a_A = a_B = 0: w가 식별되지 않음")
        A = a[gi][:, None] * Z                                  # (y - b_g) ~ a_g·zᵀw, 상수항 없음
        w = np.linalg.lstsq(A, y - b[gi], rcond=None)[0]
        w, a = _normalize(w, a)
        a, b = _ab_given_s(Z @ w, y, gi)
        new = _sse(Z, y, gi, w, a, b)
        done = (sse - new) <= TOL * max(sse, 1e-300)
        sse = new
        if done:
            break
    w, a = _normalize(w, a)
    # 수렴 점검: 모든 모수(w, a, b)에 대한 SSE 기울기의 노름 (척도: SSE)
    r = y - (a[gi] * (Z @ w) + b[gi])
    gw = -2 * (a[gi][:, None] * Z).T @ r
    ga = np.array([-2 * np.sum(r[gi == g] * (Z[gi == g] @ w)) for g in (0, 1)])
    gb = np.array([-2 * np.sum(r[gi == g]) for g in (0, 1)])
    grad = float(np.linalg.norm(np.concatenate([gw - (gw @ w) * w, ga, gb])) / max(sse, 1e-300))   # w는 구면 접선 성분
    return {"w": w, "a": a, "b": b, "sse": sse, "n_iter": it, "grad_rel": grad}


def standardize(Xtr, Xev):
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=0)
    assert (sd > 0).all()
    return (Xtr - mu) / sd, (Xev - mu) / sd, mu, sd


def shared_fold(M, y, gi, tr, ev, rng):
    """바깥 fold 하나. 학습 쪽(tr)만으로 적합, 평가 쪽(ev) 예측. y[ev]는 쓰지 않는다."""
    Ztr, Zev, mu, sd = standardize(M[tr], M[ev])
    ytr, gtr = y[tr], gi[tr]
    main = fit_shared_linear(Ztr, ytr, gtr, svd_start(Ztr, ytr, gtr))
    gaps, angles, iters = [], [], [main["n_iter"]]
    for _ in range(N_RANDOM_STARTS):
        r = fit_shared_linear(Ztr, ytr, gtr, rng.standard_normal(Ztr.shape[1]))
        gaps.append((r["sse"] - main["sse"]) / main["sse"])
        angles.append(np.degrees(np.arccos(np.clip(abs(r["w"] @ main["w"]), -1, 1))))
        iters.append(r["n_iter"])
    pred = main["a"][gi[ev]] * (Zev @ main["w"]) + main["b"][gi[ev]]
    stab = {"sse_train": main["sse"], "n_iter_svd_start": main["n_iter"], "grad_rel": main["grad_rel"],
            "max_rel_sse_gap_random": float(max(gaps)), "min_rel_sse_gap_random": float(min(gaps)),
            "max_angle_deg_random": float(max(angles)), "max_iter_any_start": int(max(iters)),
            "unstable": bool(max(abs(g) for g in gaps) > UNSTABLE)}
    return pred, main, stab, mu, sd


# ================================================================ 공통
def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def write_yaml(path, meta):
    lines = []
    for k, v in meta.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"  {kk}: {vv}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def env():
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}


def inputs_hash():
    p = sj.PROC if hasattr(sj, "PROC") else sj.ROOT / "data" / "processed"
    files = [p / "features_v1.csv", p / "wafer_table.csv", p / "folds_v1.csv",
             sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_selection.csv",
             Path(f"{OUT5}_selection.csv"), Path(f"{OUT5}_oof.csv"), Path(f"{OUT5}_structure.csv")]
    return {f.name: sha16(f) for f in files if f.exists()}


def full_f(X, y, gi):
    """2026-11-04 전체 데이터 최종 f, 전체 데이터 재적합 상수 (structure_table과 같은 계산)."""
    sel = pd.read_csv(sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_selection.csv", encoding="utf-8-sig")
    text = sel[sel["rule"] == "1se"]["equation"].iloc[0]
    eq = ss.parse_equation(text)
    tf, af, bf, _ = sj.joint_refit(eq.node, eq.theta, X, y, gi)
    return ss.predict(eq.node, tf, X, len(y)), text, tf


# ================================================================ #2
def cmd_baseline():
    t0 = time.perf_counter()
    out = sj.RESULTS / "diagnostics" / f"{STAMP}_{sj.AUTHOR}_sr_joint_fullf_baseline_corr.csv"
    sj.guard(out, out.with_suffix(".yaml"))
    d, X, names, _ = so.joint_data()
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    f, text, tf = full_f(X, y, gi)
    # 점검: 저장된 fold f와의 상관 재현
    st = pd.read_csv(f"{OUT5}_structure.csv", encoding="utf-8-sig", dtype={"fold": str})
    sel5 = pd.read_csv(f"{OUT5}_selection.csv", encoding="utf-8-sig")
    for _, r in sel5.iterrows():
        fk = so.f_values(r["equation_search"], [], X, len(y))
        for i, g in enumerate("AB"):
            c = np.corrcoef(fk[gi == i], f[gi == i])[0, 1]
            saved = st[st["fold"] == str(r["fold"])][f"corr_f_full_{g}"].iloc[0]
            assert abs(c - saved) < 1e-9, (r["fold"], g, c, saved)
    g1 = np.log(X["P1_on"]) - X["auc_RETAINER_s1"]
    assert np.min(np.abs(g1)) > 1e-6
    base = {
        "B1_MAIN_OUTER": X["auc_MAIN_OUTER_s1"],
        "B2_log_MAIN_OUTER": np.log(X["auc_MAIN_OUTER_s1"]),
        "B3_sum_3integrals": X["auc_MAIN_OUTER_s1"] + X["auc_SLURRY_C_s1"] + X["auc_PCHAMBER_s1"],
        "B4_logprod_3integrals": np.log(X["auc_MAIN_OUTER_s1"]) + np.log(X["auc_SLURRY_C_s1"]) + np.log(X["auc_PCHAMBER_s1"]),
        "B5_dur_POLISH": X["dur_POLISH_s1"],
        "B6_log_dur_POLISH": np.log(X["dur_POLISH_s1"]),
        "G1_factor1_retainer": g1,
        "G2_factor2_log_integrals": f / g1,
        "f_full": f,
    }
    keep_all = np.ones(len(y), bool)
    drop = ~((d["WAFER_ID"] == so.DROP[0]) & (d["STAGE"] == so.DROP[1])).to_numpy()
    rows = []
    for tag, keep in (("all", keep_all), ("drop_extreme", drop)):
        for i, g in enumerate(GROUPS):
            m = (gi == i) & keep
            for nm, v in base.items():
                ok = m & np.isfinite(v)
                rows.append({"featureset": sj.FEATURESET, "subset": tag, "group": g, "baseline": nm, "n": int(ok.sum()),
                             "pearson_with_f": float(np.corrcoef(v[ok], f[ok])[0, 1]),
                             "spearman_with_f": float(spearmanr(v[ok], f[ok]).statistic),
                             "pearson_with_mrr_insample": float(np.corrcoef(v[ok], y[ok])[0, 1]),
                             "spearman_with_mrr_insample": float(spearmanr(v[ok], y[ok]).statistic)})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    single = df[(df["subset"] == "all") & df["baseline"].isin(["B1_MAIN_OUTER", "B2_log_MAIN_OUTER", "B5_dur_POLISH",
                                                               "B6_log_dur_POLISH"])]
    hit = single.groupby("baseline")["pearson_with_f"].apply(lambda s: bool((s.abs() >= 0.95).all()))
    write_yaml(out.with_suffix(".yaml"), {
        "nature": NATURE, "date_run": STAMP, "author": sj.AUTHOR, "plan": PLAN, "section": "#2 (계획 2절)",
        "f_full_equation": text, "f_full_theta_refit": [float(t) for t in tf],
        "scaling": "합산 |중앙값| (공동 SR과 같음, 라벨 미사용)",
        "check": "fold f와 f_full의 상관이 2026-11-05 _structure.csv 값과 1e-9 이내로 일치 (통과)",
        "rule_single_baseline_abs_r_ge_095_both_tables": {k: v for k, v in hit.items()},
        "mrr_corr_note": "pearson/spearman_with_mrr_insample은 train+val 표본 내 값. f_full은 같은 라벨로 찾았으므로 낙관적. 판정에 쓰지 않음",
        "inputs_sha256_16": inputs_hash(), "env": env(), "runtime_sec": round(time.perf_counter() - t0, 1)})
    print(df[df["subset"] == "all"].round(3).to_string(index=False))
    print("단일 기준선 규칙(|r|≥0.95 두 테이블):", dict(hit))


# ================================================================ #5
def boot_ci(y, p1, p2, folds, rng):
    def rmse(e):
        return np.sqrt(np.mean(e ** 2))
    obs = rmse(y - p1) - rmse(y - p2)
    idx = [np.flatnonzero(folds == k) for k in np.unique(folds)]
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx])
        bs[b] = rmse(y[i] - p1[i]) - rmse(y[i] - p2[i])
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def classify(oof_diff, wins_first, n_folds, lo, hi, labels):
    """계획 3.5절. labels = (첫 모형 우세, 둘째 모형 우세, 불확실)."""
    if oof_diff < 0 and wins_first >= 4:
        return labels[0] + (" (구간 0 미포함)" if hi < 0 else " (구간 0 포함)")
    if oof_diff > 0 and (n_folds - wins_first) >= 4:
        return labels[1] + (" (구간 0 미포함)" if lo > 0 else " (구간 0 포함)")
    return labels[2]


def cmd_shared():
    t0 = time.perf_counter()
    base = sj.RESULTS / f"{STAMP}_{sj.AUTHOR}_FV1_sr_joint_sharedlinear"
    outs = {s: base.with_name(f"{base.name}{s}.csv") for s in ("", "_oof", "_summary", "_coefs")}
    sj.guard(*outs.values(), *[p.with_suffix(".yaml") for p in outs.values()])
    d, X, names, masks = so.joint_data()
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    sel5 = pd.read_csv(f"{OUT5}_selection.csv", encoding="utf-8-sig")
    oof5 = pd.read_csv(f"{OUT5}_oof.csv", encoding="utf-8-sig")
    rng_starts = np.random.default_rng(SEED)
    metric_rows, oof_rows, coef_rows = [], [], []
    for k in so.FOLDS:
        ev = masks[k]
        tr = ~ev
        vars_k = sel5[sel5["fold"] == k]["variables"].iloc[0].split()
        M = np.column_stack([X[c] for c in vars_k])
        p_sh, fit, stab, mu, sd = shared_fold(M, y, gi, tr, ev, rng_starts)
        p_lr, _ = so.lr_fit_predict(M, y, gi, tr, ev)
        keys = d.loc[ev, sj.KEYS].reset_index(drop=True)
        saved = oof5[oof5["fold"] == k]
        for model, p in (("lr_group_samevars", p_lr),):
            s = keys.merge(saved[saved["model"] == model], on=sj.KEYS, how="left")
            assert np.allclose(s["mrr_pred"].to_numpy(), p, atol=1e-8), f"LR 재현 실패 fold {k}"
        sj_ = keys.merge(saved[saved["model"] == "joint_sr"], on=sj.KEYS, how="left")
        p_js = sj_["mrr_pred"].to_numpy()
        assert np.allclose(sj_["mrr_true"].to_numpy(), y[ev])
        for model, p in (("joint_sr", p_js), ("shared_linear_samevars", p_sh), ("lr_group_samevars", p_lr)):
            for g in (0, 1):
                m = gi[ev] == g
                metric_rows.append({"featureset": sj.FEATURESET, "group": GROUPS[g], "model": model, "fold": k,
                                    **sj._metrics(p[m], y[ev][m]), "n_train": int((tr & (gi == g)).sum()),
                                    "n_vars": len(vars_k)})
            oof_rows.append(pd.DataFrame({"featureset": sj.FEATURESET, **{c: keys[c] for c in sj.KEYS},
                                          "group": [GROUPS[g] for g in gi[ev]], "model": model, "fold": k,
                                          "mrr_true": y[ev], "mrr_pred": p}))
        # 원래 척도(공동 SR과 같은 |중앙값| 척도화 변수) 기준 기울기: β_g = a_g·w/sd
        beta = {g: fit["a"][i] * fit["w"] / sd for i, g in enumerate(GROUPS)}
        coef_rows.append({"featureset": sj.FEATURESET, "fold": k, "variables": " ".join(vars_k), "n_vars": len(vars_k),
                          "w_std": " ".join(f"{v:.6f}" for v in fit["w"]),
                          "a_A": fit["a"][0], "a_B": fit["a"][1], "b_A": fit["b"][0], "b_B": fit["b"][1],
                          "ratio_aA_aB": fit["a"][0] / fit["a"][1],
                          "beta_A_scaledvars": " ".join(f"{v:.4f}" for v in beta["A-456"]),
                          "beta_B_scaledvars": " ".join(f"{v:.4f}" for v in beta["B-456"]),
                          "z_mean": " ".join(f"{v:.6g}" for v in mu), "z_sd": " ".join(f"{v:.6g}" for v in sd),
                          "n_params_shared": len(vars_k) + 3, "n_params_lr": 2 * (len(vars_k) + 1), **stab})
    met, oof, coefs = pd.DataFrame(metric_rows), pd.concat(oof_rows, ignore_index=True), pd.DataFrame(coef_rows)
    drop = ~((oof["WAFER_ID"] == so.DROP[0]) & (oof["STAGE"] == so.DROP[1]))
    rng = np.random.default_rng(SEED)
    summ = []
    pairs = [("joint_sr", "shared_linear_samevars", ("공동 SR 우세", "공유 선형 우세", "차이 불확실")),
             ("shared_linear_samevars", "lr_group_samevars", ("공유 선형 우세(공유 비용 없음)", "LR 우세(공유 비용 있음)", "차이 불확실"))]
    for subset, keep in (("all", np.ones(len(oof), bool)), ("drop_extreme_scoreonly", drop.to_numpy())):
        o = oof[keep]
        for g in GROUPS:
            og = o[o["group"] == g]
            w = og.pivot_table(index=sj.KEYS + ["fold", "mrr_true"], columns="model", values="mrr_pred").reset_index()
            yy, ff = w["mrr_true"].to_numpy(), w["fold"].to_numpy()
            for model in ("joint_sr", "shared_linear_samevars", "lr_group_samevars"):
                fr = [np.sqrt(np.mean((w[w["fold"] == k][model] - w[w["fold"] == k]["mrr_true"]) ** 2)) for k in so.FOLDS]
                fm = [np.mean(np.abs(w[w["fold"] == k][model] - w[w["fold"] == k]["mrr_true"])) for k in so.FOLDS]
                summ.append({"featureset": sj.FEATURESET, "subset": subset, "group": g, "comparison": model,
                             "rmse_fold_mean": np.mean(fr), "rmse_fold_sd": np.std(fr, ddof=1),
                             "mae_fold_mean": np.mean(fm), "mae_fold_sd": np.std(fm, ddof=1),
                             "rmse_oof": float(np.sqrt(np.mean((w[model] - yy) ** 2))),
                             "mae_oof": float(np.mean(np.abs(w[model] - yy))), "n_oof": len(w)})
            for m1, m2, labels in pairs:
                fr1 = np.array([np.sqrt(np.mean((w[w["fold"] == k][m1] - w[w["fold"] == k]["mrr_true"]) ** 2)) for k in so.FOLDS])
                fr2 = np.array([np.sqrt(np.mean((w[w["fold"] == k][m2] - w[w["fold"] == k]["mrr_true"]) ** 2)) for k in so.FOLDS])
                diff = fr1 - fr2
                obs, lo, hi = boot_ci(yy, w[m1].to_numpy(), w[m2].to_numpy(), ff, rng)
                wins = int((diff < 0).sum())
                summ.append({"featureset": sj.FEATURESET, "subset": subset, "group": g, "comparison": f"{m1} - {m2}",
                             "rmse_fold_mean": diff.mean(), "rmse_fold_sd": diff.std(ddof=1),
                             "rmse_oof": obs, "rmse_oof_rel": obs / np.sqrt(np.mean((w[m2] - yy) ** 2)),
                             "boot_ci_lo": lo, "boot_ci_hi": hi, "first_better_folds": wins,
                             "diff_by_fold": " ".join(f"{v:+.3f}" for v in diff), "n_oof": len(w),
                             "class": classify(obs, wins, len(so.FOLDS), lo, hi, labels) if subset == "all" else "(사후 민감도)"})
    summ = pd.DataFrame(summ)
    met.to_csv(outs[""], index=False, encoding="utf-8-sig")
    oof.to_csv(outs["_oof"], index=False, encoding="utf-8-sig")
    summ.to_csv(outs["_summary"], index=False, encoding="utf-8-sig")
    coefs.to_csv(outs["_coefs"], index=False, encoding="utf-8-sig")
    meta = {"nature": NATURE, "date_run": STAMP, "author": sj.AUTHOR, "plan": PLAN, "section": "#5 (계획 3절)",
            "model": "MRR_i = a_g·(wᵀz_i) + b_g. z = 그 fold 공동 SR f의 변수(합산 |중앙값| 척도) → 학습 쪽 평균·SD로 표준화",
            "identifiability": "‖w‖₂ = 1, a_A > 0, w에 상수항 없음. 자유 모수 p+3",
            "estimation": f"교대 최소제곱, 상대 SSE 감소 < {TOL} 또는 {MAX_ITER}회. 초기값 = 테이블별 OLS 기울기 행렬 rank-1 SVD",
            "stability": f"fold마다 무작위 시작 {N_RANDOM_STARTS}개(SEED {SEED}); 상대 SSE 차이 > {UNSTABLE}이면 unstable",
            "regularization": "없음 (공동 SR 재적합·테이블별 LR과 같은 무정규화 최소제곱)",
            "eval_labels": "평가 쪽 MRR은 채점에만 사용",
            "checks": "테이블별 LR 예측이 2026-11-05 저장 OOF와 1e-8 이내 일치 (통과); 공동 SR은 저장 OOF 사용",
            "bootstrap": f"테이블별, fold 층화 웨이퍼 재표집 {N_BOOT}회, SEED {SEED}, 적합된 예측 고정",
            "classification": "계획 3.5절: OOF RMSE 낮고 4/5 fold 이상 → 우세; 구간 0 미포함일 때만 '실질적 이점'",
            "inputs_sha256_16": inputs_hash(), "env": env(), "runtime_sec": round(time.perf_counter() - t0, 1)}
    for p in outs.values():
        write_yaml(p.with_suffix(".yaml"), {**meta, "content": p.name})
    pd.set_option("display.width", 250)
    print(coefs.drop(columns=["z_mean", "z_sd", "w_std", "beta_A_scaledvars", "beta_B_scaledvars", "variables"]).round(6).to_string(index=False))
    print(summ.round(4).to_string(index=False))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"baseline": cmd_baseline, "shared": cmd_shared}[cmd]()


if __name__ == "__main__":
    main()

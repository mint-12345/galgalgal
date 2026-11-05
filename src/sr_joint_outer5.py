"""공동 SR 바깥 5-fold 검증 (사후 추가 검증, 계획 docs/plans/2026-11-05_joint_sr_outer5_plan.md).

`sr_joint_holdout.py`(바깥 fold 0 하나)를 반복 0의 fold 0~4로 일반화한다. 개별 SR은 다시 탐색하지 않는다.
각 바깥 fold k에서
  1. 평가 쪽(folds_v1 r = k, 약 20%)을 빼고 나머지 80%만으로 공동 SR을 처음부터 탐색 (sr_joint_search 정식 설정, 시드 42~51)
  2. 안쪽 4 fold(반복 0의 나머지 fold)에서 후보식 평가, 1-표준오차 규칙(SE = sd/√4)으로 f 선택
  3. 바깥 학습 쪽 전체로 f 공통 상수 + a_g·b_g 재적합 → 평가 쪽 예측
  4. 같은 변수의 테이블별 선형회귀(바깥 학습 쪽 적합) → 같은 평가 쪽 예측
바깥 평가 쪽 MRR은 채점에만 쓴다 (tests/test_sr_joint_outer5_전준혁.py가 확인).
fold 0은 기존 탐색 캐시(holdout_joint_p0~p4.jsonl)를 그대로 읽어 다시 계산하고 기존 결과와 일치하는지 확인한다.

명령:
  python src/sr_joint_outer5.py search --folds 1 2 3 4     (여러 프로세스로 동시에 띄워도 됨: 시드·fold 단위 claim 파일)
  python src/sr_joint_outer5.py unclaim                    (실행 중인 작업자가 없을 때만: 끊긴 claim 정리)
  python src/sr_joint_outer5.py status
  python src/sr_joint_outer5.py evaluate                   (5개 fold 모두 끝난 뒤)
캐시: ../galgalgal_joint_cache/outer5/fold{k}_seed{s}.json (fold별·시드별 파일, 원자적 쓰기)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_joint as sj  # noqa: E402
import sr_select as ss  # noqa: E402

STAMP = "2026-11-05"
PLAN = "docs/plans/2026-11-05_joint_sr_outer5_plan.md"
FOLDS = [0, 1, 2, 3, 4]               # folds_v1 반복 0 (r = 0~4)
SEEDS = list(range(42, 52))
OUT_CACHE = sj.CACHE / "outer5"
DROP = (-4230160594, "B")             # B-456 연마 시간 98초 웨이퍼 (2026-11-04 결과를 본 뒤 주목, 사후 민감도 전용)
INTEG = {"auc_SLURRY_C_s1", "auc_PCHAMBER_s1", "auc_MAIN_OUTER_s1"}
RETAINER = "auc_RETAINER_s1"
NATURE = ("사후 추가 검증(2026-11-04 공동 SR 결과와 바깥 fold 0 결과를 본 뒤 계획). 기존 판정(계획 3949cb4 9절)은 바꾸지 않음."
          " 후보 14개는 전체 train+val로 고른 정식 SR 후보의 합집합이라 완전한 외부 검증이 아님")


def inner_of(k: int) -> list[int]:
    return [r for r in FOLDS if r != k]


# ---------------------------------------------------------------- 데이터
def joint_data():
    """(웨이퍼 표, 척도 적용 X, 후보 14개, fold 마스크). sr_joint_holdout.joint_data와 같다."""
    d, full, names = sj.joint_frame()
    X = sj.design(d, names, sj.scale_table(full, names), "pooled")
    masks = ss.fold_masks(d[sj.KEYS])
    return d, X, names, masks


def eval_key_hash(d, ev) -> str:
    keys = d.loc[ev, sj.KEYS].astype(str).agg("|".join, axis=1).sort_values()
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- 탐색 (fold·시드 단위 캐시)
def seed_path(k: int, seed: int) -> Path:
    return OUT_CACHE / f"fold{k}_seed{seed}.json"


def search_one(k, seed, d, X, names, masks):
    import sr_joint_search as sjs
    import sr_run
    tr = ~masks[k]
    M = np.column_stack([X[c][tr] for c in names] + [d["gi"].to_numpy(float)[tr] + 1.0])
    y = d["MRR"].to_numpy(float)[tr]
    t0 = time.perf_counter()
    m = sjs.make_model(names, seed, sr_run.NITERATIONS)
    m.fit(M, y, variable_names=names + ["g"])
    eqs = [{"complexity": int(e["complexity"]), "loss": float(e["loss"]),
            "equation": sjs.split_template(str(e["equation"]), names)["f"]} for _, e in m.equations_.iterrows()]
    rec = {"mode": "joint", "fold": k, "seed": seed, "n_train": int(tr.sum()),
           "n_train_by_group": {g: int((tr & (d["gi"].to_numpy() == i)).sum()) for i, g in enumerate(sj.GROUPS)},
           "eval_keys_sha16": eval_key_hash(d, masks[k]), "niterations": sr_run.NITERATIONS,
           "fit_seconds": round(time.perf_counter() - t0, 1), "pid": os.getpid(), "equations": eqs}
    p = seed_path(k, seed)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return rec


def cmd_search(args):
    assert 0 not in args.folds, "fold 0은 기존 캐시(holdout_joint_p*.jsonl)를 재사용한다"
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    d, X, names, masks = joint_data()
    for k in args.folds:
        for seed in args.seeds:
            p = seed_path(k, seed)
            if p.exists():
                continue
            claim = p.with_suffix(".claim")
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            try:
                if p.exists():
                    continue
                print(f"{time.strftime('%H:%M:%S')} fold {k} 시드 {seed} 시작 (pid {os.getpid()})", flush=True)
                rec = search_one(k, seed, d, X, names, masks)
                print(f"{time.strftime('%H:%M:%S')} fold {k} 시드 {seed}: {rec['fit_seconds']}초 "
                      f"(n_train {rec['n_train']})", flush=True)
            finally:
                claim.unlink(missing_ok=True)


def cmd_unclaim(args):
    for c in OUT_CACHE.glob("*.claim"):
        print("claim 삭제:", c.name)
        c.unlink()
    for t in OUT_CACHE.glob("*.tmp*"):
        print("임시 파일 삭제:", t.name)
        t.unlink()


def cmd_status(args):
    for k in FOLDS[1:]:
        done = [s for s in SEEDS if seed_path(k, s).exists()]
        run = [s for s in SEEDS if seed_path(k, s).with_suffix(".claim").exists()]
        print(f"fold {k}: 완료 {len(done)}/10 {done}  진행 중 {run}")


def load_records(k: int) -> list[dict]:
    if k == 0:
        recs = []
        for p in sorted(sj.CACHE.glob("holdout_joint_p*.jsonl")):       # 기존 fold 0 탐색 (2026-11-05 커밋 c3542df)
            recs += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    else:
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT_CACHE.glob(f"fold{k}_seed*.json"))]
    assert sorted(r["seed"] for r in recs) == SEEDS, f"fold {k} 시드: {sorted(r['seed'] for r in recs)}"
    return recs


def candidates(recs: list[dict]) -> pd.DataFrame:
    """sr_joint_holdout.load_cache와 같은 규칙: 지수 범위 밖 식 제외, 같은 식은 하나로."""
    import sr_run
    rows = pd.DataFrame([{"seed": r["seed"], **e} for r in recs for e in r["equations"]])
    rows = rows[rows["equation"].map(sr_run.exponent_ok)]
    return rows.sort_values("complexity").drop_duplicates("equation")


# ---------------------------------------------------------------- 한 바깥 fold 평가
def score_inner(cands, X, y, gi, masks, k):
    rows = []
    for _, c in cands.iterrows():
        eq = ss.parse_equation(c["equation"])
        inner = []
        for r in inner_of(k):
            ev = masks[r]
            tr = ~masks[k] & ~ev
            t, a, b, _ = sj.joint_refit(eq.node, eq.theta, ss._subset(X, tr), y[tr], gi[tr])
            p = sj.joint_predict(eq.node, t, a, b, ss._subset(X, ev), gi[ev])
            ms = [sj._metrics(p[gi[ev] == g], y[ev][gi[ev] == g]) for g in (0, 1)]
            inner.append(np.inf if any(m["n_eval_nan"] for m in ms) else (ms[0]["rmse"] + ms[1]["rmse"]) / 2)
        inner = np.array(inner)
        rows.append({"equation": c["equation"], "complexity": int(c["complexity"]), "mean": inner.mean(),
                     "se": inner.std(ddof=1) / np.sqrt(len(inner)) if np.isfinite(inner).all() else np.inf})
    return pd.DataFrame(rows)


def pick_1se(scores: pd.DataFrame) -> pd.Series:
    ok = scores[np.isfinite(scores["mean"])]
    curve = ok.sort_values(["complexity", "mean", "equation"]).drop_duplicates("complexity")
    best = curve.loc[curve["mean"].idxmin()]
    return curve[curve["mean"] <= best["mean"] + best["se"]].sort_values("complexity").iloc[0]


def lr_fit_predict(M, y, gi, tr, ev):
    """같은 변수의 테이블별 선형회귀 (가). 학습 쪽 tr만으로 적합."""
    p, coefs = np.full(ev.sum(), np.nan), {}
    for g in (0, 1):
        mt, me = tr & (gi == g), gi[ev] == g
        m = LinearRegression().fit(M[mt], y[mt])
        p[me] = m.predict(M[ev][me])
        coefs[sj.GROUPS[g]] = {"intercept": float(m.intercept_), "coef": [float(v) for v in m.coef_]}
    return p, coefs


def run_fold(k, cands, X, y, gi, masks, names):
    """바깥 fold k 하나. 반환: 선택 정보, 안쪽 점수표, 평가 쪽 예측(공동·LR). y[masks[k]]는 채점에만 쓰인다."""
    scores = score_inner(cands, X, y, gi, masks, k)
    pick = pick_1se(scores)
    eq = ss.parse_equation(pick["equation"])
    ev, tr = masks[k], ~masks[k]
    t, a, b, info = sj.joint_refit(eq.node, eq.theta, ss._subset(X, tr), y[tr], gi[tr])
    p_joint = sj.joint_predict(eq.node, t, a, b, ss._subset(X, ev), gi[ev])
    used = [c for c in names if c in ss.variables(eq.node)]
    M = np.column_stack([X[c] for c in used])
    p_lr, lr_coefs = lr_fit_predict(M, y, gi, tr, ev)
    sel = {"fold": k, "equation_search": pick["equation"], "equation_refit": render(eq.node, t),
           "skeleton": ss.skeleton(eq.node), "complexity": int(pick["complexity"]), "inner_mean": float(pick["mean"]),
           "inner_se": float(pick["se"]), "variables": used, "theta": t.tolist(), "a": a.tolist(), "b": b.tolist(),
           "refit": info, "lr": lr_coefs, "n_candidates": int(len(cands))}
    return sel, scores, p_joint, p_lr


def render(node, theta) -> str:
    kind = node[0]
    if kind == "const":
        return f"{theta[node[1]]:.4g}"
    if kind == "var":
        return node[1]
    if kind == "neg":
        return f"(-{render(node[1], theta)})"
    if kind == "call":
        return f"{node[1]}({render(node[2], theta)})"
    op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}[kind]
    return f"({render(node[1], theta)} {op} {render(node[2], theta)})"


# ---------------------------------------------------------------- 구조 비교 (계획 5절, 결과 전 정의)
def strip_affine(node):
    """맨 바깥의 상수 덧셈·상수 곱·부호·상수 나눗셈을 벗긴다 (a_g·b_g가 흡수)."""
    while True:
        kind = node[0]
        if kind == "neg":
            node = node[1]
        elif kind in ("add", "sub", "mul") and node[1][0] == "const":
            node = node[2]
        elif kind in ("add", "sub", "mul", "div") and node[2][0] == "const":
            node = node[1]
        else:
            return node


def is_skeleton_star(node) -> bool:
    """S*: (auc_RETAINER_s1가 든 인수) × log(INTEG 중 2개 이상이 든 인자). 맨 바깥 상수 변환은 무시."""
    n = strip_affine(node)
    if n[0] != "mul":
        return False
    for lg, other in ((n[1], n[2]), (n[2], n[1])):
        lg = strip_affine(lg)
        if lg[0] == "call" and lg[1] == "log" and len(ss.variables(lg[2]) & INTEG) >= 2 \
                and RETAINER in ss.variables(other):
            return True
    return False


def const_free(text: str) -> str:
    return re.sub(r"(?<![A-Za-z_0-9.])-?\d+(\.\d+)?(e-?\d+)?", "C", text)


def f_values(eq_text, theta, X, n):
    eq = ss.parse_equation(eq_text)
    return ss.predict(eq.node, np.asarray(theta, float), X, n)


# ---------------------------------------------------------------- 명령: 평가
def cmd_evaluate(args):
    t0 = time.perf_counter()
    base = sj.RESULTS / f"{STAMP}_{sj.AUTHOR}_FV1_sr_joint_outer5"
    outs = {s: base.with_name(f"{base.name}{s}.csv") for s in
            ("", "_oof", "_selection", "_inner", "_summary", "_structure", "_posthoc_onewafer")}
    sj.guard(*outs.values(), *[p.with_suffix(".yaml") for p in outs.values()])
    d, X, names, masks = joint_data()
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    assert np.sum([masks[k] for k in FOLDS], axis=0).max() == 1 and np.sum([masks[k] for k in FOLDS], axis=0).min() == 1
    for k in FOLDS:
        sj.check_fold(d, masks[k])
    sels, inner_rows, metric_rows, oof, secs = [], [], [], [], {}
    for k in FOLDS:
        recs = load_records(k)
        secs[k] = {r["seed"]: r["fit_seconds"] for r in recs}
        if k > 0:
            assert all(r["eval_keys_sha16"] == eval_key_hash(d, masks[k]) for r in recs)
        cands = candidates(recs)
        sel, scores, p_j, p_l = run_fold(k, cands, X, y, gi, masks, names)
        sels.append(sel)
        inner_rows.append(scores.assign(fold=k))
        ev = masks[k]
        for model, p in (("joint_sr", p_j), ("lr_group_samevars", p_l)):
            for g in (0, 1):
                m = gi[ev] == g
                metric_rows.append({"group": sj.GROUPS[g], "model": model, "fold": k,
                                    **sj._metrics(p[m], y[ev][m]), "n_train": int((~ev & (gi == g)).sum()),
                                    "complexity": sel["complexity"], "n_vars": len(sel["variables"])})
            oof.append(pd.DataFrame({"WAFER_ID": d.loc[ev, "WAFER_ID"].to_numpy(), "STAGE": d.loc[ev, "STAGE"].to_numpy(),
                                     "group": [sj.GROUPS[g] for g in gi[ev]], "model": model, "fold": k,
                                     "mrr_true": y[ev], "mrr_pred": p}))
        # 참고 (판정·결론에 쓰지 않음): 학습 쪽 평균, 후보 14개 테이블별 선형회귀
        p14, _ = lr_fit_predict(np.column_stack([X[c] for c in names]), y, gi, ~ev, ev)
        pm = np.array([y[~ev & (gi == g)].mean() for g in gi[ev]])
        for model, p in (("lr_group_14cands_ref", p14), ("train_mean_ref", pm)):
            for g in (0, 1):
                m = gi[ev] == g
                metric_rows.append({"group": sj.GROUPS[g], "model": model, "fold": k, **sj._metrics(p[m], y[ev][m]),
                                    "n_train": int((~ev & (gi == g)).sum()), "complexity": np.nan, "n_vars": np.nan})
        print(f"fold {k}: c{sel['complexity']} {sel['equation_search']}", flush=True)
    met = pd.DataFrame(metric_rows)
    oof = pd.concat(oof, ignore_index=True)

    # fold 0 재현 확인 (기존 2026-11-04_..._holdout.csv)
    old = pd.read_csv(sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_holdout.csv", encoding="utf-8-sig")
    repro = {}
    for new_m, old_m in (("joint_sr", "joint_sr_holdout"), ("lr_group_samevars", "lr_group_jointvars")):
        for g in sj.GROUPS:
            a = met[(met["model"] == new_m) & (met["fold"] == 0) & (met["group"] == g)]["rmse"].iloc[0]
            b = old[(old["model"] == old_m) & (old["group"] == g)]["rmse"].iloc[0]
            assert abs(a - b) < 1e-5, f"fold 0 재현 실패 {new_m} {g}: {a} vs {b}"
            repro[f"{new_m}/{g}"] = round(float(a), 6)
    print("fold 0 재현 일치:", repro)

    summ = summarize(met, oof)
    struct = structure_table(sels, d, X, names, y, gi)
    post = posthoc_onewafer(d, X, names, y, gi, masks, sels)

    met.insert(0, "featureset", sj.FEATURESET)
    met.round(6).to_csv(outs[""], index=False, encoding="utf-8-sig")
    oof.insert(0, "featureset", sj.FEATURESET)
    oof.round(6).to_csv(outs["_oof"], index=False, encoding="utf-8-sig")
    sel_df = pd.DataFrame([{**{k: v for k, v in s.items() if k not in ("lr", "refit", "theta", "a", "b", "variables")},
                            "variables": " ".join(s["variables"]), "n_vars": len(s["variables"]),
                            "a_A": s["a"][0], "a_B": s["a"][1], "b_A": s["b"][0], "b_B": s["b"][1],
                            "theta": json.dumps(s["theta"]), "refit_nfev": s["refit"]["nfev"],
                            "lr_coefs": json.dumps(s["lr"], ensure_ascii=False)} for s in sels])
    sel_df.insert(0, "featureset", sj.FEATURESET)
    sel_df.to_csv(outs["_selection"], index=False, encoding="utf-8-sig")
    inner = pd.concat(inner_rows, ignore_index=True)
    inner.insert(0, "featureset", sj.FEATURESET)
    inner.round(6).to_csv(outs["_inner"], index=False, encoding="utf-8-sig")
    summ.insert(0, "featureset", sj.FEATURESET)
    summ.round(6).to_csv(outs["_summary"], index=False, encoding="utf-8-sig")
    struct.insert(0, "featureset", sj.FEATURESET)
    struct.to_csv(outs["_structure"], index=False, encoding="utf-8-sig")
    post.insert(0, "featureset", sj.FEATURESET)
    post.round(6).to_csv(outs["_posthoc_onewafer"], index=False, encoding="utf-8-sig")

    print(met[met["model"].isin(["joint_sr", "lr_group_samevars"])]
          .pivot_table(index=["group", "fold"], columns="model", values="rmse").round(3))
    print(summ.round(3).to_string(index=False))
    print(struct.drop(columns=["equation_refit", "skeleton"]).to_string(index=False))
    print(post.round(3).to_string(index=False))

    y0 = {**common_yaml(t0), "fold0_reproduction_rmse": repro, "fit_seconds": secs,
          "cache": "../galgalgal_joint_cache/outer5/fold{k}_seed{s}.json (fold 1~4), holdout_joint_p*.jsonl (fold 0)"}
    notes = {"": "fold별 지표. joint_sr·lr_group_samevars가 주 비교, *_ref는 참고",
             "_oof": "반복 0의 5개 평가 fold 예측을 합친 OOF (웨이퍼마다 한 번)",
             "_selection": "fold별 선택 f, 재적합 상수, a·b, LR 계수",
             "_inner": "fold별 후보식 안쪽 4 fold 점수 (1-표준오차 선택 근거)",
             "_summary": "5-fold 평균·표준편차, OOF RMSE·MAE, 차이(공동 − LR), 우세 fold 수. 보정 t는 참고",
             "_structure": "fold별 f 구조·변수·S* 일치·전체 데이터 최종 f와의 유사성",
             "_posthoc_onewafer": "사후 민감도: B-456 웨이퍼 1장 제외. 주 결론에 쓰지 않음"}
    for s, p in outs.items():
        sj.write_yaml(p.with_suffix(".yaml"), {**y0, "content": notes[s]})
    print(f"저장 {len(outs)}개 ({time.perf_counter() - t0:.0f}초)")


def summarize(met, oof) -> pd.DataFrame:
    from cv_common import corrected_ttest
    rows = []
    for g in sj.GROUPS:
        mj = met[(met["model"] == "joint_sr") & (met["group"] == g)].sort_values("fold")
        ml = met[(met["model"] == "lr_group_samevars") & (met["group"] == g)].sort_values("fold")
        for model, m in (("joint_sr", mj), ("lr_group_samevars", ml)):
            o = oof[(oof["model"] == model) & (oof["group"] == g)]
            e = o["mrr_pred"] - o["mrr_true"]
            rows.append({"group": g, "model": model, "rmse_fold_mean": m["rmse"].mean(), "rmse_fold_sd": m["rmse"].std(ddof=1),
                         "mae_fold_mean": m["mae"].mean(), "mae_fold_sd": m["mae"].std(ddof=1),
                         "rmse_oof": float(np.sqrt(np.mean(e ** 2))), "mae_oof": float(np.mean(np.abs(e))),
                         "n_oof": len(o), "n_folds": len(m)})
        diff = mj["rmse"].to_numpy() - ml["rmse"].to_numpy()
        tt = corrected_ttest(diff, mj["n_train"].to_numpy(), mj["n_eval"].to_numpy())
        rows.append({"group": g, "model": "diff_joint_minus_lr", "rmse_fold_mean": diff.mean(), "rmse_fold_sd": diff.std(ddof=1),
                     "mae_fold_mean": (mj["mae"].to_numpy() - ml["mae"].to_numpy()).mean(),
                     "rmse_oof": rows[-2]["rmse_oof"] - rows[-1]["rmse_oof"], "mae_oof": rows[-2]["mae_oof"] - rows[-1]["mae_oof"],
                     "joint_better_folds": int((diff < 0).sum()), "n_folds": len(diff),
                     "diff_by_fold": " ".join(f"{v:+.3f}" for v in diff),
                     "ref_corrected_t": tt["t"], "ref_corrected_p": tt["p"]})
    return pd.DataFrame(rows)


def structure_table(sels, d, X, names, y, gi) -> pd.DataFrame:
    """fold별 f와 전체 데이터 최종 f 비교. 기능 유사성은 라벨 없이 f 값끼리의 그룹 내 상관."""
    sel = pd.read_csv(sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_selection.csv", encoding="utf-8-sig")
    full_text = sel[sel["rule"] == "1se"]["equation"].iloc[0]
    eqf = ss.parse_equation(full_text)
    tf, af, bf, _ = sj.joint_refit(eqf.node, eqf.theta, X, y, gi)       # 전체 데이터 재적합 (2026-11-04 보고와 같은 방식)
    n = len(y)
    f_full = ss.predict(eqf.node, tf, X, n)
    vf = ss.variables(eqf.node)
    rows = [{"fold": "full", "equation_search": full_text, "equation_refit": render(eqf.node, tf), "skeleton": ss.skeleton(eqf.node),
             "complexity": int(sel[sel["rule"] == "1se"]["complexity"].iloc[0]), "variables": " ".join(c for c in names if c in vf),
             "n_vars": len(vf), "S_star": is_skeleton_star(eqf.node), "same_text_as_full": True, "same_skeleton_as_full": True,
             "jaccard_vars_full": 1.0, "corr_f_full_A": 1.0, "corr_f_full_B": 1.0,
             "a_A": af[0], "a_B": af[1], "b_A": bf[0], "b_B": bf[1]}]
    for s in sels:
        eq = ss.parse_equation(s["equation_search"])
        v = set(s["variables"])
        fk = ss.predict(eq.node, np.asarray(s["theta"]), X, n)
        corr = {g: float(np.corrcoef(fk[gi == i], f_full[gi == i])[0, 1]) for i, g in enumerate("AB")}
        rows.append({"fold": str(s["fold"]), "equation_search": s["equation_search"], "equation_refit": s["equation_refit"],
                     "skeleton": s["skeleton"], "complexity": s["complexity"], "variables": " ".join(s["variables"]),
                     "n_vars": len(v), "S_star": is_skeleton_star(eq.node),
                     "same_text_as_full": s["equation_search"] == full_text,
                     "same_skeleton_as_full": ss.skeleton(eq.node) == ss.skeleton(eqf.node),
                     "jaccard_vars_full": len(v & vf) / len(v | vf),
                     "corr_f_full_A": corr["A"], "corr_f_full_B": corr["B"],
                     "a_A": s["a"][0], "a_B": s["a"][1], "b_A": s["b"][0], "b_B": s["b"][1]})
    out = pd.DataFrame(rows)
    use = pd.DataFrame({c: [c in r.split() for r in out["variables"]] for c in names}, index=out.index).astype(int)
    return pd.concat([out, use.add_prefix("use_")], axis=1)


def posthoc_onewafer(d, X, names, y, gi, masks, sels) -> pd.DataFrame:
    """사후 민감도 (계획 6절). 선택된 f는 그대로, 탐색은 다시 하지 않는다.
    (i) score_only: 기존 예측에서 그 웨이퍼만 채점 제외 / (ii) refit_without: 두 모형 계수를 그 웨이퍼 없이 다시 적합 + 채점 제외."""
    drop = ((d["WAFER_ID"] == DROP[0]) & (d["STAGE"] == DROP[1])).to_numpy()
    assert drop.sum() == 1
    rows = []
    for variant in ("all_wafers", "score_only", "refit_without"):
        for s in sels:
            k = s["fold"]
            ev, tr = masks[k], ~masks[k]
            if variant == "refit_without":
                tr = tr & ~drop
            eq = ss.parse_equation(s["equation_search"])
            t, a, b, _ = sj.joint_refit(eq.node, eq.theta, ss._subset(X, tr), y[tr], gi[tr])
            pj = sj.joint_predict(eq.node, t, a, b, ss._subset(X, ev), gi[ev])
            M = np.column_stack([X[c] for c in s["variables"]])
            pl, _ = lr_fit_predict(M, y, gi, tr, ev)
            keep = np.ones(ev.sum(), bool) if variant == "all_wafers" else ~drop[ev]
            for model, p in (("joint_sr", pj), ("lr_group_samevars", pl)):
                for g in (0, 1):
                    m = (gi[ev] == g) & keep
                    rows.append({"variant": variant, "group": sj.GROUPS[g], "model": model, "fold": k,
                                 **sj._metrics(p[m], y[ev][m]), "n_train": int((tr & (gi == g)).sum())})
    r = pd.DataFrame(rows)
    piv = r.pivot_table(index=["variant", "group", "fold"], columns="model", values="rmse").reset_index()
    piv["diff"] = piv["joint_sr"] - piv["lr_group_samevars"]
    agg = piv.groupby(["variant", "group"]).agg(joint_rmse_fold_mean=("joint_sr", "mean"), lr_rmse_fold_mean=("lr_group_samevars", "mean"),
                                                diff_fold_mean=("diff", "mean"), joint_better_folds=("diff", lambda v: int((v < 0).sum())),
                                                diff_fold2=("diff", lambda v: float(v[piv.loc[v.index, "fold"] == 2].iloc[0])))
    return r.merge(agg.reset_index(), on=["variant", "group"], how="left")


def common_yaml(t0) -> dict:
    y = sj.common_yaml(t0, {})
    y["nature"] = NATURE
    y["plan"] = f"{PLAN} + 기존 {sj.PLAN} ({sj.PLAN_COMMIT})"
    y["design"] = ("folds_v1 반복 0의 fold 0~4를 바깥 평가 쪽으로 차례로 두고, 바깥 학습 쪽만으로 공동 SR(sr_joint_search 정식 설정,"
                   " niterations 500·populations 31·population_size 27·deterministic·serial, 시드 42~51)을 탐색,"
                   " 안쪽 4 fold 1-표준오차(SE = sd/√4)로 f 선택, 바깥 학습 쪽 전체 재적합 후 평가 쪽 채점."
                   " 비교: 같은 변수의 테이블별 선형회귀. fold 0은 기존 탐색 재사용")
    y["eval_label_influence"] = ("후보 14개(정식 SR 후보 합집합)는 전체 train+val 라벨로 선정 → 평가 fold 라벨이 후보 선택에 영향."
                                 " 척도(|중앙값|)·fold 배정·교체 구간은 라벨 미사용. 이번 추가 검증 자체가 기존 결과를 본 뒤 결정됨")
    return y


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    s.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    sub.add_parser("unclaim")
    sub.add_parser("status")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    {"search": cmd_search, "unclaim": cmd_unclaim, "status": cmd_status, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

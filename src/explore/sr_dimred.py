"""PLS 성분으로 공동 SR (사후 탐색, 계획 docs/plans/2026-11-16_김혜연_dimred_plan.md 8절).

모형: MRR_g = a[g] · f(Z) + b[g]. Z = dimred.fit_transform의 PLS 성분 점수, f는 두 테이블 공통, a·b만 테이블별.
탐색 설정은 정식 SR과 같고, 템플릿·지수 차단 loss는 공동 SR(sr/joint의 sr_joint_search.py, 전준혁)과 같은 규칙이다.

작업(job):
  A   PLS k=2, 주 집합 전체로 탐색 → 25-fold 선택·기준선 비교 (판정)
  C   PLS k=5, 같은 절차 (민감도, 판정 미사용)
  B   PLS k=2, 바깥 fold 0~4(folds_v1 반복 0)마다 학습 쪽만으로 탐색 (B1, 구조 재현 판정)
  B2  PLS k=3, 바깥 fold 1·2 (fold별 선택 설정. fold 0·3·4는 B 결과를 그대로 씀, 보고만)

명령:
  python src/explore/sr_dimred.py toy                        인공 데이터로 템플릿·지수 차단 확인 (라벨 없음)
  python src/explore/sr_dimred.py search --jobs A C B B2     작업자 하나. 여러 개를 동시에 띄워도 된다(시드 단위 claim 파일)
  python src/explore/sr_dimred.py status
  python src/explore/sr_dimred.py evaluate                   모든 탐색이 끝난 뒤 선택·비교·판정·저장
캐시: ../galgalgal_dimred_cache/{job}_{fold}_seed{s}.json (저장소 밖)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dimred as dr  # noqa: E402
import sr_run  # noqa: E402
import sr_select as ss  # noqa: E402
from cv_common import corrected_ttest  # noqa: E402

ROOT = HERE.parents[1]
CACHE = ROOT.parent / "galgalgal_dimred_cache"
PLAN = "docs/plans/2026-11-16_김혜연_dimred_plan.md"
AUTHOR = "박종화"
SEEDS = list(range(42, 52))
METHOD = "PLS"
JOBS = {"A": (2, ["full"]), "C": (5, ["full"]), "B": (2, [0, 1, 2, 3, 4]), "B2": (3, [1, 2])}
B2_FROM_B = [0, 3, 4]                 # selection.csv 반복 0에서 PLS 2가 뽑힌 fold
OUTER5 = [0, 1, 2, 3, 4]              # folds_v1 반복 0 (r = 0~4)
ALPHA = 0.05
NATURE = "사후 탐색 (교수님 제안, dimred 계획 8절). 정식 SR·공동 SR 판정은 바꾸지 않음"

# 공동 SR 템플릿용 지수 차단 loss (sr_joint_search.LOSS_TEMPLATE과 같은 규칙)
LOSS_TEMPLATE = f"""
function pow_guard(ex, dataset::Dataset{{T,L}}, options) where {{T,L}}
    subs = values(SymbolicRegression.get_contents(ex))
    bad = any(subs) do sub
        tree = SymbolicRegression.get_tree(sub)
        any(tree) do node
            node.degree == 2 && options.operators.binops[node.op] === SymbolicRegression.safe_pow &&
                !(node.r.degree == 0 && node.r.constant && abs(node.r.val) <= {sr_run.MAX_ABS_EXPONENT})
        end
    end
    bad && return {sr_run.BLOCKED_LOSS}
    pred, ok = eval_tree_array(ex, dataset.X, options)
    !ok && return {sr_run.BLOCKED_LOSS}
    return sum(abs2, pred .- dataset.y) / length(dataset.y)
end
"""


def make_model(names, seed, niterations=sr_run.NITERATIONS):
    """정식 sr_run.make_model과 같은 설정 + 공동 템플릿."""
    from pysr import PySRRegressor, TemplateExpressionSpec
    spec = TemplateExpressionSpec(combine=f"a[g] * f({', '.join(names)}) + b[g]", expressions=["f"],
                                  variable_names=list(names) + ["g"], parameters={"a": 2, "b": 2})
    nested = {a: {b: 0 for b in sr_run.UNARY} for a in sr_run.UNARY}
    return PySRRegressor(
        niterations=niterations, populations=31, population_size=27,
        binary_operators=["+", "-", "*", "/", "^"], unary_operators=sr_run.UNARY,
        constraints={"^": (-1, 1)}, nested_constraints=nested,
        complexity_of_operators={"exp": 3, "log": 3, "sqrt": 2, "^": 2},
        maxsize=20, deterministic=True, parallelism="serial", random_state=seed,
        progress=False, verbosity=0, temp_equation_file=True,
        loss_function_expression=LOSS_TEMPLATE, expression_spec=spec)


ARG = re.compile(r"#(\d+)")


def split_template(text, names):
    """템플릿 식 문자열에서 f(변수 이름으로 바꿈)와 a·b를 꺼낸다."""
    parts = {}
    for line in re.split(r"[\n;]", text):
        line = line.strip().lstrip("╭├╰│ ").strip()
        if "=" in line:
            key, val = line.split("=", 1)
            parts[key.strip()] = val.strip()
    f = parts["f"]
    assert all(1 <= int(k) <= len(names) for k in ARG.findall(f)), f"f 인자 번호가 범위 밖: {f}"
    return ARG.sub(lambda m: names[int(m.group(1)) - 1], f), parts.get("a"), parts.get("b")


# ---------------------------------------------------------------- 데이터
_DATA = {}


def main_data():
    if "main" not in _DATA:
        d = dr.load_data()
        d = d[d["in_main"]].reset_index(drop=True)
        for g in dr.GROUPS:
            assert (d["group"] == g).sum() == dr.EXPECTED_MAIN[g]
        d["gi"] = (d["group"] == "B-456").astype(int)
        _DATA["main"] = d
        _DATA["folds"] = dr.outer_folds(d)
    return _DATA["main"], _DATA["folds"]


def outer_mask(r):
    """반복 0·fold r의 평가 쪽 마스크."""
    _, folds = main_data()
    rr, repeat, fold, tr, ev = folds[r]
    assert rr == r and repeat == 0
    return ev


def components(tr, ev, k):
    """학습 쪽 tr로만 PLS k 성분을 만들고 평가 쪽 ev에 적용한다. 반환 (Ztr 배열 dict, Zev 배열 dict, 이름)."""
    d, _ = main_data()
    if ev is None:
        Ztr, _, _ = dr.fit_transform(d[tr], [], METHOD, k)
        Zev = None
    else:
        dr.assert_disjoint(d[tr], d[ev])
        Ztr, (Zev,), _ = dr.fit_transform(d[tr], [d[ev]], METHOD, k)
    names = list(Ztr.columns)
    to = lambda Z: {c: Z[c].to_numpy(float) for c in names}
    return to(Ztr), (to(Zev) if Zev is not None else None), names


# ---------------------------------------------------------------- 탐색
def seed_path(job, fold, seed):
    return CACHE / f"{job}_{fold}_seed{seed}.json"


def train_mask(fold):
    d, _ = main_data()
    return np.ones(len(d), bool) if fold == "full" else ~outer_mask(fold)


def search_one(job, fold, seed):
    k, _ = JOBS[job]
    d, _ = main_data()
    tr = train_mask(fold)
    Z, _, names = components(tr, None, k)
    M = np.column_stack([Z[c] for c in names] + [d["gi"].to_numpy(float)[tr] + 1.0])
    y = d["MRR"].to_numpy(float)[tr]
    t0 = time.perf_counter()
    m = make_model(names, seed)
    m.fit(M, y, variable_names=names + ["g"])
    rows = []
    for _, e in m.equations_.iterrows():
        f, a, b = split_template(str(e["equation"]), names)
        rows.append({"complexity": int(e["complexity"]), "loss": float(e["loss"]), "equation": f, "a": a, "b": b,
                     "exponent_ok": bool(sr_run.exponent_ok(f))})
    return {"job": job, "k": k, "fold": fold, "seed": seed, "n_train": int(tr.sum()),
            "fit_seconds": round(time.perf_counter() - t0, 1), "equations": rows}


def cmd_search(args):
    CACHE.mkdir(exist_ok=True)
    for job in args.jobs:
        for fold in JOBS[job][1]:
            for seed in SEEDS:
                out = seed_path(job, fold, seed)
                claim = out.with_suffix(".claim")
                if out.exists():
                    continue
                try:
                    os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
                try:
                    rec = search_one(job, fold, seed)
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                    os.replace(tmp, out)
                    print(f"{job} fold {fold} 시드 {seed}: {rec['fit_seconds']}초, 파레토 {len(rec['equations'])}개", flush=True)
                finally:
                    claim.unlink(missing_ok=True)


def cmd_status(args):
    for job, (k, folds) in JOBS.items():
        for fold in folds:
            done = [s for s in SEEDS if seed_path(job, fold, s).exists()]
            run = [s for s in SEEDS if seed_path(job, fold, s).with_suffix(".claim").exists()]
            secs = [json.loads(seed_path(job, fold, s).read_text(encoding="utf-8"))["fit_seconds"] for s in done]
            print(f"{job:2s} k={k} fold {fold!s:4s}: 완료 {len(done)}/10, 실행 중 {run}, 평균 {np.mean(secs) if secs else 0:.0f}초")


# ---------------------------------------------------------------- 인공 데이터 확인
def cmd_toy(args):
    rng = np.random.default_rng(0)
    n, names = 300, ["PLS1", "PLS2"]
    Z = rng.normal(0, 1, (n, 2))
    g = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)]
    gi = (g - 1).astype(int)
    f = np.exp(0.5 * Z[:, 0]) - Z[:, 1]
    y = np.array([3.0, 5.0])[gi] * f + np.array([70.0, 80.0])[gi] + rng.normal(0, 0.01, n)
    m = make_model(names, 42, niterations=args.niterations)
    m.fit(np.column_stack([Z, g]), y, variable_names=names + ["g"])
    best = m.equations_.loc[m.equations_["loss"].idxmin()]
    fx, a, b = split_template(str(best["equation"]), names)
    eq = ss.parse_equation(fx)
    print("최저 loss", float(best["loss"]), "f:", fx, "a:", a, "b:", b)
    t, aa, bb, _ = joint_refit(eq.node, eq.theta, {c: Z[:, j] for j, c in enumerate(names)}, y, gi)
    p = joint_predict(eq.node, t, aa, bb, {c: Z[:, j] for j, c in enumerate(names)}, gi)
    print("재적합 RMSE", float(np.sqrt(np.mean((p - y) ** 2))), "f에 g 없음:", "g" not in ss.variables(eq.node))


# ---------------------------------------------------------------- 재적합·평가
def _init_ab(f, y, gi):
    a, b = np.ones(2), np.zeros(2)
    for g in (0, 1):
        m = (gi == g) & np.isfinite(f)
        if m.sum() >= 3 and np.var(f[m]) > 0:
            a[g], b[g] = np.polyfit(f[m], y[m], 1)
        else:
            b[g] = y[gi == g].mean()
    return a, b


def joint_refit(node, theta0, X, y, gi, max_nfev=200):
    """f 상수(공통) + a_g·b_g(테이블별)를 학습 쪽으로 함께 맞춘다 (sr_joint.joint_refit과 같은 방식)."""
    ids = sorted(ss.const_ids(node))
    theta = np.array(theta0, dtype=float)
    n = len(y)
    a0, b0 = _init_ab(ss.predict(node, theta, X, n), y, gi)
    powers = ss.exponent_ids(node)
    lo = np.r_[[-sr_run.MAX_ABS_EXPONENT if i in powers else -np.inf for i in ids], [-np.inf] * 4]
    hi = np.r_[[sr_run.MAX_ABS_EXPONENT if i in powers else np.inf for i in ids], [np.inf] * 4]
    k = len(ids)

    def unpack(p):
        t = theta.copy()
        t[ids] = p[:k]
        return t, p[k:k + 2], p[k + 2:k + 4]

    def residual(p):
        t, a, b = unpack(p)
        r = a[gi] * ss.predict(node, t, X, n) + b[gi] - y
        r[np.isnan(r)] = 1e6
        return r

    x0 = np.r_[np.clip(theta[ids], lo[:k], hi[:k]), a0, b0]
    res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=max_nfev)
    t, a, b = unpack(res.x)
    return t, a, b, {"nfev": int(res.nfev), "status": int(res.status)}


def joint_predict(node, theta, a, b, X, gi):
    return a[gi] * ss.predict(node, theta, X, len(gi)) + b[gi]


def errors(p, y):
    e = p - y
    return {"mse": float(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


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


def structure(text):
    eq = ss.parse_equation(text)
    return ss.skeleton(strip_affine(eq.node)), tuple(sorted(ss.variables(eq.node)))


def load_cands(job, fold):
    recs = [json.loads(seed_path(job, fold, s).read_text(encoding="utf-8")) for s in SEEDS]
    rows = pd.DataFrame([{"seed": r["seed"], **e} for r in recs for e in r["equations"]])
    n_bad = int((~rows["exponent_ok"]).sum())
    rows = rows[rows["exponent_ok"]]
    rows = rows[rows["equation"].map(lambda t: len(ss.variables(ss.parse_equation(t).node)) > 0)]
    return rows.sort_values(["complexity", "loss"]).drop_duplicates("equation"), recs, n_bad


class FoldSet:
    """평가 단위(학습·평가 마스크)마다 PLS 성분을 한 번만 만든다."""

    def __init__(self, k, pairs):
        d, _ = main_data()
        self.y, self.gi = d["MRR"].to_numpy(float), d["gi"].to_numpy(int)
        self.units = []
        for key, tr, ev in pairs:
            Ztr, Zev, names = components(tr, ev, k)
            self.units.append((key, tr, ev, Ztr, Zev))

    def score(self, text):
        """식 하나를 모든 단위에서 재적합·채점. 반환: 단위별 [(A 지표), (B 지표)] 또는 None(계산 불가)."""
        eq = ss.parse_equation(text)
        out = []
        for key, tr, ev, Ztr, Zev in self.units:
            t, a, b, _ = joint_refit(eq.node, eq.theta, Ztr, self.y[tr], self.gi[tr])
            p = joint_predict(eq.node, t, a, b, Zev, self.gi[ev])
            if not np.isfinite(p).all():
                return None
            out.append([errors(p[self.gi[ev] == g], self.y[ev][self.gi[ev] == g]) for g in (0, 1)])
        return out

    def lr(self):
        out = []
        for key, tr, ev, Ztr, Zev in self.units:
            names = list(Ztr)
            Mtr = np.column_stack([Ztr[c] for c in names])
            Mev = np.column_stack([Zev[c] for c in names])
            g = np.array(dr.GROUPS)
            p = dr.per_table_lr(Mtr, self.y[tr], g[self.gi[tr]], Mev, g[self.gi[ev]])
            out.append([errors(p[self.gi[ev] == h], self.y[ev][self.gi[ev] == h]) for h in (0, 1)])
        return out


def curve_and_pick(fs, cands):
    rows = []
    for _, c in cands.iterrows():
        sc = fs.score(c["equation"])
        if sc is None:
            continue
        unit = np.array([(u[0]["rmse"] + u[1]["rmse"]) / 2 for u in sc])
        rows.append({"equation": c["equation"], "complexity": int(c["complexity"]), "seed": int(c["seed"]),
                     "mean": unit.mean(), "se": unit.std(ddof=1) / np.sqrt(len(unit)), "scores": sc})
    allc = pd.DataFrame(rows)
    curve = allc.sort_values(["complexity", "mean", "equation"]).drop_duplicates("complexity").reset_index(drop=True)
    best = curve.loc[curve["mean"].idxmin()]
    limit = best["mean"] + best["se"]
    pick = curve[curve["mean"] <= limit].sort_values("complexity").iloc[0]
    return curve, pick, float(limit), len(cands) - len(allc)


def render(node, theta):
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


# ---------------------------------------------------------------- 분석 A·C (25-fold)
def analysis_25(job):
    k = JOBS[job][0]
    d, folds = main_data()
    cands, recs, n_bad = load_cands(job, "full")
    fs = FoldSet(k, [(r, tr, ev) for r, _, _, tr, ev in folds])
    curve, pick, limit, n_invalid = curve_and_pick(fs, cands)
    lr = fs.lr()
    rows, tests = [], []
    n_tr = {g: [int((tr & (fs.gi == g)).sum()) for _, tr, _, _, _ in fs.units] for g in (0, 1)}
    n_ev = {g: [int((ev & (fs.gi == g)).sum()) for _, _, ev, _, _ in fs.units] for g in (0, 1)}
    for j, (r, *_rest) in enumerate(fs.units):
        for g in (0, 1):
            for model, sc in ((f"joint_sr_pls{k}", pick["scores"]), (f"lr_pls{k}", lr)):
                rows.append({"analysis": job, "group": dr.GROUPS[g], "model": model, "featureset": dr.FEATURESET, "fold": r,
                             **sc[j][g], "n_train": n_tr[g][j], "n_eval": n_ev[g][j]})
    folds_df = pd.DataFrame(rows)
    for g in (0, 1):
        x = folds_df[folds_df["group"] == dr.GROUPS[g]].pivot(index="fold", columns="model", values="mse")
        t = corrected_ttest((x[f"joint_sr_pls{k}"] - x[f"lr_pls{k}"]).to_numpy(), n_tr[g], n_ev[g])
        rm = folds_df[folds_df["group"] == dr.GROUPS[g]].groupby("model")["rmse"].mean()
        tests.append({"analysis": job, "group": dr.GROUPS[g], "a": f"joint_sr_pls{k}", "b": f"lr_pls{k}",
                      "rmse_a": rm[f"joint_sr_pls{k}"], "rmse_b": rm[f"lr_pls{k}"],
                      "wins_a": int((x[f"joint_sr_pls{k}"] < x[f"lr_pls{k}"]).sum()), **t})
    # 전체 데이터로 최종 f 재적합 (보고용)
    Zall, _, names = components(np.ones(len(d), bool), None, k)
    eq = ss.parse_equation(pick["equation"])
    t, a, b, _ = joint_refit(eq.node, eq.theta, Zall, fs.y, fs.gi)
    final = {"analysis": job, "k": k, "equation_search": pick["equation"], "equation_refit": render(eq.node, t),
             "complexity": int(pick["complexity"]), "seed_found": int(pick["seed"]), "mean_rmse": float(pick["mean"]),
             "se": float(pick["se"]), "limit_1se": limit, "a_A": float(a[0]), "a_B": float(a[1]), "b_A": float(b[0]),
             "b_B": float(b[1]), "variables": " ".join(sorted(ss.variables(eq.node))),
             "skeleton": structure(pick["equation"])[0], "n_pareto_rows": int(sum(len(r["equations"]) for r in recs)),
             "n_candidates": int(len(cands)), "n_removed_exponent": n_bad, "n_invalid_eval": n_invalid,
             "fit_seconds_mean": float(np.mean([r["fit_seconds"] for r in recs]))}
    curve = curve.drop(columns="scores").assign(analysis=job, k=k)
    pareto = pd.DataFrame([{"analysis": job, "k": k, "seed": r["seed"], "featureset": dr.FEATURESET, **e}
                           for r in recs for e in r["equations"]])
    return final, folds_df, pd.DataFrame(tests), curve, pareto


# ---------------------------------------------------------------- 분석 B (바깥 5-fold)
def analysis_outer(job, fold, ref_structure):
    k = JOBS[job][0]
    d, _ = main_data()
    ev_out = outer_mask(fold)
    tr_out = ~ev_out
    inner = [(j, tr_out & ~outer_mask(j), outer_mask(j)) for j in OUTER5 if j != fold]
    cands, recs, n_bad = load_cands(job, fold)
    fs_in = FoldSet(k, inner)
    curve, pick, limit, n_invalid = curve_and_pick(fs_in, cands)
    fs_out = FoldSet(k, [(fold, tr_out, ev_out)])
    sc = fs_out.score(pick["equation"])
    lr = fs_out.lr()
    st = structure(pick["equation"])
    sel = {"analysis": job, "k": k, "fold": fold, "equation_search": pick["equation"], "complexity": int(pick["complexity"]),
           "inner_mean": float(pick["mean"]), "inner_se": float(pick["se"]), "limit_1se": limit, "skeleton": st[0],
           "variables": " ".join(st[1]), "same_structure_as_A": bool(st == ref_structure),
           "n_candidates": int(len(cands)), "n_removed_exponent": n_bad, "n_invalid_inner": n_invalid,
           "fit_seconds_mean": float(np.mean([r["fit_seconds"] for r in recs]))}
    rows = []
    for g in (0, 1):
        for model, s in ((f"joint_sr_pls{k}", sc), (f"lr_pls{k}", lr)):
            rows.append({"analysis": job, "group": dr.GROUPS[g], "model": model, "featureset": dr.FEATURESET, "fold": fold,
                         **(s[0][g] if s else {"mse": np.nan, "mae": np.nan, "rmse": np.nan}),
                         "n_train": int((tr_out & (fs_out.gi == g)).sum()), "n_eval": int((ev_out & (fs_out.gi == g)).sum())})
    return sel, rows


# ---------------------------------------------------------------- 판정·저장
def judge(tests_A, sels_B1):
    perf = {r["group"]: bool(r["mean_diff"] < 0 and r["p"] < ALPHA) for _, r in tests_A.iterrows()}
    n_same = int(sum(s["same_structure_as_A"] for s in sels_B1))
    c1, c2 = all(perf.values()), n_same >= 3
    if c1 and c2:
        v = "공동식 인정"
    elif c1:
        v = "선형 대비 이득은 있으나 구조가 재현되지 않음"
    elif c2:
        v = "구조는 재현되나 선형 대비 이득이 확인되지 않음"
    else:
        v = "공동식 인정 안 됨"
    if any(perf.values()) and not c1:
        v += " (부분적: 성능 조건 " + ", ".join(g for g, ok in perf.items() if ok) + "만 만족)"
    border = [f"{r['group']} p = {r['p']:.3f}" for _, r in tests_A.iterrows() if 0.04 <= r["p"] <= 0.06]
    if border:
        v += " (경계: " + "; ".join(border) + ")"
    return {"verdict": v, "perf_condition": perf, "n_same_structure_B1": n_same, "structure_condition": c2}


def git_head():
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def write(df, name, meta, diag=False):
    base = ROOT / "results" / ("diagnostics" if diag else "")
    stamp = date.today().isoformat()
    tag = "" if diag else "FV1_"
    path = base / f"{stamp}_{AUTHOR}_{tag}dimred_sr_{name}.csv"
    while path.exists():
        path = path.with_name(path.stem + "_v2.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    import pysr
    y = {"date": stamp, "author": AUTHOR, "script": "src/explore/sr_dimred.py", "code_commit": git_head(), "plan": PLAN,
         "nature": NATURE, "featureset": dr.FEATURESET, "wafers": "load_data() in_main == True (A-456 940, B-456 951)",
         "components": "dimred.fit_transform, PLS, 학습 쪽만으로 학습", "seeds": SEEDS, "niterations": sr_run.NITERATIONS,
         "pysr": pysr.__version__, **meta}
    path.with_suffix(".yaml").write_text("\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in y.items()) + "\n",
                                          encoding="utf-8")
    print("저장", path.relative_to(ROOT))


def cmd_evaluate(args):
    t0 = time.perf_counter()
    finals, folds_all, tests_all, curves, paretos = [], [], [], [], []
    for job in ("A", "C"):
        f, fo, te, cu, pa = analysis_25(job)
        finals.append(f); folds_all.append(fo); tests_all.append(te); curves.append(cu); paretos.append(pa)
    ref = structure(finals[0]["equation_search"])
    sels, outer_rows = [], []
    for fold in OUTER5:
        s, r = analysis_outer("B", fold, ref)
        sels.append({**s, "set": "B1"}); outer_rows += [{**x, "set": "B1"} for x in r]
    for fold in OUTER5:
        if fold in B2_FROM_B:
            s = next(x for x in sels if x["set"] == "B1" and x["fold"] == fold)
            sels.append({**s, "set": "B2"})
            outer_rows += [{**x, "set": "B2"} for x in outer_rows if x["set"] == "B1" and x["fold"] == fold]
        else:
            s, r = analysis_outer("B2", fold, ref)
            sels.append({**s, "set": "B2"}); outer_rows += [{**x, "set": "B2"} for x in r]
    tests = pd.concat(tests_all, ignore_index=True)
    verdict = judge(tests[tests["analysis"] == "A"], [s for s in sels if s["set"] == "B1"])
    meta = {"verdict": verdict["verdict"], "seconds": round(time.perf_counter() - t0)}
    write(pd.concat(folds_all, ignore_index=True), "folds", {**meta, "note": "분석 A(k=2, 판정)·C(k=5, 민감도). 탐색에 주 집합 전체를 써서 낙관적"})
    write(tests, "tests", {**meta, "test": "cv_common.corrected_ttest, MSE(공동 SR − PLS k 고정 테이블별 선형회귀)"})
    write(pd.DataFrame(finals), "final", meta)
    write(pd.DataFrame(sels), "outer5_selection", {**meta, "note": "B1 = k=2 고정(판정), B2 = fold별 선택 설정(보고만)"})
    write(pd.DataFrame(outer_rows), "outer5", meta)
    write(pd.DataFrame([{**verdict, "perf_condition": json.dumps(verdict["perf_condition"], ensure_ascii=False)}]), "judgment", meta)
    write(pd.concat(paretos, ignore_index=True), "pareto", meta)
    write(pd.concat(curves, ignore_index=True), "curve", {**meta, "note": "복잡도별 최저 식, 25-fold 두 테이블 RMSE 평균(낙관적)"}, diag=True)
    print("판정:", verdict)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toy")
    t.add_argument("--niterations", type=int, default=40)
    s = sub.add_parser("search")
    s.add_argument("--jobs", nargs="+", default=list(JOBS), choices=list(JOBS))
    sub.add_parser("status")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    {"toy": cmd_toy, "search": cmd_search, "status": cmd_status, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

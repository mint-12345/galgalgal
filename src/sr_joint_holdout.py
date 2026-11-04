"""바깥 fold 하나를 뺀 탐색 검증 (계획 변경 기록 2026-11-05, 사후 후속 실험, 기술 통계만).

folds_v1 반복 0의 fold 0(r = 0)을 바깥 평가 쪽으로 두고, 바깥 학습 쪽만으로
  (1) 공동 SR: sr_joint_search와 같은 템플릿·설정, 합산 |중앙값| 척도, 공통 후보 14개
  (2) 개별 SR: 정식 sr_run 설정, 테이블별 정식 후보 10개, 테이블별 |중앙값| 척도
을 시드 42~51로 새로 탐색한다. 식은 반복 0의 나머지 4개 fold(안쪽)로 1-표준오차 규칙에 따라 고르고,
바깥 학습 쪽 전체로 재적합해 r = 0 평가 쪽에서 채점한다. 척도 통계는 각 방식의 원래 규칙(라벨 미사용, 전체 웨이퍼).

명령:
  python src/sr_joint_holdout.py search --mode joint|A-456|B-456 --seeds 42 43 ... --tag p0
  python src/sr_joint_holdout.py evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_joint as sj  # noqa: E402
import sr_joint_search as sjs  # noqa: E402
import sr_run  # noqa: E402
import sr_select as ss  # noqa: E402

OUTER = 0                     # r = 0 (반복 0, fold 0)
INNER = [1, 2, 3, 4]          # 반복 0의 나머지 fold
SEEDS = list(range(42, 52))


def joint_data():
    d, full, names = sj.joint_frame()
    X = sj.design(d, names, sj.scale_table(full, names), "pooled")
    masks = ss.fold_masks(d[sj.KEYS])
    return d, X, names, masks


def group_data(g):
    frame, names, _ = sr_run.load_frame(g)          # 정식과 같은 척도·웨이퍼
    masks = ss.fold_masks(frame[sj.KEYS])
    X = {n: frame[n].to_numpy(float) for n in names}
    return frame, X, names, masks


def cmd_search(args):
    cache = sj.CACHE / f"holdout_{args.mode}_{args.tag}.jsonl"
    done = set()
    if cache.exists():
        done = {json.loads(line)["seed"] for line in cache.read_text(encoding="utf-8").splitlines()}
    if args.mode == "joint":
        d, X, names, masks = joint_data()
        tr = ~masks[OUTER]
        M = np.column_stack([X[c][tr] for c in names] + [d["gi"].to_numpy(float)[tr] + 1.0])
        y = d["MRR"].to_numpy(float)[tr]
    else:
        frame, X, names, masks = group_data(args.mode)
        tr = ~masks[OUTER]
        M = np.column_stack([X[c][tr] for c in names])
        y = frame["MRR"].to_numpy(float)[tr]
    for seed in args.seeds:
        if seed in done:
            continue
        t0 = time.perf_counter()
        if args.mode == "joint":
            m = sjs.make_model(names, seed, sr_run.NITERATIONS)
            m.fit(M, y, variable_names=names + ["g"])
            eqs = [{"complexity": int(e["complexity"]), "loss": float(e["loss"]),
                    "equation": sjs.split_template(str(e["equation"]), names)["f"]} for _, e in m.equations_.iterrows()]
        else:
            m = sr_run.make_model(seed, sr_run.NITERATIONS)
            m.fit(M, y, variable_names=names)
            eqs = [{"complexity": int(e["complexity"]), "loss": float(e["loss"]), "equation": str(e["equation"])}
                   for _, e in m.equations_.iterrows()]
        rec = {"mode": args.mode, "seed": seed, "n_train": int(tr.sum()), "fit_seconds": round(time.perf_counter() - t0, 1),
               "equations": eqs}
        with cache.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{args.mode} 시드 {seed}: {rec['fit_seconds']}초 (n_train {rec['n_train']})", flush=True)


def load_cache(mode):
    recs = []
    for p in sorted(sj.CACHE.glob(f"holdout_{mode}_*.jsonl")):
        recs += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["seed"] for r in recs) == SEEDS, f"{mode} 시드: {sorted(r['seed'] for r in recs)}"
    rows = pd.DataFrame([{"seed": r["seed"], **e} for r in recs for e in r["equations"]])
    rows = rows[rows["equation"].map(sr_run.exponent_ok)]
    return rows.sort_values("complexity").drop_duplicates("equation"), {r["seed"]: r["fit_seconds"] for r in recs}


def pick_1se(scores: pd.DataFrame):
    ok = scores[np.isfinite(scores["mean"])]
    curve = ok.sort_values(["complexity", "mean", "equation"]).drop_duplicates("complexity")
    best = curve.loc[curve["mean"].idxmin()]
    return curve[curve["mean"] <= best["mean"] + best["se"]].sort_values("complexity").iloc[0]


def eval_joint_candidates(cands, d, X, masks):
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    outer = masks[OUTER]
    rows = []
    for _, c in cands.iterrows():
        eq = ss.parse_equation(c["equation"])
        inner = []
        for r in INNER:
            ev = masks[r]
            tr = ~outer & ~ev
            t, a, b, _ = sj.joint_refit(eq.node, eq.theta, ss._subset(X, tr), y[tr], gi[tr])
            p = sj.joint_predict(eq.node, t, a, b, ss._subset(X, ev), gi[ev])
            ms = [sj._metrics(p[gi[ev] == g], y[ev][gi[ev] == g]) for g in (0, 1)]
            inner.append(np.inf if any(m["n_eval_nan"] for m in ms) else (ms[0]["rmse"] + ms[1]["rmse"]) / 2)
        inner = np.array(inner)
        rows.append({"equation": c["equation"], "complexity": c["complexity"], "mean": inner.mean(),
                     "se": inner.std(ddof=1) / 2 if np.isfinite(inner).all() else np.inf})
    return pd.DataFrame(rows)


def outer_joint(eq_text, d, X, masks):
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    ev = masks[OUTER]
    eq = ss.parse_equation(eq_text)
    t, a, b, _ = sj.joint_refit(eq.node, eq.theta, ss._subset(X, ~ev), y[~ev], gi[~ev])
    p = sj.joint_predict(eq.node, t, a, b, ss._subset(X, ev), gi[ev])
    return {sj.GROUPS[g]: sj._metrics(p[gi[ev] == g], y[ev][gi[ev] == g]) for g in (0, 1)}


def eval_group_candidates(cands, frame, X, masks):
    y = frame["MRR"].to_numpy(float)
    outer = masks[OUTER]
    rows = []
    for _, c in cands.iterrows():
        eq = ss.parse_equation(c["equation"])
        inner = []
        for r in INNER:
            ev = masks[r]
            tr = ~outer & ~ev
            th = ss.refit(eq.node, eq.theta, ss._subset(X, tr), y[tr])
            e = ss.predict(eq.node, th, ss._subset(X, ev), int(ev.sum())) - y[ev]
            inner.append(np.inf if np.isnan(e).any() else float(np.sqrt(np.mean(e ** 2))))
        inner = np.array(inner)
        rows.append({"equation": c["equation"], "complexity": c["complexity"], "mean": inner.mean(),
                     "se": inner.std(ddof=1) / 2 if np.isfinite(inner).all() else np.inf})
    return pd.DataFrame(rows)


def cmd_evaluate(args):
    t0 = time.perf_counter()
    out = sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_holdout.csv"
    sj.guard(out, out.with_suffix(".yaml"))
    rows, picks, secs = [], {}, {}
    d, X, names, masks = joint_data()
    cj, secs["joint"] = load_cache("joint")
    sj_scores = eval_joint_candidates(cj, d, X, masks)
    pj = pick_1se(sj_scores)
    picks["joint"] = {"equation": pj["equation"], "complexity": int(pj["complexity"]), "inner_mean": float(pj["mean"])}
    for g, m in outer_joint(pj["equation"], d, X, masks).items():
        rows.append({"group": g, "model": "joint_sr_holdout", **m})
    # 같은 변수의 (가) 테이블별 선형회귀
    for r in sj.baseline_rows(d, X, sorted(ss.variables(ss.parse_equation(pj["equation"]).node)),
                              {OUTER: masks[OUTER]}, kinds=("lr_group", "lr_common", "mean")):
        rows.append({**r, "model": f"{r['model']}_jointvars"})
    for g in sj.GROUPS:
        frame, Xg, gnames, gm = group_data(g)
        cg, secs[g] = load_cache(g)
        sc = eval_group_candidates(cg, frame, Xg, gm)
        pg = pick_1se(sc)
        picks[g] = {"equation": pg["equation"], "complexity": int(pg["complexity"]), "inner_mean": float(pg["mean"])}
        eq = ss.parse_equation(pg["equation"])
        y = frame["MRR"].to_numpy(float)
        ev = gm[OUTER]
        th = ss.refit(eq.node, eq.theta, ss._subset(Xg, ~ev), y[~ev])
        p = ss.predict(eq.node, th, ss._subset(Xg, ev), int(ev.sum()))
        rows.append({"group": g, "model": "individual_sr_holdout", **sj._metrics(p, y[ev])})
        Mg = np.column_stack([Xg[c] for c in gnames])
        lr = LinearRegression().fit(Mg[~ev], y[~ev]).predict(Mg[ev])
        rows.append({"group": g, "model": "lr_group_10cands", **sj._metrics(lr, y[ev])})
    # 참고: 본 분석(전체 데이터로 탐색)의 최종 f·정식 SR 식을 같은 fold에서 (바깥 fold 라벨을 본 식이라 낙관적)
    comp = pd.read_csv(sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_compare.csv", encoding="utf-8-sig")
    for model in ("joint", "formal_sr"):
        for _, r in comp[(comp["model"] == model) & (comp["fold"] == OUTER)].iterrows():
            rows.append({"group": r["group"], "model": f"{model}_fullsearch_reference", "rmse": r["rmse"],
                         "mse": r["mse"], "mae": r["mae"], "n_eval": r["n_eval"]})
    res = pd.DataFrame(rows)
    res.insert(0, "featureset", sj.FEATURESET)
    res["fold"] = OUTER
    res.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(res.pivot_table(index="model", columns="group", values="rmse").round(4))
    print(json.dumps(picks, ensure_ascii=False, indent=1))
    sj.write_yaml(out.with_suffix(".yaml"), sj.common_yaml(t0, {
        "stage": "바깥 fold 하나(r = 0)를 뺀 탐색 검증. 기술 통계만, 검정·판정에 쓰지 않음 (계획 변경 기록 2026-11-05)",
        "design": "바깥 학습 쪽만으로 공동 SR·개별 SR을 시드 42~51 정식 설정으로 탐색, 안쪽 4 fold(반복 0의 fold 1~4) 1-표준오차로 선택,"
                  " 바깥 학습 쪽 전체로 재적합해 r = 0에서 채점",
        "picked": picks, "fit_seconds": secs,
        "reference": "*_fullsearch_reference는 전체 데이터로 탐색한 식이라 r = 0 라벨을 본 낙관적 값",
    }))
    print(f"저장: {out.name} ({time.perf_counter() - t0:.0f}초)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--mode", required=True, choices=["joint", "A-456", "B-456"])
    s.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    s.add_argument("--tag", default="p0")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    {"search": cmd_search, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

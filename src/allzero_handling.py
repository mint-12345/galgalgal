"""A-123 전체-0 센서 웨이퍼 처리 방식 비교 (진단, CLAUDE.md 13절 미결정 항목). 방식은 확정하지 않는다.

조건은 `src/models/repro_R_selected.py`의 R-fold와 같다.
  A-123만, R 피처(125개), 13절 임시 선택 기준(p < 0.05 AND OOB 중요도 > 0)을 분할마다 학습 조각에서만 계산,
  Monte Carlo CV 20회, 같은 시드와 같은 분할.

방식
  1 baseline  표시만 하고 그대로 사용 (= 커밋된 R-fold의 A-123. 수치가 같아야 한다 → 검산)
  2 exclude   전체-0 센서 웨이퍼를 학습·평가 조각에서 모두 뺀다 (분할은 1과 같고, 조각에서 빼기만 한다)
  3 dropfeat  A-123 train에서 한 번이라도 전체-0이 나오는 센서의 피처를 모든 웨이퍼에서 뺀다
  4 flags     1 + `n_allzero_sensors` + 센서별 전체-0 여부(0/1) 12개. 추가 피처도 같은 선택 절차를 거친다

MSE는 두 가지로 낸다.
  eval=all    그 방식의 평가 조각 전체
  eval=clean  평가 조각 중 전체-0 센서가 없는 웨이퍼만 (네 방식을 같은 웨이퍼로 비교)

센서별 전체-0 여부는 집계표의 R 피처에서 역산한다: 그 센서의 mean·std가 두 구간 모두 0.
train 1,981장 전부에서 12개 합이 `n_allzero_sensors`와 일치하는 것을 확인하고 쓴다 (어긋나면 멈춘다).
이 플래그는 진단용이다. 방식 4를 채택하면 `src/features.py`로 옮긴다 (3절 5번).

분할·OOB 순열의 난수는 원 스크립트가 세 그룹을 A-123 → A-456 → B-456 순으로 돌며 소비한다.
A-123만 돌리면서 같은 난수를 받으려고, 다른 두 그룹은 학습 없이 난수 소비만 똑같이 흉내 낸다
(분할: 같은 길이의 permutation, OOB: 같은 트리 시드로 얻은 OOB 크기만큼 shuffle).

실행: python src/allzero_handling.py
출력: results/diagnostics/2026-09-27_김혜연_allzero_handling.csv       분할별 MSE (+ .yaml)
      results/diagnostics/2026-09-27_김혜연_allzero_handling_votes.csv 방식별 피처 선택 횟수 (+ .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from aggregate import ALLZERO_SENSORS, FLOW, PRESSURE, ROTATION, r_feature_names  # noqa: E402
from load import KEY, ROOT  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from repro_R import INPUT, N_JOBS, N_REP, SEED, TRAIN_FRAC, build  # noqa: E402
from repro_R_selected import DI, MIN_VOTES, P_MAX, oob_importance, ttest_pass  # noqa: E402

GROUP = "A-123"
DATE = "2026-09-27"
OUT = ROOT / "results" / "diagnostics" / f"{DATE}_{AUTHOR}_allzero_handling.csv"
VOTES = ROOT / "results" / "diagnostics" / f"{DATE}_{AUTHOR}_allzero_handling_votes.csv"
REF = ROOT / "results" / f"2026-09-26_{AUTHOR}_R_montecarlo_cv_selected.csv"   # 읽기만 한다
METHODS = ("baseline", "exclude", "dropfeat", "flags")
ABBR = {**PRESSURE, **FLOW, **ROTATION}
FLAG_COLS = [f"allzero_{ABBR[s]}" for s in ALLZERO_SENSORS]
PRESSURE_FEATS = [c for c in r_feature_names() if any(f"_{a}_" in c for a in PRESSURE.values())]
ROTATION_FEATS = [c for c in r_feature_names() if any(f"_{a}_" in c for a in ROTATION.values())]
assert len(PRESSURE_FEATS) == 36 and len(ROTATION_FEATS) == 6


def allzero_flags(tab: pd.DataFrame) -> pd.DataFrame:
    """센서별 웨이퍼 전체-0 여부. 두 구간의 mean·std가 모두 0 (빈 구간은 0으로 본다)."""
    out = {}
    for s in ALLZERO_SENSORS:
        cols = [f"{st}_{ABBR[s]}_{seg}" for st in ("mean", "std") for seg in ("s1", "s2")]
        cols = [c for c in cols if c in tab.columns]                 # 회전은 mean만 있다
        out[f"allzero_{ABBR[s]}"] = (tab[cols].fillna(0) == 0).all(axis=1).astype(int)
    return pd.DataFrame(out, index=tab.index)


def tree_seeds() -> list[int]:
    """RandomForestRegressor(random_state=SEED)의 트리 시드. 데이터와 무관하다."""
    rf = RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=1)
    rf.fit(np.zeros((10, 1)), np.arange(10, dtype=float))
    return [t.random_state for t in rf.estimators_]


def skip_oob_draws(rng: np.random.Generator, seeds: list[int], n: int, n_feat: int) -> None:
    """`oob_importance`가 n행·n_feat열 학습 조각에서 소비하는 난수만큼 rng를 진행시킨다."""
    for s in seeds:
        m = len(_generate_unsampled_indices(s, n, n, None))
        if m < 3:
            continue
        buf = np.empty(m)
        for _ in range(n_feat):
            rng.shuffle(buf)


def fit_eval(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> dict:
    """LR과 tree bagging의 검증 조각 예측 오차 (웨이퍼별)."""
    errs = {}
    for model, est in (("lr", LinearRegression()),
                       ("tree_bag", RandomForestRegressor(max_features=1.0, random_state=SEED,
                                                          n_jobs=N_JOBS))):
        est.fit(tr[cols], tr["MRR"])
        errs[model] = te["MRR"].to_numpy(float) - est.predict(te[cols])
    return errs


def main() -> None:
    for p in (OUT, VOTES):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (CLAUDE.md 3절)")
    d, cols = build()
    tab = pd.read_csv(ROOT / INPUT)
    tr_all = tab[tab["split"] == "train"]
    flags_all = allzero_flags(tr_all)
    if not (flags_all.sum(axis=1) == tr_all["n_allzero_sensors"]).all():
        raise ValueError("역산한 센서별 전체-0 여부의 합이 n_allzero_sensors와 다르다")
    d = d.merge(tr_all[KEY + ["n_allzero_sensors"]].join(flags_all), on=KEY, how="left")
    assert d["n_allzero_sensors"].notna().all()

    a = d[d["group"] == GROUP]
    zero_sensors = [c for c in FLAG_COLS if a[c].any()]
    dropped = [c for c in cols if any(f"_{f.removeprefix('allzero_')}_" in c for f in zero_sensors)]
    feats = {"baseline": cols, "exclude": cols,
             "dropfeat": [c for c in cols if c not in dropped],
             "flags": cols + ["n_allzero_sensors"] + FLAG_COLS}
    print(f"{GROUP}: {len(a)}장 중 전체-0 센서 웨이퍼 {int((a['n_allzero_sensors'] > 0).sum())}장")
    print(f"  한 번이라도 전체-0인 센서 {len(zero_sensors)}개 → 방식 3에서 뺀 피처 {len(dropped)}개, "
          f"남은 {len(feats['dropfeat'])}개")

    # 분할: 원 스크립트와 같은 순서로 세 그룹의 permutation을 뽑고 A-123만 쓴다
    split_rng = np.random.default_rng(SEED)
    groups = list(d.groupby("group").size().items())                  # A-123, A-456, B-456 순
    assert groups[0][0] == GROUP
    splits, other_n = [], []
    for _ in range(N_REP):
        row = []
        for g, n in groups:
            idx = split_rng.permutation(n)
            cut = int(n * TRAIN_FRAC)
            if g == GROUP:
                splits.append((a.iloc[idx[:cut]], a.iloc[idx[cut:]]))
            else:
                row.append(cut)
        other_n.append(row)
    seeds = tree_seeds()

    rows, votes = [], {}
    for method in METHODS:
        use = feats[method]
        rng = np.random.default_rng(SEED)                             # 방식마다 원 스크립트와 같은 출발점
        votes[method] = np.zeros(len(use), int)
        for rep, (tr, te) in enumerate(splits):
            if method == "exclude":
                tr, te = tr[tr["n_allzero_sensors"] == 0], te[te["n_allzero_sensors"] == 0]
            X, y = tr[use].to_numpy(float), tr["MRR"].to_numpy(float)
            rf = RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=N_JOBS).fit(X, y)
            keep = ttest_pass(X, y) & (oob_importance(rf, X, y, rng) > 0)
            for n in other_n[rep]:                                    # A-456, B-456 난수 소비 흉내
                skip_oob_draws(rng, seeds, n, len(cols))
            votes[method] += keep
            sel = [c for c, k in zip(use, keep) if k]
            errs = fit_eval(tr, te, sel)
            clean = (te["n_allzero_sensors"] == 0).to_numpy()
            for model, err in errs.items():
                for ev, m in (("all", np.ones(len(te), bool)), ("clean", clean)):
                    e = err[m]
                    rows.append({"group": GROUP, "method": method, "model": model, "eval": ev,
                                 "fold": rep, "mse": float((e ** 2).mean()),
                                 "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                 "n_train": len(tr), "n_eval": int(m.sum()),
                                 "n_features": len(sel),
                                 "n_pressure": sum(c in PRESSURE_FEATS for c in sel),
                                 "n_rotation": sum(c in ROTATION_FEATS for c in sel),
                                 "n_flag": sum(c in FLAG_COLS or c == "n_allzero_sensors" for c in sel)})
            print(f"  [{method}] {rep + 1}/{N_REP} 회", flush=True)
    res = pd.DataFrame(rows).round(4)

    # 검산: 방식 1은 커밋된 R-fold의 A-123과 같아야 한다
    ref = pd.read_csv(REF)
    ref = ref[(ref["group"] == GROUP) & (ref["featureset"] == "R-fold")].set_index(["model", "fold"])
    mine = res[(res["method"] == "baseline") & (res["eval"] == "all")].set_index(["model", "fold"])
    diff = (mine["mse"] - ref.loc[mine.index, "mse"]).abs().max()
    nf_same = (mine["n_features"] == ref.loc[mine.index, "n_features"]).all()
    print(f"\n검산: 방식 1 vs 커밋된 R-fold A-123: MSE 최대 차이 {diff:.4f}, 피처 수 일치 {nf_same}")

    vt = pd.DataFrame({"feature": cols + ["n_allzero_sensors"] + FLAG_COLS})
    for m in METHODS:
        v = dict(zip(feats[m], votes[m]))
        vt[f"votes_{m}"] = [v.get(f, np.nan) for f in vt["feature"]]  # 그 방식에 없는 피처는 NaN
    voted = {m: [f for f, v in zip(feats[m], votes[m]) if v >= MIN_VOTES] for m in METHODS}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    vt.to_csv(VOTES, index=False)
    common = [
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/allzero_handling.py",
        f"input: {INPUT} (14.2에서 동결한 버전)", f"group: {GROUP}",
        f"seed: {SEED}",
        f"n_jobs: {N_JOBS}  # 병렬 처리 설정. random_state 고정이라 결과는 n_jobs와 무관하게 같다",
        f"cv: Monte Carlo CV {N_REP}회, {TRAIN_FRAC:.1f}/{1 - TRAIN_FRAC:.1f}. "
        "분할은 repro_R_selected.py와 같다 (다른 두 그룹은 난수 소비만 흉내)",
        f"selection: R-fold. p < {P_MAX} (단순회귀 t-검정) AND OOB permutation importance > 0, "
        "분할마다 학습 조각 안에서만",
        "methods: {baseline: 표시만 하고 그대로, exclude: 전체-0 센서 웨이퍼를 학습·평가에서 제외, "
        f"dropfeat: 전체-0이 한 번이라도 나온 센서 {len(zero_sensors)}개의 피처 {len(dropped)}개 제거, "
        "flags: baseline + n_allzero_sensors + 센서별 전체-0 여부 12개 (선택 절차를 같이 거침)}",
        f"zero_sensors: {[f.removeprefix('allzero_') for f in zero_sensors]}",
        "flag_source: R 피처 mean·std가 두 구간 모두 0 → 전체-0. train 1,981장에서 합이 n_allzero_sensors와 일치",
        "labels: train 라벨만. test 라벨과 data/locked/ 는 열지 않았다",
    ]
    OUT.with_suffix(".yaml").write_text("\n".join(common + [
        "content: 방식별·모델별·평가 집합별 분할 MSE",
        "eval: {all: 그 방식의 평가 조각 전체, clean: 평가 조각 중 전체-0 센서가 없는 웨이퍼만}",
        "models: {lr: LinearRegression, tree_bag: RandomForestRegressor(max_features=1.0, "
        "random_state=42)}",
        f"check: baseline/all 이 {REF.relative_to(ROOT).as_posix()} 의 A-123 R-fold와 "
        f"MSE 최대 차이 {diff:.4f}, 피처 수 일치 {bool(nf_same)}",
        "note: 진단 결과 (14.4). 처리 방식은 확정하지 않았다",
        "",
    ]), encoding="utf-8")
    VOTES.with_suffix(".yaml").write_text("\n".join(common + [
        f"content: 방식별 피처 선택 횟수 ({N_REP}회 중). 그 방식에 없는 피처는 빈 칸",
        f"voted_ge_{MIN_VOTES}: " + str({m: len(voted[m]) for m in METHODS}),
        "related: " + OUT.relative_to(ROOT).as_posix(),
        "",
    ]), encoding="utf-8")

    print(f"\n[{GROUP} MSE 평균 ± 표준편차]  배율 = MSE / Di Table 4 (LR {DI['lr'][GROUP]}, "
          f"tree bagging {DI['tree_bag'][GROUP]})")
    for model in ("lr", "tree_bag"):
        for m in METHODS:
            cells = []
            for ev in ("all", "clean"):
                s = res[(res["method"] == m) & (res["model"] == model) & (res["eval"] == ev)]
                cells.append(f"{ev} {s['mse'].mean():6.2f}±{s['mse'].std(ddof=1):5.2f} "
                             f"(×{s['mse'].mean() / DI[model][GROUP]:.2f}, n_eval {s['n_eval'].median():.0f})")
            print(f"  {model:9s} {m:9s} " + " | ".join(cells))
    print("\n[선택된 피처 수] 분할별 중앙값 (최소~최대) / 20회 중 10회 이상")
    for m in METHODS:
        s = res[(res["method"] == m) & (res["model"] == "lr") & (res["eval"] == "all")]
        vp = sum(c in PRESSURE_FEATS for c in voted[m])
        vr = sum(c in ROTATION_FEATS for c in voted[m])
        line = [f"{k} {s[k].median():.0f} ({s[k].min()}~{s[k].max()})"
                for k in ("n_features", "n_pressure", "n_rotation", "n_flag")]
        print(f"  {m:9s} " + ", ".join(line) + f" / 10회 이상 {len(voted[m])}개 (압력 {vp}, 회전 {vr})")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {VOTES.relative_to(ROOT)} (+ 각 .yaml)")


if __name__ == "__main__":
    main()

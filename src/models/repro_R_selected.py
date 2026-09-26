"""R 세트 재현 + 피처 선택 (CLAUDE.md 13절 임시 결정, 2026-09-26). 결과는 새 파일로 저장한다.

선택 기준 (13절, 김혜연 임시 결정 → 팀 회의에서 확정)
  각 Monte Carlo CV 분할의 **학습 조각 안에서만** 계산한다.
  ① t-검정: 피처 하나씩 MRR에 단순회귀했을 때 p < 0.05
  ② OOB 중요도: tree bagging의 permutation importance > 0 (Breiman 방식, 트리별 OOB 표본에서 측정)
  ③ 둘 다 통과하면 그 분할에서 "선택"
  ④ 20회 중 10회 이상 선택된 피처를 최종 피처로 사용

평가 모델 (같은 20개 분할, 같은 검증 조각)
  all         피처 125개 전부 (`repro_R.py`와 같은 조건, 비교용)
  voted       ④의 최종 피처 (Di 원문 방식. 투표는 20개 분할의 **학습 조각**만 보고 모은다)
  fold        그 분할에서 ③을 통과한 피처만 (분할마다 다름. 분할 간에도 정보가 새지 않는 엄격한 버전)

`voted`는 20개 분할의 학습 조각을 모아 만든 목록이라 같은 분할의 검증 조각으로 평가하면 완전히
중첩된 설계는 아니다. 그래서 `fold`를 같이 낸다. 두 값이 크게 다르면 `fold`를 믿는다.

실행: python src/models/repro_R_selected.py
출력: results/2026-09-26_김혜연_R_montecarlo_cv_selected.csv (+ 같은 이름의 .yaml)
      results/diagnostics/2026-09-26_김혜연_R_feature_votes.csv (+ 같은 이름의 .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import ROOT  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from repro_R import GROUPS, N_JOBS, N_REP, SEED, TRAIN_FRAC, build  # noqa: E402

P_MAX, MIN_VOTES = 0.05, 10
OUT = ROOT / "results" / f"2026-09-26_{AUTHOR}_R_montecarlo_cv_selected.csv"
VOTES = ROOT / "results" / "diagnostics" / f"2026-09-26_{AUTHOR}_R_feature_votes.csv"
DI = {"lr": {"A-456": 5.77, "B-456": 10.95, "A-123": 5.00},
      "tree_bag": {"A-456": 5.65, "B-456": 8.32, "A-123": 5.32}}


def ttest_pass(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """피처 하나씩 MRR에 단순회귀했을 때 기울기의 p < 0.05. 단순회귀의 t-검정 = 상관의 t-검정."""
    n = len(y)
    xc = X - X.mean(0)
    yc = y - y.mean()
    sx = np.sqrt((xc ** 2).sum(0))
    r = np.divide(xc.T @ yc, sx * np.sqrt((yc ** 2).sum()),
                  out=np.zeros(X.shape[1]), where=sx > 0)      # 상수 피처는 r = 0
    r = np.clip(r, -1 + 1e-12, 1 - 1e-12)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return stats.t.sf(np.abs(t), n - 2) * 2 < P_MAX


def oob_importance(rf: RandomForestRegressor, X: np.ndarray, y: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """Breiman OOB permutation importance: 트리마다 자기 OOB 표본에서 MSE 증가량."""
    imp = np.zeros(X.shape[1])
    for tree in rf.estimators_:
        oob = _generate_unsampled_indices(tree.random_state, len(X), len(X), None)
        if len(oob) < 3:
            continue
        Xo, yo = X[oob].copy(), y[oob]
        base = float(((yo - tree.predict(Xo)) ** 2).mean())
        for j in range(X.shape[1]):
            col = Xo[:, j].copy()
            rng.shuffle(Xo[:, j])
            imp[j] += float(((yo - tree.predict(Xo)) ** 2).mean()) - base
            Xo[:, j] = col
    return imp / len(rf.estimators_)


def evaluate(name: str, tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> list[dict]:
    """고른 피처로 LR과 tree bagging을 학습해 검증 조각의 오차를 낸다."""
    out = []
    for model, est in (("lr", LinearRegression()),
                       ("tree_bag", RandomForestRegressor(max_features=1.0, random_state=SEED,
                                                          n_jobs=N_JOBS))):
        if not cols:
            out.append({"model": model, "mse": np.nan, "mae": np.nan, "rmse": np.nan})
            continue
        est.fit(tr[cols], tr["MRR"])
        err = te["MRR"].to_numpy(float) - est.predict(te[cols])
        out.append({"model": model, "mse": float((err ** 2).mean()),
                    "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))})
    for o in out:
        o["featureset"] = f"R-{name}"
        o["n_features"] = len(cols)
    return out


def main() -> None:
    for p in (OUT, VOTES):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (CLAUDE.md 3절)")
    d, cols = build()
    rng = np.random.default_rng(SEED)
    split_rng = np.random.default_rng(SEED)
    votes = {g: np.zeros(len(cols), int) for g in GROUPS}
    splits = {g: [] for g in GROUPS}
    per_fold = {g: [] for g in GROUPS}

    print(f"\n피처 선택 (13절 임시 기준: p < {P_MAX} AND OOB 중요도 > 0, {N_REP}회 중 {MIN_VOTES}회 이상)")
    for rep in range(N_REP):
        for group, gd in d.groupby("group"):
            idx = split_rng.permutation(len(gd))
            cut = int(len(gd) * TRAIN_FRAC)
            tr, te = gd.iloc[idx[:cut]], gd.iloc[idx[cut:]]
            splits[group].append((tr, te))
            X, y = tr[cols].to_numpy(float), tr["MRR"].to_numpy(float)   # 학습 조각만 본다
            rf = RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=N_JOBS).fit(X, y)
            keep = ttest_pass(X, y) & (oob_importance(rf, X, y, rng) > 0)
            votes[group] += keep
            per_fold[group].append([c for c, k in zip(cols, keep) if k])
        print(f"  {rep + 1}/{N_REP} 회", end="\r", flush=True)

    final = {g: [c for c, v in zip(cols, votes[g]) if v >= MIN_VOTES] for g in GROUPS}
    print("\n\n[선택된 피처 수] 그룹별")
    for g in GROUPS:
        nf = [len(f) for f in per_fold[g]]
        print(f"  {g}: 최종 {len(final[g]):3d}개 / 125   (분할별 선택 수 중앙값 {int(np.median(nf))}개)")

    rows = []
    for group in GROUPS:
        for rep, (tr, te) in enumerate(splits[group]):
            for name, use in (("all", cols), ("voted", final[group]), ("fold", per_fold[group][rep])):
                for r in evaluate(name, tr, te, use):
                    rows.append({"group": group, "fold": rep, "n_train": len(tr), "n_eval": len(te),
                                 **r})
    res = pd.DataFrame(rows)[["group", "model", "featureset", "fold", "mse", "mae", "rmse",
                              "n_train", "n_eval", "n_features"]].round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-26", f"author: {AUTHOR}", "script: src/models/repro_R_selected.py",
        "featureset: R (Di 125개) + 13절 임시 선택 기준",
        "input: data/processed/wafer_table.csv (14.2에서 동결한 버전)",
        f"seed: {SEED}",
        f"n_jobs: {N_JOBS}  # 병렬 처리 설정. random_state 고정이라 결과는 n_jobs와 무관하게 같다",
        f"cv: Monte Carlo CV {N_REP}회, 그룹별 무작위 {TRAIN_FRAC}/{1 - TRAIN_FRAC}",
        "revealed: train 전체 (R 재현 설정, CLAUDE.md 7절)",
        f"selection: p < {P_MAX} (단순회귀 t-검정) AND OOB permutation importance > 0, "
        f"{N_REP}회 중 {MIN_VOTES}회 이상. 각 분할의 학습 조각 안에서만 계산",
        "variants: {all: 125개 전부, voted: 최종 투표 피처, fold: 분할별 선택 피처(엄격)}",
        "models: {lr: LinearRegression, tree_bag: RandomForestRegressor(max_features=1.0, "
        "random_state=42)}",
        "votes_file: results/diagnostics/2026-09-26_김혜연_R_feature_votes.csv",
        "selected_features:",
        *[f"  {g}: {final[g]}" for g in GROUPS],
        "labels: train 라벨만. test 라벨과 data/locked/ 는 열지 않았다",
        "caveat: voted 는 20개 분할의 학습 조각을 모아 만든 목록이라 완전 중첩 설계가 아니다 → fold 와 비교",
        "",
    ]), encoding="utf-8")

    vt = pd.DataFrame({"feature": cols})
    for g in GROUPS:
        vt[f"votes_{g}"] = votes[g]
        vt[f"selected_{g}"] = (votes[g] >= MIN_VOTES).astype(int)
    vt.to_csv(VOTES, index=False)
    VOTES.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-26", f"author: {AUTHOR}", "script: src/models/repro_R_selected.py",
        "content: R 피처 125개의 그룹별 선택 투표 수 (20회 중)",
        f"rule: p < {P_MAX} AND OOB permutation importance > 0, 각 분할의 학습 조각 안에서만",
        f"selected: votes >= {MIN_VOTES}",
        "related: results/2026-09-26_김혜연_R_montecarlo_cv_selected.csv",
        "note: 진단 결과 (14.4)",
        "",
    ]), encoding="utf-8")

    print("\n[MSE 평균 ± 표준편차]  괄호는 Di Table 4")
    for model in ("lr", "tree_bag"):
        for name in ("all", "voted", "fold"):
            s = res[(res["model"] == model) & (res["featureset"] == f"R-{name}")]
            cells = []
            for g in GROUPS:
                x = s[s["group"] == g]["mse"]
                cells.append(f"{g} {x.mean():7.2f}±{x.std(ddof=1):5.2f} (Di {DI[model][g]})")
            print(f"  {model:9s} {name:6s} " + " | ".join(cells))
    print(f"\n저장: {OUT.relative_to(ROOT)}, {VOTES.relative_to(ROOT)} (+ 각 .yaml)")


if __name__ == "__main__":
    main()

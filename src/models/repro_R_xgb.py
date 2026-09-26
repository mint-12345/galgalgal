"""XGBoost로 같은 재현 실험 (가이드 2-5, CLAUDE.md 12절·14.3절). 결과는 새 파일로 저장한다.

Di 원문에 없는 모델이라 목표값이 없다. **tree bagging과 비슷한 수준인지**만 본다.

피처는 `R-fold`: 각 Monte Carlo CV 분할의 학습 조각 안에서 13절 임시 기준
(t-검정 p < 0.05 AND OOB permutation importance > 0)을 통과한 피처만 쓴다.
분할과 선택은 `repro_R_selected.py`와 **같은 난수열로 재현**한다 (같은 SEED, 같은 루프 순서).
그 검증을 위해 tree bagging도 같은 분할에서 다시 돌려 기존 결과와 맞는지 확인한다
(`results/2026-09-26_김혜연_R_montecarlo_cv_selected.csv`의 R-fold 값과 같아야 한다).

# TODO(결정 필요): XGBoost 하이퍼파라미터는 탐색하지 않고 아래 값으로 고정했다.
#   튜닝을 하려면 학습 조각 안에서만 해야 한다 (13절 피처 선택과 같은 이유).

실행: python src/models/repro_R_xgb.py
출력: results/2026-09-26_김혜연_R_xgb.csv (+ 같은 이름의 .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import ROOT  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from repro_R import GROUPS, N_JOBS, N_REP, SEED, TRAIN_FRAC, build  # noqa: E402
from repro_R_selected import P_MAX, oob_importance, ttest_pass  # noqa: E402

OUT = ROOT / "results" / f"2026-09-26_{AUTHOR}_R_xgb.csv"
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                  objective="reg:squarederror", tree_method="hist",
                  random_state=SEED, n_jobs=N_JOBS)
REF = {"A-456": 6.80, "B-456": 10.41, "A-123": 7.00}   # 기존 R-fold tree bagging (재현 확인용)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    d, cols = build()
    rng = np.random.default_rng(SEED)          # OOB 순열용 (repro_R_selected와 같은 순서로 소비)
    split_rng = np.random.default_rng(SEED)    # 분할용
    rows = []
    print(f"\n분할별 피처 선택 + XGBoost/tree bagging ({N_REP}회 × {len(GROUPS)}그룹)")
    for rep in range(N_REP):
        for group, gd in d.groupby("group"):
            idx = split_rng.permutation(len(gd))
            cut = int(len(gd) * TRAIN_FRAC)
            tr, te = gd.iloc[idx[:cut]], gd.iloc[idx[cut:]]
            X, y = tr[cols].to_numpy(float), tr["MRR"].to_numpy(float)   # 학습 조각만 본다
            rf = RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=N_JOBS).fit(X, y)
            keep = ttest_pass(X, y) & (oob_importance(rf, X, y, rng) > 0)
            sel = [c for c, k in zip(cols, keep) if k]
            for model, est in (("xgb", XGBRegressor(**XGB_PARAMS)),
                               ("tree_bag", RandomForestRegressor(max_features=1.0,
                                                                  random_state=SEED,
                                                                  n_jobs=N_JOBS))):
                est.fit(tr[sel], tr["MRR"])
                err = te["MRR"].to_numpy(float) - est.predict(te[sel])
                rows.append({"group": group, "model": model, "featureset": "R-fold", "fold": rep,
                             "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                             "rmse": float(np.sqrt((err ** 2).mean())), "n_train": len(tr),
                             "n_eval": len(te), "n_features": len(sel)})
        print(f"  {rep + 1}/{N_REP} 회", end="\r", flush=True)
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-26", f"author: {AUTHOR}", "script: src/models/repro_R_xgb.py",
        "featureset: R-fold (분할별로 13절 임시 기준을 통과한 피처만)",
        "input: data/processed/wafer_table.csv (14.2에서 동결한 버전)",
        f"seed: {SEED}",
        f"n_jobs: {N_JOBS}  # 병렬 처리 설정. random_state 고정이라 결과는 n_jobs와 무관하게 같다",
        f"cv: Monte Carlo CV {N_REP}회, 그룹별 무작위 {TRAIN_FRAC}/{1 - TRAIN_FRAC} "
        "(repro_R_selected.py와 같은 난수열로 재현한 같은 분할)",
        f"selection: p < {P_MAX} AND OOB permutation importance > 0, 학습 조각 안에서만 (13절)",
        "models:",
        f"  xgb: XGBRegressor({', '.join(f'{k}={v!r}' for k, v in XGB_PARAMS.items())})",
        "  tree_bag: RandomForestRegressor(max_features=1.0, random_state=42, n_jobs=-1) "
        "# 분할 재현 확인용",
        "tuning: 없음. 하이퍼파라미터는 고정값 (TODO 결정 필요)",
        "di_target: 없음 (Di 원문에 XGBoost 없음). tree bagging 과 비슷한 수준인지만 본다",
        "labels: train 라벨만. test 라벨과 data/locked/ 는 열지 않았다",
        "",
    ]), encoding="utf-8")

    print("\n[MSE 평균 ± 표준편차, R-fold]")
    for model in ("xgb", "tree_bag"):
        cells = []
        for g in GROUPS:
            x = res[(res["model"] == model) & (res["group"] == g)]["mse"]
            cells.append(f"{g} {x.mean():6.2f}±{x.std(ddof=1):4.2f}")
        print(f"  {model:9s} " + " | ".join(cells))
    same = all(abs(res[(res["model"] == "tree_bag") & (res["group"] == g)]["mse"].mean() - REF[g])
               < 0.005 for g in GROUPS)
    print(f"  분할 재현 확인: tree bagging 이 기존 R-fold 값과 {'일치' if same else '불일치 → 확인 필요'}"
          f" ({', '.join(f'{g} {REF[g]}' for g in GROUPS)})")
    print(f"  분할별 선택 피처 수 중앙값: "
          f"{ {g: int(res[res['group'] == g]['n_features'].median()) for g in GROUPS} }")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""R-fold tree bagging에 압력 시간 무리 표시(press_long)를 더하면 오차가 달라지는가 (CLAUDE.md 12절·14.3절).

press_long은 `src/features.py`에서 만들어 `data/processed/features_S.csv`에 저장한 값을 키로 병합한다
(모델 파일 안에서 피처를 만들지 않는다, 3절 5번). 456 그룹에만 정의된다 (A-123은 NaN).

분할과 피처 선택은 `repro_R_selected.py`와 **같은 난수열로 재현**한다 (같은 SEED, 같은 루프 순서).
A-123도 난수열을 맞추려고 선택까지 돌리고, 기준 모델만 평가한다.
비교 (같은 분할, 같은 검증 조각, 같은 random_state)
  R-fold             그 분할의 학습 조각에서 13절 임시 기준을 통과한 피처 (기존 결과 재현 확인용)
  R-fold+press_long  위 피처 + press_long (선택 절차를 거치지 않고 강제로 넣는다)
참고로 학습 조각에서 press_long이 t-검정(p < 0.05)을 통과하는지와, R 피처 중 press_long과
상관이 가장 큰 피처를 함께 기록한다.

실행: python src/models/repro_R_press_cluster.py   (먼저 python src/features.py)
출력: results/2026-10-14_김혜연_R_press_cluster.csv (+ 같은 이름의 .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import OUT as FEATURES  # noqa: E402
from load import KEY, ROOT  # noqa: E402
from pv_check import AUTHOR  # noqa: E402
from repro_R import GROUPS, N_JOBS, N_REP, SEED, TRAIN_FRAC, build  # noqa: E402
from repro_R_selected import P_MAX, oob_importance, ttest_pass  # noqa: E402

OUT = ROOT / "results" / f"2026-10-14_{AUTHOR}_R_press_cluster.csv"
FLAG = "press_long"
REF = {"A-456": 6.80, "B-456": 10.41, "A-123": 7.00}   # 기존 R-fold tree bagging (재현 확인용)


def tree_bag(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    est = RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=N_JOBS)
    est.fit(tr[cols], tr["MRR"])
    return te["MRR"].to_numpy(float) - est.predict(te[cols])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    d, cols = build()
    n0 = len(d)
    d = d.merge(pd.read_csv(FEATURES)[KEY + [FLAG]], on=KEY, how="left")
    if len(d) != n0:
        raise ValueError("features_S.csv 병합 후 행 수가 바뀌었다")
    if d.loc[d["group"] != "A-123", FLAG].isna().any():
        raise ValueError("456 그룹에 press_long 이 비어 있다")

    corr = []
    for g in ("A-456", "B-456"):
        gd = d[d["group"] == g]
        r = gd[cols].corrwith(gd[FLAG]).abs().sort_values(ascending=False)
        corr.append(f"  {g}: " + ", ".join(f"{c} {v:.3f}" for c, v in r.head(5).items()))
    print("press_long 과 상관이 큰 R 피처 (|r|, train)\n" + "\n".join(corr))

    rng = np.random.default_rng(SEED)          # OOB 순열용 (repro_R_selected와 같은 순서로 소비)
    split_rng = np.random.default_rng(SEED)    # 분할용
    rows = []
    print(f"\n분할별 피처 선택 + tree bagging ({N_REP}회 × {len(GROUPS)}그룹)", flush=True)
    for rep in range(N_REP):
        for group, gd in d.groupby("group"):
            idx = split_rng.permutation(len(gd))
            cut = int(len(gd) * TRAIN_FRAC)
            tr, te = gd.iloc[idx[:cut]], gd.iloc[idx[cut:]]
            X, y = tr[cols].to_numpy(float), tr["MRR"].to_numpy(float)   # 학습 조각만 본다
            rf = RandomForestRegressor(max_features=1.0, random_state=SEED, n_jobs=N_JOBS).fit(X, y)
            keep = ttest_pass(X, y) & (oob_importance(rf, X, y, rng) > 0)
            sel = [c for c, k in zip(cols, keep) if k]
            variants = [("R-fold", sel)]
            flag_t = np.nan
            if group != "A-123":
                variants.append((f"R-fold+{FLAG}", sel + [FLAG]))
                flag_t = int(ttest_pass(tr[[FLAG]].to_numpy(float), y)[0])
            for name, use in variants:
                err = tree_bag(tr, te, use)
                rows.append({"group": group, "model": "tree_bag", "featureset": name, "fold": rep,
                             "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                             "rmse": float(np.sqrt((err ** 2).mean())), "n_train": len(tr),
                             "n_eval": len(te), "n_features": len(use), "flag_ttest_pass": flag_t,
                             "share_long_eval": float(te[FLAG].mean()) if group != "A-123" else np.nan})
        print(f"  {rep + 1}/{N_REP} 회", flush=True)
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-10-14", f"author: {AUTHOR}", "script: src/models/repro_R_press_cluster.py",
        f"featureset: R-fold (분할별 13절 임시 기준 통과 피처), R-fold+{FLAG} (위 + {FLAG} 강제 포함)",
        "input: data/processed/wafer_table.csv (14.2에서 동결한 버전) + data/processed/features_S.csv (press_long)",
        f"{FLAG}: src/features.py. 456 그룹에서 압력 6개 중 하나라도 > 0 인 시간(빈칸 dt > 2초 제외) >= 220초면 1",
        f"seed: {SEED}",
        f"n_jobs: {N_JOBS}  # 병렬 처리 설정. random_state 고정이라 결과는 n_jobs와 무관하게 같다",
        f"cv: Monte Carlo CV {N_REP}회, 그룹별 무작위 {TRAIN_FRAC}/{round(1 - TRAIN_FRAC, 2)} "
        "(repro_R_selected.py와 같은 난수열로 재현한 같은 분할)",
        f"selection: p < {P_MAX} AND OOB permutation importance > 0, 학습 조각 안에서만 (13절). {FLAG} 는 선택을 거치지 않음",
        "model: RandomForestRegressor(max_features=1.0, random_state=42, n_jobs=-1)",
        f"flag_ttest_pass: 그 분할 학습 조각에서 {FLAG} 단독 t-검정 p < {P_MAX} 이면 1 (참고)",
        "flag_corr_top5 (|r|, train 전체):",
        *corr,
        "labels: train 라벨만. test 라벨과 data/locked/ 는 열지 않았다",
        "",
    ]), encoding="utf-8")

    print("\n[MSE 평균 ± 표준편차]")
    for g in GROUPS:
        for name in res[res["group"] == g]["featureset"].unique():
            x = res[(res["group"] == g) & (res["featureset"] == name)]["mse"]
            print(f"  {g} {name:20s} {x.mean():6.3f} ± {x.std(ddof=1):5.3f}")
    for g in ("A-456", "B-456"):
        a = res[(res["group"] == g) & (res["featureset"] == "R-fold")].set_index("fold")["mse"]
        b = res[(res["group"] == g) & (res["featureset"] == f"R-fold+{FLAG}")].set_index("fold")["mse"]
        diff = b - a
        print(f"  {g} 짝지은 차이 (+{FLAG} − 기준): {diff.mean():+.4f} ± {diff.std(ddof=1):.4f}, "
              f"개선 {int((diff < 0).sum())}/{len(diff)}회")
    same = all(abs(res[(res["group"] == g) & (res["featureset"] == "R-fold")]["mse"].mean() - REF[g])
               < 0.005 for g in GROUPS)
    print(f"  분할 재현 확인: R-fold 가 기존 값과 {'일치' if same else '불일치 → 확인 필요'} ({REF})")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

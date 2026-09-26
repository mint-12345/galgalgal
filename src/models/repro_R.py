"""R 세트 재현 모델 (CLAUDE.md 9절, 12절, 14.3). 지속 모델 / LR / Tree bagging.

피처는 R 세트만 쓴다 (6.1절).
  ID 1~21   `features_repro.neighbor_features` (F4). 재현이므로 `revealed`에 train 전체를 넘긴다 (7절).
  ID 22~125 동결된 `data/processed/wafer_table.csv`의 104개 (`aggregate.r_feature_names()`).
물리 파트의 P·V 후보나 진단 결과는 쓰지 않는다.

평가는 Di 2.4절의 Monte Carlo CV: train을 무작위 80/20으로 나누기를 20회, 그룹별로 따로.
  지속 모델  r_t = r_{t−1} (학습 없음). `lag_MRR_01`을 그대로 예측값으로 쓴다.
  LR         LinearRegression (원문의 t-검정 피처 선택은 아직 임계값 미정 → 전체 피처)
  tree_bag   RandomForestRegressor(max_features=1.0, random_state=SEED)  (9절)

# TODO(결정 필요): 6.1절 피처 선택 절차(Monte Carlo CV 20회 투표)의 임계값. 지금은 선택 없이 전부 쓴다.
# 주의: 지속 모델 기준선이 Di Table 4의 1.6배다 (docs/reports/repro_김혜연.md "지속 모델 기준선 불일치").
#       그래서 절대값이 아니라 **지속 모델 대비 개선폭**으로 읽어야 한다.

실행: python src/models/repro_R.py
출력: results/2026-09-26_김혜연_R_montecarlo_cv.csv (+ 같은 이름의 .yaml)  ← 12절·14.3절 규칙
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate import OUTLIER_MRR, r_feature_names  # noqa: E402
from features_repro import K_USAGE, N_LAG, neighbor_features  # noqa: E402
from load import KEY, ROOT  # noqa: E402
from pv_check import AUTHOR  # noqa: E402

SEED, N_REP, TRAIN_FRAC = 42, 20, 0.8
FEATURESET = "R"
INPUT = "data/processed/wafer_table.csv"
OUT = ROOT / "results" / f"2026-09-26_{AUTHOR}_R_montecarlo_cv.csv"
GROUPS = ["A-456", "B-456", "A-123"]
DI = {"persistence": {"A-456": 7.55, "B-456": 11.4, "A-123": 5.62, "전체": 8.78},
      "lr": {"A-456": 5.77, "B-456": 10.95, "A-123": 5.00, "전체": 7.76},
      "tree_bag": {"A-456": 5.65, "B-456": 8.32, "A-123": 5.32, "전체": 6.48}}


def build() -> tuple[pd.DataFrame, list[str]]:
    """train 웨이퍼의 R 피처 행렬 (ID 1~125)."""
    tab = pd.read_csv(ROOT / INPUT)
    tab = tab[~(tab["MRR"] > OUTLIER_MRR)].reset_index(drop=True)
    f4 = neighbor_features(tab, tab[tab["split"] == "train"][KEY])   # 재현: train 전체 공개 (7절)
    lag = [f"lag_MRR_{i + 1:02d}" for i in range(N_LAG)]
    nn = [f"nn_MRR_{i + 1:02d}" for i in range(K_USAGE)]
    cols = lag + nn + r_feature_names()                              # ID 1~21 + 22~125
    d = pd.concat([tab[KEY + ["split", "group", "MRR"]], f4[lag + nn]], axis=1)
    d = d.join(tab[r_feature_names()])
    d = d[d["split"] == "train"]
    n_before = len(d)
    d = d.dropna(subset=cols + ["MRR"]).reset_index(drop=True)
    print(f"train {n_before:,}장 → 피처가 온전한 {len(d):,}장 "
          f"(지연이 없는 첫 웨이퍼·구간이 빈 웨이퍼 {n_before - len(d)}장 제외), 피처 {len(cols)}개")
    return d, cols


def fit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if name == "persistence":
        return te["lag_MRR_01"].to_numpy(float)                      # 학습 없음
    model = (LinearRegression() if name == "lr"
             else RandomForestRegressor(max_features=1.0, random_state=SEED))
    model.fit(tr[cols], tr["MRR"])
    return model.predict(te[cols])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    d, cols = build()
    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(N_REP):
        for group, gd in d.groupby("group"):
            idx = rng.permutation(len(gd))
            cut = int(len(gd) * TRAIN_FRAC)
            tr, te = gd.iloc[idx[:cut]], gd.iloc[idx[cut:]]
            for name in ("persistence", "lr", "tree_bag"):
                err = te["MRR"].to_numpy(float) - fit_predict(name, tr, te, cols)
                rows.append({"group": group, "model": name, "featureset": FEATURESET, "fold": rep,
                             "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                             "rmse": float(np.sqrt((err ** 2).mean())),
                             "n_train": len(tr), "n_eval": len(te)})
        print(f"  {rep + 1}/{N_REP} 회 완료", end="\r")
    res = pd.DataFrame(rows).round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-26", f"author: {AUTHOR}", "script: src/models/repro_R.py",
        f"featureset: {FEATURESET} (Di 125개: ID 1~21 neighbor_features + ID 22~125 집계표 104개)",
        f"input: {INPUT} (14.2에서 동결한 버전)",
        f"seed: {SEED}", f"cv: Monte Carlo CV {N_REP}회, 그룹별 무작위 {TRAIN_FRAC}/{1 - TRAIN_FRAC}",
        "revealed: train 전체 (R 재현 설정, CLAUDE.md 7절)",
        "models: {persistence: r_t = r_{t-1}, lr: LinearRegression, "
        "tree_bag: RandomForestRegressor(max_features=1.0, random_state=42)}",
        "feature_selection: 없음 (6.1절 임계값 미정 → TODO 결정 필요)",
        "wafers: MRR > 400 이상치 4장 제외, 피처에 NaN이 있는 웨이퍼 제외",
        "labels: train 라벨만. test 라벨과 data/locked/ 는 열지 않았다",
        "caveat: 지속 모델 기준선이 Di Table 4의 1.6배다 "
        "(docs/reports/repro_김혜연.md '지속 모델 기준선 불일치'). 절대값 대신 지속 대비 개선폭으로 읽는다",
        "",
    ]), encoding="utf-8")

    print("\n[Monte Carlo CV 20회 평균 MSE]  괄호는 Di Table 4")
    piv = res.pivot_table(index="model", columns="group", values="mse", aggfunc=["mean", "std"])
    for name in ("persistence", "lr", "tree_bag"):
        base = res[(res["model"] == "persistence")].groupby("group")["mse"].mean()
        line = []
        for g in GROUPS:
            m = piv[("mean", g)][name]
            s = piv[("std", g)][name]
            line.append(f"{g} {m:6.2f}±{s:4.2f} (Di {DI[name][g]}, 지속 대비 "
                        f"{(1 - m / base[g]) * 100:+5.1f}%)")
        print(f"  {name:11s} " + " | ".join(line))
    print(f"\n  참고 Di의 지속 대비 개선폭: LR {1 - 7.76 / 8.78:.1%}, "
          f"tree bagging {1 - 6.48 / 8.78:.1%} (전체 기준)")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

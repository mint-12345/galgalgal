"""Step 8: Step 7 표(1,977행)를 5개 묶음(fold)으로 나누는 평가 구성 만들기.

- training 에서 만든 Step 7 표만 사용한다. test / validation 은 사용하지 않는다.
- 조건 1: 같은 WAFER_ID 는 반드시 같은 fold (StratifiedGroupKFold 의 group = WAFER_ID)
- 조건 2: fold 마다 STAGE x CHAMBER_PATH 구성이 전체와 비슷하도록 층화
- 조건 3: 각 행의 fold 번호를 파일로 저장해 이후 모든 모델 비교에서 그대로 사용
- WAFER_ID 는 나누는 용도로만 쓴다. 모델은 학습하지 않는다.

fold 선택 방법:
  random_state 0~99 로 StratifiedGroupKFold 를 만들어 보고,
  (1) fold 별 STAGE x CHAMBER_PATH 비율과 전체 비율의 차이 합이 가장 작고,
  (2) 같으면 fold 행 수의 최대-최소 차이가 가장 작고,
  (3) 그래도 같으면 random_state 가 가장 작은 구성을 고른다.
출력: data/processed/step8_cv_folds_v1.csv
실행: python -B scripts/step8_make_cv_folds.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

pd.set_option("display.width", 200)

ROOT = Path(__file__).resolve().parents[1]
TABLE_FILE = ROOT / "data" / "processed" / "step7_wafer_stage_table_v1.csv"
OUT_FILE = ROOT / "data" / "processed" / "step8_cv_folds_v1.csv"
N_FOLDS = 5
SEEDS = range(100)

# A|other 층은 1행뿐이라 sklearn 이 "least populated class ... less than n_splits" 경고를 매번 낸다.
# 예상된 상황이므로 이 경고만 숨긴다.
warnings.filterwarnings("ignore", message="The least populated class in y has only")


def assign(table, strata, seed):
    fold = np.zeros(len(table), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(splitter.split(table, strata, groups=table["WAFER_ID"]), start=1):
        fold[test_idx] = k
    return fold


def score(strata, fold):
    overall = strata.value_counts(normalize=True)
    comp = pd.crosstab(fold, strata, normalize="index").reindex(columns=overall.index, fill_value=0)
    deviation = float((comp - overall).abs().to_numpy().sum())
    sizes = np.bincount(fold)[1:]
    return deviation, int(sizes.max() - sizes.min())


def main():
    table = pd.read_csv(TABLE_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    assert len(table) == 1977 and not table.duplicated(["WAFER_ID", "STAGE"]).any()
    strata = (table["STAGE"] + "|" + table["CHAMBER_PATH"]).rename("STAGE|CHAMBER_PATH")
    print(f"Step 7 표: {len(table):,}행, 고유 WAFER_ID {table['WAFER_ID'].nunique():,}개")
    print("층화 기준(STAGE|CHAMBER_PATH) 전체 행 수:", strata.value_counts().to_dict())

    results = []
    for seed in SEEDS:
        fold = assign(table, strata, seed)
        dev, size_gap = score(strata, fold)
        results.append((dev, size_gap, seed, fold))
    dev, size_gap, seed, fold = min(results, key=lambda r: (r[0], r[1], r[2]))
    print(f"\n선택한 random_state={seed} (비율 차이 합 {dev:.4f}, fold 행 수 최대-최소 차이 {size_gap}) "
          f"/ 후보 {len(SEEDS)}개 중 비율 차이 합 범위 {min(r[0] for r in results):.4f}~{max(r[0] for r in results):.4f}")

    folds = table[["WAFER_ID", "STAGE", "CHAMBER_PATH"]].assign(FOLD=fold)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT_FILE, index=False)
    print(f"저장: {OUT_FILE.relative_to(ROOT).as_posix()} (Step 7 표와 같은 행 순서, 키는 WAFER_ID + STAGE)")

    # ---- 확인 ----
    print("\n[1] fold 별 행 수와 고유 wafer 수")
    size = folds.groupby("FOLD").agg(rows=("WAFER_ID", "size"), unique_wafers=("WAFER_ID", "nunique"))
    size.loc["전체"] = [len(folds), folds["WAFER_ID"].nunique()]
    print(size.to_string())

    print("\n[2] fold 별 STAGE / CHAMBER_PATH 구성 (행 수)")
    print(pd.crosstab(folds["FOLD"], [folds["STAGE"], folds["CHAMBER_PATH"]], margins=True, margins_name="전체").to_string())
    print("\n    fold 별 STAGE 비율")
    print(pd.crosstab(folds["FOLD"], folds["STAGE"], normalize="index").round(3).to_string())
    print("\n    fold 별 CHAMBER_PATH 비율")
    print(pd.crosstab(folds["FOLD"], folds["CHAMBER_PATH"], normalize="index").round(3).to_string())
    print("    전체 비율: STAGE", folds["STAGE"].value_counts(normalize=True).round(3).to_dict(),
          "/ CHAMBER_PATH", folds["CHAMBER_PATH"].value_counts(normalize=True).round(3).to_dict())

    print("\n[3] 같은 WAFER_ID 가 두 fold 에 나뉜 경우")
    per_wafer = folds.groupby("WAFER_ID")["FOLD"].nunique()
    split = per_wafer[per_wafer > 1]
    both = folds.groupby("WAFER_ID")["STAGE"].nunique().eq(2)
    print(f"    두 fold 이상에 걸친 WAFER_ID: {len(split)}개")
    print(f"    Stage A/B 가 모두 있는 wafer {int(both.sum())}개 중 A/B 가 같은 fold 인 wafer: "
          f"{int((per_wafer[both[both].index] == 1).sum())}개")
    for k in range(1, N_FOLDS + 1):
        overlap = set(folds.loc[folds["FOLD"] == k, "WAFER_ID"]) & set(folds.loc[folds["FOLD"] != k, "WAFER_ID"])
        assert not overlap, f"fold {k} 평가용과 학습용에 같은 WAFER_ID 가 있습니다"
    print("    각 fold 를 평가용으로 뒀을 때 나머지 4개 fold 와 겹치는 WAFER_ID: 모든 fold 에서 0개")
    assert len(split) == 0 and folds["FOLD"].between(1, N_FOLDS).all()


if __name__ == "__main__":
    main()

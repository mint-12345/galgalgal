"""지속 모델이 Di Table 4와 왜 다른지 확인 (진단, CLAUDE.md 14.4). 새 파일로 저장한다.

Di 2.4절: 지속 모델은 r_t = r_{t−1}, 평가는 Monte Carlo CV (train을 무작위로 train/validation으로
나누기를 20회, 비율은 논문에 없음 → 80/20으로 둔다).

확인 1  `src/features_repro.py`의 main()이 쓴 `revealed`가 무엇이었는지 (코드로 재현해 확인)
확인 2  train·val·test가 시간상 섞여 있는지. 그룹별로 train만 줄 세웠을 때와 셋을 합쳐 줄 세웠을 때
        연속 웨이퍼 사이 간격의 중앙값 비교 (센서 시각 `t_start`만 쓰고 라벨은 쓰지 않는다)
확인 3  revealed = train + val 라벨(data/raw/labels/만)로 두고 train 웨이퍼의 지속 모델 MSE
확인 4  확인 3의 조건으로 Monte Carlo CV 20회 (무작위 80/20, SEED 고정) → 그룹별 평균 ± 표준편차
참고    직전 웨이퍼를 같은 그룹이 아니라 같은 챔버군(456 = A·B 합침)에서 찾으면 어떻게 되는지

test 라벨은 어떤 경로로도 읽지 않는다. test 웨이퍼는 라벨이 없으므로 "직전 웨이퍼" 후보에서 빠지고,
시간 간격 확인(2)에만 센서 시각으로 등장한다.

예측은 `features_repro.neighbor_features(n_lag=1)`의 `lag_MRR_01`을 그대로 쓴다. 직전 공개 웨이퍼가
없어 채워 넣은 행(`lag_missing > 0`)은 평가에서 뺀다.

실행: python src/persistence_check.py
출력: results/diagnostics/2026-09-26_김혜연_persistence_check.csv (+ 같은 이름의 .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import OUTLIER_MRR  # noqa: E402
from features_repro import neighbor_features  # noqa: E402
from load import KEY, ROOT, load_labels  # noqa: E402
from pv_check import AUTHOR  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"2026-09-26_{AUTHOR}_persistence_check.csv"
SEED, N_REP, TRAIN_FRAC = 42, 20, 0.8
GROUPS = ["A-456", "B-456", "A-123"]
DI = {"A-456": (7.55, 1.27), "B-456": (11.4, 1.36), "A-123": (5.62, 1.06)}


def wafers() -> pd.DataFrame:
    """집계표 + val 라벨(data/raw/labels/). test 라벨은 읽지 않는다."""
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    val = load_labels("val").rename(columns={"AVG_REMOVAL_RATE": "MRR_val"})
    tab = tab.merge(val, on=KEY, how="left")
    v = tab["split"] == "val"
    if not np.allclose(tab.loc[v, "MRR"], tab.loc[v, "MRR_val"], equal_nan=True):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")
    tab = tab.drop(columns="MRR_val").copy()
    tab["chamberset"] = tab["group"].str.split("-").str[1]          # 456 / 123
    n_out = int((tab["MRR"] > OUTLIER_MRR).sum())
    tab = tab[~(tab["MRR"] > OUTLIER_MRR)].reset_index(drop=True)   # 이상치 4장 (전부 train)
    print(f"웨이퍼 {len(tab):,}장 (MRR > {OUTLIER_MRR} 인 {n_out}장 제외), "
          f"val 라벨 {int((tab['split'] == 'val').sum()):,}장 병합")
    return tab


def persistence(tab: pd.DataFrame, revealed: pd.DataFrame, evaluate: np.ndarray,
                group_col: str = "group") -> dict[str, tuple[float, int]]:
    """revealed를 공개 라벨로 두고 evaluate 행의 지속 모델 MSE를 group_col 단위로 낸다."""
    d = tab if group_col == "group" else tab.assign(group=tab[group_col])
    f = neighbor_features(d, revealed, n_lag=1, k_usage=1)
    ok = evaluate & (f["lag_missing"] == 0).to_numpy() & tab["MRR"].notna().to_numpy()
    err = (tab.loc[ok, "MRR"] - f.loc[ok, "lag_MRR_01"]) ** 2
    return {g: (float(e.mean()), len(e)) for g, e in err.groupby(tab.loc[ok, "group"])}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    tab = wafers()
    is_train = (tab["split"] == "train").to_numpy()
    is_val = (tab["split"] == "val").to_numpy()
    rows = []

    # 확인 1: features_repro.main()의 revealed = train 전체 (CV 조각이 아니다)
    rev_train = tab[is_train][KEY]
    base = persistence(tab, rev_train, is_train)
    print(f"\n[1] features_repro.main()의 revealed = split == 'train' 전체 {len(rev_train):,}장 "
          f"(CV 학습 조각 아님)")
    for g in GROUPS:
        mse, n = base[g]
        rows.append({"check": "1_revealed_train_only", "group": g, "pool": "train 전체",
                     "n_revealed": len(rev_train), "n_eval": n, "mse": mse, "mse_std": None,
                     "di_mse": DI[g][0], "di_std": DI[g][1]})
        print(f"    {g}: MSE {mse:6.2f} (평가 {n:,}장)  Di {DI[g][0]}")

    # 확인 2: 시간상 섞임 (센서 시각만)
    print("\n[2] 연속 웨이퍼 사이 시간 간격 중앙값 (초)")
    for g in GROUPS:
        gd = tab[tab["group"] == g]
        gaps = {}
        for pool, sel in {"train만": ["train"], "train+val": ["train", "val"],
                          "train+val+test": ["train", "val", "test"]}.items():
            s = gd[gd["split"].isin(sel)]
            gaps[pool] = s["t_start"].sort_values().diff().median()
            rows.append({"check": "2_interleave", "group": g, "pool": pool, "n_revealed": None,
                         "n_eval": len(s), "mse": None, "mse_std": None, "gap_median_s": gaps[pool]})
        print(f"    {g}: train만 {gaps['train만']:7.0f} / train+val {gaps['train+val']:7.0f} / "
              f"셋 합쳐 {gaps['train+val+test']:7.0f}  "
              f"(train만이 셋 합친 것의 {gaps['train만'] / gaps['train+val+test']:.1f}배)")

    # 확인 3: revealed = train + val
    rev_tv = tab[is_train | is_val][KEY]
    with_val = persistence(tab, rev_tv, is_train)
    print(f"\n[3] revealed = train + val 라벨 {len(rev_tv):,}장 → train 웨이퍼 평가")
    for g in GROUPS:
        mse, n = with_val[g]
        rows.append({"check": "3_revealed_train_val", "group": g, "pool": "train+val",
                     "n_revealed": len(rev_tv), "n_eval": n, "mse": mse, "mse_std": None,
                     "di_mse": DI[g][0], "di_std": DI[g][1]})
        print(f"    {g}: MSE {mse:6.2f} (평가 {n:,}장)  Di {DI[g][0]}  "
              f"[1]에서 {base[g][0]:.2f} → {mse:.2f}")

    # 확인 4: Monte Carlo CV 20회 (무작위 80/20)
    print(f"\n[4] Monte Carlo CV {N_REP}회 (무작위 {TRAIN_FRAC:.0%}/{1 - TRAIN_FRAC:.0%}, SEED {SEED})")
    rng = np.random.default_rng(SEED)
    tr_idx = np.flatnonzero(is_train)
    fold = {p: {g: [] for g in GROUPS} for p in ("train80", "train80+val")}
    for _ in range(N_REP):
        pick = rng.permutation(tr_idx)
        fit, hold = pick[:int(len(pick) * TRAIN_FRAC)], pick[int(len(pick) * TRAIN_FRAC):]
        ev = np.zeros(len(tab), bool)
        ev[hold] = True
        for pool, extra in (("train80", np.zeros(len(tab), bool)), ("train80+val", is_val)):
            m = np.zeros(len(tab), bool)
            m[fit] = True
            for g, (mse, _) in persistence(tab, tab[m | extra][KEY], ev).items():
                fold[pool][g].append(mse)
    for pool in fold:
        for g in GROUPS:
            v = np.array(fold[pool][g])
            rows.append({"check": "4_mccv", "group": g, "pool": pool, "n_revealed": None,
                         "n_eval": N_REP, "mse": float(v.mean()), "mse_std": float(v.std(ddof=1)),
                         "di_mse": DI[g][0], "di_std": DI[g][1]})
            print(f"    {pool:12s} {g}: {v.mean():6.2f} ± {v.std(ddof=1):4.2f}   "
                  f"Di {DI[g][0]} ± {DI[g][1]}")

    # 참고: 직전 웨이퍼를 같은 챔버군에서 찾기 (456 = A·B 합침)
    cs = persistence(tab, rev_tv, is_train, group_col="chamberset")
    print("\n[참고] 직전 웨이퍼를 같은 챔버군에서 찾기 (revealed = train+val)")
    for g in GROUPS:
        mse, n = cs[g]
        rows.append({"check": "5_chamberset", "group": g, "pool": "train+val, 챔버군 단위",
                     "n_revealed": len(rev_tv), "n_eval": n, "mse": mse, "mse_std": None,
                     "di_mse": DI[g][0], "di_std": DI[g][1]})
        print(f"    {g}: MSE {mse:6.2f} (그룹 단위 {with_val[g][0]:.2f})")

    res = pd.DataFrame(rows).round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-26", f"author: {AUTHOR}", "script: src/persistence_check.py",
        "question: 지속 모델 MSE가 Di Table 4와 다른 원인",
        "input: data/processed/wafer_table.csv + data/raw/labels/CMP-validation-removalrate.csv",
        "model: 지속 모델 r_t = r_{t-1} (features_repro.neighbor_features 의 lag_MRR_01)",
        "wafers: MRR > 400 인 이상치 4장 제외. 직전 공개 웨이퍼가 없는 행은 평가에서 제외",
        f"mccv: {N_REP}회, 무작위 {TRAIN_FRAC}/{1 - TRAIN_FRAC} 분할, seed {SEED}",
        "checks: {1: revealed=train 전체, 2: split 시간 섞임(간격 중앙값), "
        "3: revealed=train+val, 4: Monte Carlo CV, 5: 챔버군 단위 직전 웨이퍼}",
        "di_table4: {A-456: 7.55±1.27, B-456: 11.4±1.36, A-123: 5.62±1.06}",
        "locked: data/locked/ 와 test 라벨은 열지 않았다 (test 웨이퍼는 확인 2에만 t_start로 등장)",
        "note: 진단 결과 (14.4)",
        "",
    ]), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""전이학습(preregistration.md 5.2·5.3절)의 라벨 공개 마스크와 전진 검증 분할 파일을 만든다.

- 대상: folds_v1.csv의 A-123 행 (이상치 제외 431장). MRR 열은 읽지 않는다.
- 시간 순서: wafer_table.csv의 t_start (웨이퍼 첫 TIMESTAMP)만 읽어 쓴다. features_v1.csv에는 시작 시각 열이 없다.
- 반올림은 사사오입 floor(x + 0.5). 정수 연산으로 계산한다.

1) 주 평가 마스크 (design = folds): folds_v1 25개 fold, r = repeat×5 + fold.
   각 fold의 A-123 학습 쪽(fold != k) N장을 t_start 순으로 정렬, 공개 장수 n = floor(비율 × N + 0.5).
   - 고정주기(fixed): j번째 공개 위치 = floor((j + r/25) × N/n), j = 0…n−1
   - 무작위(random): numpy default_rng(42 + r).choice(N, n, replace=False)
2) 전진 검증 분할: A-123 431장을 t_start 순으로 정렬, 단계 s = 0…4에서
   학습 = 앞 floor(N × (0.5 + 0.1s) + 0.5)장, 평가 = 그다음부터 floor(N × (0.6 + 0.1s) + 0.5)장까지.
   456 웨이퍼는 그 단계 eval_start(첫 평가 웨이퍼의 t_start)보다 먼저 시작한 것만 쓴다 (cutoffs 파일로 적용).
3) 전진 검증 마스크 (design = walkforward): 단계 s의 학습 구간에 같은 규칙. 고정주기 r = 5×s, 무작위 시드 42 + s.

비율: 2 / 4 / 8 / 16 / 100%.
마스크 파일에는 공개된 행만 넣는다 (pos = 시간순 정렬한 학습 쪽 안의 위치 0…N−1).

실행: python src/make_masks.py [출력 폴더]   (기본 data/processed, 출력 파일이 이미 있으면 멈춘다)
출력: reveal_masks_v1.csv, walkforward_v1.csv, walkforward_cutoffs_v1.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "data" / "processed" / "folds_v1.csv"
TABLE = ROOT / "data" / "processed" / "wafer_table.csv"
KEY = ["WAFER_ID", "STAGE"]
TARGET = "A-123"
N_TARGET = 431
RATIOS = [2, 4, 8, 16, 100]  # %
N_STEPS = 5
N_R = 25  # 고정주기 식의 r/25
SEED = 42
NAMES = ["reveal_masks_v1.csv", "walkforward_v1.csv", "walkforward_cutoffs_v1.csv"]


def half_up(num: int, den: int) -> int:
    """floor(num/den + 0.5), 정수 연산."""
    return (2 * num + den) // (2 * den)


def n_reveal(ratio: int, n: int) -> int:
    return half_up(ratio * n, 100)


def fixed_positions(n_train: int, n: int, r: int) -> list[int]:
    # floor((j + r/25) × N/n) = floor((25j + r) × N / (25n))
    pos = [((N_R * j + r) * n_train) // (N_R * n) for j in range(n)]
    if len(set(pos)) != n or pos[-1] >= n_train:
        raise SystemExit(f"고정주기 위치 오류: N={n_train}, n={n}, r={r}")
    return pos


def random_positions(n_train: int, n: int, seed: int) -> list[int]:
    return sorted(np.random.default_rng(seed).choice(n_train, size=n, replace=False).tolist())


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(FOLDS)
    f = folds[folds["group"] == TARGET].copy()
    t = pd.read_csv(TABLE, usecols=KEY + ["t_start"])
    f = f.merge(t, on=KEY, how="left", validate="m:1")
    a = f[f["repeat"] == 0][KEY + ["t_start"]].reset_index(drop=True)
    if len(a) != N_TARGET or a.duplicated(KEY).any():
        raise SystemExit(f"A-123 대상 {len(a)}장 (기대 {N_TARGET}). 멈춘다.")
    if not a["WAFER_ID"].is_unique:
        raise SystemExit("A-123 안에 같은 WAFER_ID가 두 번 이상 있다")
    if a["t_start"].isna().any() or a["t_start"].duplicated().any():
        raise SystemExit("t_start 결측 또는 동률이 있다 (시간 순서가 정해지지 않음)")
    return f, a


def mask_rows(train: pd.DataFrame, r: int, seed: int, meta: dict) -> list[dict]:
    train = train.sort_values("t_start", kind="mergesort").reset_index(drop=True)
    n_train = len(train)
    rows = []
    for ratio in RATIOS:
        n = n_reveal(ratio, n_train)
        for method, pos in (("fixed", fixed_positions(n_train, n, r)),
                            ("random", random_positions(n_train, n, seed))):
            for p in pos:
                rows.append({**meta, "method": method, "ratio": ratio, "n_train": n_train, "n_reveal": n,
                             "pos": p, "WAFER_ID": train.at[p, "WAFER_ID"], "STAGE": train.at[p, "STAGE"]})
    return rows


def build(f: pd.DataFrame, a: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = []
    for (rep, k), g in f.groupby(["repeat", "fold"]):
        r = int(rep) * 5 + int(k)
        r_all = f[f["repeat"] == rep]
        train = r_all[r_all["fold"] != k][KEY + ["t_start"]]
        rows += mask_rows(train, r, SEED + r,
                          {"design": "folds", "repeat": int(rep), "fold": int(k), "r": r, "step": pd.NA})

    a = a.sort_values("t_start", kind="mergesort").reset_index(drop=True)
    n_all = len(a)
    wf, cut = [], []
    for s in range(N_STEPS):
        tr_end = half_up(n_all * (5 + s), 10)
        ev_end = half_up(n_all * (6 + s), 10)
        role = np.where(a.index < tr_end, "train", np.where(a.index < ev_end, "eval", "unused"))
        wf.append(a[KEY].assign(step=s, role=role, t_rank=a.index))
        cut.append({"step": s, "n_train": tr_end, "n_eval": ev_end - tr_end,
                    "train_end": a.at[tr_end - 1, "t_start"], "eval_start": a.at[tr_end, "t_start"],
                    "eval_end": a.at[ev_end - 1, "t_start"]})
        rows += mask_rows(a.iloc[:tr_end], 5 * s, SEED + s,
                          {"design": "walkforward", "repeat": pd.NA, "fold": pd.NA, "r": 5 * s, "step": s})

    masks = pd.DataFrame(rows)
    for c in ["repeat", "fold", "step"]:
        masks[c] = masks[c].astype("Int64")
    masks = masks[["design", "repeat", "fold", "r", "step", "method", "ratio", "n_train", "n_reveal",
                   "pos", "WAFER_ID", "STAGE"]]
    if masks.duplicated(["design", "r", "step", "method", "ratio", "WAFER_ID", "STAGE"]).any():
        raise SystemExit("마스크에 중복 행이 있다")
    return {NAMES[0]: masks, NAMES[1]: pd.concat(wf, ignore_index=True), NAMES[2]: pd.DataFrame(cut)}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "processed"
    for name in NAMES:
        if (out / name).exists():
            raise SystemExit(f"{out / name} 가 이미 있다. 덮어쓰지 않는다.")
    f, a = load()
    tables = build(f, a)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / name, index=False, lineterminator="\n")
        print(f"{name}: {len(df)}행", flush=True)


if __name__ == "__main__":
    main()

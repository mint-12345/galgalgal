"""R 세트 재현용 F4 피처 (CLAUDE.md 6.1절 ID 1~21, 7절 규칙). 재현 확정 후 **동결**한다 (14.1).

이 파일에는 Di et al. 2017의 R 피처만 넣는다. 물리 파트에서 만든 P·V 후보나 진단 결과
(pv_check 계열)는 쓰지 않는다.

ID 1~11  MRR 시간 지연 r(t−1)…r(t−11): 같은 그룹에서 **먼저 시작한** 공개 라벨 웨이퍼의 MRR,
         가까운 것부터.
ID 12~21 사용량 최근접 이웃 10개의 MRR: 거리 공간 = z-score한 사용량 6개의 웨이퍼 평균,
         같은 그룹, **자기 자신 제외**.

7절 규칙
- F4는 이 함수에서만 계산한다.
- 항상 같은 그룹 안에서만 찾는다.
- 시간 지연은 대상 웨이퍼보다 **먼저 시작한** 웨이퍼만 (`t_start` 가 같으면 순서를 알 수 없으므로 제외).
- 사용량 이웃은 자기 자신 제외.
- test·val 웨이퍼의 이웃도 공개(train) 웨이퍼에서만 찾는다 → 호출자가 `revealed`로 넘긴다.
- 공개 라벨이 `n_lag`개보다 적으면 **그룹 과거 평균**으로 채우고 `lag_missing`에 채운 개수를 적는다 (6.2절).

`revealed` 밖의 MRR은 어떤 경로로도 읽지 않는다. 비공개 웨이퍼의 MRR은 함수 시작에서 NaN으로
지우고 그 배열만 쓴다.

사용량 6개의 "웨이퍼 평균"
  집계표(`wafer_table.csv`, 14.2에서 동결)에는 구간 평균 `mean_{U}_s1`, `mean_{U}_s2`만 있고
  웨이퍼 전체 평균 컬럼이 없다. 그래서 두 구간 평균을 **연마 시간(`dur_POLISH_s1/s2`, ID 22~23)으로
  가중 평균**해서 쓴다. 사용량 카운터는 웨이퍼 안에서 거의 일정해 어떤 가중이든 값이 비슷하다.
  s2가 비어 있는 웨이퍼(챔버 4만 있는 2장)는 s1만 쓴다.
  # TODO(결정 필요): 원문이 말하는 "웨이퍼 평균"을 행 단위 평균으로 다시 계산할지 (재집계 필요).

실행: python src/features_repro.py   (train 공개 가정으로 만들어 요약과 지속 모델 MSE를 출력)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import KEY, ROOT  # noqa: E402

USAGE6 = ["BACKING_FILM", "DRESSER", "POLISH_TABLE", "DRESSER_TABLE", "MEMBRANE", "PSHEET"]
SEGMENTS = ["s1", "s2"]
N_LAG, K_USAGE = 11, 10


def usage_means(wafers: pd.DataFrame, usage: list[str] = USAGE6) -> pd.DataFrame:
    """사용량 6개의 웨이퍼 평균 (구간 평균을 연마 시간으로 가중)."""
    dur = {s: wafers[f"dur_POLISH_{s}"].astype(float).fillna(0).clip(lower=0) for s in SEGMENTS}
    out = {}
    for u in usage:
        seg = {s: wafers[f"mean_{u}_{s}"].astype(float) for s in SEGMENTS}
        wt = {s: dur[s].where(seg[s].notna(), 0.0) for s in SEGMENTS}
        tot = wt["s1"] + wt["s2"]
        num = sum(seg[s].fillna(0.0) * wt[s] for s in SEGMENTS)
        simple = pd.concat(seg.values(), axis=1).mean(axis=1)   # 가중치가 0이면 단순 평균
        out[u] = (num / tot).where(tot > 0, simple)
    return pd.DataFrame(out, index=wafers.index)


def _zscore(u: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """pool 행으로 계산한 평균·표준편차로 전체를 z-score. 표준편차 0이면 1로 둔다."""
    mu = np.nanmean(u[pool], axis=0) if pool.any() else np.zeros(u.shape[1])
    sd = np.nanstd(u[pool], axis=0) if pool.any() else np.ones(u.shape[1])
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (u - np.where(np.isfinite(mu), mu, 0.0)) / sd


def neighbor_features(wafers: pd.DataFrame, revealed, n_lag: int = N_LAG,
                      k_usage: int = K_USAGE, window: int | None = None) -> pd.DataFrame:
    """R 세트의 F4 피처 (ID 1~21)를 만든다.

    wafers: `(WAFER_ID, STAGE)`, `group`, `t_start`, `MRR`, 그리고 사용량 구간 평균
            (`mean_{U}_s1/s2`, `dur_POLISH_s1/s2`)이 있는 웨이퍼 단위 표.
    revealed: MRR이 공개된 웨이퍼 키의 집합 (`(WAFER_ID, STAGE)` 튜플).
        - R (재현): train 1,977장 전체를 넘긴다.
        - R-c (시뮬레이션): 정책이 고른 웨이퍼 중 해당 시점 이전 것만 넘긴다.
    window: 사용량 이웃 탐색 풀을 "대상 웨이퍼 직전 window장" 안의 공개 웨이퍼로 제한한다
            (6.2절의 최근 L개 창). None이면 같은 그룹의 공개 웨이퍼 전체 (Di 원문).

    반환 (wafers와 같은 순서·인덱스):
      lag_MRR_01…{n_lag}   ID 1~11.  가까운 과거부터. 모자라면 그룹 과거 평균으로 채움
      nn_MRR_01…{k_usage}  ID 12~21. 사용량 거리가 가까운 순
      label_age_s          가장 최근 공개 웨이퍼와의 시작 시각 차이 (초)
      lag_missing          채워 넣은 지연 개수 (0이면 온전)
      nn_missing           찾지 못한 사용량 이웃 개수
    """
    need = KEY + ["group", "t_start", "MRR"]
    missing = [c for c in need if c not in wafers.columns]
    if missing:
        raise KeyError(f"wafers에 없는 컬럼: {missing}")
    if n_lag < 1 or k_usage < 1:
        raise ValueError("n_lag와 k_usage는 1 이상이어야 한다")
    if window is not None and window < 1:
        raise ValueError("window는 1 이상이거나 None이어야 한다")

    rev = {tuple(k) for k in (revealed.values if isinstance(revealed, pd.DataFrame) else revealed)}
    keys = list(map(tuple, wafers[KEY].to_numpy()))
    is_rev = np.array([k in rev for k in keys])
    mrr_all = wafers["MRR"].to_numpy(float)
    if np.isnan(mrr_all[is_rev]).any():
        raise ValueError("revealed 웨이퍼 중 MRR이 비어 있는 것이 있다")
    mrr = np.where(is_rev, mrr_all, np.nan)      # 비공개 라벨은 여기서 지운다. 이 배열만 읽는다
    t = wafers["t_start"].to_numpy(float)
    group = wafers["group"].to_numpy()
    wid = wafers["WAFER_ID"].to_numpy()
    u = _zscore(usage_means(wafers).to_numpy(float), is_rev)

    lag = np.full((len(wafers), n_lag), np.nan)
    nn = np.full((len(wafers), k_usage), np.nan)
    age = np.full(len(wafers), np.nan)
    n_filled = np.zeros(len(wafers), int)
    n_nn_missing = np.full(len(wafers), k_usage, int)

    for g in pd.unique(group):
        gi = np.flatnonzero(group == g)                      # 그룹 밖은 절대 보지 않는다
        order = gi[np.lexsort((wid[gi], t[gi]))]
        past: list[int] = []                                 # 시작 시각이 더 이른 공개 웨이퍼 (시간순)
        ptr = 0
        for pos, j in enumerate(order):
            while ptr < pos and t[order[ptr]] < t[j]:        # 같은 t_start는 넣지 않는다
                if is_rev[order[ptr]]:
                    past.append(order[ptr])
                ptr += 1
            # ID 1~11: 가까운 과거부터
            take = past[-n_lag:][::-1]
            lag[j, :len(take)] = mrr[take]
            if take:
                age[j] = t[j] - t[take[0]]
            if len(take) < n_lag:                            # 6.2절: 그룹 과거 평균으로 채움
                lag[j, len(take):] = np.mean(mrr[past]) if past else np.nan
                n_filled[j] = n_lag - len(take)
            # ID 12~21: 사용량 z-공간 최근접 이웃, 자기 자신 제외
            if window is None:
                pool = order[is_rev[order]]
            else:
                pool = order[max(0, pos - window):pos]
                pool = pool[is_rev[pool]]
            pool = pool[pool != j]
            if len(pool):
                cols = np.flatnonzero(np.isfinite(u[j]))     # 대상의 값이 없는 축은 빼고 잰다
                diff = u[np.ix_(pool, cols)] - u[j, cols]
                dist = np.sqrt((diff ** 2).sum(axis=1))
                ok = np.isfinite(dist)                       # 값이 없는 후보는 버린다
                cand, dist = pool[ok], dist[ok]
                pick = cand[np.lexsort((t[cand], dist))][:k_usage]   # 거리 같으면 이른 웨이퍼부터
                nn[j, :len(pick)] = mrr[pick]
                n_nn_missing[j] = k_usage - len(pick)

    out = pd.DataFrame(
        {**{f"lag_MRR_{i + 1:02d}": lag[:, i] for i in range(n_lag)},
         **{f"nn_MRR_{i + 1:02d}": nn[:, i] for i in range(k_usage)},
         "label_age_s": age, "lag_missing": n_filled, "nn_missing": n_nn_missing},
        index=wafers.index)
    return out


def main() -> None:
    """train을 전부 공개로 두고(R 재현 설정) 만든 뒤 요약과 지속 모델 MSE를 출력한다."""
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[tab["is_outlier"] == 0].reset_index(drop=True)
    revealed = tab[(tab["split"] == "train")][KEY]
    print(f"웨이퍼 {len(tab):,}장 (이상치 제외), 공개 라벨 {len(revealed):,}장")

    f = neighbor_features(tab, revealed)
    print(f"\n피처 {f.shape[1]}개 = 지연 {N_LAG} + 사용량 이웃 {K_USAGE} + 3")
    print(f"지연을 채운 웨이퍼: {(f['lag_missing'] > 0).sum()}장, "
          f"이웃이 모자란 웨이퍼: {(f['nn_missing'] > 0).sum()}장")
    print(f"label_age_s 중앙값: {f['label_age_s'].median():,.0f}초")

    d = pd.concat([tab[KEY + ["split", "group", "MRR"]], f], axis=1)
    tr = d[d["split"] == "train"]
    mse = ((tr["MRR"] - tr["lag_MRR_01"]) ** 2)
    # TODO(결정 필요): 아래 지속 모델 MSE가 Di Table 4의 1.4~1.8배다 (2026-09-26).
    #   확인해 본 것: WAFER_ID 순 정렬(더 나쁨 32.1/56.7), 전체-0 센서 웨이퍼 제외(변화 없음),
    #   직전 2·3·5·11장 평균(가장 좋은 2장 평균도 10.8/16.3), 시간상 가장 가까운 웨이퍼(미래 포함,
    #   8.7/13.5 — 미래를 보므로 반칙). 어느 것도 7.55/11.4에 닿지 않는다.
    #   → 정렬 기준이 아니라 Di의 평가 프로토콜(20회 CV 구성, 웨이퍼 정의) 쪽을 논문에서 확인해야 한다.
    print("\n[지속 모델] 직전 공개 웨이퍼의 MRR을 그대로 예측값으로 (CLAUDE.md 9절 Di Table 4)")
    got = mse.groupby(tr["group"]).mean()
    want = {"A-456": 7.55, "B-456": 11.4, "A-123": 5.62}
    for g in ["A-456", "B-456", "A-123"]:
        print(f"  {g}: {got[g]:7.2f}   (Di {want[g]})")
    print(f"  전체 : {mse.mean():7.2f}   (Di 8.78)")


if __name__ == "__main__":
    main()

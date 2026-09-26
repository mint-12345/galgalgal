"""'직전 웨이퍼'의 순서 기준만 바꿔 지속 모델 MSE 비교 (진단, CLAUDE.md 14.4). 새 파일로 저장한다.

`persistence_check.py`에서 revealed(공개 라벨 범위)와 CV 구성을 바꿔도 Di Table 4의 1.6배였다.
남은 변수는 "직전 웨이퍼를 무엇으로 정하는가"뿐이라 순서 기준만 바꿔 본다.

순서 기준 (revealed = train, 그룹별)
  1 t_start          웨이퍼 첫 기록 시각 (현재 기준값)
  2 main_chamber     주 연마 챔버(456 → 4, 123 → 1)에 처음 들어간 시각
  3 t_end            웨이퍼 마지막 기록 시각
  4 polish_table     start_POLISH_TABLE_w 순 (리셋 구간 안에서 증가 순, 구간은 시간순)
  5 dresser          리셋이 가장 적은 사용량 카운터 순 (DRESSER 3회, 참고로 DRESSER_TABLE 0회도 같이)

각 기준에서 지속 모델 MSE(그룹별)와 MRR의 lag-1 자기상관을 같이 낸다.
추가: revealed = train + 자기보다 앞선 val 로 두고 **val 웨이퍼**의 지속 모델 MSE
      (Di Table 5의 test 지속 모델 8.23과 비교용. val 라벨은 data/raw/labels/에서만 읽는다).

예측은 `features_repro.neighbor_features(n_lag=1)`의 `lag_MRR_01`이다. 순서 기준은 `t_start` 자리에
그 기준의 순위를 넣어 바꾼다 (같은 그룹·더 이른 순서·공개 라벨 규칙은 그대로 적용된다).
순서값이 같은 두 웨이퍼는 앞뒤를 알 수 없으므로 서로의 직전 웨이퍼가 되지 않는다.

test 라벨과 data/locked/는 어떤 경로로도 읽지 않는다.

실행: python src/persistence_order.py
출력: results/diagnostics/2026-09-26_김혜연_persistence_order.csv (+ 같은 이름의 .yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import OUTLIER_MRR, dedup  # noqa: E402
from features_repro import neighbor_features  # noqa: E402
from load import KEY, ROOT, load_labels, load_sensors  # noqa: E402
from pv_check import AUTHOR  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / f"2026-09-26_{AUTHOR}_persistence_order.csv"
GROUPS = ["A-456", "B-456", "A-123"]
DI4 = {"A-456": 7.55, "B-456": 11.4, "A-123": 5.62}      # Table 4 (train CV)
DI5_TEST = 8.23                                          # Table 5 (test 지속 모델)
MAIN_CHAMBER = {"456": 4, "123": 1}


def main_chamber_time(train: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    """주 연마 챔버에 처음 들어간 시각. 그 챔버가 없는 웨이퍼는 NaN."""
    want = train.set_index(KEY)["group"].str.split("-").str[1].map(MAIN_CHAMBER)
    r = raw.join(want.rename("main"), on=KEY)
    hit = r[r["CHAMBER"] == r["main"]]
    return hit.groupby(KEY)["TIMESTAMP"].min()


def order_key(train: pd.DataFrame, name: str, mc_time: pd.Series) -> pd.Series:
    """순서 기준별 정렬 키 (작을수록 먼저). 리셋 구간이 있는 것은 (구간 번호, 시작값)."""
    if name == "t_start":
        return train["t_start"]
    if name == "t_end":
        return train["t_end"]
    if name == "main_chamber":
        return train.set_index(KEY).index.map(mc_time).to_series(index=train.index)
    if name in ("polish_table", "dresser", "dresser_table"):
        u = {"polish_table": "POLISH_TABLE", "dresser": "DRESSER",
             "dresser_table": "DRESSER_TABLE"}[name]
        seg = train[f"nreset_{u}_w"].astype(float)       # 리셋 구간 번호 (시간순으로 증가)
        start = train[f"start_{u}_w"].astype(float)
        rank = pd.Series(np.lexsort((start.to_numpy(), seg.to_numpy())), index=train.index)
        out = pd.Series(np.empty(len(train)), index=train.index)
        out.iloc[rank.to_numpy()] = np.arange(len(train), dtype=float)
        return out
    raise ValueError(name)


def persistence(tab: pd.DataFrame, revealed: pd.DataFrame, key: pd.Series,
                evaluate: np.ndarray) -> dict[str, tuple[float, int]]:
    """key 순서로 직전 공개 웨이퍼를 찾아 evaluate 행의 MSE를 그룹별로 낸다."""
    d = tab.assign(t_start=key.to_numpy(float))
    keep = d["t_start"].notna().to_numpy()
    f = neighbor_features(d[keep].reset_index(drop=True), revealed, n_lag=1, k_usage=1)
    e = tab[keep].reset_index(drop=True)
    ok = evaluate[keep] & (f["lag_missing"] == 0).to_numpy() & e["MRR"].notna().to_numpy()
    err = (e.loc[ok, "MRR"] - f.loc[ok, "lag_MRR_01"]) ** 2
    return {g: (float(x.mean()), len(x)) for g, x in err.groupby(e.loc[ok, "group"])}


def autocorr(train: pd.DataFrame, key: pd.Series) -> dict[str, float]:
    """그 순서로 줄 세웠을 때 MRR의 lag-1 자기상관."""
    d = train.assign(_k=key.to_numpy(float)).dropna(subset=["_k"])
    return {g: float(gd.sort_values("_k")["MRR"].autocorr(1)) for g, gd in d.groupby("group")}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} 가 이미 있다 (CLAUDE.md 3절)")
    tab = pd.read_csv(ROOT / "data" / "processed" / "wafer_table.csv")
    tab = tab[~(tab["MRR"] > OUTLIER_MRR)].reset_index(drop=True)
    train = tab[tab["split"] == "train"].reset_index(drop=True)
    raw, _ = dedup(load_sensors("train"))
    raw = raw.merge(train[KEY], on=KEY)
    mc = main_chamber_time(train, raw)
    print(f"train {len(train):,}장 (이상치 제외), 주 연마 챔버 시각 {mc.notna().sum():,}장")

    rows = []
    orders = ["t_start", "main_chamber", "t_end", "polish_table", "dresser", "dresser_table"]
    ev = np.ones(len(train), bool)
    base_key = order_key(train, "t_start", mc)
    for name in orders:
        key = order_key(train, name, mc)
        mse = persistence(train, train[KEY], key, ev)
        ac = autocorr(train, key)
        print(f"\n[{name}]")
        for g in GROUPS:
            gi = train.index[train["group"] == g]
            # t_start 순서와 실제로 다른 줄을 세우는지 (자리 바뀐 웨이퍼 수)
            a = np.argsort(base_key[gi].to_numpy(float), kind="stable")
            b = np.argsort(key[gi].to_numpy(float), kind="stable")
            rows.append({"check": "order", "order": name, "group": g, "n_eval": mse[g][1],
                         "mse": mse[g][0], "lag1_autocorr": ac[g], "di_mse": DI4[g],
                         "ratio_vs_di": mse[g][0] / DI4[g], "n_pos_diff_vs_t_start": int((a != b).sum())})
            print(f"    {g}: MSE {mse[g][0]:6.2f} (Di {DI4[g]}, {mse[g][0] / DI4[g]:.2f}배), "
                  f"자기상관 {ac[g]:.3f}, 평가 {mse[g][1]:,}장, "
                  f"t_start 순서와 자리 다른 웨이퍼 {int((a != b).sum())}장")

    # val 웨이퍼의 지속 모델 (revealed = train + 자기보다 앞선 val)
    val = load_labels("val")
    if not np.allclose(tab[tab["split"] == "val"].merge(val, on=KEY)["MRR"],
                       tab[tab["split"] == "val"].merge(val, on=KEY)["AVG_REMOVAL_RATE"]):
        raise ValueError("집계표의 val MRR이 data/raw/labels/의 값과 다르다")
    tv = tab[tab["split"].isin(["train", "val"])].reset_index(drop=True)
    mse_val = persistence(tv, tv[KEY], tv["t_start"], (tv["split"] == "val").to_numpy())
    n_all = sum(n for _, n in mse_val.values())
    pooled = sum(m * n for m, n in mse_val.values()) / n_all
    print(f"\n[val 지속 모델] revealed = train + 앞선 val, 평가 = val {n_all:,}장 "
          f"(Di Table 5 test 8.23)")
    for g in GROUPS:
        m, n = mse_val[g]
        rows.append({"check": "val", "order": "t_start", "group": g, "n_eval": n, "mse": m,
                     "lag1_autocorr": None, "di_mse": DI5_TEST, "ratio_vs_di": m / DI5_TEST})
        print(f"    {g}: MSE {m:6.2f} (평가 {n:,}장)")
    rows.append({"check": "val", "order": "t_start", "group": "전체", "n_eval": n_all,
                 "mse": pooled, "lag1_autocorr": None, "di_mse": DI5_TEST,
                 "ratio_vs_di": pooled / DI5_TEST})
    print(f"    전체 : MSE {pooled:6.2f}  (Di Table 5 test 8.23, {pooled / DI5_TEST:.2f}배)")

    res = pd.DataFrame(rows).round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-09-26", f"author: {AUTHOR}", "script: src/persistence_order.py",
        "question: 직전 웨이퍼의 순서 기준을 바꾸면 지속 모델이 Di Table 4에 가까워지는가",
        "input: data/processed/wafer_table.csv + data/raw/train 센서 + "
        "data/raw/labels/CMP-validation-removalrate.csv",
        "model: 지속 모델 r_t = r_{t-1} (features_repro.neighbor_features 의 lag_MRR_01)",
        "wafers: MRR > 400 인 이상치 4장 제외. 직전 공개 웨이퍼가 없는 행은 평가에서 제외",
        "orders: {t_start: 첫 기록 시각, main_chamber: 주 연마 챔버(456→4, 123→1) 첫 진입 시각, "
        "t_end: 마지막 기록 시각, polish_table: (nreset, start) 순, dresser: 같은 방식, "
        "dresser_table: 리셋 0회 카운터}",
        "val_check: revealed = train + 자기보다 앞선 val, 평가 = val. Di Table 5 test 지속 모델 8.23",
        "di_table4: {A-456: 7.55, B-456: 11.4, A-123: 5.62}",
        "locked: data/locked/ 와 test 라벨은 열지 않았다",
        "note: 진단 결과 (14.4)",
        "",
    ]), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}, {OUT.with_suffix('.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

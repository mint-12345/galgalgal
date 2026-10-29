"""`src/features_repro.neighbor_features`가 CLAUDE.md 7절 규칙을 지키는지 확인한다.

확인 대상
1. 시간 이웃(ID 1~11)이 **같은 그룹**, **더 이른 시작 시각**의 웨이퍼만 쓰는지
2. 사용량 이웃(ID 12~21)에 **자기 자신**이 들어가지 않는지
3. `revealed`에 없는 웨이퍼의 MRR이 **절대** 쓰이지 않는지

실행: .venv/Scripts/python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features_repro import USAGE6, neighbor_features  # noqa: E402

KEY = ["WAFER_ID", "STAGE"]
ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "wafer_table.csv"


def make_wafers(rows: list[tuple], usage: list[list[float]] | None = None) -> pd.DataFrame:
    """rows = (WAFER_ID, STAGE, group, t_start, MRR). usage는 웨이퍼당 6개 값."""
    d = pd.DataFrame(rows, columns=KEY + ["group", "t_start", "MRR"])
    u = np.array(usage if usage is not None else [[i] * 6 for i in range(len(d))], float)
    for k, name in enumerate(USAGE6):
        d[f"mean_{name}_s1"] = u[:, k]
        d[f"mean_{name}_s2"] = u[:, k]
    d["dur_POLISH_s1"] = 100.0
    d["dur_POLISH_s2"] = 100.0
    return d


def lag_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c.startswith("lag_MRR_")]


def nn_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c.startswith("nn_MRR_")]


# ---------------------------------------------------------------- 1. 시간 이웃

# 두 그룹이 시간상 뒤섞여 있고, A-456에는 시작 시각이 같은 웨이퍼 두 장(3, 4)이 있다.
MIXED = [
    (1, "A", "A-456", 0.0, 50.0),
    (11, "B", "B-456", 5.0, 10.0),
    (2, "A", "A-456", 10.0, 60.0),
    (12, "B", "B-456", 15.0, 20.0),
    (3, "A", "A-456", 20.0, 70.0),
    (4, "A", "A-456", 20.0, 80.0),
    (13, "B", "B-456", 25.0, 30.0),
    (5, "A", "A-456", 30.0, 90.0),
]


def test_lag_uses_only_same_group_and_earlier_wafers():
    d = make_wafers(MIXED)
    f = neighbor_features(d, d[KEY], n_lag=2, k_usage=1)
    got = pd.concat([d[KEY + ["group", "t_start"]], f[lag_cols(f) + ["lag_missing"]]], axis=1)
    by_id = got.set_index("WAFER_ID")

    # 첫 웨이퍼는 과거가 없다 → 전부 채움(NaN), 이웃도 없다
    assert by_id.loc[1, "lag_missing"] == 2
    assert by_id.loc[1, ["lag_MRR_01", "lag_MRR_02"]].isna().all()
    # 과거가 1장뿐이면 나머지는 그룹 과거 평균(= 50)으로 채운다
    assert by_id.loc[2, "lag_MRR_01"] == 50.0 and by_id.loc[2, "lag_MRR_02"] == 50.0
    assert by_id.loc[2, "lag_missing"] == 1
    # 가까운 과거부터 채워진다
    assert list(by_id.loc[3, ["lag_MRR_01", "lag_MRR_02"]]) == [60.0, 50.0]
    # 시작 시각이 같은 웨이퍼 3은 4의 이웃이 될 수 없다 (70.0이 쓰이면 안 된다)
    assert list(by_id.loc[4, ["lag_MRR_01", "lag_MRR_02"]]) == [60.0, 50.0]
    # 같은 t_start 안에서는 WAFER_ID 순 → 5의 직전은 4, 그다음이 3
    assert list(by_id.loc[5, ["lag_MRR_01", "lag_MRR_02"]]) == [80.0, 70.0]

    # 그룹을 넘지 않는다: B-456 행에 A-456의 MRR이 하나도 없어야 한다
    a_mrr = {50.0, 60.0, 70.0, 80.0, 90.0}
    b_rows = got[got["group"] == "B-456"][lag_cols(f)].to_numpy()
    assert not set(b_rows[np.isfinite(b_rows)]) & a_mrr
    assert list(by_id.loc[12, ["lag_MRR_01", "lag_MRR_02"]]) == [10.0, 10.0]  # 과거는 11장뿐


def test_lag_never_uses_later_wafers_on_shuffled_input():
    """입력 행 순서가 달라도 결과가 같고, 미래 웨이퍼의 MRR은 쓰이지 않는다."""
    d = make_wafers(MIXED)
    f = neighbor_features(d, d[KEY], n_lag=3, k_usage=1)
    shuffled = d.iloc[[5, 0, 7, 2, 6, 1, 4, 3]].reset_index(drop=True)
    g = neighbor_features(shuffled, shuffled[KEY], n_lag=3, k_usage=1)
    g = pd.concat([shuffled[KEY], g], axis=1).set_index(KEY).loc[
        list(map(tuple, d[KEY].to_numpy()))].reset_index(drop=True)
    pd.testing.assert_frame_equal(f, g)

    # 각 행의 지연값은 모두 "더 이른 시작 시각, 같은 그룹" 웨이퍼의 MRR 집합 안에 있어야 한다
    for i, row in d.iterrows():
        past = d[(d["group"] == row["group"]) & (d["t_start"] < row["t_start"])]["MRR"]
        allowed = set(past) | ({past.mean()} if len(past) else set())
        vals = f.loc[i, lag_cols(f)].dropna()
        assert set(vals) <= allowed, f"웨이퍼 {row['WAFER_ID']}에 미래·타그룹 값이 섞였다"


# ------------------------------------------------------- 2. 사용량 이웃 (자기 자신)

def test_usage_neighbor_excludes_self():
    """자기 자신(거리 0)을 빼고, 그다음으로 가까운 웨이퍼를 고른다."""
    rows = [(1, "A", "A-456", 0.0, 50.0), (2, "A", "A-456", 10.0, 60.0),
            (3, "A", "A-456", 20.0, 70.0), (4, "A", "A-456", 30.0, 80.0)]
    usage = [[0] * 6, [1] * 6, [5] * 6, [9] * 6]     # 1번과 2번이 서로 가장 가깝다
    d = make_wafers(rows, usage)
    f = neighbor_features(d, d[KEY], n_lag=1, k_usage=3)
    own = d["MRR"].to_numpy()[:, None]
    assert not (f[nn_cols(f)].to_numpy() == own).any(), "자기 자신의 MRR이 이웃에 들어갔다"
    assert f.loc[0, "nn_MRR_01"] == 60.0            # 1번의 최근접은 2번
    assert f.loc[1, "nn_MRR_01"] == 50.0            # 2번의 최근접은 1번
    assert f.loc[3, "nn_MRR_01"] == 70.0            # 4번의 최근접은 3번
    assert (f["nn_missing"] == 0).all()             # 후보가 3장씩 있다


def test_usage_neighbor_keeps_identical_twin_but_not_self():
    """사용량이 완전히 같은 쌍둥이는 골라야 한다 (거리로 자기 자신을 지우는 게 아니다)."""
    rows = [(1, "A", "A-456", 0.0, 50.0), (2, "A", "A-456", 10.0, 60.0),
            (3, "A", "A-456", 20.0, 70.0)]
    usage = [[2] * 6, [2] * 6, [9] * 6]              # 1번과 2번은 사용량이 같다
    d = make_wafers(rows, usage)
    f = neighbor_features(d, d[KEY], n_lag=1, k_usage=1)
    assert f.loc[0, "nn_MRR_01"] == 60.0
    assert f.loc[1, "nn_MRR_01"] == 50.0


def test_usage_neighbor_stays_in_group():
    """다른 그룹에 사용량이 똑같은 웨이퍼가 있어도 쓰지 않는다."""
    rows = [(1, "A", "A-456", 0.0, 50.0), (2, "A", "A-456", 10.0, 60.0),
            (99, "B", "B-456", 5.0, 999.0)]
    usage = [[3] * 6, [8] * 6, [3] * 6]              # 99번은 1번과 사용량이 같다
    d = make_wafers(rows, usage)
    f = neighbor_features(d, d[KEY], n_lag=1, k_usage=1)
    assert f.loc[0, "nn_MRR_01"] == 60.0             # 같은 그룹의 2번
    assert 999.0 not in set(f[nn_cols(f)].to_numpy().ravel())


# ------------------------------------------------------------- 3. 비공개 라벨

HALF = [(i, "A", "A-456", float(i), 100.0 + i) for i in range(12)]


def test_unrevealed_mrr_is_never_used():
    d = make_wafers(HALF)
    revealed = d[KEY].iloc[:6]                        # 앞 6장만 공개
    hidden = d.index[6:]
    d.loc[hidden, "MRR"] = 999.0                      # 비공개 라벨에 눈에 띄는 값을 넣는다
    f = neighbor_features(d, revealed, n_lag=3, k_usage=2)
    vals = f[lag_cols(f) + nn_cols(f)].to_numpy()
    assert 999.0 not in set(vals[np.isfinite(vals)]), "비공개 웨이퍼의 MRR이 쓰였다"
    assert set(vals[np.isfinite(vals)]) <= set(d["MRR"].iloc[:6]) | {d["MRR"].iloc[:6].mean()} \
        | {d["MRR"].iloc[:1].mean(), d["MRR"].iloc[:2].mean(), d["MRR"].iloc[:3].mean(),
           d["MRR"].iloc[:4].mean(), d["MRR"].iloc[:5].mean()}


def test_output_unchanged_when_hidden_labels_change():
    """비공개 라벨을 통째로 바꿔도 결과가 한 글자도 달라지지 않아야 한다."""
    d = make_wafers(HALF)
    revealed = d[KEY].iloc[:6]
    base = neighbor_features(d, revealed, n_lag=3, k_usage=2)
    for replacement in (np.nan, -1e9, 0.0):
        alt = d.copy()
        alt.loc[alt.index[6:], "MRR"] = replacement
        pd.testing.assert_frame_equal(base, neighbor_features(alt, revealed, n_lag=3, k_usage=2))


def test_revealed_label_must_exist():
    d = make_wafers(HALF)
    d.loc[0, "MRR"] = np.nan
    with pytest.raises(ValueError):
        neighbor_features(d, d[KEY], n_lag=3, k_usage=2)


# ------------------------------------------------- 실제 집계표로 같은 규칙 재확인

@pytest.mark.skipif(not TABLE.exists(), reason="data/processed/wafer_table.csv 가 없다")
def test_real_table_respects_all_three_rules():
    tab = pd.read_csv(TABLE)
    tab = tab[tab["is_outlier"] == 0].reset_index(drop=True)
    revealed = tab[tab["split"] == "train"][KEY]
    f = neighbor_features(tab, revealed)
    d = pd.concat([tab[KEY + ["split", "group", "t_start", "MRR"]], f], axis=1)

    train_mrr = set(tab[tab["split"] == "train"]["MRR"])
    vals = d[lag_cols(f) + nn_cols(f)].to_numpy()
    seen = set(vals[np.isfinite(vals)])
    # val·test 라벨은 공개되지 않았다 → 그 MRR이 나타나면 안 된다 (train 값과 겹치는 경우 제외)
    hidden_mrr = set(tab[tab["split"] == "val"]["MRR"].dropna()) - train_mrr
    assert not seen & hidden_mrr

    # 지연 1은 항상 같은 그룹에서 더 이른 시작 시각의 공개 웨이퍼 MRR이다
    tr = tab[tab["split"] == "train"]
    for group, gd in d.groupby("group"):
        pool = tr[tr["group"] == group]
        for _, row in gd.sample(20, random_state=42).iterrows():
            if row["lag_missing"] > 0:
                continue
            earlier = pool[pool["t_start"] < row["t_start"]]["MRR"]
            assert row["lag_MRR_01"] in set(earlier)

    # 자기 자신은 사용량 이웃에 들어갈 수 없다 → 자기 MRR과 같은 값이 나오더라도
    # 그것을 만든 웨이퍼가 자기 자신인지 확인할 수 없으므로, 여기서는 그룹 규칙만 본다
    for group, gd in d.groupby("group"):
        allowed = set(tr[tr["group"] == group]["MRR"])
        v = gd[nn_cols(f)].to_numpy()
        assert set(v[np.isfinite(v)]) <= allowed

"""정식 SR과 시드 시험이 함께 쓰는 설정 (preregistration.md 4.3, 4.8절).

- 입력: 그룹별 후보 10개(sr_candidates.py 결과), features_v1.csv(FV1), wafer_table.csv의 MRR.
  train·val, is_outlier = 0만 쓴다. test 라벨과 data/locked/는 읽지 않는다.
- 척도: 변수마다 그룹 train+val 전체의 절댓값 중앙값으로 나눈다 (부호 유지). MRR은 원척도.
- 웨이퍼: 후보 10개에 값이 모두 있는 웨이퍼만 쓴다.
"""
import re
from pathlib import Path

import pandas as pd
from pysr import PySRRegressor

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CANDIDATES = ROOT / "results" / "diagnostics" / "2026-10-30_박종화_sr_candidates.csv"
UNARY = ["exp", "log", "sqrt"]
MAX_ABS_EXPONENT = 3.0
POW_TOKEN = re.compile(r"\^\s*\(?\s*([^\s()]+)")


def load_xy(group):
    """(X, y, 후보 이름, 전체 웨이퍼 수)를 돌려준다. X는 척도를 적용한 값."""
    c = pd.read_csv(CANDIDATES)
    names = c[(c["group"] == group) & (c["selected"] == 1)].sort_values("rank")["feature"].tolist()
    assert len(names) == 10, f"{group} 후보가 {len(names)}개다"

    feats = pd.read_csv(PROC / "features_v1.csv")
    table = pd.read_csv(PROC / "wafer_table.csv",
                        usecols=["WAFER_ID", "STAGE", "split", "MRR", "is_outlier"])
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    data = feats[feats["group"] == group].merge(table, on=["WAFER_ID", "STAGE"], validate="one_to_one")

    scale = data[names].abs().median()
    assert (scale >= 1e-9).all(), "절댓값 중앙값이 1e-9 미만인 후보가 있다"
    keep = data[names].notna().all(axis=1)
    return (data.loc[keep, names] / scale).to_numpy(), data.loc[keep, "MRR"].to_numpy(), names, len(data)


def make_model(seed, niterations):
    nested = {a: {b: 0 for b in UNARY} for a in UNARY}
    return PySRRegressor(
        niterations=niterations,
        populations=31,
        population_size=27,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=UNARY,
        constraints={"^": (-1, 1)},
        nested_constraints=nested,
        complexity_of_operators={"exp": 3, "log": 3, "sqrt": 2, "^": 2},
        maxsize=20,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )


def exponent_ok(equation):
    """모든 `^`의 지수가 상수이고 −3~3 안에 있으면 True (4.8 5번).

    constraints로 지수 크기가 1이라 오른쪽은 상수나 변수 하나다.
    """
    for token in POW_TOKEN.findall(equation):
        try:
            value = float(token)
        except ValueError:
            return False
        if abs(value) > MAX_ABS_EXPONENT:
            return False
    return True


def filter_pareto(equations):
    """(통과한 식 표, 뺀 개수)를 돌려준다."""
    mask = equations["equation"].map(exponent_ok)
    return equations[mask], int((~mask).sum())

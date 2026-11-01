"""`data/processed/source_preds_v1.csv`(전이학습 소스 예측값)의 구조와 누수 조건을 확인한다.

1. 소스 2 × 입력 2 × (folds 1 + walkforward 5) 조합마다 A-123 431장, NaN·중복 없음
2. walkforward의 role이 walkforward_v1.csv와 같다
3. 각 단계 소스 학습 장수 = t_start < eval_start인 456 웨이퍼 수 (미래 456 미사용)

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv_common as cv  # noqa: E402

PREDS = cv.PROCESSED / "source_preds_v1.csv"
pytestmark = pytest.mark.skipif(not (PREDS.exists() and cv.TABLE.exists()), reason="파일 없음")


@pytest.fixture(scope="module")
def sp():
    return pd.read_csv(PREDS)


def test_every_combination_covers_a123(sp):
    n = sp.groupby(["design", "step", "source", "featureset"]).size()
    assert len(n) == 24 and (n == 431).all()
    assert not sp["pred"].isna().any()
    assert not sp.duplicated(["design", "step", "source", "featureset", "WAFER_ID", "STAGE"]).any()
    assert set(sp["max_depth"]) == {4}


def test_walkforward_roles_match_split_file(sp):
    walk = pd.read_csv(cv.PROCESSED / "walkforward_v1.csv")
    w = sp[sp["design"] == "walkforward"].merge(walk, on=cv.KEY + ["step"], suffixes=("", "_ref"))
    assert len(w) == (sp["design"] == "walkforward").sum()
    assert (w["role"] == w["role_ref"]).all()


def test_walkforward_source_counts_exclude_future(sp):
    data = cv.load_dataset()
    cuts = pd.read_csv(cv.PROCESSED / "walkforward_cutoffs_v1.csv").set_index("step")
    groups = {"S1": ["A-456"], "S2": ["A-456", "B-456"]}
    n = sp.groupby(["design", "step", "source"])["n_source"].unique()
    for (design, step, source), vals in n.items():
        assert len(vals) == 1
        g = data["group"].isin(groups[source])
        if design == "folds":
            assert vals[0] == g.sum()
        else:
            assert vals[0] == (g & (data["t_start"] < cuts.loc[step, "eval_start"])).sum()

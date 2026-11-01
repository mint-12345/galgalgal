"""A-123 분리 대 통합 결과 파일(`results/*_전준혁_a123_split_vs_pooled*`)의 누수·구성 조건을 다시 확인한다.

1. 주 평가: 25 fold, 분리 n_train + n_eval = 431 (A-123만 학습), 통합 n_train = 그 fold 학습 쪽 전체, 두 모델 n_eval 같음
2. 전진 검증: A-123 학습·평가 장수 = cutoffs 파일, 통합의 456 장수 = t_start < eval_start인 456 수 (미래 456 미사용)
3. 검정표의 비율은 0.25 근처(A-123 평가/학습)이고 df = 24

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv_common as cv  # noqa: E402

RES = cv.ROOT / "results"
MAIN = sorted(p for p in RES.glob("*_전준혁_a123_split_vs_pooled*.csv")
              if not p.stem.endswith(("_tests", "_walkforward")))
pytestmark = pytest.mark.skipif(not MAIN or not cv.TABLE.exists(), reason="결과 파일 없음")


@pytest.fixture(scope="module")
def data():
    return cv.load_dataset()


@pytest.mark.parametrize("path", MAIN, ids=lambda p: p.stem)
def test_main_eval_structure(path, data):
    r = pd.read_csv(path)
    sep = r[r["model"] == "separate"].set_index("fold").sort_index()
    poo = r[r["model"] == "pooled"].set_index("fold").sort_index()
    assert list(sep.index) == list(range(25)) and list(poo.index) == list(range(25))
    assert (sep["n_train"] + sep["n_eval"] == 431).all()
    assert (poo["n_train"] + 480 >= cv.N_ROWS - 2).all()           # 통합 = fold 학습 쪽 전체 (1,920~1,922)
    assert (sep["n_eval"] == poo["n_eval"]).all()
    assert (sep.groupby("repeat")["n_eval"].sum() == 431).all()
    t = pd.read_csv(path.with_name(path.stem + "_tests.csv"))
    assert (t["df"] == 24).all() and t["ratio"].between(0.24, 0.26).all()


@pytest.mark.parametrize("path", MAIN, ids=lambda p: p.stem)
def test_walkforward_no_future(path, data):
    wf = pd.read_csv(path.with_name(path.stem + "_walkforward.csv"))
    cuts = pd.read_csv(cv.PROCESSED / "walkforward_cutoffs_v1.csv").set_index("step")
    is456 = data["group"] != "A-123"
    for _, row in wf.iterrows():
        c = cuts.loc[row["step"]]
        assert row["n_train_a123"] == c["n_train"] and row["n_eval"] == c["n_eval"]
        expect_456 = 0 if row["model"] == "separate" else int((is456 & (data["t_start"] < c["eval_start"])).sum())
        assert row["n_train_456"] == expect_456

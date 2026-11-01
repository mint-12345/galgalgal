"""`src/since_reset_check.py`의 리셋 누락 진단(라벨 미사용) 점검.

1. 가짜 표: 20% 미만 감소 뒤 낮은 값이 이어지면 restart, 다음 웨이퍼가 원래 수준이면 bounce로 분류하고,
   restart부터 다음 감지 리셋 직전까지를 영향 웨이퍼로 잡는다.
2. 실제 집계표: 음수 since는 모두 영향 웨이퍼에 들어 있고, POLISH_TABLE 외 4종은 미감지 감소가 없다.

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv_common as cv  # noqa: E402
import since_reset_check as sr  # noqa: E402


def _toy() -> pd.DataFrame:
    # 스테이지 A 한 줄. 4번에서 감지 리셋(100 → 0), 7번에서 놓친 리셋(40 → 30, 이후 계속 낮음),
    # 10번 bounce(60 → 55 → 70), 12번에서 감지 리셋.
    start = [10, 20, 30, 100, 0, 10, 40, 30, 35, 60, 55, 70, 0, 10]
    reset = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    n = len(start)
    t = pd.DataFrame({"WAFER_ID": range(n), "STAGE": "A", "split": "train", "group": "A-456",
                      "t_start": np.arange(n, dtype=float)})
    s = pd.Series(start, dtype=float)
    base = s.where(pd.Series(reset) == 1).ffill()
    for u in sr.U5:
        t[f"start_{u}_w"] = s if u == "POLISH_TABLE" else np.arange(n, dtype=float)
        t[f"reset_{u}_w"] = reset if u == "POLISH_TABLE" else 0
        t[f"since_{u}_w"] = (s - base) if u == "POLISH_TABLE" else np.nan
    return t


def test_audit_toy_classifies_restart_and_bounce():
    summ, wafers = sr.audit(_toy())
    pt = summ.set_index("variable").loc["since_POLISH_TABLE_w"]
    assert (pt.n_reset_detected, pt.n_unflagged_drop, pt.n_restart, pt.n_bounce) == (2, 2, 1, 1)
    got = wafers.set_index("WAFER_ID")["reason"].to_dict()
    assert {k for k, v in got.items() if v.startswith("missed_reset")} == {7, 8, 9, 10, 11}   # 12번 감지 리셋 직전까지
    assert pt.n_affected == 5                         # bounce(10번)는 이미 restart 구간에 포함
    assert set(wafers["variable"]) == {"since_POLISH_TABLE_w"}


@pytest.mark.skipif(not cv.TABLE.exists(), reason="집계표 없음")
def test_audit_real_table_catches_all_negatives():
    cols = cv.KEY + ["split", "group", "t_start"] + [f"{p}_{u}_w" for u in sr.U5 for p in ("start", "reset", "since")]
    tab = pd.read_csv(cv.TABLE, usecols=cols)
    summ, wafers = sr.audit(tab)
    neg = tab[tab["since_POLISH_TABLE_w"] < 0]
    caught = set(map(tuple, wafers.loc[wafers["negative"] == 1, cv.KEY].to_numpy()))
    assert caught == set(map(tuple, neg[cv.KEY].to_numpy()))
    other = summ.set_index("variable").drop(index="since_POLISH_TABLE_w")
    assert (other[["n_unflagged_drop", "n_negative", "n_affected"]] == 0).all().all()

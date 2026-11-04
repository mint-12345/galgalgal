"""`src/sr_cross_group.py`의 정의역 점검과 fold 평가 점검 (가짜 데이터, 라벨·SR 결과 파일 불필요).

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sr_cross_group as cg  # noqa: E402
import sr_select as ss  # noqa: E402


def test_domain_issues_counts_each_problem():
    eq = ss.parse_equation("sqrt(x) + (y / z) + ((x - 1.0) ^ 1.5) + log(z)")
    X = {"x": np.array([4.0, -1.0, 0.5, 2.0]), "y": np.ones(4), "z": np.array([1.0, 1.0, 0.0, 2.0])}
    got = cg.domain_issues(eq.node, eq.theta, X, 4)
    assert got["sqrt_neg"] == 1            # x = −1
    assert got["div_zero"] == 1            # z = 0
    assert got["pow_neg_base"] == 2        # x − 1 < 0 이고 지수 1.5 (x = −1, 0.5)
    assert got["log_nonpos"] == 1          # z = 0
    assert got["nonfinite"] == 2           # 행 0, 3만 계산됨
    assert np.isnan(ss.predict(eq.node, eq.theta, X, 4)).sum() == got["nonfinite"]


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = {"a": rng.uniform(1, 2, n), "b": rng.uniform(1, 2, n)}
    y = 3.0 * X["a"] / X["b"] + 10.0 + rng.normal(0, 0.1, n)
    masks = {r: (np.arange(n) % 4) == r for r in range(4)}
    return X, y, masks


def test_fixed_uses_given_constants_and_refit_recovers():
    X, y, masks = _toy()
    eq = ss.parse_equation("((a * 1.0) / b) + 0.0")        # 형태는 같고 상수만 틀린 식
    fixed = cg.evaluate("fixed", eq, X, y, masks)
    for row in fixed:
        ev = masks[row["fold"]]
        p = ss.predict(eq.node, eq.theta, ss._subset(X, ev), int(ev.sum()))
        assert row["rmse"] == pytest.approx(np.sqrt(np.mean((p - y[ev]) ** 2)))
        assert row["n_eval_nan"] == 0
    refit = cg.evaluate("refit", eq, X, y, masks)
    assert max(r["rmse"] for r in refit) < 0.2 < min(r["rmse"] for r in fixed)
    shifted = cg.evaluate("fixed+b", eq, X, y, masks)       # 절편만 맞추면 고정보다 낫다
    assert np.mean([r["rmse"] for r in shifted]) < np.mean([r["rmse"] for r in fixed])

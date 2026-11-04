"""공동 SR(`src/sr_joint.py`, `src/sr_joint_search.py`) 점검. 가짜 데이터만 쓰고 PySR·Julia는 부르지 않는다.

1. 공동 재적합이 공통 f 상수와 테이블별 a·b를 복원한다
2. fold 평가에서 평가 쪽 MRR을 바꿔도 예측(따라서 계수)이 바뀌지 않는다 → 평가 라벨이 계수 추정에 안 쓰임
3. 템플릿 식 문자열에서 f를 꺼내 변수 이름으로 바꾼다 (g는 f에 없다)
4. 판정 규칙이 계획 9절대로 나뉜다

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sr_joint as sj  # noqa: E402
import sr_joint_search as sjs  # noqa: E402
import sr_select as ss  # noqa: E402


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = {"x1": rng.uniform(1, 2, n), "x2": rng.uniform(1, 2, n)}
    gi = np.arange(n) % 2
    f = X["x1"] / (X["x2"] + 0.5)
    y = np.array([2.0, -1.0])[gi] * f + np.array([10.0, 50.0])[gi] + rng.normal(0, 0.01, n)
    return X, y, gi


def test_joint_refit_recovers_shared_and_group_constants():
    X, y, gi = _toy()
    eq = ss.parse_equation("x1 / (x2 + 3.0)")
    t, a, b, _ = sj.joint_refit(eq.node, eq.theta, X, y, gi)
    assert t[0] == pytest.approx(0.5, abs=0.01)
    assert a == pytest.approx([2.0, -1.0], abs=0.02)
    assert b == pytest.approx([10.0, 50.0], abs=0.05)


def test_eval_labels_not_used_for_fitting():
    X, y, gi = _toy()
    eq = ss.parse_equation("x1 / (x2 + 3.0)")
    masks = {r: (np.arange(len(y)) % 5) == r for r in range(5)}
    base = sj.eval_joint(eq.node, eq.theta, X, y, gi, masks)
    y2 = y.copy()
    y2[masks[0]] += 1000.0                       # fold 0 평가 쪽 라벨만 바꿈
    alt = sj.eval_joint(eq.node, eq.theta, X, y2, gi, masks)
    for r0, r1 in zip(base, alt):
        if r0["fold"] == 0:
            assert r0["a"] == r1["a"] and r0["b"] == r1["b"]          # 계수 같음
            assert r1["rmse"] > 100                                    # 채점에만 반영
        else:
            assert r0["a"] != r1["a"] or r0["b"] != r1["b"]          # 다른 fold는 학습 쪽에 들어가 바뀜


def test_split_template_maps_args_and_excludes_g():
    names = ["auc_MAIN_OUTER_s1", "dur_POLISH_s1", "frac_hi"]
    text = "f = #1 / (#3 + exp(#2 * -0.5)); a = [3.0, -2.0]; b = [10.0, 40.0]"
    p = sjs.split_template(text, names)
    assert p["f"] == "auc_MAIN_OUTER_s1 / (frac_hi + exp(dur_POLISH_s1 * -0.5))"
    assert p["args_used"] == [1, 2, 3] and "g" not in p["f"]
    assert ss.variables(ss.parse_equation(p["f"]).node) == set(names)


@pytest.mark.parametrize("p,gap,expect", [
    ((0.01, 0.01), (0.0, 0.0), "공통 형태 있음"),
    ((0.01, 0.01), (0.5, 0.0), "부분적"),
    ((0.01, 0.30), (0.0, 0.0), "부분적"),
    ((0.30, 0.30), (0.0, 0.0), "공통 형태 없음"),
])
def test_judge_rule(p, gap, expect):
    tests = pd.DataFrame({"group": sj.GROUPS, "mean_diff": [-1.0, -1.0], "p": list(p)})
    rmse_c = {"A-456": 4.0, "B-456": 4.0}
    rmse_j = {g: rmse_c[g] + gp for g, gp in zip(sj.GROUPS, gap)}
    out = sj.judge(tests, rmse_j, rmse_c, {"A-456": 0.04, "B-456": 0.04})
    assert out["verdict"].startswith(expect)

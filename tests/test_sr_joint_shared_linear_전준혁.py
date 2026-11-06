"""공유 선형 함수 모형(`src/sr_joint_shared_linear.py`) 점검. 가짜 데이터만 쓴다.

1. 교대 최소제곱이 알려진 rank-1 모형(w, a_g, b_g)을 식별성 규약(‖w‖=1, a_A>0)대로 되찾는다
2. 무작위 시작과 SVD 시작이 같은 SSE로 수렴한다
3. 평가 쪽 MRR을 바꿔도 예측이 그대로다 (채점에만 쓰임)
4. 입력 변수의 아핀 변환에 예측이 변하지 않는다
5. 분류 규칙이 계획 3.5절대로 동작한다

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sr_joint_shared_linear as sl  # noqa: E402


def _toy(n=800, p=4, seed=1, noise=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p))
    gi = np.arange(n) % 2
    w = np.array([0.6, -0.5, 0.4, 0.2][:p])
    w = w / np.linalg.norm(w)
    a, b = np.array([3.0, 4.5]), np.array([70.0, 85.0])
    y = a[gi] * (Z @ w) + b[gi] + rng.normal(0, noise, n)
    return Z, y, gi, w, a, b


def test_recovers_rank1_truth_with_convention():
    Z, y, gi, w, a, b = _toy()
    fit = sl.fit_shared_linear(Z, y, gi, sl.svd_start(Z, y, gi))
    assert np.linalg.norm(fit["w"]) == pytest.approx(1.0)
    assert fit["a"][0] > 0
    assert np.allclose(fit["w"], w, atol=0.02)
    assert np.allclose(fit["a"], a, atol=0.1) and np.allclose(fit["b"], b, atol=0.1)
    assert fit["grad_rel"] < 1e-6


def test_sign_convention_flips_when_truth_negative():
    Z, y, gi, w, a, b = _toy()
    y2 = -(y - b[gi]) + b[gi]                       # a를 음수로 → 규약상 w가 뒤집혀야 함
    fit = sl.fit_shared_linear(Z, y2, gi, sl.svd_start(Z, y2, gi))
    assert fit["a"][0] > 0 and np.allclose(fit["w"], -w, atol=0.02)


def test_random_starts_reach_same_sse():
    Z, y, gi, *_ = _toy(noise=1.0)
    main = sl.fit_shared_linear(Z, y, gi, sl.svd_start(Z, y, gi))
    rng = np.random.default_rng(0)
    for _ in range(10):
        r = sl.fit_shared_linear(Z, y, gi, rng.standard_normal(Z.shape[1]))
        assert abs(r["sse"] - main["sse"]) / main["sse"] < 1e-8


def test_eval_labels_not_used_and_affine_invariance():
    Z, y, gi, *_ = _toy()
    M = Z * np.array([3.0, 0.5, 10.0, 2.0]) + np.array([5.0, -1.0, 100.0, 0.3])
    ev = np.zeros(len(y), bool)
    ev[::5] = True
    tr = ~ev
    p1, *_ = sl.shared_fold(M, y, gi, tr, ev, np.random.default_rng(0))
    y2 = y.copy()
    y2[ev] += 1000.0
    p2, *_ = sl.shared_fold(M, y2, gi, tr, ev, np.random.default_rng(0))
    assert np.allclose(p1, p2)
    p3, *_ = sl.shared_fold(Z, y, gi, tr, ev, np.random.default_rng(0))      # 척도·위치가 다른 같은 정보
    assert np.allclose(p1, p3, atol=1e-6)


def test_rank1_matches_lr_when_tables_share_direction():
    """참 β_B가 β_A의 배수이면 공유 선형과 테이블별 LR의 학습 SSE가 거의 같다."""
    Z, y, gi, *_ = _toy(noise=0.5)
    fit = sl.fit_shared_linear(Z, y, gi, sl.svd_start(Z, y, gi))
    from sklearn.linear_model import LinearRegression
    sse_lr = sum(float(np.sum((y[gi == g] - LinearRegression().fit(Z[gi == g], y[gi == g]).predict(Z[gi == g])) ** 2))
                 for g in (0, 1))
    assert fit["sse"] >= sse_lr - 1e-6                                 # 제약 모형이 더 작을 수 없음
    assert fit["sse"] / sse_lr < 1.01


@pytest.mark.parametrize("diff,wins,lo,hi,expect", [
    (-0.1, 4, -0.2, -0.01, "A 우세 (구간 0 미포함)"),
    (-0.1, 4, -0.2, 0.05, "A 우세 (구간 0 포함)"),
    (-0.1, 3, -0.2, -0.01, "불확실"),
    (0.1, 1, 0.02, 0.2, "B 우세 (구간 0 미포함)"),
    (0.1, 2, 0.02, 0.2, "불확실"),
])
def test_classify(diff, wins, lo, hi, expect):
    assert sl.classify(diff, wins, 5, lo, hi, ("A 우세", "B 우세", "불확실")) == expect

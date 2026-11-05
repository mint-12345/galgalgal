"""공동 SR 바깥 5-fold 검증(`src/sr_joint_outer5.py`) 점검. 가짜 데이터만 쓰고 PySR·Julia는 부르지 않는다.

1. 안쪽 fold는 바깥 fold를 빼고, 캐시 경로는 fold·시드마다 다르다
2. 바깥 평가 쪽 MRR을 바꿔도 선택 식·계수·예측이 그대로다 (채점에만 쓰임)
3. 사전 정의 구조 S* 판정이 계획 4절대로 동작한다
4. 같은 변수 선형회귀는 학습 쪽만으로 적합된다

실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sr_joint_outer5 as so  # noqa: E402
import sr_select as ss  # noqa: E402

FULL_F = ("(log(P1_on) - auc_RETAINER_s1) * log(auc_SLURRY_C_s1 * ((auc_PCHAMBER_s1 + 0.0898) * auc_MAIN_OUTER_s1))")
FOLD0_F = ("(auc_RETAINER_s1 + log(std_POLISH_TABLE_s1)) * log(auc_SLURRY_C_s1 * (auc_PCHAMBER_s1 * auc_MAIN_OUTER_s1))")


def test_inner_folds_and_cache_paths():
    for k in so.FOLDS:
        assert k not in so.inner_of(k) and len(so.inner_of(k)) == 4
    paths = {so.seed_path(k, s) for k in so.FOLDS for s in so.SEEDS}
    assert len(paths) == len(so.FOLDS) * len(so.SEEDS)
    assert so.seed_path(1, 42).parent == so.OUT_CACHE


def _toy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = {"x1": rng.uniform(1, 2, n), "x2": rng.uniform(1, 2, n), "x3": rng.uniform(1, 2, n)}
    gi = np.arange(n) % 2
    f = X["x1"] / (X["x2"] + 0.5)
    y = np.array([2.0, -1.0])[gi] * f + np.array([10.0, 50.0])[gi] + rng.normal(0, 0.05, n)
    fold = rng.permutation(np.arange(n) % 5)
    masks = {r: fold == r for r in range(5)}
    cands = pd.DataFrame({"equation": ["x1", "x1 / (x2 + 3.0)", "x1 / (x2 + x3 * 0.1 + 3.0)"], "complexity": [1, 5, 9]})
    return X, y, gi, masks, cands


@pytest.mark.parametrize("k", [1, 3])
def test_outer_eval_labels_only_used_for_scoring(k):
    X, y, gi, masks, cands = _toy()
    names = ["x1", "x2", "x3"]
    sel, scores, pj, pl = so.run_fold(k, cands, X, y, gi, masks, names)
    y2 = y.copy()
    y2[masks[k]] += 500.0                                   # 바깥 평가 쪽 라벨만 크게 바꿈
    sel2, scores2, pj2, pl2 = so.run_fold(k, cands, X, y2, gi, masks, names)
    assert sel["equation_search"] == sel2["equation_search"]
    assert np.allclose(sel["theta"], sel2["theta"]) and np.allclose(sel["a"], sel2["a"]) and np.allclose(sel["b"], sel2["b"])
    pd.testing.assert_frame_equal(scores, scores2)
    assert np.allclose(pj, pj2) and np.allclose(pl, pl2)
    assert sel["lr"] == sel2["lr"]
    # 다른 fold 라벨을 바꾸면 계수가 바뀐다 (학습 쪽에 들어가므로)
    y3 = y.copy()
    y3[masks[(k + 1) % 5]] += 5.0
    sel3, *_ = so.run_fold(k, cands, X, y3, gi, masks, names)
    assert not np.allclose(sel["b"], sel3["b"])


def test_toy_picks_true_form_and_recovers_constants():
    X, y, gi, masks, cands = _toy()
    sel, *_ = so.run_fold(0, cands, X, y, gi, masks, ["x1", "x2", "x3"])
    assert sel["equation_search"] == "x1 / (x2 + 3.0)"
    assert sel["theta"][0] == pytest.approx(0.5, abs=0.05)
    assert sel["a"] == pytest.approx([2.0, -1.0], abs=0.1)
    assert sel["variables"] == ["x1", "x2"]


@pytest.mark.parametrize("text,expect", [
    (FULL_F, True),
    (FOLD0_F, True),
    ("81.6 - ((auc_RETAINER_s1 + log(std_POLISH_TABLE_s1)) * log(auc_MAIN_OUTER_s1 * auc_SLURRY_C_s1)) * 73.3", True),
    ("log(auc_SLURRY_C_s1 * auc_MAIN_OUTER_s1) * auc_RETAINER_s1", True),
    ("(log(P1_on) - auc_RETAINER_s1) * log(auc_SLURRY_C_s1 * dur_POLISH_s1)", False),      # 적분 1개뿐
    ("(log(P1_on) - frac_hi) * log(auc_SLURRY_C_s1 * auc_MAIN_OUTER_s1)", False),           # 리테이너 없음
    ("auc_RETAINER_s1 + log(auc_SLURRY_C_s1 * auc_MAIN_OUTER_s1)", False),                  # 곱이 아님
    ("auc_PCHAMBER_s1 * dur_POLISH_s1", False),
])
def test_skeleton_star(text, expect):
    assert so.is_skeleton_star(ss.parse_equation(text).node) is expect


def test_lr_uses_train_side_only():
    X, y, gi, masks, _ = _toy()
    M = np.column_stack([X["x1"], X["x2"]])
    ev = masks[2]
    p, _ = so.lr_fit_predict(M, y, gi, ~ev, ev)
    y2 = y.copy()
    y2[ev] = 0.0
    p2, _ = so.lr_fit_predict(M, y2, gi, ~ev, ev)
    assert np.allclose(p, p2)

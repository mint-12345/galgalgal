"""SR 식 선택·항 정의를 가짜 데이터로 확인한다. MRR·SR 실행 결과를 읽지 않는다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sr_run import exponent_ok, filter_pareto  # noqa: E402
from sr_select import (  # noqa: E402
    HEAD_SINCE,
    consistent, const_ids, exponent_ids, fold_rmse, main_terms, parse_equation, predict, refit, remove_term,
    score_candidates, select_1se, select_multiplier, skeleton, split_terms, stability, term_signs, variables,
)


def fake_data(n=400, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = {"a": rng.uniform(0.5, 2.0, n), "b": rng.uniform(0.5, 2.0, n), "z": rng.uniform(0.5, 2.0, n)}
    y = 10 * X["a"] ** 0.37 + 2 * X["b"] + 5 + rng.normal(0, noise, n)
    return X, y


def fake_masks(n, seed=1):
    """5조각 × 5회. 실제 folds_v1과 같은 25개 평가 마스크."""
    rng = np.random.default_rng(seed)
    masks = {}
    for repeat in range(5):
        order = rng.permutation(n)
        for fold, chunk in enumerate(np.array_split(order, 5)):
            m = np.zeros(n, bool)
            m[chunk] = True
            masks[repeat * 5 + fold] = m
    return masks


def test_parse_keeps_form_and_negative_constant():
    eq = parse_equation("(a ^ 0.37) * -1.5 + exp(b)")
    assert sorted(eq.theta.tolist()) == [-1.5, 0.37]
    assert variables(eq.node) == {"a", "b"}
    assert len(exponent_ids(eq.node)) == 1
    assert skeleton(eq.node) == "(((a pow c) mul c) add exp(b))"


def test_predict_matches_python_and_marks_invalid_as_nan():
    X = {"a": np.array([4.0, -4.0])}
    eq = parse_equation("sqrt(a) + 1.0")
    out = predict(eq.node, eq.theta, X, 2)
    assert out[0] == pytest.approx(3.0)
    assert np.isnan(out[1])


def test_refit_recovers_constants_and_keeps_exponent_in_range():
    X, y = fake_data()
    eq = parse_equation("((a ^ 1.0) * 5.0) + (b * 1.0) + 1.0")
    theta = refit(eq.node, eq.theta, X, y)
    # 지수·계수·절편은 서로 보완되므로 값 대신 예측 오차(잡음 0.05)와 잘 정해지는 b 계수를 본다
    assert np.sqrt(np.mean((predict(eq.node, theta, X, len(y)) - y) ** 2)) < 0.06
    assert theta[2] == pytest.approx(2.0, abs=0.02)
    wild = parse_equation("(a ^ 1.0) * 1.0")
    y_steep = X["a"] ** 8
    fit = refit(wild.node, wild.theta, X, y_steep)
    assert abs(fit[sorted(exponent_ids(wild.node))[0]]) <= 3.0 + 1e-9


def test_refit_is_only_constants():
    X, y = fake_data()
    eq = parse_equation("a + b")
    assert len(const_ids(eq.node)) == 0
    assert refit(eq.node, eq.theta, X, y).size == 0


def test_fold_rmse_uses_train_side_only():
    X, y = fake_data()
    masks = fake_masks(len(y))
    eq = parse_equation("a * 1.0 + 1.0")
    base = fold_rmse(eq.node, eq.theta, X, y, masks)
    assert len(base) == 25
    # 평가 쪽 y를 크게 바꿔도 그 fold의 상수는 달라지면 안 된다: 학습 쪽으로만 맞춘 상수와 같아야 한다
    ev = masks[0]
    y2 = y.copy()
    y2[ev] += 1000
    t1 = refit(eq.node, eq.theta, {k: v[~ev] for k, v in X.items()}, y[~ev])
    t2 = refit(eq.node, eq.theta, {k: v[~ev] for k, v in X.items()}, y2[~ev])
    assert np.allclose(t1, t2)


def test_invalid_equation_is_excluded_from_curve():
    X, y = fake_data()
    X = {**X, "z": X["z"] - 1.25}          # 음수가 섞인 변수
    masks = fake_masks(len(y))
    curve, invalid = score_candidates([("log(z) + 1.0", 4), ("a * 1.0 + 1.0", 5)], X, y, masks)
    assert invalid == 1
    assert curve["complexity"].tolist() == [5]


def test_same_complexity_keeps_lower_error():
    X, y = fake_data()
    masks = fake_masks(len(y))
    curve, _ = score_candidates([("z * 1.0 + 1.0", 5), ("a * 1.0 + 1.0", 5)], X, y, masks)
    assert curve["equation"].tolist() == ["a * 1.0 + 1.0"]


def test_select_rules():
    curve = pd.DataFrame({"equation": list("abcd"), "complexity": [1, 3, 5, 9],
                          "mean_rmse": [6.0, 4.3, 4.1, 4.0], "se": [0.2, 0.15, 0.12, 0.2]})
    pick, limit = select_1se(curve)
    assert limit == pytest.approx(4.2) and pick["equation"] == "c"
    assert select_multiplier(curve, 1.00)[0]["equation"] == "d"
    assert select_multiplier(curve, 1.05)[0]["equation"] == "c"      # 4.2 이내
    assert select_multiplier(curve, 1.10)[0]["equation"] == "b"      # 4.4 이내


def test_1se_recovers_true_form_on_fake_data():
    X, y = fake_data()
    masks = fake_masks(len(y))
    eqs = [("10.0", 1), ("a * 3.0 + 7.0", 5), ("(a ^ 0.4) * 10.0 + b * 2.0 + 5.0", 11),
           ("(a ^ 0.4) * 10.0 + b * 2.0 + 5.0 + z * 0.01", 15)]
    curve, _ = score_candidates(eqs, X, y, masks)
    pick, _ = select_1se(curve)
    assert pick["complexity"] == 11


def test_terms_sum_drops_constant_term():
    eq = parse_equation("a * 2.0 - b * 3.0 + 7.0")
    terms = split_terms(eq.node)
    assert [variables(t.node) for t in terms] == [{"a"}, {"b"}]
    assert [t.negated for t in terms] == [False, True]
    assert term_signs(terms[0], eq.theta) == (1,) and term_signs(terms[1], eq.theta) == (-1,)


def test_terms_single_product_uses_factors_and_coefficient_sign():
    eq = parse_equation("(a ^ 0.3) * b * -2.0")
    terms = split_terms(eq.node)
    assert [variables(t.node) for t in terms] == [{"a"}, {"b"}]
    signs = term_signs(terms[1], eq.theta)
    assert signs == (-1,)                 # b 인자에는 바깥 계수 -2의 부호가 붙는다


def test_terms_whole_expression_when_not_sum_or_product():
    eq = parse_equation("log(a * -2.0 + 9.0)")
    terms = split_terms(eq.node)
    assert len(terms) == 1 and terms[0].node[0] == "call"
    assert set(term_signs(terms[0], eq.theta)) == {-1, 1}      # 안쪽 계수 부호로 판단


def test_remove_term_rebuilds_expression():
    eq = parse_equation("a * 2.0 - b * 3.0 + 7.0")
    terms = split_terms(eq.node)
    reduced = remove_term(eq.node, terms[0].node)
    assert variables(reduced) == {"b"}
    whole = parse_equation("log(a)")
    assert remove_term(whole.node, split_terms(whole.node)[0].node) is None


def test_remove_factor_keeps_other_factor():
    eq = parse_equation("(a ^ 0.3) * b * 2.0")
    terms = split_terms(eq.node)
    reduced = remove_term(eq.node, terms[0].node)
    assert variables(reduced) == {"b"}


def test_main_terms_find_real_and_ignore_noise():
    X, y = fake_data()
    masks = fake_masks(len(y))
    eq = parse_equation("(a ^ 0.4) * 10.0 + b * 2.0 + z * 0.01 + 5.0")
    full = fold_rmse(eq.node, eq.theta, X, y, masks).mean()
    found = {tuple(sorted(variables(m["obj"].node))): m["is_main"] for m in main_terms(eq, X, y, masks, full)}
    assert found == {("a",): True, ("b",): True, ("z",): False}


def test_stability_detects_sign_flip_and_skips_small_interval():
    rng = np.random.default_rng(3)
    n = 300
    a = rng.uniform(0.5, 2.0, n)
    intervals = np.r_[np.zeros(145), np.ones(150), np.full(5, 2)].astype(int)
    y = np.where(intervals == 1, -2.0, 2.0) * a + rng.normal(0, 0.01, n)
    X = {"a": a}
    eq = parse_equation("a * 1.0")
    mains = [{"term": "a", "obj": split_terms(eq.node)[0]}]
    rows, stable, cv = stability(eq, mains, X, y, intervals)
    assert stable == {"a": False}
    assert [r["used"] for r in rows] == [1, 1, 0]
    assert len(cv) == 1


def test_consistent_requires_signs_and_same_form():
    ok = {"A": {"main_signs_stable": True, "form": "f"}, "B": {"main_signs_stable": True, "form": "f"}}
    assert consistent(ok)
    assert not consistent({**ok, "B": {"main_signs_stable": True, "form": "g"}})
    assert not consistent({**ok, "B": {"main_signs_stable": False, "form": "f"}})


def test_head_since_counts_as_same_form_only_for_head_parts():
    a = parse_equation("since_MEMBRANE_w * 2.0 + x * 1.0").node
    b = parse_equation("since_BACKING_FILM_w * 3.0 + x * 1.0").node
    c = parse_equation("since_MEMBRANE_w * 2.0 + y * 1.0").node
    assert skeleton(a, HEAD_SINCE) == skeleton(b, HEAD_SINCE)
    assert skeleton(a) != skeleton(b)
    assert skeleton(a, HEAD_SINCE) != skeleton(c, HEAD_SINCE)
    assert "since_DRESSER_w" not in HEAD_SINCE


def test_exponent_rule_on_equations():
    assert exponent_ok("(a ^ -2.5) + b")
    assert not exponent_ok("a ^ b")
    assert not exponent_ok("a ^ 3.5")
    passed, removed = filter_pareto(pd.DataFrame({"equation": ["a ^ 0.3", "a ^ b"]}))
    assert removed == 1 and len(passed) == 1

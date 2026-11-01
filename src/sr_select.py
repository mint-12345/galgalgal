"""정식 SR 파레토에서 최종 식을 고르고 기준선·안정성을 계산한다 (preregistration.md 4.4~4.7, 4.8 5~7·9번).

- 식은 문자열을 파싱해 형태를 고정하고, 상수만 비선형 최소제곱으로 fold마다 다시 맞춘다.
  `^`의 지수는 재적합 중에도 −3~3 안에 둔다 (#11).
- 선택: 복잡도마다 fold RMSE 평균이 가장 작은 식만 남긴 뒤, 최솟값 + 표준오차 1개 안에서 복잡도가 가장 낮은 식.
  표준오차 = 25개 fold RMSE의 표준편차(ddof=1)/√25. 민감도는 최솟값의 1.00·1.02·1.05·1.10배.
- 기준선: 같은 변수의 선형회귀, F-free RF. 같은 웨이퍼·같은 fold로 비교한다.
- 안정성: 헤드 부품 리셋(스테이지 단위)으로 나눈 구간마다 상수를 다시 맞춰 부호와 변동계수를 본다.
- folds_v1 오차는 식 선택에 쓴 데이터라 낙관적이다. 성능으로 보고하지 않는다 (4.1, 4.8 7번).
- test 라벨과 data/locked/는 읽지 않는다.

실행: python src/sr_select.py results/{날짜}_박종화_FV1_sr_pareto.csv
출력: results/{날짜}_박종화_FV1_sr_folds.csv (+ .yaml)
      results/diagnostics/{날짜}_박종화_sr_curve.csv, _sr_selection.csv, _sr_stability.csv
      이미 있으면 멈춘다 (덮어쓰지 않음).
"""
import ast
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import corrected_ttest  # noqa: E402
from sr_run import GROUPS, MAX_ABS_EXPONENT, PROC, RESULTS, exponent_ok, load_frame  # noqa: E402

KEYS = ["WAFER_ID", "STAGE"]
SEED = 42
FEATURESET = "features_v1"
MULTIPLIERS = [1.00, 1.02, 1.05, 1.10]
MAIN_TERM_RATIO = 1.05
MAX_COMPLEXITY = 20
EXPECTED_HEAD_RESETS = 5
# 구간별 재적합의 최소 웨이퍼 수는 max(10, 2 × (상수 수 + 1))이다 (4.8 9번).
MIN_INTERVAL_WAFERS = 10
# RF 입력에서도 since_POLISH_TABLE_w를 뺀다 (4.8 11번).
RF_EXCLUDED = {"since_POLISH_TABLE_w"}
# 헤드 부품 3종의 경과량은 같은 웨이퍼에서 리셋되므로 그룹 간 형태 비교에서 같은 계열로 본다 (4.6). 변수명은 각각 표기한다.
HEAD_SINCE = {f"since_{u}_w": "since_head" for u in ("BACKING_FILM", "MEMBRANE", "PSHEET")}
UNARY = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt}


# ---------------------------------------------------------------- 식 파싱
@dataclass
class Equation:
    """문자열 식. node는 중첩 튜플, theta는 상수 값. 형태는 node로 고정된다."""
    node: tuple
    theta: np.ndarray


def parse_equation(text):
    """PySR 식 문자열을 Equation으로 바꾼다. 음수 상수는 상수 하나로 본다."""
    theta = []

    def add_const(value):
        theta.append(float(value))
        return ("const", len(theta) - 1)

    def walk(n):
        if isinstance(n, ast.Constant):
            return add_const(n.value)
        if isinstance(n, ast.Name):
            return ("var", n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            if isinstance(n.operand, ast.Constant):
                return add_const(-n.operand.value)
            return ("neg", walk(n.operand))
        if isinstance(n, ast.BinOp):
            ops = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
            return (ops[type(n.op)], walk(n.left), walk(n.right))
        if isinstance(n, ast.Call) and n.func.id in UNARY and len(n.args) == 1:
            return ("call", n.func.id, walk(n.args[0]))
        raise ValueError(f"해석할 수 없는 식: {text}")

    node = walk(ast.parse(text.replace("^", "**"), mode="eval").body)
    return Equation(node, np.array(theta, dtype=float))


def _children(node):
    if node[0] in ("add", "sub", "mul", "div", "pow"):
        return [node[1], node[2]]
    if node[0] == "neg":
        return [node[1]]
    if node[0] == "call":
        return [node[2]]
    return []


def const_ids(node):
    ids = {node[1]} if node[0] == "const" else set()
    for c in _children(node):
        ids |= const_ids(c)
    return ids


def variables(node):
    found = {node[1]} if node[0] == "var" else set()
    for c in _children(node):
        found |= variables(c)
    return found


def exponent_ids(node):
    ids = {node[2][1]} if node[0] == "pow" and node[2][0] == "const" else set()
    for c in _children(node):
        ids |= exponent_ids(c)
    return ids


def skeleton(node, rename=None):
    """상수를 c로 바꾼 형태 문자열. rename이 있으면 변수 이름을 바꿔 그룹 간 비교에 쓴다."""
    kind = node[0]
    if kind == "const":
        return "c"
    if kind == "var":
        return rename.get(node[1], node[1]) if rename else node[1]
    if kind == "neg":
        return f"(-{skeleton(node[1], rename)})"
    if kind == "call":
        return f"{node[1]}({skeleton(node[2], rename)})"
    return f"({skeleton(node[1], rename)} {kind} {skeleton(node[2], rename)})"


def _evaluate(node, theta, X):
    kind = node[0]
    if kind == "const":
        return theta[node[1]]
    if kind == "var":
        return X[node[1]]
    if kind == "neg":
        return -_evaluate(node[1], theta, X)
    if kind == "call":
        return UNARY[node[1]](_evaluate(node[2], theta, X))
    a, b = _evaluate(node[1], theta, X), _evaluate(node[2], theta, X)
    if kind == "add":
        return a + b
    if kind == "sub":
        return a - b
    if kind == "mul":
        return a * b
    if kind == "div":
        return a / b
    return np.power(np.asarray(a, dtype=float), b)


def predict(node, theta, X, n):
    """X = {변수 이름: 배열}. 계산이 안 되는 값은 NaN으로 돌려준다."""
    with np.errstate(all="ignore"):
        value = _evaluate(node, theta, X)
    out = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def refit(node, theta0, X, y):
    """node 안의 상수만 비선형 최소제곱으로 맞춘다. 지수는 −3~3 안에 둔다."""
    ids = sorted(const_ids(node))
    theta = np.array(theta0, dtype=float)
    if not ids:
        return theta
    powers = exponent_ids(node)
    low = np.array([-MAX_ABS_EXPONENT if i in powers else -np.inf for i in ids])
    high = np.array([MAX_ABS_EXPONENT if i in powers else np.inf for i in ids])

    def residual(values):
        trial = theta.copy()
        trial[ids] = values
        r = predict(node, trial, X, len(y)) - y
        r[np.isnan(r)] = 1e6
        return r

    theta[ids] = least_squares(residual, np.clip(theta[ids], low, high), bounds=(low, high), max_nfev=200).x
    return theta


# ---------------------------------------------------------------- fold 오차와 선택
def fold_masks(table):
    """{r: 평가 쪽 마스크}. r = repeat × 5 + fold (5.7 1번). 학습 쪽은 같은 반복의 나머지 조각이다."""
    assign = pd.read_csv(PROC / "folds_v1.csv")
    assign["r"] = assign["repeat"] * 5 + assign["fold"]
    masks = {}
    for repeat, part in assign.groupby("repeat"):
        merged = table[KEYS].merge(part[KEYS + ["r"]], on=KEYS, how="left", validate="one_to_one")
        assert merged["r"].notna().all(), f"fold가 없는 웨이퍼가 있다: repeat={repeat}"
        for r in sorted(merged["r"].unique()):
            masks[int(r)] = merged["r"].to_numpy() == r
    assert len(masks) == 25, f"fold가 {len(masks)}개다"
    return masks


def _subset(X, mask):
    return {k: v[mask] for k, v in X.items()}


def fold_rmse(node, theta0, X, y, masks):
    """fold마다 상수를 다시 맞춘 평가 쪽 RMSE. 계산이 안 되는 예측이 있으면 inf."""
    out = []
    for r in sorted(masks):
        ev = masks[r]
        theta = refit(node, theta0, _subset(X, ~ev), y[~ev])
        err = predict(node, theta, _subset(X, ev), int(ev.sum())) - y[ev]
        out.append(np.inf if np.isnan(err).any() else float(np.sqrt(np.mean(err ** 2))))
    return np.array(out)


def score_candidates(equations, X, y, masks):
    """(복잡도마다 fold RMSE 평균이 가장 작은 식의 표, 계산이 안 되어 뺀 식 수). 4.8 6번."""
    rows = []
    for text, complexity in equations:
        eq = parse_equation(text)
        scores = fold_rmse(eq.node, eq.theta, X, y, masks)
        rows.append({
            "equation": text, "complexity": int(complexity), "mean_rmse": scores.mean(),
            "se": scores.std(ddof=1) / np.sqrt(len(scores)) if np.isfinite(scores).all() else np.inf,
        })
    table = pd.DataFrame(rows)
    finite = np.isfinite(table["mean_rmse"])
    curve = table[finite].sort_values(["complexity", "mean_rmse", "equation"]).drop_duplicates("complexity")
    return curve.reset_index(drop=True), int((~finite).sum())


def select_1se(curve):
    best = curve.loc[curve["mean_rmse"].idxmin()]
    limit = best["mean_rmse"] + best["se"]
    return curve[curve["mean_rmse"] <= limit].sort_values("complexity").iloc[0], float(limit)


def select_multiplier(curve, multiplier):
    limit = curve["mean_rmse"].min() * multiplier
    return curve[curve["mean_rmse"] <= limit].sort_values("complexity").iloc[0], float(limit)


# ---------------------------------------------------------------- 항 정의 (4.8 9번)
@dataclass
class Term:
    node: tuple
    sign_ids: list      # 부호를 볼 상수
    negated: bool


def _flatten_sum(node, negated=False):
    if node[0] == "add":
        return _flatten_sum(node[1], negated) + _flatten_sum(node[2], negated)
    if node[0] == "sub":
        return _flatten_sum(node[1], negated) + _flatten_sum(node[2], not negated)
    if node[0] == "neg":
        return _flatten_sum(node[1], not negated)
    return [(node, negated)]


def _flatten_product(node):
    return _flatten_product(node[1]) + _flatten_product(node[2]) if node[0] == "mul" else [node]


def split_terms(node):
    """상수항을 뺀 최상위 덧셈 항. 곱 하나면 각 인자. 덧셈도 곱도 아니면 식 전체가 한 항."""
    parts = [(n, neg) for n, neg in _flatten_sum(node) if variables(n)]
    if len(parts) == 1 and parts[0][0][0] == "mul":
        factors = _flatten_product(parts[0][0])
        outer = {i for f in factors if not variables(f) for i in const_ids(f)}
        return [Term(f, sorted(outer | const_ids(f)), parts[0][1]) for f in factors if variables(f)]
    return [Term(n, sorted(const_ids(n)), neg) for n, neg in parts]


def term_signs(term, theta):
    """항의 부호 묶음. 상수가 없으면 형태로 고정되어 빼기 여부만 남는다."""
    flip = -1 if term.negated else 1
    return tuple(int(np.sign(theta[i])) * flip for i in term.sign_ids) or (flip,)


def remove_term(node, target):
    """target 항을 뺀 식. 남는 것이 없으면 None (평균 하나로 맞춘다)."""
    parts = _flatten_sum(node)
    variable_parts = [p for p in parts if variables(p[0])]
    if len(variable_parts) == 1 and variable_parts[0][0][0] == "mul":
        factors = [f for f in _flatten_product(variable_parts[0][0]) if f is not target]
        keep = factors[0] if factors else None
        for f in factors[1:]:
            keep = ("mul", keep, f)
        rebuilt = [(keep, neg) if n is variable_parts[0][0] else (n, neg) for n, neg in parts]
        rebuilt = [p for p in rebuilt if p[0] is not None]
    else:
        rebuilt = [p for p in parts if p[0] is not target]
    if not rebuilt:
        return None
    out = ("neg", rebuilt[0][0]) if rebuilt[0][1] else rebuilt[0][0]
    for n, neg in rebuilt[1:]:
        out = ("sub" if neg else "add", out, n)
    return out


def _mean_fold_rmse(y, masks):
    out = []
    for r in sorted(masks):
        ev = masks[r]
        out.append(float(np.sqrt(np.mean((y[ev] - y[~ev].mean()) ** 2))))
    return np.array(out)


def main_terms(eq, X, y, masks, full_mean):
    """항별로 빼고 상수를 다시 맞춘 RMSE. 5% 이상 커지면 주요 항이다 (4.8 9번)."""
    out = []
    for term in split_terms(eq.node):
        reduced = remove_term(eq.node, term.node)
        scores = _mean_fold_rmse(y, masks) if reduced is None else fold_rmse(reduced, eq.theta, X, y, masks)
        ratio = scores.mean() / full_mean
        out.append({"term": skeleton(term.node), "obj": term, "rmse_without": float(scores.mean()),
                    "ratio": float(ratio), "is_main": bool(ratio >= MAIN_TERM_RATIO)})
    return out


# ---------------------------------------------------------------- 구간 안정성 (4.6)
def head_reset_intervals():
    """스테이지별 시작 시각 순으로 헤드 부품 리셋을 누적한 구간 번호. 세 부품은 같은 웨이퍼에서 리셋된다."""
    t = pd.read_csv(PROC / "wafer_table.csv", usecols=KEYS + ["t_start", "reset_MEMBRANE_w"])
    t = t.sort_values(["STAGE", "t_start"]).reset_index(drop=True)
    t["interval"] = t.groupby("STAGE")["reset_MEMBRANE_w"].transform(lambda s: s.fillna(0).cumsum()).astype(int)
    counts = t.groupby("STAGE")["reset_MEMBRANE_w"].sum()
    assert (counts == EXPECTED_HEAD_RESETS).all(), f"헤드 부품 리셋 수가 다르다: {counts.to_dict()}"
    return t[KEYS + ["interval"]]


def stability(eq, mains, X, y, intervals):
    """구간마다 상수를 다시 맞춰 (구간별 행, 항별 부호 일치 여부, 상수별 변동계수)를 돌려준다."""
    min_n = max(MIN_INTERVAL_WAFERS, 2 * (len(eq.theta) + 1))
    rows, signs, values = [], {m["term"]: [] for m in mains}, []
    for k in sorted(np.unique(intervals)):
        pick = intervals == k
        row = {"interval": int(k), "n": int(pick.sum()), "used": int(pick.sum() >= min_n)}
        if row["used"]:
            theta = refit(eq.node, eq.theta, _subset(X, pick), y[pick])
            values.append(theta)
            for m in mains:
                s = term_signs(m["obj"], theta)
                signs[m["term"]].append(s)
                row[f"sign[{m['term']}]"] = str(s)
        rows.append(row)
    stable = {t: len(set(s)) <= 1 for t, s in signs.items()}
    cv = []
    if len(values) > 1:
        v = np.array(values)
        with np.errstate(all="ignore"):
            cv = (v.std(axis=0, ddof=1) / np.abs(v.mean(axis=0))).tolist()
    return rows, stable, cv


def consistent(results):
    """모든 주기에서 주요 항의 부호가 같고 두 그룹의 함수 형태가 같으면 True (4.6)."""
    return all(r["main_signs_stable"] for r in results.values()) and len({r["form"] for r in results.values()}) == 1


# ---------------------------------------------------------------- 기준선
def fold_rows(group, model, y, masks, predict_fold):
    """12절 형식의 fold별 행. predict_fold(ev)는 평가 쪽 예측을 돌려준다."""
    rows = []
    for r in sorted(masks):
        ev = masks[r]
        err = predict_fold(ev) - y[ev]
        rows.append({"group": group, "model": model, "featureset": FEATURESET, "fold": r,
                     "mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
                     "rmse": float(np.sqrt(np.mean(err ** 2))), "n_train": int((~ev).sum()), "n_eval": int(ev.sum())})
    return rows


def rf_columns():
    fam = pd.read_csv(PROC / "feature_families_v1.csv")
    return [c for c in fam["feature"] if c not in RF_EXCLUDED]


def analyze_group(group, pareto):
    frame, names, n_total = load_frame(group)
    table = frame.merge(head_reset_intervals(), on=KEYS, how="left", validate="one_to_one")
    assert table["interval"].notna().all()
    masks = fold_masks(table)
    y = table["MRR"].to_numpy(float)
    X = {n: table[n].to_numpy(float) for n in names}

    part = pareto[pareto["group"] == group]
    n_removed = int((~part["equation"].map(exponent_ok)).sum())
    usable = part[part["equation"].map(exponent_ok)].drop_duplicates("equation")
    curve, n_invalid = score_candidates(list(zip(usable["equation"], usable["complexity"])), X, y, masks)

    final, limit = select_1se(curve)
    selection = [{"group": group, "rule": "1se", "limit": limit, **final.to_dict()}]
    for m in MULTIPLIERS:
        pick, lim = select_multiplier(curve, m)
        selection.append({"group": group, "rule": f"x{m:.2f}", "limit": lim, **pick.to_dict()})

    eq = parse_equation(final["equation"])
    mains = main_terms(eq, X, y, masks, final["mean_rmse"])
    rows, stable, cv = stability(eq, mains, X, y, table["interval"].to_numpy())

    def final_pred(ev):
        theta = refit(eq.node, eq.theta, _subset(X, ~ev), y[~ev])
        return predict(eq.node, theta, _subset(X, ev), int(ev.sum()))

    M = table[names].to_numpy(float)
    R = table[rf_columns()].to_numpy(float)
    folds = fold_rows(group, "sr_final", y, masks, final_pred)
    folds += fold_rows(group, "lr", y, masks, lambda ev: LinearRegression().fit(M[~ev], y[~ev]).predict(M[ev]))
    folds += fold_rows(group, "rf", y, masks, lambda ev: RandomForestRegressor(
        max_features=1.0, random_state=SEED, n_jobs=-1).fit(R[~ev], y[~ev]).predict(R[ev]))
    frame_folds = pd.DataFrame(folds)
    means = frame_folds.groupby("model")["rmse"].mean()
    pair = frame_folds.pivot(index="fold", columns="model", values="mse")
    size = frame_folds[frame_folds["model"] == "lr"].set_index("fold")
    test = corrected_ttest((pair["sr_final"] - pair["lr"]).to_numpy(),
                           size["n_train"].to_numpy(), size["n_eval"].to_numpy())

    main_list = [m["term"] for m in mains if m["is_main"]]
    summary = {
        "group": group, "equation": final["equation"], "complexity": int(final["complexity"]),
        "at_max_complexity": int(final["complexity"]) >= MAX_COMPLEXITY, "mean_rmse": float(final["mean_rmse"]),
        "n_wafers": len(y), "n_wafers_total": n_total, "n_pareto": len(part), "n_removed_exponent": n_removed,
        "n_invalid_refit": n_invalid, "rmse_lr": float(means["lr"]), "rmse_rf": float(means["rf"]),
        "worse_than_lr": bool(final["mean_rmse"] > means["lr"]), "p_vs_lr": test.get("p"),
        "main_terms": main_list, "main_signs_stable": all(stable[t] for t in main_list),
        "n_intervals": len(rows), "n_intervals_skipped": sum(1 for row in rows if not row["used"]),
        "form": skeleton(eq.node, HEAD_SINCE), "raw_form": skeleton(eq.node),
        "variables": sorted(variables(eq.node)), "cv_constants": cv,
    }
    curve = curve.assign(group=group)[["group", "complexity", "mean_rmse", "se", "equation"]]
    stab = pd.DataFrame(rows).assign(group=group)
    terms = pd.DataFrame([{k: m[k] for k in ("term", "rmse_without", "ratio", "is_main")} for m in mains]).assign(group=group)
    return summary, curve, pd.DataFrame(selection), stab, terms, frame_folds


def main():
    if len(sys.argv) != 2:
        sys.exit("사용법: python src/sr_select.py results/{날짜}_박종화_FV1_sr_pareto.csv")
    pareto_path = Path(sys.argv[1])
    pareto = pd.read_csv(pareto_path)
    done = pareto.groupby("group")["seed"].nunique()
    assert all(done.get(g, 0) == 10 for g in GROUPS), f"시드가 모자란다: {done.to_dict()}"

    stamp = date.today().isoformat()
    diag = RESULTS / "diagnostics"
    out = {"folds": RESULTS / f"{stamp}_박종화_FV1_sr_folds.csv",
           "curve": diag / f"{stamp}_박종화_sr_curve.csv", "selection": diag / f"{stamp}_박종화_sr_selection.csv",
           "stability": diag / f"{stamp}_박종화_sr_stability.csv", "terms": diag / f"{stamp}_박종화_sr_terms.csv"}
    for p in out.values():
        if p.exists():
            sys.exit(f"이미 있어서 멈춘다: {p}")

    parts, results = {k: [] for k in out}, {}
    for g in GROUPS:
        summary, curve, selection, stab, terms, folds = analyze_group(g, pareto)
        results[g] = summary
        for k, v in (("curve", curve), ("selection", selection), ("stability", stab), ("terms", terms), ("folds", folds)):
            parts[k].append(v)
    for k, p in out.items():
        pd.concat(parts[k], ignore_index=True).to_csv(p, index=False, encoding="utf-8-sig")

    lines = [f"date: {stamp}", "author: 박종화", "script: src/sr_select.py", f"input: {pareto_path.as_posix()}",
             f"featureset: {FEATURESET}", "note: folds_v1 오차는 식 선택에 쓴 데이터라 낙관적, 성능으로 보고하지 않음",
             f"same_form_head_since_as_family: {consistent(results)}"]
    for g, r in results.items():
        lines.append(f"{g}:")
        lines += [f"  {k}: {v}" for k, v in r.items()]
    out["folds"].with_suffix(".yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("저장:", ", ".join(p.name for p in out.values()))


if __name__ == "__main__":
    main()

"""공동 SR 2단계 탐색: PySR 2.7.2 TemplateExpressionSpec으로 MRR_g = a[g]·f(x) + b[g] (사후 후속 실험).

계획: docs/plans/2026-11-04_joint_sr_plan.md 7절. 설정은 정식 SR(`src/sr_run.py`)과 같게 하고,
지수 차단 loss만 템플릿 안의 f 트리를 검사하도록 바꾼다.
- f는 공통 후보 14개만 받는다(g는 받지 않음). f 안의 상수는 하나라 두 테이블이 공유한다.
- a, b는 길이 2 매개변수 벡터, g = 1(A-456) / 2(B-456)로 고른다.
- 입력: 두 테이블 합산 |중앙값|으로 나눈 14개 변수(sr_joint.design, pooled), 출력: 원척도 MRR.

명령:
  python src/sr_joint_search.py toy              인공 데이터로 템플릿 구조·지수 차단 검증 (라벨 없음)
  python src/sr_joint_search.py search [--seeds 42 43 ...] [--niterations 500]
      시드마다 파레토를 캐시에 덧붙인다(다시 실행하면 끝난 시드 건너뜀). 기본값은 정식 설정.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_joint as sj  # noqa: E402
import sr_run  # noqa: E402

SEEDS = list(range(42, 52))
NITERATIONS = sr_run.NITERATIONS          # 500
MAXSIZE = 20
SEARCH_CACHE = sj.CACHE / "search_pareto.jsonl"

# 정식 sr_run.LOSS_EXPRESSION과 같은 규칙. 템플릿이면 하위 식(f)마다 트리를 검사한다.
LOSS_TEMPLATE = f"""
function pow_guard(ex, dataset::Dataset{{T,L}}, options) where {{T,L}}
    subs = values(SymbolicRegression.get_contents(ex))
    bad = any(subs) do sub
        tree = SymbolicRegression.get_tree(sub)
        any(tree) do node
            node.degree == 2 && options.operators.binops[node.op] === SymbolicRegression.safe_pow &&
                !(node.r.degree == 0 && node.r.constant && abs(node.r.val) <= {sr_run.MAX_ABS_EXPONENT})
        end
    end
    bad && return {sr_run.BLOCKED_LOSS}
    pred, ok = eval_tree_array(ex, dataset.X, options)
    !ok && return {sr_run.BLOCKED_LOSS}
    return sum(abs2, pred .- dataset.y) / length(dataset.y)
end
"""
PLAIN_LOSS = """
function plain_mse(ex, dataset::Dataset{T,L}, options) where {T,L}
    pred, ok = eval_tree_array(ex, dataset.X, options)
    !ok && return L(Inf)
    return sum(abs2, pred .- dataset.y) / length(dataset.y)
end
"""


def make_spec(names: list[str]):
    from pysr import TemplateExpressionSpec
    return TemplateExpressionSpec(
        combine=f"a[g] * f({', '.join(names)}) + b[g]",
        expressions=["f"], variable_names=list(names) + ["g"], parameters={"a": 2, "b": 2})


def make_model(names, seed, niterations, loss=LOSS_TEMPLATE, maxsize=MAXSIZE):
    """정식 sr_run.make_model과 같은 설정 + expression_spec."""
    from pysr import PySRRegressor
    nested = {a: {b: 0 for b in sr_run.UNARY} for a in sr_run.UNARY}
    return PySRRegressor(
        niterations=niterations, populations=31, population_size=27,
        binary_operators=["+", "-", "*", "/", "^"], unary_operators=sr_run.UNARY,
        constraints={"^": (-1, 1)}, nested_constraints=nested,
        complexity_of_operators={"exp": 3, "log": 3, "sqrt": 2, "^": 2},
        maxsize=maxsize, deterministic=True, parallelism="serial", random_state=seed,
        progress=False, verbosity=0, temp_equation_file=True,
        loss_function_expression=loss, expression_spec=make_spec(names))


ARG = re.compile(r"#(\d+)")


def split_template(text: str, names: list[str]) -> dict:
    """템플릿 식 문자열에서 f 식(변수 이름으로 바꿈)과 a·b 값을 꺼낸다."""
    parts = {}
    for line in re.split(r"[\n;]", text):
        line = line.strip().lstrip("╭├╰│ ").strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        parts[key.strip()] = val.strip()
    f = parts.get("f")
    if f is None:
        raise ValueError(f"f를 찾지 못함: {text!r}")
    used = sorted({int(k) for k in ARG.findall(f)})
    assert all(1 <= k <= len(names) for k in used), f"f 인자 번호가 범위 밖: {used}"
    f_named = ARG.sub(lambda m: names[int(m.group(1)) - 1], f)
    return {"f": f_named, "f_raw": f, "a": parts.get("a"), "b": parts.get("b"), "args_used": used}


# ---------------------------------------------------------------- 인공 데이터 검증
def cmd_toy(args):
    """(1) 템플릿 실행 (2) f가 g를 받지 않음 (3) a·b 그룹별 추정 (4) 지수 차단 작동."""
    rng = np.random.default_rng(0)
    n = 300
    names = ["x1", "x2", "x3"]
    X = rng.uniform(1.0, 2.0, (n, 3))
    g = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)]
    gi = (g - 1).astype(int)
    out = {}
    # 검증 1~3: f = x1 / x2, a = (3, −2), b = (10, 40)
    f = X[:, 0] / X[:, 1]
    y = np.array([3.0, -2.0])[gi] * f + np.array([10.0, 40.0])[gi] + rng.normal(0, 0.01, n)
    t0 = time.perf_counter()
    m = make_model(names, 42, args.niterations)
    m.fit(np.column_stack([X, g]), y, variable_names=names + ["g"])
    eq = m.equations_
    best = eq.loc[eq["loss"].idxmin()]
    parts = split_template(str(best["equation"]), names)
    print("열:", list(eq.columns))
    print("최저 loss 식:", repr(best["equation"]), "loss", best["loss"], "complexity", best["complexity"])
    print("f:", parts, f"({time.perf_counter() - t0:.0f}초)")
    out["template_runs"] = True
    out["f_has_no_g"] = "g" not in parts["f"] and all(k <= 3 for k in parts["args_used"])
    out["best_loss_lt_1e-3"] = bool(best["loss"] < 1e-3)
    # 검증 4: 참 식 x1 ^ x2 (변수 지수). 차단 loss면 변수 지수 식이 파레토에 없어야 한다.
    y4 = np.array([1.0, 2.0])[gi] * X[:, 0] ** (3 * X[:, 1]) + np.array([0.0, 5.0])[gi]
    res = {}
    for name, loss in (("guard", LOSS_TEMPLATE), ("plain", PLAIN_LOSS)):
        m4 = make_model(names, 42, args.niterations, loss=loss)
        m4.fit(np.column_stack([X, g]), y4, variable_names=names + ["g"])
        fs = [split_template(str(e), names)["f"] for e in m4.equations_["equation"]]
        res[name] = {"n": len(fs), "n_bad_exponent": int(sum(not sr_run.exponent_ok(f) for f in fs)),
                     "min_loss": float(m4.equations_["loss"].min())}
    print("지수 차단 시험:", res)
    out["guard_blocks_variable_exponent"] = res["guard"]["n_bad_exponent"] == 0
    out["plain_finds_variable_exponent"] = res["plain"]["n_bad_exponent"] > 0
    out["guard_detail"] = res
    print("검증 결과:", out)
    path = sj.CACHE / "toy_validation.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = all(out[k] for k in ("template_runs", "f_has_no_g", "best_loss_lt_1e-3", "guard_blocks_variable_exponent"))
    print("방식 A(템플릿) 사용 가능:" if ok else "방식 A 불가 → 방식 B:", ok)


# ---------------------------------------------------------------- 탐색
def cmd_search(args):
    d, full, names = sj.joint_frame()
    sc = sj.scale_table(full, names)
    X = sj.design(d, names, sc, "pooled")
    M = np.column_stack([X[c] for c in names] + [d["gi"].to_numpy(float) + 1.0])
    y = d["MRR"].to_numpy(float)
    assert not np.isnan(M).any() and not np.isnan(y).any()
    cache = Path(args.cache) if args.cache else SEARCH_CACHE
    done = set()
    if cache.exists():
        done = {(json.loads(l)["seed"], json.loads(l)["niterations"]) for l in cache.read_text(encoding="utf-8").splitlines()}
    for seed in args.seeds:
        if (seed, args.niterations) in done:
            print(f"시드 {seed} 이미 있음, 건너뜀", flush=True)
            continue
        t0 = time.perf_counter()
        m = make_model(names, seed, args.niterations)
        m.fit(M, y, variable_names=names + ["g"])
        sec = round(time.perf_counter() - t0, 1)
        rows = []
        for _, e in m.equations_.iterrows():
            parts = split_template(str(e["equation"]), names)
            rows.append({"complexity": int(e["complexity"]), "loss": float(e["loss"]), "template": str(e["equation"]),
                         "f": parts["f"], "a": parts["a"], "b": parts["b"],
                         "exponent_ok": bool(sr_run.exponent_ok(parts["f"]))})
        rec = {"seed": seed, "niterations": args.niterations, "fit_seconds": sec, "n_rows": len(y), "equations": rows}
        with cache.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"시드 {seed}: {sec}초, 파레토 {len(rows)}개 (식 내용은 출력하지 않음)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toy")
    t.add_argument("--niterations", type=int, default=40)
    s = sub.add_parser("search")
    s.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    s.add_argument("--niterations", type=int, default=NITERATIONS)
    s.add_argument("--cache", default=None, help="속도 시험용 별도 캐시 경로 (정식 탐색은 생략)")
    args = ap.parse_args()
    {"toy": cmd_toy, "search": cmd_search}[args.cmd](args)


if __name__ == "__main__":
    main()

"""정식 SR 실행과 시드 시험이 함께 쓰는 설정 (preregistration.md 4.3, 4.8절).

- 입력: 그룹별 후보 10개(sr_candidates.py 결과), features_v1.csv(FV1), wafer_table.csv의 MRR.
  train·val, is_outlier = 0만 쓴다. test 라벨과 data/locked/는 읽지 않는다.
- 척도: 변수마다 그룹 train+val 전체의 절댓값 중앙값으로 나눈다 (부호 유지). MRR은 원척도.
- 웨이퍼: 후보 10개에 값이 모두 있는 웨이퍼만 쓴다.
- 지수 규칙: 탐색 중 차단, 사후 제외는 검증용 (4.8 5번).

실행: python src/sr_run.py   (A-456, B-456 × 시드 42~51, niterations=500)
출력: results/{날짜}_박종화_FV1_sr_pareto.csv (+ .yaml)
      시드 하나가 끝날 때마다 행을 덧붙인다. 다시 실행하면 끝난 시드는 건너뛴다.
      --date YYYY-MM-DD 로 이어 돌릴 파일을 지정한다. 식 내용은 화면에 출력하지 않는다.
"""
import argparse
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CANDIDATES = RESULTS / "diagnostics" / "2026-11-01_박종화_sr_candidates.csv"
GROUPS = ["A-456", "B-456"]
SEEDS = range(42, 52)
NITERATIONS = 500
FEATURESET = "features_v1"
UNARY = ["exp", "log", "sqrt"]
MAX_ABS_EXPONENT = 3.0
POW_TOKEN = re.compile(r"\^\s*\(?\s*([^\s()]+)")
# 탐색 중 차단 (4.8 5번): `^`의 지수가 상수가 아니거나 −3~3 밖이면 이 loss를 준다.
# 무한대 때문에 경고가 나거나 멈추면 "L(1e10)"으로 바꾼다.
BLOCKED_LOSS = "L(Inf)"
LOSS_EXPRESSION = f"""
function pow_guard(ex, dataset::Dataset{{T,L}}, options) where {{T,L}}
    tree = SymbolicRegression.get_tree(ex)
    bad = any(tree) do node
        node.degree == 2 && options.operators.binops[node.op] === SymbolicRegression.safe_pow &&
            !(node.r.degree == 0 && node.r.constant && abs(node.r.val) <= {MAX_ABS_EXPONENT})
    end
    bad && return {BLOCKED_LOSS}
    pred, ok = eval_tree_array(ex, dataset.X, options)
    !ok && return {BLOCKED_LOSS}
    return sum(abs2, pred .- dataset.y) / length(dataset.y)
end
"""


def load_frame(group):
    """(후보가 모두 있는 웨이퍼 표, 후보 이름, 전체 웨이퍼 수). 후보 열은 척도를 적용한 값."""
    c = pd.read_csv(CANDIDATES)
    names = c[(c["group"] == group) & (c["selected"] == 1)].sort_values("rank")["feature"].tolist()
    assert len(names) == 10, f"{group} 후보가 {len(names)}개다"

    feats = pd.read_csv(PROC / "features_v1.csv")
    table = pd.read_csv(PROC / "wafer_table.csv",
                        usecols=["WAFER_ID", "STAGE", "split", "MRR", "is_outlier"])
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    data = feats[feats["group"] == group].merge(table, on=["WAFER_ID", "STAGE"], validate="one_to_one")

    scale = data[names].abs().median()
    assert (scale >= 1e-9).all(), "절댓값 중앙값이 1e-9 미만인 후보가 있다"
    keep = data[names].notna().all(axis=1)
    frame = data[keep].copy()
    frame[names] = frame[names] / scale
    return frame.reset_index(drop=True), names, len(data)


def load_xy(group):
    """(X, y, 후보 이름, 전체 웨이퍼 수)를 돌려준다. X는 척도를 적용한 값."""
    frame, names, n_total = load_frame(group)
    return frame[names].to_numpy(), frame["MRR"].to_numpy(), names, n_total


def make_model(seed, niterations, maxsize=20):
    from pysr import PySRRegressor   # 식 선택·테스트는 Julia가 필요 없다

    nested = {a: {b: 0 for b in UNARY} for a in UNARY}
    return PySRRegressor(
        niterations=niterations,
        populations=31,
        population_size=27,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=UNARY,
        constraints={"^": (-1, 1)},
        nested_constraints=nested,
        complexity_of_operators={"exp": 3, "log": 3, "sqrt": 2, "^": 2},
        maxsize=maxsize,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        loss_function_expression=LOSS_EXPRESSION,
    )


def exponent_ok(equation):
    """모든 `^`의 지수가 상수이고 −3~3 안에 있으면 True (4.8 5번의 사후 검증용).

    constraints로 지수 크기가 1이라 오른쪽은 상수나 변수 하나다.
    """
    for token in POW_TOKEN.findall(equation):
        try:
            value = float(token)
        except ValueError:
            return False
        if abs(value) > MAX_ABS_EXPONENT:
            return False
    return True


def filter_pareto(equations):
    """(통과한 식 표, 뺀 개수)를 돌려준다."""
    mask = equations["equation"].map(exponent_ok)
    return equations[mask], int((~mask).sum())


def run_all(stamp, groups=GROUPS, seeds=SEEDS, niterations=NITERATIONS):
    """그룹 × 시드를 차례로 돌려 파레토를 덧붙인다. 끝난 시드는 건너뛴다."""
    out = RESULTS / f"{stamp}_박종화_FV1_sr_pareto.csv"
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["group"], old["seed"]))
    for group in groups:
        X, y, names, n_total = load_xy(group)
        assert not np.isnan(X).any() and not np.isnan(y).any()
        for seed in seeds:
            if (group, seed) in done:
                continue
            model = make_model(seed, niterations)
            start = time.perf_counter()
            model.fit(X, y, variable_names=names)
            seconds = round(time.perf_counter() - start, 1)
            e = model.equations_
            passed, n_removed = filter_pareto(e)
            rows = pd.DataFrame({
                "group": group, "seed": seed, "featureset": FEATURESET, "niterations": niterations,
                "complexity": e["complexity"], "loss": e["loss"], "equation": e["equation"],
                "exponent_ok": e["equation"].map(exponent_ok), "n_wafers": len(y), "n_wafers_total": n_total,
                "fit_seconds": seconds,
            })
            rows.to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8")
            print(f"{group} seed={seed} {seconds}s pareto={len(e)} removed={n_removed}", flush=True)
    return out


def write_yaml(out):
    p = pd.read_csv(out)
    out.with_suffix(".yaml").write_text("\n".join([
        f"date: {out.name[:10]}",
        "author: 박종화",
        "script: src/sr_run.py",
        f"input: data/processed/features_v1.csv + wafer_table.csv(MRR) + {CANDIDATES.name}",
        f"featureset: {FEATURESET}",
        f"groups: {GROUPS}",
        f"seeds: {list(SEEDS)}",
        f"niterations: {NITERATIONS}",
        "populations: 31",
        "population_size: 27",
        "maxsize: 20",
        "deterministic: true",
        "parallelism: serial",
        f"blocked_loss: {BLOCKED_LOSS}",
        f"n_rows: {len(p)}",
        f"n_removed_exponent: {int((~p['exponent_ok']).sum())}",
        "",
    ]), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(), help="이어 돌릴 결과 파일의 날짜")
    out = run_all(ap.parse_args().date)
    write_yaml(out)
    print(f"저장: {out.name}, {out.with_suffix('.yaml').name}")


if __name__ == "__main__":
    main()

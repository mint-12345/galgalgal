"""정식 SR 전 시드 시험: A-456, 시드 42 (preregistration.md 4.8 4번).

- niterations 250과 500을 따로 실행해 시간과 파레토 최저 RMSE를 비교한다.
- RMSE는 A-456 전체에 맞춘 학습 loss의 제곱근이다. fold 평가 RMSE가 아니다.
- 개선이 1% 이상이면 1000도 실행한다. 1000에서 또 1% 이상 개선되면 1000에서 멈추고
  "수렴 덜 됐을 수 있음"을 한계로 적는다.
- 식 내용은 출력하거나 저장하지 않는다. 이 결과는 정식 결과에 넣지 않는다.

실행: python src/sr_pilot.py          (--smoke: 2회 반복으로 동작만 확인, 저장 안 함)
출력: results/diagnostics/{날짜}_박종화_sr_pilot.csv, 같은 이름의 .yaml
"""
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_run import BLOCKED_LOSS, filter_pareto, load_xy, make_model  # noqa: E402

GROUP = "A-456"
SEED = 42
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "diagnostics"
MIN_IMPROVEMENT = 0.01


def run_once(X, y, names, niterations):
    model = make_model(SEED, niterations)
    start = time.perf_counter()
    model.fit(X, y, variable_names=names)
    seconds = time.perf_counter() - start
    passed, n_removed = filter_pareto(model.equations_)
    assert len(passed) > 0, "지수 규칙을 통과한 식이 없다"
    return {
        "group": GROUP, "seed": SEED, "niterations": niterations, "fit_seconds": round(seconds, 1),
        "n_wafers": len(y), "n_pareto": len(model.equations_), "n_removed_exponent": n_removed,
        "train_loss_min": float(passed["loss"].min()),
        "train_rmse_min": float(np.sqrt(passed["loss"].min())),
        "max_complexity": int(passed["complexity"].max()),
    }


def improvement(before, after):
    return (before["train_rmse_min"] - after["train_rmse_min"]) / before["train_rmse_min"]


def main():
    X, y, names, n_total = load_xy(GROUP)
    assert not np.isnan(X).any() and not np.isnan(y).any()

    if "--smoke" in sys.argv:
        r = run_once(X, y, names, 2)
        print({k: r[k] for k in ("n_wafers", "n_pareto", "n_removed_exponent", "fit_seconds")})
        print("smoke ok, 저장 안 함")
        return

    stamp = date.today().isoformat()
    out_csv = OUT_DIR / f"{stamp}_박종화_sr_pilot.csv"
    out_yaml = out_csv.with_suffix(".yaml")
    if out_csv.exists() or out_yaml.exists():
        sys.exit(f"이미 있어서 멈춘다: {out_csv}")

    rows = []
    for niter in (250, 500):
        rows.append(run_once(X, y, names, niter))
        print(rows[-1], flush=True)
    gain = improvement(rows[0], rows[1])
    decision = "keep_500"
    if gain >= MIN_IMPROVEMENT:
        rows.append(run_once(X, y, names, 1000))
        print(rows[-1], flush=True)
        gain_1000 = improvement(rows[1], rows[2])
        decision = "use_1000" if gain_1000 < MIN_IMPROVEMENT else "use_1000_may_not_have_converged"
    else:
        gain_1000 = None

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_yaml.write_text("\n".join([
        f"date: {stamp}",
        "author: 박종화",
        "script: src/sr_pilot.py",
        "input: data/processed/features_v1.csv + wafer_table.csv(MRR) + 2026-10-30_박종화_sr_candidates.csv",
        f"group: {GROUP}",
        f"seed: {SEED}",
        f"n_wafers_total: {n_total}",
        "comparison: 각 niterations를 따로 실행 (같은 실행의 중간 값 아님)",
        f"train_rmse: {GROUP} 전체에 맞춘 학습 loss의 제곱근, fold 평가 RMSE 아님",
        f"improvement_250_to_500: {gain:.6f}",
        f"improvement_500_to_1000: {'null' if gain_1000 is None else f'{gain_1000:.6f}'}",
        f"decision: {decision}",
        "exponent_rule: 탐색 중 차단 (loss_function_expression), 사후 제외는 검증용",
        f"blocked_loss: {BLOCKED_LOSS}",
        "equations: 출력·저장하지 않음",
        "",
    ]), encoding="utf-8")
    print(f"improvement_250_to_500={gain:.4%} decision={decision}")
    print(f"저장: {out_csv.name}, {out_yaml.name}")


if __name__ == "__main__":
    main()

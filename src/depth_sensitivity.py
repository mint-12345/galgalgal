"""트리 깊이 민감도 요약 (preregistration.md 3.2절, 6절 "트리 깊이 4 → 3, 6").

새로 학습하지 않는다. 아래 실행 결과를 읽어 깊이별로 나란히 놓는다.
- `src/a123_split_vs_pooled.py --depth {3,4,6}` → 분리 대 통합 결론이 깊이에 따라 바뀌는지
- `src/source_models.py --depth {3,4,6}` → 소스 모델(456 안 교차검증) 품질

실행: python src/depth_sensitivity.py [날짜]   (기본 오늘, 출력이 있으면 멈춘다)
출력: results/{날짜}_전준혁_depth_sensitivity.csv (+ .yaml)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cv  # noqa: E402

AUTHOR = "전준혁"
DEPTHS = [3, 4, 6]
ALPHA = 0.05


def suffix(d: int) -> str:
    return "" if d == 4 else f"_d{d}"


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    res = cv.ROOT / "results"
    out = res / f"{date.today().isoformat()}_{AUTHOR}_depth_sensitivity.csv"
    if out.exists() or out.with_suffix(".yaml").exists():
        raise FileExistsError(f"{out} 가 이미 있다")
    rows = []
    for d in DEPTHS:
        stem = f"{day}_{AUTHOR}_a123_split_vs_pooled{suffix(d)}"
        t = pd.read_csv(res / f"{stem}_tests.csv")
        t = t[(t["metric"] == "mse") & (t["subset"] == "all")].iloc[0]
        wf = pd.read_csv(res / f"{stem}_walkforward.csv").pivot(index="step", columns="model", values="mse")
        verdict = ("통합 우위" if t["mean_diff"] < 0 else "분리 우위") if t["p"] < ALPHA else "차이 없음(분리 유지)"
        rows.append({"analysis": "a123_split_vs_pooled", "max_depth": d,
                     "mse_separate": t["mean_separate"], "mse_pooled": t["mean_pooled"],
                     "mean_diff": t["mean_diff"], "t": t["t"], "p": t["p"], "verdict": verdict,
                     "wf_mse_separate": wf["separate"].mean(), "wf_mse_pooled": wf["pooled"].mean(),
                     "wf_wins_separate": int((wf["separate"] < wf["pooled"]).sum()),
                     "wf_wins_pooled": int((wf["pooled"] < wf["separate"]).sum())})
        sc = pd.read_csv(res / f"{day}_{AUTHOR}_source456_cv{suffix(d)}.csv")
        for (src, fs, g), x in sc.groupby(["source", "featureset", "group"]):
            rows.append({"analysis": "source456_cv", "max_depth": d, "source": src, "featureset": fs,
                         "group": g, "rmse": x["rmse"].mean(), "mse": x["mse"].mean()})
    tab = pd.DataFrame(rows)
    tab.round(6).to_csv(out, index=False, lineterminator="\n")
    a = tab[tab["analysis"] == "a123_split_vs_pooled"]
    stable = a["verdict"].nunique() == 1
    print(a.round(4).to_string())
    print(tab[tab["analysis"] == "source456_cv"].pivot_table(index=["source", "featureset", "group"],
                                                             columns="max_depth", values="rmse").round(3))
    print(f"분리 대 통합 결론이 깊이 3/4/6에서 같은가: {stable}")
    out.with_suffix(".yaml").write_text("\n".join([
        f"date: {date.today().isoformat()}", f"author: {AUTHOR}", "script: src/depth_sensitivity.py",
        "preregistration: v0.2.1 3.2절 민감도(트리 깊이 3/4/6), 6절",
        "inputs:",
        *[f"  - results/{day}_{AUTHOR}_a123_split_vs_pooled{suffix(d)}_tests.csv, _walkforward.csv" for d in DEPTHS],
        *[f"  - results/{day}_{AUTHOR}_source456_cv{suffix(d)}.csv" for d in DEPTHS],
        "changed: XGBoost max_depth만 (나머지는 3.2절 고정값)",
        "featureset: F-free (113) (+ group·STAGE one-hot for pooled) / source: F-free, F-free-usage",
        "retrained: 없음 (기존 결과 파일만 요약)",
        f"conclusion_stable_across_depths: {stable}",
        "conclusion: " + ", ".join(f"d{int(r.max_depth)} {r.verdict} (p = {r.p:.3f})" for r in a.itertuples()),
        "labels: train+val만. test·data/locked/ 사용 안 함", ""]), encoding="utf-8")
    print(f"저장: {out.relative_to(cv.ROOT)}")


if __name__ == "__main__":
    main()

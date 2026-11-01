"""보정 t-검정(Nadeau & Bengio 2003)의 n_train 해석 민감도 (진단, 사전확정 기준은 바꾸지 않는다).

전이학습(5.2·5.5절)은 각 fold의 A-123 학습 쪽 중 일부만 라벨을 공개한다. 보정 항 n_eval/n_train의 n_train을
- `full`: A-123 학습 쪽 전체(약 345장) — 5.7절 16번 "평가/학습 비율은 실제 fold 크기"의 해석,
  `cv_common.corrected_ttest`와 PR #20 `nb_test`가 쓰는 값
- `revealed`: 그 fold의 공개 라벨 수(8%면 약 28장) — 모델이 실제로 라벨을 본 수
로 넣을 때 p값이 얼마나 달라지는지 본다.

1. 재검증: `corrected_ttest`를 독립 계산식과 비교. 비율을 fold별 비의 평균(`cv_common`)으로 구할 때와
   평균 크기의 비(PR #20 `nb_test`)로 구할 때의 차이. #8 결과로 두 방식의 p 비교.
2. 해석 표(결과 없이 계산 가능): 공개 비율 2/4/8/16/100%마다 두 해석의 보정 비율, 표준오차 배율,
   `full` 기준 t가 얼마 이상이어야 `revealed` 기준으로도 p < 0.05인지, `full` p = 0.05/0.01/0.001일 때 `revealed` p.
3. `--transfer 경로`: 전이학습 fold별 결과(PR #20 `a123_transfer.py`의 주 csv 형식:
   ratio, r, source, mode, model, subset, mse, n_public, n_train, n_eval)가 있으면 실제 비교마다 두 해석의 p와
   판정이 달라지는지 계산. 이 스크립트는 전이학습을 실행하지 않는다.

실행: python src/ttest_ntrain_sensitivity.py [--transfer results/..._a123_transfer.csv]   (출력이 있으면 멈춘다)
출력: results/diagnostics/{날짜}_전준혁_ttest_ntrain_sensitivity.csv (+ .yaml), --transfer면 _transfer.csv 추가
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cv  # noqa: E402

AUTHOR = "전준혁"
J = 25
ALPHA = 0.05
N_A123 = 431


def fold_sizes() -> pd.DataFrame:
    """공개 마스크(주 평가, 고정주기)에서 fold·비율별 A-123 학습 쪽 장수와 공개 장수. 라벨 미사용."""
    m = pd.read_csv(cv.PROCESSED / "reveal_masks_v1.csv")
    m = m[(m["design"] == "folds") & (m["method"] == "fixed")]
    s = m.groupby(["ratio", "r"])[["n_train", "n_reveal"]].first().reset_index()
    s["n_eval"] = N_A123 - s["n_train"]
    assert (s.groupby("ratio").size() == J).all()
    return s


def verify(sizes: pd.DataFrame) -> list[dict]:
    rows = []
    s = sizes[sizes["ratio"] == 8].sort_values("r")
    rng = np.random.default_rng(cv.SEED)
    d = rng.normal(-0.3, 1.0, J)
    got = cv.corrected_ttest(d, s["n_train"], s["n_eval"])
    ratio = float((s["n_eval"] / s["n_train"]).mean())
    t_ref = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
    rows.append({"kind": "verify", "item": "corrected_ttest = 독립 계산식 (무작위 차이, A-123 fold 크기)",
                 "value_a": got["t"], "value_b": t_ref, "abs_diff": abs(got["t"] - t_ref)})
    r_pr20 = float(s["n_eval"].mean() / s["n_train"].mean())
    rows.append({"kind": "verify", "item": "보정 비율: fold별 비의 평균(cv_common) 대 평균 크기의 비(PR #20)",
                 "value_a": ratio, "value_b": r_pr20, "abs_diff": abs(ratio - r_pr20)})
    t8 = pd.read_csv(cv.ROOT / "results" / "2026-10-31_전준혁_a123_split_vs_pooled.csv")
    t8 = t8.pivot(index="fold", columns="model", values="mse").join(
        t8[t8["model"] == "separate"].set_index("fold")[["n_train", "n_eval"]])
    a = cv.corrected_ttest(t8["pooled"] - t8["separate"], t8["n_train"], t8["n_eval"])
    b = cv.corrected_ttest(t8["pooled"] - t8["separate"], t8["n_train"].mean(),
                           t8["n_eval"].mean())
    rows.append({"kind": "verify", "item": "#8 p: fold별 비 평균 대 평균 크기의 비",
                 "value_a": a["p"], "value_b": b["p"], "abs_diff": abs(a["p"] - b["p"])})
    return rows


def analytic(sizes: pd.DataFrame) -> list[dict]:
    rows = []
    t_crit = stats.t.ppf(1 - ALPHA / 2, J - 1)
    for ratio, s in sizes.groupby("ratio"):
        r_full = float((s["n_eval"] / s["n_train"]).mean())
        r_rev = float((s["n_eval"] / s["n_reveal"]).mean())
        factor = float(np.sqrt((1 / J + r_rev) / (1 / J + r_full)))   # revealed SE / full SE
        rec = {"kind": "analytic", "reveal_pct": int(ratio), "n_train_mean": s["n_train"].mean(),
               "n_reveal_mean": s["n_reveal"].mean(), "n_reveal_min": int(s["n_reveal"].min()),
               "n_reveal_max": int(s["n_reveal"].max()), "n_eval_mean": s["n_eval"].mean(),
               "ratio_full": r_full, "ratio_revealed": r_rev, "se_factor": factor,
               "t_full_needed_for_revealed_sig": t_crit * factor}
        for p_full in (0.05, 0.01, 0.001):
            t_full = stats.t.ppf(1 - p_full / 2, J - 1)
            rec[f"p_revealed_if_p_full_{p_full}"] = float(2 * stats.t.sf(t_full / factor, J - 1))
        rows.append(rec)
    return rows


def apply_transfer(path: Path) -> pd.DataFrame:
    R = pd.read_csv(path)
    need = {"ratio", "r", "source", "mode", "model", "subset", "mse", "n_public", "n_train", "n_eval"}
    if missing := need - set(R.columns):
        raise ValueError(f"{path.name}: 열 없음 {sorted(missing)}")
    A = R[R["subset"] == "all"]
    out = []
    for (ratio, src, mode), g in A.groupby(["ratio", "source", "mode"]):
        P = g.pivot(index="r", columns="model", values="mse")
        sz = g.drop_duplicates("r").set_index("r").loc[P.index]
        for a, b in (("T3", "T1"), ("T3", "T2"), ("T2", "T1")):
            res = cv.ttest_ntrain_sensitivity(P[a] - P[b], sz["n_eval"], sz["n_train"], sz["n_public"], ALPHA)
            w = res.set_index("n_train_basis")
            out.append({"ratio": ratio, "source": src, "mode": mode, "compare": f"{a}-{b}",
                        "mean_diff": w.loc["full", "mean_diff"],
                        "p_full": w.loc["full", "p"], "p_revealed": w.loc["revealed", "p"],
                        "ratio_full": w.loc["full", "ratio"], "ratio_revealed": w.loc["revealed", "ratio"],
                        "sig_full": w.loc["full", "significant"], "sig_revealed": w.loc["revealed", "significant"],
                        "verdict_changes": bool(w.loc["full", "significant"] != w.loc["revealed", "significant"])})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", type=Path, default=None, help="전이학습 fold별 결과 csv (PR #20 형식)")
    args = ap.parse_args()
    today = date.today().isoformat()
    out = cv.ROOT / "results" / "diagnostics" / f"{today}_{AUTHOR}_ttest_ntrain_sensitivity.csv"
    tr_out = out.with_name(f"{out.stem}_transfer.csv")
    if args.transfer is None and (out.exists() or out.with_suffix(".yaml").exists()):
        raise FileExistsError(f"{out} 가 이미 있다 (덮어쓰지 않음)")
    if args.transfer is not None and tr_out.exists():
        raise FileExistsError(f"{tr_out} 가 이미 있다 (덮어쓰지 않음)")

    sizes = fold_sizes()
    if args.transfer is not None:
        tr = apply_transfer(cv.ROOT / args.transfer if not args.transfer.is_absolute() else args.transfer)
        tr.round(6).to_csv(tr_out, index=False, lineterminator="\n")
        print(tr.round(4).to_string(index=False))
        print(f"판정이 바뀌는 비교: {int(tr['verdict_changes'].sum())}개 / {len(tr)}개. 저장: {tr_out.relative_to(cv.ROOT)}")
        return

    tab = pd.DataFrame(verify(sizes) + analytic(sizes))
    tab.round(6).to_csv(out, index=False, lineterminator="\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[tab["kind"] == "verify"].dropna(axis=1, how="all").to_string(index=False))
        print(tab[tab["kind"] == "analytic"].dropna(axis=1, how="all").round(4).to_string(index=False))
    a8 = tab[(tab["kind"] == "analytic") & (tab["reveal_pct"] == 8)].iloc[0]
    out.with_suffix(".yaml").write_text("\n".join([
        f"date: {today}", f"author: {AUTHOR}", "script: src/ttest_ntrain_sensitivity.py",
        "purpose: 보정 t-검정 n_train 해석(A-123 학습 쪽 전체 대 공개 라벨 수) 민감도. 사전확정 기준은 바꾸지 않음",
        "preregistration: v0.2.1 5.5절 주 판정, 5.7절 16번 (평가/학습 비율은 실제 fold 크기)",
        "input: data/processed/reveal_masks_v1.csv (design=folds, method=fixed; n_train, n_reveal만, 라벨 미사용),"
        " results/2026-10-31_전준혁_a123_split_vs_pooled.csv (재검증 예시)",
        f"formula: t = mean(d) / sqrt((1/J + n_eval/n_train) var(d)), J = {J}, df = {J - 1}, 양측",
        "result:",
        f"  ratio_8pct: full {a8.ratio_full:.3f}, revealed {a8.ratio_revealed:.3f}",
        f"  se_factor_8pct: {a8.se_factor:.3f} (revealed SE / full SE)",
        f"  t_full_needed_8pct: {a8.t_full_needed_for_revealed_sig:.3f}"
        f" (full 기준 |t|가 이 값 이상이어야 revealed 기준으로도 p < {ALPHA})",
        f"  p_revealed_when_p_full_0.05_8pct: {a8['p_revealed_if_p_full_0.05']:.3f}",
        "note: 실제 전이학습 결과의 판정 변화는 --transfer로 계산 (전이학습 결과 파일이 나온 뒤)", ""]),
        encoding="utf-8")
    print(f"저장: {out.relative_to(cv.ROOT)}")


if __name__ == "__main__":
    main()

"""A-123 분리 학습 대 통합 학습 판정 (preregistration.md 3.2절 #17, 민감도는 트리 깊이 3·4·6).

- 분리: A-123 학습 쪽만으로 XGBoost(F-free 113, 범주형 없음).
- 통합: 세 그룹 학습 쪽 전체로 XGBoost(F-free 113 + group·STAGE one-hot).
- 두 모델 모두 A-123 평가 쪽으로만 채점. 하이퍼파라미터는 3.2절 고정값(깊이만 `--depth`로 바꿈).
- 주 평가: folds_v1.csv 25 fold, 지표 MSE, 보정 t-검정(`cv_common.corrected_ttest`), 차이 = 통합 − 분리.
  보정 비율 n_eval/n_train은 A-123 평가 쪽 / A-123 학습 쪽(fold 분할 단위, 약 0.25).
  판정: p < 0.05면 평균이 낮은 쪽이 우위. 유의하지 않으면 기본값(분리) 유지.
- 보조 평가: 전진 검증 5단계(walkforward_v1.csv, 앞 50%→다음 10% … 90%→마지막 10%).
  A-123 학습 = 그 단계 role == train(라벨 전부 사용), 평가 = role == eval.
  통합 모델의 456은 t_start < eval_start(walkforward_cutoffs_v1.csv)인 웨이퍼만(5.2절 #21과 같은 경계).
  단계별 표와 이긴 단계 수를 보고. 주 평가와 결론이 다르면 그대로 적고 기본값(분리) 유지.

실행: python src/a123_split_vs_pooled.py [--depth 4]   (출력이 있으면 멈춘다)
출력: results/{날짜}_전준혁_a123_split_vs_pooled[_d{깊이}].csv (+ _walkforward.csv, _tests.csv, .yaml)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cv  # noqa: E402
import model_zoo as mz  # noqa: E402

AUTHOR = "전준혁"
TARGET = "A-123"
ALPHA = 0.05


def is_target(df: pd.DataFrame) -> pd.Series:
    return df["group"] == TARGET


def fit_predict_pair(depth: int):
    cols = cv.feature_columns("full")
    return {"separate": mz.make_fit_predict("xgb", cols, None, depth),
            "pooled": mz.make_fit_predict("xgb", cols, "onehot", depth)}


def run_folds(data: pd.DataFrame, depth: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fps = fit_predict_pair(depth)
    scores, preds = [], []
    for name, fp in fps.items():
        tf = is_target if name == "separate" else None
        s, p = cv.cv_evaluate(data, fp, train_filter=tf, eval_filter=is_target, report_groups=False)
        if name == "separate":
            assert (s["n_train"] + s["n_eval"] == 431).all()
        scores.append(s.assign(model=name))
        preds.append(p.assign(model=name))
    return pd.concat(scores, ignore_index=True), pd.concat(preds, ignore_index=True)


def split_allzero(preds: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """보고용: fold별 MSE를 센서 정상(allzero = 0) / 전체-0(allzero = 1) 웨이퍼로 나눔."""
    p = preds.merge(data[cv.KEY + ["allzero"]], on=cv.KEY, how="left", validate="m:1")
    p["se"] = (p["y"] - p["pred"]) ** 2
    return (p.groupby(["model", "r_id", "allzero"])["se"].agg(["mean", "size"])
            .rename(columns={"mean": "mse", "size": "n_eval"}).reset_index())


def run_walkforward(data: pd.DataFrame, depth: int) -> pd.DataFrame:
    walk = pd.read_csv(cv.PROCESSED / "walkforward_v1.csv")
    cuts = pd.read_csv(cv.PROCESSED / "walkforward_cutoffs_v1.csv").set_index("step")
    fps = fit_predict_pair(depth)
    rows = []
    for step in sorted(walk["step"].unique()):
        w = data.merge(walk[walk["step"] == step][cv.KEY + ["role"]], on=cv.KEY, how="left", validate="1:1")
        eval_start = cuts.loc[step, "eval_start"]
        tgt_tr = w[w["role"] == "train"]
        ev = w[w["role"] == "eval"]
        src = w[(w["group"] != TARGET) & (w["t_start"] < eval_start)]
        if len(tgt_tr) != cuts.loc[step, "n_train"] or len(ev) != cuts.loc[step, "n_eval"]:
            raise ValueError(f"단계 {step}: 학습·평가 장수가 cutoffs 파일과 다르다")
        for name, fp in fps.items():
            tr = tgt_tr if name == "separate" else pd.concat([tgt_tr, src])
            if not (tr["t_start"] < eval_start).all():
                raise ValueError(f"단계 {step} {name}: eval_start 이후 웨이퍼가 학습에 있다")
            cv.check_no_leak(w, w.index.get_indexer(tr.index), w.index.get_indexer(ev.index))
            p = fp(tr, ev)
            rows.append({"step": int(step), "model": name, "n_train": len(tr), "n_train_a123": len(tgt_tr),
                         "n_train_456": len(tr) - len(tgt_tr), "n_eval": len(ev),
                         **cv.metrics(ev["MRR"].to_numpy(float), p)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()
    depth = args.depth
    today = date.today().isoformat()
    stem = f"{today}_{AUTHOR}_a123_split_vs_pooled" + ("" if depth == 4 else f"_d{depth}")
    out = cv.ROOT / "results" / f"{stem}.csv"
    wf_out = out.with_name(f"{stem}_walkforward.csv")
    t_out = out.with_name(f"{stem}_tests.csv")
    for p in (out, wf_out, t_out, out.with_suffix(".yaml")):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (덮어쓰지 않음)")

    data = cv.load_dataset()
    scores, preds = run_folds(data, depth)
    sep = scores[scores["model"] == "separate"].set_index("r_id").sort_index()
    poo = scores[scores["model"] == "pooled"].set_index("r_id").sort_index()
    assert (sep["n_eval"] == poo["n_eval"]).all()

    tests = []
    for metric in ("mse", "rmse", "mae"):
        t = cv.corrected_ttest(poo[metric] - sep[metric], sep["n_train"], sep["n_eval"])
        tests.append({"comparison": "pooled - separate", "metric": metric, "subset": "all",
                      "mean_pooled": poo[metric].mean(), "mean_separate": sep[metric].mean(), **t})
    az = split_allzero(preds, data)
    for flag in (0, 1):
        a = az[(az["allzero"] == flag)].pivot(index="r_id", columns="model", values="mse")
        n = az[(az["allzero"] == flag) & (az["model"] == "separate")].set_index("r_id")["n_eval"]
        t = cv.corrected_ttest(a["pooled"] - a["separate"], sep["n_train"], sep["n_eval"])
        tests.append({"comparison": "pooled - separate", "metric": "mse",
                      "subset": f"allzero={flag} (보고만)", "mean_pooled": a["pooled"].mean(),
                      "mean_separate": a["separate"].mean(), "n_eval_mean": n.mean(), **t})
    tests = pd.DataFrame(tests)
    main_t = tests.iloc[0]
    if main_t["p"] < ALPHA:
        verdict = "통합 우위" if main_t["mean_diff"] < 0 else "분리 우위"
    else:
        verdict = "유의한 차이 없음 → 기본값(분리) 유지"

    wf = run_walkforward(data, depth)
    wfp = wf.pivot(index="step", columns="model", values="mse")
    sep_wins = int((wfp["separate"] < wfp["pooled"]).sum())
    poo_wins = int((wfp["pooled"] < wfp["separate"]).sum())
    if main_t["p"] < ALPHA and main_t["mean_diff"] < 0:
        flipped = poo_wins < 3
    elif main_t["p"] < ALPHA:
        flipped = sep_wins < 3
    else:
        flipped = None   # 주 평가가 유의하지 않으면 뒤집힐 결론이 없다

    # fold = 0~24 (r_id). 반복 안 번호(fold 0~4)는 r_id와 repeat로 정해지므로 뺀다 (열 이름 중복 방지)
    res = scores.drop(columns="fold").rename(columns={"r_id": "fold"}).assign(group=TARGET, featureset="F-free(+cat for pooled)",
                                                         max_depth=depth)
    res = res[["group", "model", "featureset", "max_depth", "fold", "repeat", "mse", "mae", "rmse",
               "n_train", "n_eval"]]
    res.round(6).to_csv(out, index=False, lineterminator="\n")
    wf.assign(group=TARGET, featureset="F-free(+cat for pooled)", max_depth=depth).round(6).to_csv(
        wf_out, index=False, lineterminator="\n")
    tests.assign(max_depth=depth).round(6).to_csv(t_out, index=False, lineterminator="\n")

    print(f"[깊이 {depth}] 주 평가 (25 fold, A-123 평가 쪽)")
    print(tests.round(4).to_string())
    print(f"판정: {verdict}")
    print("전진 검증:")
    print(wf.round(3).to_string())
    print(f"분리 승 {sep_wins} / 통합 승 {poo_wins}, 주 평가 결론 뒤집힘: {flipped}")

    lines = [
        f"date: {today}", f"author: {AUTHOR}", "script: src/a123_split_vs_pooled.py",
        f"command: python src/a123_split_vs_pooled.py --depth {depth}",
        "preregistration: v0.2.1 3.2절 #17 (분리 대 통합), 민감도 트리 깊이 3/4/6",
        "input: data/processed/features_v1.csv + wafer_table.csv (MRR, t_start; train+val, is_outlier=0)",
        "folds: data/processed/folds_v1.csv 25 fold (주 평가)",
        "walkforward: data/processed/walkforward_v1.csv + walkforward_cutoffs_v1.csv (보조 평가 5단계)",
        "models:",
        f"  separate: A-123 학습 쪽만, F-free 113, {mz.params_text('xgb', depth)}",
        f"  pooled: 세 그룹 학습 쪽 전체, F-free 113 + group·STAGE one-hot(고정 수준), {mz.params_text('xgb', depth)}",
        "eval: A-123 평가 쪽만",
        "test: 보정 t-검정(Nadeau & Bengio 2003), 차이 = pooled − separate, J = 25, df = 24,"
        " 비율 = fold별 (A-123 평가 쪽 / A-123 학습 쪽) 평균",
        f"rule: p < {ALPHA}이면 평균이 낮은 쪽 우위, 아니면 기본값(분리) 유지. allzero 분해는 보고만",
        "walkforward_rule: 통합 456은 t_start < eval_start만. A-123 학습 라벨은 그 단계 train 전부",
        "flip_rule: 주 평가에서 유의하게 이긴 쪽이 전진 검증 5단계 중 3단계 미만에서 이기면 '뒤집힘'"
        " (5.5절 보조 판정 규칙을 준용한 해석, 3.2절에는 단계 수 기준이 명시되지 않음)",
        "labels: train+val만. test·data/locked/ 사용 안 함",
        "result:",
        f"  mse_separate_mean: {main_t['mean_separate']:.4f}",
        f"  mse_pooled_mean: {main_t['mean_pooled']:.4f}",
        f"  mean_diff_pooled_minus_separate: {main_t['mean_diff']:.4f}",
        f"  t: {main_t['t']:.4f}", f"  df: {int(main_t['df'])}", f"  p: {main_t['p']:.6f}",
        f"  ratio: {main_t['ratio']:.4f}",
        f"  verdict: {verdict}",
        f"  walkforward_wins: {{separate: {sep_wins}, pooled: {poo_wins}}}",
        f"  walkforward_flipped: {flipped}",
    ]
    out.with_suffix(".yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out.relative_to(cv.ROOT)} 외 3개")


if __name__ == "__main__":
    main()

"""XGBoost 스레드 수(n_jobs)에 따른 결과 차이 점검 (진단, 하이퍼파라미터 탐색 아님).

배경: 3.2절 고정 설정은 n_jobs를 정하지 않는다. xgboost 3.2.0에서 subsample < 1이면 행 표집이
스레드 수에 따라 달라져, 같은 데이터·피처·하이퍼파라미터라도 n_jobs가 다르면 예측값이 달라진다.
- 전준혁 #7 `source_preds_v1.csv`: n_jobs = 4 (`cv_common.XGB_PARAMS`)
- 성수빈 PR #20 `src/a123_transfer.py` `fit_sources()`: n_jobs = 2
  → `source_models.fit_predict`를 n_jobs = 2로 돌리면 PR #20 예측과 4종 모두 비트 단위로 같다
    (2026-11-01, PR #20 커밋 ece0357로 확인. PR #20은 main에 없어 이 스크립트는 그 코드를 읽지 않는다).

점검 (n_jobs = 1, 2, 4, 8):
1. 소스 모델 4종의 A-123 예측값(주 평가용, 456 전체 학습)이 n_jobs = 4 대비 얼마나 다른가. A-123 라벨은 읽지 않는다.
2. 소스 모델 품질(456 안 25 fold 교차검증 RMSE)이 달라지는가
3. #8 분리 대 통합 주 평가(25 fold, 보정 t-검정)의 판정이 달라지는가

실행: python src/xgb_thread_check.py   (출력이 있으면 멈춘다)
출력: results/diagnostics/{날짜}_전준혁_xgb_thread_check.csv (+ .yaml)
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a123_split_vs_pooled as sp  # noqa: E402
import cv_common as cv  # noqa: E402
import source_models as sm  # noqa: E402

AUTHOR = "전준혁"
THREADS = [1, 2, 4, 8]
BASE = 4          # source_preds_v1, #6~#9 결과의 n_jobs
ALPHA = 0.05


@contextmanager
def n_jobs(k: int):
    old = cv.XGB_PARAMS["n_jobs"]
    cv.XGB_PARAMS["n_jobs"] = k
    try:
        yield
    finally:
        cv.XGB_PARAMS["n_jobs"] = old


def source_preds(data: pd.DataFrame) -> dict:
    tgt = data[data["group"] == sm.TARGET]
    out = {}
    for src_name, groups in sm.SOURCES.items():
        src = data[data["group"].isin(groups)]
        for fs, fs_label in sm.FEATURESETS.items():
            out[(src_name, fs_label)] = sm.fit_predict(src, tgt, cv.feature_columns(fs), 4)
    return out


def main() -> None:
    today = date.today().isoformat()
    out = cv.ROOT / "results" / "diagnostics" / f"{today}_{AUTHOR}_xgb_thread_check.csv"
    if out.exists() or out.with_suffix(".yaml").exists():
        raise FileExistsError(f"{out} 가 이미 있다 (덮어쓰지 않음)")
    data = cv.load_dataset()
    rows = []

    preds = {}
    for k in THREADS:
        with n_jobs(k):
            preds[k] = source_preds(data)
    v1 = pd.read_csv(cv.PROCESSED / "source_preds_v1.csv").query("design == 'folds'")
    for (src_name, fs_label), p4 in preds[BASE].items():
        f = v1[(v1["source"] == src_name) & (v1["featureset"] == fs_label)]
        assert np.allclose(f["pred"].to_numpy(), p4, rtol=0, atol=1e-5), "n_jobs = 4 재학습이 source_preds_v1과 다르다"
        for k in THREADS:
            d = np.abs(preds[k][(src_name, fs_label)] - p4)
            rows.append({"check": "source_pred_A123", "n_jobs": k, "source": src_name, "featureset": fs_label,
                         "max_abs_diff_vs4": d.max(), "mean_abs_diff_vs4": d.mean(),
                         "pred_sd": float(np.std(preds[k][(src_name, fs_label)]))})

    for k in THREADS:
        with n_jobs(k):
            for src_name, groups in sm.SOURCES.items():
                in_src = lambda df, g=groups: df["group"].isin(g)  # noqa: E731
                fp = lambda tr, ev, fd: sm.fit_predict(tr, ev, cv.feature_columns("full"), 4)  # noqa: E731
                s, _ = cv.cv_evaluate(data, fp, train_filter=in_src, eval_filter=in_src, report_groups=False)
                rows.append({"check": "source456_cv", "n_jobs": k, "source": src_name, "featureset": "F-free",
                             "rmse": s["rmse"].mean(), "mse": s["mse"].mean()})

    for k in THREADS:
        with n_jobs(k):
            scores, _ = sp.run_folds(data, 4)
        sep = scores[scores["model"] == "separate"].set_index("r_id").sort_index()
        poo = scores[scores["model"] == "pooled"].set_index("r_id").sort_index()
        t = cv.corrected_ttest(poo["mse"] - sep["mse"], sep["n_train"], sep["n_eval"])
        verdict = (("통합 우위" if t["mean_diff"] < 0 else "분리 우위") if t["p"] < ALPHA
                   else "차이 없음(분리 유지)")
        rows.append({"check": "a123_split_vs_pooled", "n_jobs": k, "mse_separate": sep["mse"].mean(),
                     "mse_pooled": poo["mse"].mean(), "mean_diff": t["mean_diff"], "t": t["t"], "p": t["p"],
                     "verdict": verdict})

    tab = pd.DataFrame(rows)
    tab.round(6).to_csv(out, index=False, lineterminator="\n")
    for c, g in tab.groupby("check", sort=False):
        print(g.dropna(axis=1, how="all").round(4).to_string(index=False))
    sp_rows = tab[tab["check"] == "a123_split_vs_pooled"]
    src_rows = tab[tab["check"] == "source_pred_A123"]
    stable = sp_rows["verdict"].nunique() == 1
    out.with_suffix(".yaml").write_text("\n".join([
        f"date: {today}", f"author: {AUTHOR}", "script: src/xgb_thread_check.py",
        "purpose: XGBoost n_jobs에 따른 재현성 점검 (튜닝 아님, 3.2절 고정값 그대로)",
        f"xgboost: {xgboost.__version__}",
        "params: " + ", ".join(f"{k}={v!r}" for k, v in cv.XGB_PARAMS.items() if k != "n_jobs"),
        f"n_jobs: {THREADS} (기준 {BASE} = source_preds_v1·#6~#9 결과)",
        "input: data/processed/features_v1.csv + wafer_table.csv (train+val, is_outlier=0), folds_v1.csv",
        "pr20_check: |",
        "  PR #20 (explore/a123-run, ece0357) fit_sources()는 같은 456 행·같은 피처 113/36개(같은 순서)·같은 3.2절 값,",
        "  n_jobs만 2. source_models.fit_predict를 n_jobs=2로 돌린 A-123 예측이 PR #20과 4종 모두 비트 단위로 같았다",
        "  (임시 비교 스크립트로 확인, A-123 라벨 미사용). n_jobs=4인 source_preds_v1과는 최대 1.79~2.91 차이.",
        "result:",
        f"  source_pred_max_abs_diff_vs4: {src_rows['max_abs_diff_vs4'].max():.4f}",
        f"  source_pred_mean_abs_diff_vs4: {src_rows['mean_abs_diff_vs4'].max():.4f} (조합별 최댓값)",
        f"  split_vs_pooled_verdict_stable: {stable}",
        "  split_vs_pooled_p: " + ", ".join(f"n_jobs {int(r.n_jobs)} p = {r.p:.3f}" for r in sp_rows.itertuples()),
        "labels: train+val만. 소스 예측 비교에는 A-123 라벨 미사용. test·data/locked/ 사용 안 함", ""]),
        encoding="utf-8")
    print(f"분리 대 통합 판정이 n_jobs 1/2/4/8에서 같은가: {stable}")
    print(f"저장: {out.relative_to(cv.ROOT)}")


if __name__ == "__main__":
    main()

"""A-123 전이학습 T2(f_456 + c)가 나쁜 원인 분해 — 공개 8%, 사후 분석 (진단 CLAUDE.md 14.4).

판정(PR #25)이 끝난 뒤 원인을 보는 분석이다. 판정에는 쓰지 않는다.

src/a123_transfer.py를 고치지 않고 import해서 본 실행과 같은 설정(load, fit_sources, split, run_fold; xgboost 3.2.0,
f_456 n_jobs=2, GP 시드 42+r)으로 공개 8% × 25 fold를 다시 돌린다.
- 재현 확인: run_fold의 fold별 mse를 본 실행 folds.csv(8%)와 비교해 최대 |차이|를 출력한다.
- 평가 웨이퍼를 센서 정상(allzero=0) / 전체-0(allzero=1) / 전체로 나눠 S1·S2 × full·usage별로 기록:
  T0~T3 RMSE (run_fold 출력), f_456 예측 평균·표준편차, 실제 MRR 평균·표준편차, 피어슨·스피어만 상관,
  T2 오차 분해: RMSE² = 편향² + 오차 표준편차²(ddof=0), 편향 = mean(MRR − T2).
  오프셋 c(공개 라벨로 추정한 값)와 그 부분집합에 맞는 오프셋 mean(MRR − f_456)도 함께 둔다.
- f_456은 456 전체로 1회 학습해 웨이퍼 예측이 fold와 무관하므로, A-123 431장 전체 기준 분포·상관도 같이 둔다.

행 종류(row_type): fold(25 fold × 4 소스·입력 × 3 부분집합), fold_mean(fold 평균), all_wafers(431장 기준, RMSE 없음).
라벨 파일·test·data/locked/는 읽지 않는다 (A-123 MRR은 a123_transfer.load와 같이 wafer_table.csv에서).
실행: OPENBLAS_NUM_THREADS=1 python src/diagnostics/t2_decomposition_a123.py
출력: results/diagnostics/2026-11-01_김혜연_t2_decomposition.{csv,yaml} (있으면 멈춤)
"""
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a123_transfer as at  # noqa: E402

DATE = "2026-11-01"
AUTHOR = "김혜연"
OUT = at.ROOT / "results" / "diagnostics" / f"{DATE}_{AUTHOR}_t2_decomposition.csv"
MAIN_FOLDS = at.RESULTS / f"{DATE}_{AUTHOR}_FV1_a123_transfer_folds.csv"
RATIO = 8
SUBSETS = {"clean": 0, "zero": 1, "all": None}


def dist_stats(f, y):
    rec = {"n": len(y), "f_mean": f.mean(), "f_sd": f.std(ddof=1), "y_mean": y.mean(), "y_sd": y.std(ddof=1),
           "ideal_offset": float(np.mean(y - f))}
    ok = len(y) >= 3 and f.std() > 0 and y.std() > 0
    rec["pearson"] = pearsonr(f, y)[0] if ok else np.nan
    rec["spearman"] = spearmanr(f, y)[0] if ok else np.nan
    return rec


def main():
    if OUT.exists() or OUT.with_suffix(".yaml").exists():
        sys.exit(f"이미 있음 (덮어쓰기 금지): {OUT.name}")
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "OPENBLAS_NUM_THREADS=1로 실행"
    if at.xgboost.__version__ != at.XGB_VERSION:
        sys.exit(f"xgboost {at.xgboost.__version__}: {at.XGB_VERSION} 필요")
    t0 = time.time()
    started = dt.datetime.now()
    git = at.git_state()
    data, a, fa, masks, cols = at.load()
    at.MODEL_COLS.clear()
    at.MODEL_COLS.update(cols)
    models = at.fit_sources(data, cols)

    rows, rec = [], []
    for r in range(25):
        tr, ev, pub = at.split(a, fa, masks, r, RATIO)
        out = at.run_fold(tr, ev, pub, models, r)
        for x in out:
            x.update(ratio=RATIO, r=r)
        rec += out
        O = pd.DataFrame(out).pivot_table(index=["source", "input", "subset"], columns="model", values="rmse")
        y_pub, y_ev = pub["MRR"].to_numpy(float), ev["MRR"].to_numpy(float)
        for (s, m), mdl in models.items():
            fp = mdl.predict(pub[cols[m]].to_numpy(float))
            fe = mdl.predict(ev[cols[m]].to_numpy(float))
            c = float(np.mean(y_pub - fp))
            t2 = fe + c
            for sub, flag in SUBSETS.items():
                msk = np.ones(len(ev), bool) if flag is None else (ev["allzero"].to_numpy() == flag)
                if msk.sum() == 0:
                    continue
                e = y_ev[msk] - t2[msk]
                row = {"row_type": "fold", "r": r, "source": s, "input": m, "subset": sub,
                       **{f"rmse_{k}": O.loc[(s, m, sub), k] for k in ["T0", "T1", "T2", "T3"]},
                       **dist_stats(fe[msk], y_ev[msk]),
                       "offset_c": c, "n_public": len(pub), "n_public_clean": int((pub["allzero"] == 0).sum()),
                       "t2_bias": float(e.mean()), "t2_err_sd": float(e.std(ddof=0))}
                assert np.isclose(row["rmse_T2"] ** 2, row["t2_bias"] ** 2 + row["t2_err_sd"] ** 2)
                rows.append(row)
    F = pd.DataFrame(rows)

    # 재현 확인 (값은 출력하지 않고 최대 |차이|만)
    M = pd.read_csv(MAIN_FOLDS)
    M = M[M["ratio"] == RATIO]
    keys = ["ratio", "r", "source", "input", "model", "subset"]
    J = M.merge(pd.DataFrame(rec), on=keys, suffixes=("_main", "_new"), validate="one_to_one")
    assert len(J) == len(M)
    repro = float(np.abs(J["mse_main"] - J["mse_new"]).max())
    print(f"재현 8%: fold별 mse 최대 |차이| {repro:.1e} ({len(J)}행)", flush=True)

    num = [c for c in F.columns if c not in ["row_type", "r", "source", "input", "subset"]]
    FM = F.groupby(["source", "input", "subset"], sort=False)[num].mean().reset_index()
    FM.insert(0, "row_type", "fold_mean")
    W = []
    for (s, m), mdl in models.items():
        f_all = mdl.predict(a[cols[m]].to_numpy(float))
        y_all = a["MRR"].to_numpy(float)
        for sub, flag in SUBSETS.items():
            msk = np.ones(len(a), bool) if flag is None else (a["allzero"].to_numpy() == flag)
            W.append({"row_type": "all_wafers", "source": s, "input": m, "subset": sub, **dist_stats(f_all[msk], y_all[msk])})
    R = pd.concat([F, FM, pd.DataFrame(W)], ignore_index=True)
    R.insert(1, "group", at.TARGET_GROUP)
    R.insert(2, "ratio", RATIO)
    R.to_csv(OUT, index=False, encoding="utf-8")

    info = {
        "experiment": "t2_decomposition_a123 (A-123 전이 T2 원인 분해, 공개 8%, 사후 분석, 14.4)",
        "note": "판정(PR #25) 이후 원인 확인용. 판정에 쓰지 않음",
        "author": AUTHOR, "group": at.TARGET_GROUP,
        "run_started": f"{started:%Y-%m-%d %H:%M:%S}", "runtime_sec": round(time.time() - t0),
        "git": git, "environment": {**at.environment(), "OPENBLAS_NUM_THREADS": "1"},
        "same_as_main": {"code": "src/a123_transfer.py (load, fit_sources, split, run_fold) import",
                         "f456_n_jobs": at.XGB_N_JOBS, "xgboost": at.XGB_VERSION, "gp_seed": at.SEED_RULE},
        "ratio": RATIO, "folds": "folds_v1 25", "reveal": "reveal_masks_v1 fixed",
        "subsets": "clean = allzero 0, zero = allzero 1, all = 전체",
        "row_types": "fold / fold_mean (25 fold 평균) / all_wafers (A-123 431장, f_456는 fold와 무관)",
        "t2_decomposition": "rmse_T2^2 = t2_bias^2 + t2_err_sd^2 (ddof=0), t2_bias = mean(MRR - T2)",
        "ideal_offset": "그 부분집합에서 mean(MRR - f_456). offset_c는 공개 라벨 전체로 추정한 T2의 c",
        "reproduction_max_abs_diff_mse": repro,
        "input_main": MAIN_FOLDS.name,
        "inputs_sha256_16": {p.name: at.sha(p) for p in [at.P_FEATURES, at.P_FAMILIES, at.P_FOLDS, at.P_MASKS, at.P_TABLE,
                                                          MAIN_FOLDS]},
        "output": OUT.name,
    }
    at.write_yaml(OUT.with_suffix(".yaml"), info)
    print("저장:", OUT.name, "+ .yaml", flush=True)


if __name__ == "__main__":
    main()

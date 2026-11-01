"""A-123 전이학습 GP 경고 종류 진단 (진단 CLAUDE.md 14.4, preregistration.md 5.7-15).

본 실행(PR #25)과 같은 설정으로 공개 2/4/8/16% × 25 fold의 T1·T3 GP를 다시 학습하고, 학습마다
경고 메시지 전문·종류와 최적화 후 하이퍼파라미터(진폭, 길이척도, 잡음), 경계에 붙었는지를 기록한다. 100%는 뺀다.

방법: src/a123_transfer.py를 고치지 않고 import한다. 모듈 안의 GaussianProcessRegressor만 기록용 하위 클래스로
바꿔 끼운다. 이 클래스는 학습 후 kernel_과 경고를 기록하고, 경고를 다시 내보내 fit_gp의 경고 건수가 본 실행과
같게 둔다. 분할·공개·δ 입력 규칙·f_456(n_jobs=2)·커널·시드(42+r)는 본 코드(load, fit_sources, split, run_fold) 그대로다.

경계에 붙음: 값 ≤ 하한 × 1.01 이면 하한, 값 ≥ 상한 × 0.99 이면 상한 (경계의 1% 이내).
normalize_y=True라 진폭·잡음은 표준화된 MRR 단위다.

재현 확인: folds.csv에는 예측값이 없어서, fold별 저장값(mse·rmse·mae·rho·b·coverage90·gp_warn_t1·gp_warn_t3)을
다시 계산한 값과 비교해 비율별 최대 |차이|만 출력한다. 오차·성능 수치 자체는 출력하지 않는다.

features·reveal mask·folds·wafer_table만 읽는다. 라벨 파일·test·data/locked/는 읽지 않는다.
실행: OPENBLAS_NUM_THREADS=1 python src/diagnostics/gp_warnings_a123.py
출력: results/diagnostics/2026-11-01_김혜연_gp_warnings.{csv,yaml} (있으면 멈춤)
"""
import datetime as dt
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a123_transfer as at  # noqa: E402

DATE = "2026-11-01"
AUTHOR = "김혜연"
DIAG = at.ROOT / "results" / "diagnostics"
OUT = DIAG / f"{DATE}_{AUTHOR}_gp_warnings.csv"
MAIN_FOLDS = at.RESULTS / f"{DATE}_{AUTHOR}_FV1_a123_transfer_folds.csv"
RATIOS = [2, 4, 8, 16]
BOUNDS = {"amplitude": (1e-2, 1e2), "length_scale": (1e-2, 1e2), "noise": (1e-3, 10.0)}  # a123_transfer.fit_gp와 같음
NAME_KO = {"amplitude": "진폭", "length_scale": "길이척도", "noise": "잡음"}
REL = 0.01
CHECK_COLS = ["mse", "rmse", "mae", "rho", "b", "coverage90", "gp_warn_t1", "gp_warn_t3"]
CATS = (ConvergenceWarning, UserWarning, RuntimeWarning)   # fit_gp가 세는 종류

LOG = []        # 학습마다 1행
CTX = {}        # 지금 학습 중인 fold


class RecordingGPR(GaussianProcessRegressor):
    def fit(self, X, y):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = super().fit(X, y)
        LOG.append({**CTX, "kernel_": self.kernel_, "warns": list(w), "n_dim": X.shape[1]})
        for x in w:   # fit_gp의 기록에도 그대로 들어가도록 다시 낸다
            warnings.warn_explicit(x.message, x.category, x.filename, x.lineno)
        return out


def bound_side(v, kind):
    lo, hi = BOUNDS[kind]
    return "하한" if v <= lo * (1 + REL) else "상한" if v >= hi * (1 - REL) else ""


def msg_type(m):
    """숫자를 #로 바꿔 같은 종류끼리 묶는다."""
    return re.sub(r"\d+(\.\d+)?(e[-+]?\d+)?", "#", m).strip()


def to_rows(entry, cols):
    k = entry["kernel_"]
    amp, ls, noise = k.k1.k1.constant_value, np.atleast_1d(k.k1.k2.length_scale), k.k2.noise_level
    iso = entry["iso"]   # 공개 < 10장이면 등방성 커널 (fit_gp와 같은 기준)
    names = ["(등방성)"] if iso else cols
    assert len(names) == len(ls), (names, ls)
    hits = []
    row = {kk: entry[kk] for kk in ["ratio", "r", "model", "source", "input", "n_public", "n_public_clean"]}
    row.update(iso_kernel=int(iso), delta_cols=",".join(cols), amplitude=amp, amplitude_bound=bound_side(amp, "amplitude"),
               noise=noise, noise_bound=bound_side(noise, "noise"))
    for kind, v in [("amplitude", amp), ("noise", noise)]:
        if bound_side(v, kind):
            hits.append(f"{NAME_KO[kind]} {bound_side(v, kind)}")
    ls_d, ls_b = {}, {}
    for n, v in zip(names, ls):
        ls_d[n] = float(v)
        b = bound_side(v, "length_scale")
        if b:
            ls_b[n] = b
            hits.append(f"길이척도 {n} {b}")
    row.update(length_scales=json.dumps(ls_d, ensure_ascii=False), length_scale_bounds=json.dumps(ls_b, ensure_ascii=False),
               n_bounds_hit=len(hits), bounds_hit="; ".join(hits))
    ws = [x for x in entry["warns"] if issubclass(x.category, CATS)]
    row.update(n_warn=len(ws),
               n_convergence=sum(issubclass(x.category, ConvergenceWarning) for x in ws),
               n_user=sum(issubclass(x.category, UserWarning) and not issubclass(x.category, ConvergenceWarning) for x in ws),
               n_runtime=sum(issubclass(x.category, RuntimeWarning) for x in ws),
               warn_categories=" || ".join(x.category.__name__ for x in ws),
               warn_messages=" || ".join(str(x.message).replace("\n", " ") for x in ws),
               warn_types=" || ".join(msg_type(str(x.message).replace("\n", " ")) for x in ws))
    return row


def main():
    if OUT.exists() or OUT.with_suffix(".yaml").exists():
        sys.exit(f"이미 있음 (덮어쓰기 금지): {OUT.name}")
    import os
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "OPENBLAS_NUM_THREADS=1로 실행"
    if at.xgboost.__version__ != at.XGB_VERSION:
        sys.exit(f"xgboost {at.xgboost.__version__}: {at.XGB_VERSION} 필요")
    t0 = time.time()
    started = dt.datetime.now()
    git = at.git_state()
    at.GaussianProcessRegressor = RecordingGPR
    data, a, fa, masks, cols = at.load()
    at.MODEL_COLS.clear()
    at.MODEL_COLS.update(cols)
    models = at.fit_sources(data, cols)
    order = [("T1", "", "")] + [("T3", s, m) for (s, m) in models]   # run_fold 안의 fit_gp 호출 순서
    rows, rec = [], []
    for ratio in RATIOS:
        for r in range(25):
            tr, ev, pub = at.split(a, fa, masks, r, ratio)
            dcols = at.select_delta(tr, pub)
            n0 = len(LOG)
            CTX.update(ratio=ratio, r=r, n_public=len(pub), n_public_clean=int((pub["allzero"] == 0).sum()),
                       iso=len(pub) < at.ISO_BELOW)
            out = at.run_fold(tr, ev, pub, models, r)
            new = LOG[n0:]
            assert len(new) == len(order), (ratio, r, len(new))
            for e, (mdl, s, m) in zip(new, order):
                e.update(model=mdl, source=s, input=m)
                rows.append(to_rows(e, dcols))
            assert all(x["delta_cols"] == ",".join(dcols) for x in out)
            for x in out:
                x.update(ratio=ratio, r=r)
            rec += out
        print(f"공개 {ratio}% 완료 ({time.time() - t0:.0f}초)", flush=True)
    D = pd.DataFrame(rows)

    # 재현 확인: 값은 출력하지 않고 최대 |차이|만
    R = pd.DataFrame(rec)
    M = pd.read_csv(MAIN_FOLDS)
    M = M[M["ratio"].isin(RATIOS) & M["model"].isin(["T1", "T3"])]
    keys = ["ratio", "r", "source", "input", "model", "subset"]
    J = M.merge(R[R["model"].isin(["T1", "T3"])], on=keys, suffixes=("_main", "_new"), validate="one_to_one")
    assert len(J) == len(M), (len(J), len(M))
    repro = {}
    for ratio, g in J.groupby("ratio"):
        repro[int(ratio)] = {c: float(np.nanmax(np.abs(g[f"{c}_main"] - g[f"{c}_new"]))) for c in CHECK_COLS}
        print(f"재현 {ratio}%: 최대 |차이| " + ", ".join(f"{c} {v:.1e}" for c, v in repro[int(ratio)].items()), flush=True)
    # 학습 단위 경고 수도 본 실행과 맞는지 (T1은 fold당 1회)
    W = M[M["subset"] == "all"].drop_duplicates(["ratio", "r", "source", "input"])
    t1_main = W.drop_duplicates(["ratio", "r"]).set_index(["ratio", "r"])["gp_warn_t1"]
    t1_new = D[D["model"] == "T1"].set_index(["ratio", "r"])["n_warn"]
    t3_main = W.set_index(["ratio", "r", "source", "input"])["gp_warn_t3"]
    t3_new = D[D["model"] == "T3"].set_index(["ratio", "r", "source", "input"])["n_warn"]
    warn_match = bool((t1_main.sort_index() == t1_new.sort_index()).all() and (t3_main.sort_index() == t3_new.sort_index()).all())
    print("학습별 경고 건수가 본 실행과 같음:", warn_match, flush=True)

    D.insert(0, "group", at.TARGET_GROUP)
    D.to_csv(OUT, index=False, encoding="utf-8")

    # 요약 (yaml에 같이 남긴다)
    D["model_key"] = np.where(D["model"] == "T1", "T1", "T3 " + D["source"] + " " + D["input"])
    types = D.assign(t=D["warn_types"].str.split(r" \|\| ")).explode("t")
    types = types[types["t"].fillna("") != ""]
    hits = D.assign(h=D["bounds_hit"].str.split("; ")).explode("h")
    hits = hits[hits["h"].fillna("") != ""]
    info = {
        "experiment": "gp_warnings_a123 (A-123 전이학습 GP 경고 진단, 14.4)",
        "author": AUTHOR, "group": at.TARGET_GROUP,
        "run_started": f"{started:%Y-%m-%d %H:%M:%S}", "run_finished": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
        "runtime_sec": round(time.time() - t0),
        "git": git, "environment": {**at.environment(), "OPENBLAS_NUM_THREADS": "1"},
        "scope": {"ratios": RATIOS, "folds": "folds_v1 25", "models": "T1, T3 (S1·S2 × full·usage)", "fits": len(D)},
        "same_as_main": {"code": "src/a123_transfer.py (load, fit_sources, split, select_delta, run_fold, fit_gp) import",
                         "f456_n_jobs": at.XGB_N_JOBS, "xgboost": at.XGB_VERSION, "gp_seed": at.SEED_RULE},
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "bound_rule": "값 ≤ 하한×1.01 → 하한, 값 ≥ 상한×0.99 → 상한",
        "units": "normalize_y=True → 진폭·잡음은 표준화된 MRR 단위, 길이척도는 표준화 입력 단위",
        "reproduction_max_abs_diff": {str(k): v for k, v in repro.items()},
        "reproduction_note": "folds.csv에 예측값이 없어 fold별 저장값(mse·rmse·mae·rho·b·coverage90·경고 수)으로 비교",
        "warn_count_matches_main": warn_match,
        "warn_type_counts": {t: int(v) for t, v in types["t"].value_counts().items()},
        "bounds_hit_counts": {h: int(v) for h, v in hits["h"].value_counts().items()},
        "input_main": MAIN_FOLDS.name,
        "inputs_sha256_16": {p.name: at.sha(p) for p in [at.P_FEATURES, at.P_FAMILIES, at.P_FOLDS, at.P_MASKS, at.P_TABLE,
                                                          MAIN_FOLDS]},
        "output": OUT.name,
    }
    at.write_yaml(OUT.with_suffix(".yaml"), info)
    print("저장:", OUT.name, "+ .yaml", flush=True)


if __name__ == "__main__":
    main()

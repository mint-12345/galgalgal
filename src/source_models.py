"""전이학습 소스 모델 f_456의 A-123 예측값 (preregistration.md 3.2·5.1·5.2·5.4절).

검증·참고용 (2026-11-01): 전이학습의 공식 f_456는 PR #20 `src/a123_transfer.py`의 `fit_sources()`가 직접 학습한다.
이 스크립트와 학습 데이터·피처·하이퍼파라미터가 같고 `n_jobs`만 다르다(여기 4, PR #20은 2 → 예측이 다름).
`cv_common.XGB_PARAMS["n_jobs"] = 2`로 두고 실행하면 PR #20 예측과 비트 단위로 같다 (`src/xgb_thread_check.py` 참고).

- 모델: XGBoost, 3.2절 고정 하이퍼파라미터(`cv_common.XGB_PARAMS`). 튜닝하지 않는다.
- 소스 (5.1): S1 = A-456 / S2 = A-456 + B-456. 라벨은 전부 쓴다(456은 전부 공개 가정).
- 입력 (5.4 #27): `full` = F-free 113개 / `usage` = 소모품 카운터 36개 (`cv_common.feature_columns`).
  그룹·스테이지 범주형은 넣지 않는다 (F-free 정의에 없음, A-123은 456 소스에 없는 범주).
- 예측 대상: A-123 train+val 431장 전부. A-123 라벨은 읽지 않는다(학습·점검 모두).
- 두 가지 설계 (5.2 #21):
  1) `folds` (주 평가): 456 전체(train+val)로 학습한 모델 1개 → A-123 431장 예측.
     A-123 fold 분할과 무관하게 같은 예측값을 25개 fold 모두에서 쓴다.
  2) `walkforward` (보조 평가): 단계 s마다 `t_start < eval_start(s)`인 456 웨이퍼만으로 학습 → 431장 예측.
     A-123 행에 그 단계의 역할(train/eval/unused, `walkforward_v1.csv`)을 붙인다.
- 점검(456 안 교차검증): 같은 설정을 folds_v1 25개 fold에서 456 평가 쪽으로 채점한다(소스 모델 품질 확인용).

실행: python src/source_models.py [--depth 4] [--out data/processed/source_preds_v1.csv] [--skip-cv]
출력: data/processed/source_preds_v1.csv + .yaml, results/{날짜}_전준혁_source456_cv.csv + .yaml
      (출력 파일이 이미 있으면 멈춘다. CLAUDE.md 3절 3번)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cv  # noqa: E402

AUTHOR = "전준혁"
WALK = cv.PROCESSED / "walkforward_v1.csv"
CUTS = cv.PROCESSED / "walkforward_cutoffs_v1.csv"
SOURCES = {"S1": ["A-456"], "S2": ["A-456", "B-456"]}
FEATURESETS = {"full": "F-free", "usage": "F-free-usage"}
TARGET = "A-123"


def xgb_params(depth: int) -> dict:
    return {**cv.XGB_PARAMS, "max_depth": depth}


def fit_predict(src: pd.DataFrame, tgt: pd.DataFrame, cols: list[str], depth: int) -> np.ndarray:
    if (src["group"] == TARGET).any():
        raise ValueError("소스 학습 데이터에 A-123이 들어 있다")
    m = XGBRegressor(**xgb_params(depth)).fit(src[cols], src["MRR"])
    p = m.predict(tgt[cols]).astype(float)
    if not np.isfinite(p).all():
        raise ValueError("예측 누락(NaN)")
    return p


def build_preds(data: pd.DataFrame, depth: int) -> tuple[pd.DataFrame, list[dict]]:
    """두 설계 × 소스 2 × 피처 2 의 A-123 예측값(긴 표)과 학습 기록."""
    tgt = data[data["group"] == TARGET].reset_index(drop=True)
    tgt_key = tgt[cv.KEY]
    walk = pd.read_csv(WALK)
    cuts = pd.read_csv(CUTS)
    out, log = [], []
    for src_name, groups in SOURCES.items():
        for fs, fs_label in FEATURESETS.items():
            cols = cv.feature_columns(fs)
            base = dict(source=src_name, featureset=fs_label, model="xgb", max_depth=depth)
            # 1) 주 평가용: 456 전체
            src = data[data["group"].isin(groups)]
            p = fit_predict(src, tgt, cols, depth)
            out.append(tgt_key.assign(design="folds", step=-1, role="all", **base,
                                      n_source=len(src), pred=p))
            log.append({**base, "design": "folds", "step": -1, "n_source": len(src),
                        **{f"n_{g}": int((src["group"] == g).sum()) for g in groups},
                        "src_t_max": float(src["t_start"].max()), "eval_start": np.nan,
                        "n_features": len(cols)})
            # 2) 보조 평가용: 단계별 eval_start 이전 456만
            for _, c in cuts.iterrows():
                s = int(c["step"])
                src = data[data["group"].isin(groups) & (data["t_start"] < c["eval_start"])]
                if not (src["t_start"] < c["eval_start"]).all():
                    raise ValueError("미래 456 웨이퍼가 학습에 들어갔다")
                p = fit_predict(src, tgt, cols, depth)
                role = tgt_key.merge(walk[walk["step"] == s][cv.KEY + ["role"]], on=cv.KEY,
                                     how="left", validate="1:1")["role"]
                if role.isna().any():
                    raise ValueError(f"단계 {s}: walkforward_v1.csv에 없는 A-123 행")
                out.append(tgt_key.assign(design="walkforward", step=s, role=role.to_numpy(), **base,
                                          n_source=len(src), pred=p))
                log.append({**base, "design": "walkforward", "step": s, "n_source": len(src),
                            **{f"n_{g}": int((src["group"] == g).sum()) for g in groups},
                            "src_t_max": float(src["t_start"].max()),
                            "eval_start": float(c["eval_start"]), "n_features": len(cols)})
    cols_out = ["design", "step", "source", "featureset", "model", "max_depth", "role",
                "WAFER_ID", "STAGE", "n_source", "pred"]
    return pd.concat(out, ignore_index=True)[cols_out], log


def validate(preds: pd.DataFrame, log: list[dict], data: pd.DataFrame) -> list[str]:
    """예측 누락·중복·누수 점검. 실패하면 멈추고, 통과 항목 설명을 돌려준다."""
    a123 = data[data["group"] == TARGET]
    checks = []
    n_comb = preds.groupby(["design", "step", "source", "featureset"]).size()
    if len(n_comb) != 2 * 2 * 6 or not (n_comb == len(a123)).all():
        raise ValueError(f"조합별 행 수가 431이 아니다:\n{n_comb}")
    checks.append(f"조합 {len(n_comb)}개(설계·단계 6 × 소스 2 × 피처 2) × A-123 {len(a123)}장, 모두 채워짐")
    if preds.duplicated(["design", "step", "source", "featureset", "WAFER_ID", "STAGE"]).any():
        raise ValueError("키 중복")
    keys = set(map(tuple, a123[cv.KEY].to_numpy()))
    if set(map(tuple, preds[cv.KEY].drop_duplicates().to_numpy())) != keys:
        raise ValueError("예측 키가 A-123 431장과 다르다")
    checks.append("키 중복 0, 예측 키 = folds_v1의 A-123 431장")
    if preds["pred"].isna().any():
        raise ValueError("예측 NaN")
    checks.append("예측 NaN 0")
    lg = pd.DataFrame(log)
    w = lg[lg["design"] == "walkforward"]
    if not (w["src_t_max"] < w["eval_start"]).all():
        raise ValueError("전진 검증 소스에 eval_start 이후 456 웨이퍼가 있다")
    checks.append("전진 검증: 모든 단계에서 소스 456의 최대 t_start < eval_start")
    full_n = {"S1": 983, "S2": 983 + 987}
    f = lg[lg["design"] == "folds"]
    if not all(f["n_source"] == f["source"].map(full_n)):
        raise ValueError("주 평가 소스 행 수가 456 전체와 다르다")
    checks.append("주 평가 소스 = S1 983장 / S2 1,970장 (A-123 0장)")
    nf = {"F-free": 113, "F-free-usage": 36}
    if not all(lg["n_features"] == lg["featureset"].map(nf)):
        raise ValueError("피처 개수 불일치")
    checks.append("피처 수 F-free 113 / F-free-usage 36")
    shared = set(a123["WAFER_ID"]) & set(data[data["group"] != TARGET]["WAFER_ID"])
    checks.append(f"A-123과 456이 공유하는 WAFER_ID {len(shared)}개 (같은 웨이퍼의 다른 스테이지 라벨이 소스에 들어갈 여지 없음)")
    return checks


def source_cv(data: pd.DataFrame, depth: int) -> pd.DataFrame:
    """소스 모델 품질 점검: folds_v1 25개 fold, 학습·평가 모두 소스 그룹 안."""
    rows = []
    for src_name, groups in SOURCES.items():
        for fs, fs_label in FEATURESETS.items():
            cols = cv.feature_columns(fs)
            in_src = lambda df, g=groups: df["group"].isin(g)  # noqa: E731

            def fp(tr, ev, fd, cols=cols):
                return XGBRegressor(**xgb_params(depth)).fit(tr[cols], tr["MRR"]).predict(ev[cols])

            s, _ = cv.cv_evaluate(data, fp, train_filter=in_src, eval_filter=in_src)
            rows.append(s.assign(source=src_name, featureset=fs_label, model="xgb", max_depth=depth))
            print(f"  CV {src_name} {fs_label}: RMSE "
                  + ", ".join(f"{g} {x:.3f}" for g, x in s.groupby("subset")["rmse"].mean().items()))
    res = pd.concat(rows, ignore_index=True)
    # fold = 0~24 (r_id). 반복 안 번호(fold 0~4)는 r_id와 repeat로 정해지므로 뺀다 (열 이름 중복 방지)
    res = res.drop(columns="fold").rename(columns={"subset": "group", "r_id": "fold"})
    return res[["group", "model", "featureset", "source", "max_depth", "fold", "repeat",
                "mse", "mae", "rmse", "n_train", "n_eval"]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", default=str(cv.PROCESSED / "source_preds_v1.csv"))
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    out = cv.ROOT / args.out   # 상대 경로는 저장소 루트 기준 (절대 경로면 그대로)
    today = date.today().isoformat()
    cv_out = cv.ROOT / "results" / f"{today}_{AUTHOR}_source456_cv{'' if args.depth == 4 else f'_d{args.depth}'}.csv"
    for p in [out, out.with_suffix(".yaml")] + ([] if args.skip_cv else [cv_out]):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (덮어쓰지 않음)")

    data = cv.load_dataset()
    preds, log = build_preds(data, args.depth)
    checks = validate(preds, log, data)
    # 재현성: 같은 설정으로 다시 학습하면 같은 예측
    again, _ = build_preds(data, args.depth)
    if not np.array_equal(again["pred"].to_numpy(), preds["pred"].to_numpy()):
        raise ValueError("같은 시드로 다시 학습한 예측이 다르다")
    checks.append("같은 시드로 두 번 학습한 예측값이 완전히 같음")
    preds.to_csv(out, index=False, float_format="%.6f", lineterminator="\n")
    lg = pd.DataFrame(log)
    print("\n".join("  [통과] " + c for c in checks))

    yaml = [
        f"date: {today}", f"author: {AUTHOR}", "script: src/source_models.py",
        f"command: python src/source_models.py --depth {args.depth}"
        + ("" if out.name == "source_preds_v1.csv" else f" --out {out.relative_to(cv.ROOT).as_posix()}"),
        f"output: {out.relative_to(cv.ROOT).as_posix()}", f"sha256: {sha(out)}",
        f"rows: {len(preds)}",
        "preregistration: v0.2.1 3.2, 5.1, 5.2(#21), 5.4(#25, #27)",
        "inputs:",
        f"  features: data/processed/features_v1.csv  # sha256 {sha(cv.FEATURES)[:16]}",
        f"  labels: data/processed/wafer_table.csv (MRR, t_start; train+val, is_outlier=0)  # sha256 {sha(cv.TABLE)[:16]}",
        f"  walkforward: data/processed/walkforward_v1.csv, walkforward_cutoffs_v1.csv",
        "model: XGBRegressor(" + ", ".join(f"{k}={v!r}" for k, v in xgb_params(args.depth).items()) + ")",
        "tuning: 없음 (3.2절 고정값)",
        "sources: {S1: [A-456], S2: [A-456, B-456]}",
        "featuresets:",
        "  F-free: features_v1.csv 피처 113개 전체 (feature_families_v1.csv 순서)",
        "  F-free-usage: family == 소모품 카운터 36개, {mean,std,slope}_{BACKING_FILM,DRESSER,POLISH_TABLE,DRESSER_TABLE,MEMBRANE,PSHEET}_{s1,s2}",
        "  # TODO(결정 필요): since_*_w 5종(리셋 후 경과량)을 사용량 버전에 넣을지",
        "categorical: 넣지 않음 (그룹·스테이지는 F-free 밖. A-123은 소스에 없는 범주)",
        "designs:",
        "  folds: 456 train+val 전체로 학습한 모델 1개. step = -1, role = all. 25개 fold 모두 같은 예측",
        "  walkforward: 단계 s마다 t_start < eval_start(s)인 456만 학습. role = walkforward_v1.csv의 그 단계 역할",
        "a123_labels: 읽지 않음 (예측만 생성. A-123 성능·상관은 계산하지 않았다)",
        "training_log:",
    ]
    for r in lg.to_dict("records"):
        yaml.append("  - " + ", ".join(f"{k}: {v}" for k, v in r.items()
                                       if not (isinstance(v, float) and np.isnan(v))))
    yaml.append("checks:")
    yaml += [f"  - {c}" for c in checks]
    out.with_suffix(".yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"저장: {out.relative_to(cv.ROOT)} ({len(preds)}행)")

    if not args.skip_cv:
        res = source_cv(data, args.depth)
        res.round(6).to_csv(cv_out, index=False, lineterminator="\n")
        cv_out.with_suffix(".yaml").write_text("\n".join([
            f"date: {today}", f"author: {AUTHOR}", "script: src/source_models.py",
            "purpose: 소스 모델 f_456의 456 안 교차검증 (품질 점검, 전이 성능 아님)",
            "input: data/processed/features_v1.csv + wafer_table.csv (MRR), folds_v1.csv 25 fold",
            "model: XGBRegressor(" + ", ".join(f"{k}={v!r}" for k, v in xgb_params(args.depth).items()) + ")",
            "featureset: F-free (113) / F-free-usage (36)",
            "rows: source × featureset × fold × group(all + 소스 그룹)",
            "columns: group(all/A-456/B-456), model, featureset, source, max_depth, fold(r_id 0~24), repeat, mse, mae, rmse, n_train, n_eval",
            f"seed: {cv.SEED}", "labels: train+val. test·data/locked/ 사용 안 함. A-123 미사용", ""]),
            encoding="utf-8")
        print(f"저장: {cv_out.relative_to(cv.ROOT)}")


if __name__ == "__main__":
    main()

"""리셋 후 경과량 `since_*_w` 점검과 예측 모델 영향 (진단 + 민감도, 공통 피처는 바꾸지 않는다).

배경: PR #21(박종화)에서 `since_POLISH_TABLE_w`에 음수 웨이퍼가 보고됐다. `since_*`는
`src/aggregate.py`에서 "웨이퍼 시작값이 직전 웨이퍼 시작값보다 스테이지 값 범위의 20% 이상 줄면 리셋"으로
감지한 뒤 `시작값 − 마지막 리셋 웨이퍼의 시작값`으로 만든다. 감소폭이 20%보다 작은 리셋은 놓치고,
그 뒤 다음 리셋까지 기준값이 이전 교체에 머물러 값이 틀린다(작게 나오거나 음수).

1. 진단 (라벨 미사용, train·val·test 센서 집계값만): 변수 5종마다
   - 감지된 리셋, 감지되지 않은 감소(시작값이 직전보다 작음, 카운터는 리셋 외에는 줄지 않음)
   - 감소 유형: `restart` = 다음 웨이퍼도 감소 전보다 낮음(리셋을 놓친 것으로 봄),
                `bounce` = 다음 웨이퍼가 다시 원래 수준(그 웨이퍼 하나만 튐)
   - 영향 웨이퍼: restart부터 그 스테이지의 다음 감지 리셋 직전까지 + bounce 웨이퍼 (`since`가 NaN이 아닌 것만)
2. 민감도 (train+val 라벨): `since_POLISH_TABLE_w` 하나만 뺀 112개 피처로 다시 학습해 기존 113개와 비교.
   피처를 새로 만들거나 값을 고치지 않는다(3절 5번, 14.1절). 고친 since는 팀 합의 후 `_v2`로 만든다.
   - #8 분리 대 통합 주 평가(25 fold, 보정 t-검정), #6 기본 모델(XGBoost 통합) 그룹별 RMSE
   - #7 소스 S1·S2 전체 피처 버전: 456 안 교차검증 RMSE, A-123 예측값 변화(라벨 미사용)
   - 각 모델에서 `since_POLISH_TABLE_w`의 중요도(total gain) 순위

실행: python src/since_reset_check.py   (출력이 있으면 멈춘다)
출력: results/diagnostics/{날짜}_전준혁_since_reset_audit.csv, _since_reset_wafers.csv (+ .yaml)
      results/{날짜}_전준혁_since_pt_sensitivity.csv, _summary.csv (+ .yaml)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a123_split_vs_pooled as sp  # noqa: E402
import cv_common as cv  # noqa: E402
import model_zoo as mz  # noqa: E402
import source_models as sm  # noqa: E402

AUTHOR = "전준혁"
U5 = ["BACKING_FILM", "DRESSER", "POLISH_TABLE", "MEMBRANE", "PSHEET"]
TARGET_VAR = "since_POLISH_TABLE_w"
ALPHA = 0.05
ROW_LEVEL = {"A": 111, "B": 100}   # CLAUDE.md 5.5절: 센서 행 단위로 센 POLISH_TABLE 리셋


# ------------------------------------------------------------------ 1. 진단
def audit(tab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = tab.sort_values(["STAGE", "t_start"], kind="mergesort").reset_index(drop=True)   # aggregate.py와 같은 순서
    summary, wafers = [], []
    for u in U5:
        s, r, since = t[f"start_{u}_w"], t[f"reset_{u}_w"], t[f"since_{u}_w"]
        d = s.groupby(t["STAGE"]).diff()
        nxt = s.groupby(t["STAGE"]).shift(-1)
        prev = s.groupby(t["STAGE"]).shift(1)
        drop = (d < 0) & (r == 0)
        kind = pd.Series(np.where(nxt < prev, "restart", "bounce"), index=t.index).where(drop)
        reason = pd.Series(pd.NA, index=t.index, dtype="object")
        for i in t.index[kind == "restart"]:
            j = i
            while j < len(t) and t.at[j, "STAGE"] == t.at[i, "STAGE"] and (j == i or r[j] == 0):
                if pd.isna(reason[j]):
                    reason[j] = f"missed_reset@{i}"
                j += 1
        reason[(kind == "bounce") & reason.isna()] = "bounce"
        bad = reason.notna() & since.notna()
        neg = since < 0
        assert not (neg & ~bad).any(), f"{u}: 음수인데 영향 웨이퍼로 잡히지 않은 행이 있다"
        rec = {"variable": f"since_{u}_w", "n_reset_detected": int(r.sum()), "n_unflagged_drop": int(drop.sum()),
               "n_restart": int((kind == "restart").sum()), "n_bounce": int((kind == "bounce").sum()),
               "n_negative": int(neg.sum()), "n_affected": int(bad.sum())}
        for stg in ("A", "B"):
            m = t["STAGE"] == stg
            rec[f"reset_{stg}_detected_plus_restart"] = int(r[m].sum() + (kind[m] == "restart").sum())
        for (spl, g), x in t[bad].groupby(["split", "group"]):
            rec[f"affected_{spl}_{g}"] = len(x)
        for (spl, g), x in t[neg].groupby(["split", "group"]):
            rec[f"negative_{spl}_{g}"] = len(x)
        rec["drop_rel_min"] = float((d / t["STAGE"].map(t.groupby("STAGE")[f"start_{u}_w"].agg(lambda x: x.max() - x.min())))[drop].min()) \
            if drop.any() else np.nan
        summary.append(rec)
        if bad.any():
            w = t.loc[bad, cv.KEY + ["split", "group", "t_start", f"start_{u}_w", f"since_{u}_w"]].copy()
            w = w.rename(columns={f"start_{u}_w": "start", f"since_{u}_w": "since_v1"})
            w.insert(0, "variable", f"since_{u}_w")
            w["reason"] = reason[bad]
            w["negative"] = neg[bad].astype(int)
            wafers.append(w)
    wf = pd.concat(wafers, ignore_index=True) if wafers else pd.DataFrame()
    return pd.DataFrame(summary), wf


# ------------------------------------------------------------------ 2. 민감도
def featuresets() -> dict[str, list[str]]:
    full = cv.feature_columns("full")
    assert TARGET_VAR in full
    return {"F-free": full, "F-free-noSincePT": [c for c in full if c != TARGET_VAR]}


def run_model_cv(data: pd.DataFrame, cols: list[str], fs: str) -> pd.DataFrame:
    """#8 분리(A-123만) + 통합(세 그룹, 평가는 그룹별) + #7 소스 S1·S2 (456 안)."""
    rows = []
    is_t = sp.is_target
    s, _ = cv.cv_evaluate(data, mz.make_fit_predict("xgb", cols, None, 4), train_filter=is_t, eval_filter=is_t,
                          report_groups=False)
    rows.append(s.assign(analysis="split_vs_pooled", model="separate", subset="A-123"))
    s, _ = cv.cv_evaluate(data, mz.make_fit_predict("xgb", cols, "onehot", 4))
    rows.append(s.assign(analysis="pooled_xgb", model="xgb+onehot"))
    for src_name, groups in sm.SOURCES.items():
        in_src = lambda df, g=groups: df["group"].isin(g)  # noqa: E731
        s, _ = cv.cv_evaluate(data, lambda tr, ev, fd: sm.fit_predict(tr, ev, cols, 4),
                              train_filter=in_src, eval_filter=in_src)
        rows.append(s.assign(analysis="source456_cv", model=f"xgb {src_name}"))
    out = pd.concat(rows, ignore_index=True).assign(featureset=fs)
    return out.drop(columns="fold").rename(columns={"subset": "group", "r_id": "fold"})


def importance_rank(model, names: list[str]) -> tuple[int, float]:
    g = model.get_booster().get_score(importance_type="total_gain")
    s = pd.Series({n: g.get(f"f{i}", g.get(n, 0.0)) for i, n in enumerate(names)}).sort_values(ascending=False)
    return int(s.index.get_loc(TARGET_VAR)) + 1, float(s[TARGET_VAR] / s.sum())


def importance_and_shift(data: pd.DataFrame, fsets: dict) -> list[dict]:
    from xgboost import XGBRegressor
    rows = []
    full, nopt = fsets["F-free"], fsets["F-free-noSincePT"]
    tgt = data[data["group"] == sm.TARGET]
    for src_name, groups in sm.SOURCES.items():
        src = data[data["group"].isin(groups)]
        m = XGBRegressor(**sm.xgb_params(4)).fit(src[full], src["MRR"])
        rank, share = importance_rank(m, full)
        d = sm.fit_predict(src, tgt, nopt, 4) - m.predict(tgt[full]).astype(float)   # A-123 라벨 미사용
        rows.append({"kind": "source_importance_and_shift", "model": f"xgb {src_name}", "n_train": len(src),
                     "importance_rank": rank, "importance_share": share, "n_features": len(full),
                     "pred_shift_mean_abs": float(np.abs(d).mean()), "pred_shift_max_abs": float(np.abs(d).max())})
    for name, sub, cat in [("separate", data[data["group"] == sm.TARGET], None), ("xgb+onehot (pooled)", data, "onehot")]:
        X = mz.design(sub, full, cat)
        m = mz.estimator("xgb", 4, cat).fit(X, sub["MRR"].to_numpy(float))
        rank, share = importance_rank(m, list(X.columns))
        rows.append({"kind": "importance_full_data", "model": name, "n_train": len(sub),
                     "importance_rank": rank, "importance_share": share, "n_features": X.shape[1]})
    return rows


def summarize(scores: pd.DataFrame) -> list[dict]:
    rows = []
    a = scores[scores["analysis"] != "split_vs_pooled"]
    for (an, mdl, g, fs), x in a.groupby(["analysis", "model", "group", "featureset"]):
        rows.append({"kind": "cv_mean", "analysis": an, "model": mdl, "group": g, "featureset": fs,
                     "rmse": x["rmse"].mean(), "mse": x["mse"].mean(), "mae": x["mae"].mean()})
    for (an, mdl, g), x in a.groupby(["analysis", "model", "group"]):
        p = x.pivot(index="fold", columns="featureset", values="mse")
        n = x[x["featureset"] == "F-free"].set_index("fold")
        t = cv.corrected_ttest(p["F-free-noSincePT"] - p["F-free"], n["n_train"], n["n_eval"])
        rows.append({"kind": "noSincePT_minus_full", "analysis": an, "model": mdl, "group": g, "metric": "mse", **t})
    for fs in ("F-free", "F-free-noSincePT"):
        sep = scores[(scores["model"] == "separate") & (scores["featureset"] == fs)].set_index("fold").sort_index()
        poo = scores[(scores["analysis"] == "pooled_xgb") & (scores["group"] == "A-123")
                     & (scores["featureset"] == fs)].set_index("fold").sort_index()
        t = cv.corrected_ttest(poo["mse"] - sep["mse"], sep["n_train"], sep["n_eval"])
        verdict = (("통합 우위" if t["mean_diff"] < 0 else "분리 우위") if t["p"] < ALPHA
                   else "차이 없음(분리 유지)")
        rows.append({"kind": "split_vs_pooled", "analysis": "split_vs_pooled", "model": "pooled - separate",
                     "group": "A-123", "featureset": fs, "metric": "mse", "mse_separate": sep["mse"].mean(),
                     "mse_pooled": poo["mse"].mean(), "verdict": verdict, **t})
    return rows


def write_yaml(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    today = date.today().isoformat()
    diag = cv.ROOT / "results" / "diagnostics"
    out_a = diag / f"{today}_{AUTHOR}_since_reset_audit.csv"
    out_w = diag / f"{today}_{AUTHOR}_since_reset_wafers.csv"
    out_s = cv.ROOT / "results" / f"{today}_{AUTHOR}_since_pt_sensitivity.csv"
    out_m = out_s.with_name(f"{out_s.stem}_summary.csv")
    for p in (out_a, out_w, out_s, out_m, out_a.with_suffix(".yaml"), out_w.with_suffix(".yaml"),
              out_s.with_suffix(".yaml")):
        if p.exists():
            raise FileExistsError(f"{p} 가 이미 있다 (덮어쓰지 않음)")

    # 1. 진단: 집계표의 센서 집계 열만 읽는다 (MRR 열은 읽지 않음)
    cols = cv.KEY + ["split", "group", "t_start"] + [f"{p}_{u}_w" for u in U5 for p in ("start", "reset", "since")]
    tab = pd.read_csv(cv.TABLE, usecols=cols)
    summ, wafers = audit(tab)
    summ.to_csv(out_a, index=False, lineterminator="\n")
    wafers.round(6).to_csv(out_w, index=False, lineterminator="\n")
    print(summ.T.to_string())
    pt = summ.set_index("variable").loc[TARGET_VAR]
    other_ok = bool((summ.set_index("variable").drop(index=TARGET_VAR)[["n_unflagged_drop", "n_negative"]] == 0)
                    .all().all())

    # 2. 민감도
    data = cv.load_dataset()
    fsets = featuresets()
    scores = pd.concat([run_model_cv(data, c, fs) for fs, c in fsets.items()], ignore_index=True)
    v1 = pd.read_csv(cv.ROOT / "results" / "2026-10-31_전준혁_a123_split_vs_pooled.csv")
    mine = scores[(scores["model"] == "separate") & (scores["featureset"] == "F-free")].sort_values("fold")
    assert np.allclose(mine["mse"].to_numpy(), v1[v1["model"] == "separate"].sort_values("fold")["mse"].to_numpy(),
                       atol=1e-5), "113개 재실행이 #8 기존 결과와 다르다"
    summary = pd.DataFrame(summarize(scores) + importance_and_shift(data, fsets))
    scores = scores[["analysis", "group", "model", "featureset", "fold", "repeat", "mse", "mae", "rmse",
                     "n_train", "n_eval"]]
    scores.round(6).to_csv(out_s, index=False, lineterminator="\n")
    summary.round(6).to_csv(out_m, index=False, lineterminator="\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.dropna(axis=1, how="all").round(4).to_string(index=False))
    svp = summary[summary["kind"] == "split_vs_pooled"].set_index("featureset")
    stable = svp["verdict"].nunique() == 1

    write_yaml(out_a.with_suffix(".yaml"), [
        f"date: {today}", f"author: {AUTHOR}", "script: src/since_reset_check.py",
        "input: data/processed/wafer_table.csv (start_/reset_/since_*_w 열만, MRR 미사용, train·val·test 센서)",
        "rule: aggregate.py 순서(STAGE, t_start). 감지되지 않은 감소 = 시작값 차이 < 0 이고 reset = 0."
        " restart = 다음 웨이퍼 시작값 < 감소 전 값, bounce = 그 외",
        "affected: restart 웨이퍼부터 그 스테이지 다음 감지 리셋 직전까지 + bounce 웨이퍼, since가 NaN이 아닌 것",
        f"row_level_reference: CLAUDE.md 5.5절 POLISH_TABLE 행 단위 리셋 A {ROW_LEVEL['A']} / B {ROW_LEVEL['B']}",
        "result:",
        f"  since_POLISH_TABLE_w: 감지 {int(pt.n_reset_detected)}, 미감지 감소 {int(pt.n_unflagged_drop)}"
        f" (restart {int(pt.n_restart)}, bounce {int(pt.n_bounce)}), 음수 {int(pt.n_negative)},"
        f" 영향 웨이퍼 {int(pt.n_affected)}",
        f"  detected_plus_restart: A {int(pt.reset_A_detected_plus_restart)} / B {int(pt.reset_B_detected_plus_restart)}",
        f"  other_4_since_vars_clean: {other_ok} (미감지 감소·음수 0)",
        "decision: 없음. 공통 피처(features_v1, wafer_table) 수정·제외는 팀 합의 사항", ""])
    write_yaml(out_w.with_suffix(".yaml"), [
        f"date: {today}", f"author: {AUTHOR}", "script: src/since_reset_check.py",
        "content: since_*_w 값이 리셋 누락 때문에 틀렸을 수 있는 웨이퍼 목록 (since_POLISH_TABLE_w만 해당)",
        "columns: variable, WAFER_ID, STAGE, split, group, t_start, start(웨이퍼 시작값), since_v1(집계표 값),"
        " reason(missed_reset@행번호 = 그 행에서 놓친 리셋 이후 / bounce), negative",
        "note: test 웨이퍼는 센서 집계값만 있다(라벨 없음). 행번호는 (STAGE, t_start) 정렬 기준", ""])
    write_yaml(out_s.with_suffix(".yaml"), [
        f"date: {today}", f"author: {AUTHOR}", "script: src/since_reset_check.py",
        "purpose: since_POLISH_TABLE_w 오류가 #6~#8 결과에 주는 영향 (민감도, 기존 결과는 그대로 둠)",
        "input: data/processed/features_v1.csv + wafer_table.csv (MRR, t_start; train+val, is_outlier=0), folds_v1.csv",
        "featuresets:",
        "  F-free: features_v1 113개 (기존 결과와 같음, 재실행 값이 #8 기존 결과와 일치함을 확인)",
        f"  F-free-noSincePT: 113개에서 {TARGET_VAR}만 뺀 112개 (값을 고치거나 새 피처를 만들지 않음)",
        f"model: {mz.params_text('xgb', 4)} (3.2절 고정값, n_jobs {cv.XGB_PARAMS['n_jobs']})",
        "analyses: split_vs_pooled(#8, A-123 평가), pooled_xgb(#6 기본 모델, 그룹별), source456_cv(#7 S1·S2 전체 피처)",
        "test: 보정 t-검정, 차이 = noSincePT − F-free (MSE). 분리 대 통합은 피처셋별로 #8과 같은 규칙",
        "output_summary: " + out_m.name + " (kind = cv_mean / noSincePT_minus_full / split_vs_pooled /"
        " source_importance_and_shift / importance_full_data)",
        "result:",
        "  split_vs_pooled: " + ", ".join(f"{fs} {r.verdict} (p = {r.p:.3f})" for fs, r in svp.iterrows()),
        f"  split_vs_pooled_verdict_stable: {stable}",
        "labels: train+val만. 소스 예측 변화에는 A-123 라벨 미사용. test·data/locked/ 사용 안 함", ""])
    print(f"분리 대 통합 판정이 since_POLISH_TABLE_w 제외와 무관한가: {stable}")
    print("저장:", *(p.relative_to(cv.ROOT) for p in (out_a, out_w, out_s, out_m)))


if __name__ == "__main__":
    main()

"""SR 최종 식의 그룹 간 상호 적용 — 사후 탐색적 검증 (사전확정 실험 아님, test 평가 아님).

목적: A-456과 B-456에서 서로 다른 식이 나왔다는 것만으로 공통 관계식이 없다고 말할 수 있는지 본다.
박종화 정식 SR 결과(sr/results 브랜치, 1-표준오차 최종 식)를 본 뒤 하는 분석이다. PySR은 다시 돌리지 않는다.

대상 그룹 T마다 (A-456, B-456)
  ① 자기 식: T의 최종 식, fold마다 상수 재적합 (`sr_select.fold_rmse`와 같은 방식, 기존 결과 재현 확인)
  ② 상대 식, 계수 고정: 상대 그룹 S의 최종 식을 PySR 상수 그대로 T에 적용 (학습 없음)
     ②+b: 계수는 고정하고 fold 학습 쪽에서 절편 하나(평균 잔차)만 더함 (수준 차이만 허용, 참고)
  ③ 상대 식, 형태·변수 고정 + 계수 재학습: folds_v1 25 fold마다 T의 학습 쪽으로 상수만 재적합
     (`sr_select.refit`, 초기값 = S 식의 상수, 지수 −3~3)
  참고: 학습 쪽 평균(상수 예측), 같은 변수의 선형회귀, 자기 식 계수 고정(T 전체로 맞춘 상수라 표본 내)

척도 (`sr_run.load_frame`과 같은 규칙: 변수 ÷ 그룹 train+val 전체의 절댓값 중앙값, 부호 유지, 라벨 미사용):
  - `source`: 상대 식 변수를 S 그룹의 척도로 나눔 → 상수의 물리 단위가 S 식과 같다 (②의 주 처리)
  - `target`: T 그룹의 척도로 나눔 → T에서 SR을 돌렸을 때와 같은 입력 (③과 ①의 주 처리)
  ②는 두 척도 모두, ③은 target(주)·source(민감도)를 보고한다.

웨이퍼 집합 (평가 웨이퍼가 달라지는 것을 막기 위해):
  - `own_sr`: T의 SR 사용 웨이퍼(T 후보 10개 값이 모두 있음, A 940 / B 951). ① 기존 결과 재현용
  - `cross_natural`: 상대 식 변수 값이 모두 있는 T 웨이퍼
  - `common`: own_sr ∩ 상대 식 변수 값이 모두 있음 → ①②③과 기준선을 같은 웨이퍼·같은 fold로 비교 (주 비교)
  - `common_noshared`: common에서 상대 그룹과 WAFER_ID가 같은 웨이퍼를 뺀 것 (누수 민감도)

실행: python src/sr_cross_group.py --selection <sr_selection.csv> --folds <FV1_sr_folds.csv>
      (두 파일은 sr/results 브랜치의 results/diagnostics/2026-11-01_박종화_sr_selection.csv,
       results/2026-11-01_박종화_FV1_sr_folds.csv. main에 없으면 git show로 꺼내 경로를 준다)
출력: results/{날짜}_전준혁_FV1_sr_cross_group.csv, _summary.csv (+ .yaml)
      results/diagnostics/{날짜}_전준혁_sr_cross_group_domain.csv (+ .yaml)
      이미 있으면 멈춘다. test 라벨과 data/locked/는 읽지 않는다.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_select as ss  # noqa: E402
from cv_common import corrected_ttest  # noqa: E402
from sr_run import CANDIDATES, GROUPS, PROC, RESULTS, load_frame  # noqa: E402

AUTHOR = "전준혁"
KEYS = ["WAFER_ID", "STAGE"]
FEATURESET = "features_v1"
OTHER = {"A-456": "B-456", "B-456": "A-456"}


# ---------------------------------------------------------------- 데이터
def group_data(group: str) -> pd.DataFrame:
    """그룹의 train+val(이상치 제외) 웨이퍼 전체, 척도 적용 전 값. `sr_run.load_frame`과 같은 병합."""
    feats = pd.read_csv(PROC / "features_v1.csv")
    table = pd.read_csv(PROC / "wafer_table.csv", usecols=KEYS + ["split", "MRR", "is_outlier"])
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    return (feats[feats["group"] == group].merge(table, on=KEYS, validate="one_to_one")
            .reset_index(drop=True))


def candidates(group: str) -> list[str]:
    c = pd.read_csv(CANDIDATES)
    return c[(c["group"] == group) & (c["selected"] == 1)].sort_values("rank")["feature"].tolist()


def load_equations(path: Path) -> dict:
    sel = pd.read_csv(path, encoding="utf-8-sig")
    sel = sel[sel["rule"] == "1se"].set_index("group")
    assert set(sel.index) == set(GROUPS), f"1se 식이 없는 그룹: {set(GROUPS) - set(sel.index)}"
    texts = {g: sel.loc[g, "equation"] for g in GROUPS}
    return {g: ss.parse_equation(t) for g, t in texts.items()}, texts


# ---------------------------------------------------------------- 정의역 점검
def domain_issues(node, theta, X, n) -> dict:
    """하위 식을 계산하며 정의역 문제를 센다 (웨이퍼 단위, 같은 웨이퍼가 여러 이유에 들 수 있음)."""
    bad = {"sqrt_neg": np.zeros(n, bool), "log_nonpos": np.zeros(n, bool), "div_zero": np.zeros(n, bool),
           "pow_neg_base": np.zeros(n, bool), "nonfinite": np.zeros(n, bool)}

    def ev(nd):
        kind = nd[0]
        if kind == "const":
            return np.full(n, theta[nd[1]])
        if kind == "var":
            return np.asarray(X[nd[1]], float)
        if kind == "neg":
            return -ev(nd[1])
        if kind == "call":
            a = ev(nd[2])
            if nd[1] == "sqrt":
                bad["sqrt_neg"] |= a < 0
            if nd[1] == "log":
                bad["log_nonpos"] |= a <= 0
            with np.errstate(all="ignore"):
                return ss.UNARY[nd[1]](a)
        a, b = ev(nd[1]), ev(nd[2])
        with np.errstate(all="ignore"):
            if kind == "div":
                bad["div_zero"] |= b == 0
                return a / b
            if kind == "pow":
                bad["pow_neg_base"] |= (a < 0) & (np.round(b) != b)
                return np.power(a, b)
            return {"add": a + b, "sub": a - b, "mul": a * b}[kind]

    out = ev(node)
    bad["nonfinite"] = ~np.isfinite(out)
    return {k: int(v.sum()) for k, v in bad.items()}


# ---------------------------------------------------------------- 평가
def design(data: pd.DataFrame, mask: np.ndarray, cols: list[str], scale: pd.Series) -> dict:
    return {c: (data.loc[mask, c] / scale[c]).to_numpy(float) for c in cols}


def check_folds(table: pd.DataFrame, masks: dict) -> None:
    """같은 그룹 안에서 fold 학습·평가 쪽의 키와 WAFER_ID가 겹치지 않는지 확인."""
    for r, ev in masks.items():
        tr_id, ev_id = set(table.loc[~ev, "WAFER_ID"]), set(table.loc[ev, "WAFER_ID"])
        assert tr_id.isdisjoint(ev_id), f"fold {r}: 학습·평가 쪽 WAFER_ID 겹침"


def evaluate(method: str, eq, X: dict, y: np.ndarray, masks: dict) -> list[dict]:
    """fold별 평가 쪽 지표. 계산이 안 되는 예측(NaN)은 개수를 세고 RMSE에서 뺀다."""
    rows = []
    cols = sorted(X)
    for r in sorted(masks):
        ev = masks[r]
        tr = ~ev
        n_ev = int(ev.sum())
        Xtr, Xev = ss._subset(X, tr), ss._subset(X, ev)
        n_tr_nan = 0
        if method in ("refit",):
            theta = ss.refit(eq.node, eq.theta, Xtr, y[tr])
            n_tr_nan = int(np.isnan(ss.predict(eq.node, theta, Xtr, int(tr.sum()))).sum())
            p = ss.predict(eq.node, theta, Xev, n_ev)
        elif method in ("fixed", "fixed+b"):
            p = ss.predict(eq.node, eq.theta, Xev, n_ev)
            if method == "fixed+b":
                f_tr = ss.predict(eq.node, eq.theta, Xtr, int(tr.sum()))
                n_tr_nan = int(np.isnan(f_tr).sum())
                p = p + np.nanmean(y[tr] - f_tr)
        elif method == "mean":
            p = np.full(n_ev, y[tr].mean())
        elif method == "lr_vars":
            M = np.column_stack([X[c] for c in cols])
            p = LinearRegression().fit(M[tr], y[tr]).predict(M[ev])
        else:
            raise ValueError(method)
        ok = np.isfinite(p)
        e = p[ok] - y[ev][ok]
        rows.append({"fold": r, "mse": float(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e))),
                     "rmse": float(np.sqrt(np.mean(e ** 2))), "n_train": int(tr.sum()), "n_eval": int(ok.sum()),
                     "n_eval_nan": int((~ok).sum()), "n_train_nan": n_tr_nan})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--max-nfev", type=int, default=2000, help="재적합 반복 상한 민감도 (기본 refit은 200)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    today = date.today().isoformat()
    out = RESULTS / f"{today}_{AUTHOR}_FV1_sr_cross_group.csv"
    out_s = out.with_name(f"{out.stem}_summary.csv")
    out_d = RESULTS / "diagnostics" / f"{today}_{AUTHOR}_sr_cross_group_domain.csv"
    for p in (out, out_s, out_d, out.with_suffix(".yaml"), out_d.with_suffix(".yaml")):
        if p.exists():
            sys.exit(f"이미 있어서 멈춘다: {p}")

    eqs, texts = load_equations(args.selection)
    data = {g: group_data(g) for g in GROUPS}
    scale = {}
    for g in GROUPS:
        cols = sorted(set(candidates(g)) | ss.variables(eqs["A-456"].node) | ss.variables(eqs["B-456"].node))
        scale[g] = data[g][cols].abs().median()   # 그룹 train+val 전체, sr_run.load_frame과 같은 규칙
    ids = {g: set(data[g]["WAFER_ID"]) for g in GROUPS}
    shared = ids["A-456"] & ids["B-456"]

    # 누수 점검: 공유 WAFER_ID와 그 쌍의 MRR 상관 (train+val 라벨만)
    pair = data["A-456"][["WAFER_ID", "MRR"]].merge(data["B-456"][["WAFER_ID", "MRR"]], on="WAFER_ID",
                                                     suffixes=("_A", "_B"))
    leak = {"n_shared_wafer_id": len(shared), "n_pairs": len(pair),
            "mrr_corr_pearson": float(pair["MRR_A"].corr(pair["MRR_B"])),
            "mrr_corr_spearman": float(pair["MRR_A"].corr(pair["MRR_B"], method="spearman"))}

    rows, domain, sets_info = [], [], []
    for T in GROUPS:
        S = OTHER[T]
        d = data[T]
        y_all = d["MRR"].to_numpy(float)
        own_vars, cross_vars = sorted(ss.variables(eqs[T].node)), sorted(ss.variables(eqs[S].node))
        own_sr = d[candidates(T)].notna().all(axis=1).to_numpy()
        frame, _, _ = load_frame(T)
        assert len(frame) == own_sr.sum() and set(map(tuple, frame[KEYS].to_numpy())) == \
            set(map(tuple, d.loc[own_sr, KEYS].to_numpy())), f"{T}: SR 사용 웨이퍼가 sr_run과 다르다"
        cross_ok = d[cross_vars].notna().all(axis=1).to_numpy()
        noshared = ~d["WAFER_ID"].isin(shared).to_numpy()
        sets = {"own_sr": own_sr, "cross_natural": cross_ok, "common": own_sr & cross_ok,
                "common_noshared": own_sr & cross_ok & noshared}
        for name, m in sets.items():
            sets_info.append({"group": T, "wafer_set": name, "n": int(m.sum()), "n_total": len(d),
                              "n_shared_id": int((m & ~noshared).sum())})
        # 계산 단위: (웨이퍼 집합, 식, 방법, 척도)
        plan = [("own_sr", T, "refit", "target"), ("own_sr", T, "fixed", "target"),
                ("cross_natural", S, "fixed", "source"), ("cross_natural", S, "fixed", "target"),
                ("cross_natural", S, "refit", "target"), ("cross_natural", S, "refit", "source")]
        for ws in ("common", "common_noshared"):
            plan += [(ws, T, "refit", "target"), (ws, T, "fixed", "target"),
                     (ws, S, "fixed", "source"), (ws, S, "fixed", "target"),
                     (ws, S, "fixed+b", "source"), (ws, S, "fixed+b", "target"),
                     (ws, S, "refit", "target"), (ws, S, "refit", "source"),
                     (ws, T, "lr_vars", "target"), (ws, S, "lr_vars", "target"), (ws, T, "mean", "target")]
        for ws, F, method, sc in plan:
            m = sets[ws]
            masks = ss.fold_masks(d.loc[m, KEYS].reset_index(drop=True))
            check_folds(d.loc[m].reset_index(drop=True), masks)
            vars_ = own_vars if F == T else cross_vars
            X = design(d, m, vars_, scale[S if sc == "source" else T])
            y = y_all[m]
            label = {"refit": "③ 재학습" if F == S else "① 자기 식 재적합", "fixed": "② 계수 고정" if F == S
                     else "참고: 자기 식 계수 고정(표본 내)", "fixed+b": "②+b 계수 고정+절편",
                     "lr_vars": "참고: 같은 변수 선형회귀", "mean": "참고: 학습 쪽 평균"}[method]
            for rec in evaluate(method, eqs[F], X, y, masks):
                rows.append({"group": T, "formula_from": F, "model": method, "label": label, "scale": sc,
                             "wafer_set": ws, "featureset": FEATURESET, **rec})
            if method in ("fixed", "refit") and ws in ("cross_natural", "own_sr", "common"):
                if method == "fixed":
                    iss = domain_issues(eqs[F].node, eqs[F].theta, X, int(m.sum()))
                else:   # fold별 재적합 상수로 평가 쪽 점검, 25 fold 합계(웨이퍼는 반복마다 1번씩 평가 → 5회 셈)
                    iss = {}
                    for r, ev in masks.items():
                        th = ss.refit(eqs[F].node, eqs[F].theta, ss._subset(X, ~ev), y[~ev])
                        for k, v in domain_issues(eqs[F].node, th, ss._subset(X, ev), int(ev.sum())).items():
                            iss[k] = iss.get(k, 0) + v
                    iss = {k: v / 5 for k, v in iss.items()}   # 반복당 평균
                domain.append({"group": T, "formula_from": F, "model": method, "scale": sc, "wafer_set": ws,
                               "n_wafers": int(m.sum()), **iss})
        # 정의역 밖에서 결측 때문에 빠진 웨이퍼 (상대 식 변수 기준)
        for v in cross_vars:
            domain.append({"group": T, "formula_from": S, "model": "missing", "variable": v,
                           "n_wafers": len(d), "n_missing": int(d[v].isna().sum()),
                           "n_missing_in_own_sr": int((d[v].isna() & own_sr).sum())})
        print(f"{T} 완료 ({time.perf_counter() - t0:.0f}초)", flush=True)

    res = pd.DataFrame(rows)

    # 재적합 반복 상한 민감도 (common, ③과 ①만)
    orig = ss.least_squares
    ss.least_squares = lambda *a, **k: orig(*a, **{**k, "max_nfev": args.max_nfev})
    try:
        for T in GROUPS:
            S = OTHER[T]
            d = data[T]
            m = (d[candidates(T)].notna().all(axis=1)
                            & d[sorted(ss.variables(eqs[S].node))].notna().all(axis=1)).to_numpy()
            masks = ss.fold_masks(d.loc[m, KEYS].reset_index(drop=True))
            for F in (T, S):
                X = design(d, m, sorted(ss.variables(eqs[F].node)), scale[T])
                for rec in evaluate("refit", eqs[F], X, d.loc[m, "MRR"].to_numpy(float), masks):
                    rows.append({"group": T, "formula_from": F, "model": "refit", "label": f"max_nfev={args.max_nfev}",
                                 "scale": "target", "wafer_set": "common_nfev", "featureset": FEATURESET, **rec})
    finally:
        ss.least_squares = orig
    res = pd.DataFrame(rows)
    elapsed = time.perf_counter() - t0

    # ① 기존 결과 재현 확인
    old = pd.read_csv(args.folds, encoding="utf-8-sig")
    repro = {}
    for T in GROUPS:
        mine = res[(res.group == T) & (res.formula_from == T) & (res.model == "refit") & (res.wafer_set == "own_sr")]
        ref = old[(old.group == T) & (old.model == "sr_final")].sort_values("fold")
        repro[T] = float(np.abs(mine.sort_values("fold")["rmse"].to_numpy() - ref["rmse"].to_numpy()).max())
        assert repro[T] < 1e-6, f"{T}: ① 재현 실패 (최대 차이 {repro[T]})"

    # 요약: 평균 ± sd, common에서 ①과의 보정 t-검정 (MSE, 탐색적 참고)
    summ = []
    for key, g in res.groupby(["group", "formula_from", "model", "scale", "wafer_set", "label"], sort=False):
        summ.append(dict(zip(["group", "formula_from", "model", "scale", "wafer_set", "label"], key),
                         rmse_mean=g["rmse"].mean(), rmse_sd=g["rmse"].std(ddof=1), mse_mean=g["mse"].mean(),
                         mae_mean=g["mae"].mean(), n_eval_mean=g["n_eval"].mean(),
                         n_eval_nan_total=int(g["n_eval_nan"].sum()), n_train_nan_total=int(g["n_train_nan"].sum())))
    summ = pd.DataFrame(summ)
    tests = []
    for T in GROUPS:
        for ws in ("common", "common_noshared"):
            base = res[(res.group == T) & (res.formula_from == T) & (res.model == "refit") & (res.wafer_set == ws)]
            base = base.set_index("fold").sort_index()
            for _, s in summ[(summ.group == T) & (summ.wafer_set == ws)].iterrows():
                if s.formula_from == T and s.model == "refit":
                    continue
                g = res[(res.group == T) & (res.formula_from == s.formula_from) & (res.model == s.model)
                        & (res.scale == s.scale) & (res.wafer_set == ws)].set_index("fold").sort_index()
                if (g["n_eval_nan"] > 0).any():
                    continue   # 평가 웨이퍼가 달라 쌍 비교 불가
                t = corrected_ttest(g["mse"] - base["mse"], base["n_train"], base["n_eval"])
                tests.append({"group": T, "wafer_set": ws, "formula_from": s.formula_from, "model": s.model,
                              "scale": s.scale, "vs": "① 자기 식 재적합", "metric": "mse", **t})
    summ = summ.merge(pd.DataFrame(tests)[["group", "wafer_set", "formula_from", "model", "scale", "mean_diff",
                                           "t", "p"]].rename(columns={"mean_diff": "mse_diff_vs_own"}),
                      on=["group", "wafer_set", "formula_from", "model", "scale"], how="left")

    res.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    summ.round(6).to_csv(out_s, index=False, encoding="utf-8-sig")
    dom = pd.DataFrame(domain)
    dom.to_csv(out_d, index=False, encoding="utf-8-sig")
    info = pd.DataFrame(sets_info)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(info.to_string(index=False))
        print(summ.drop(columns=["label"]).round(4).to_string(index=False))
        print(dom.round(2).to_string(index=False))
    print("누수 점검:", leak, " ① 재현 최대 차이:", repro, f" 실행 {elapsed:.0f}초")

    common_lines = []
    for T in GROUPS:
        S = OTHER[T]
        c = summ[(summ.group == T) & (summ.wafer_set == "common")].set_index(["formula_from", "model", "scale"])
        common_lines += [f"  {S}→{T} (common {int(info[(info.group == T) & (info.wafer_set == 'common')]['n'].iloc[0])}장):",
                         f"    own_refit: {c.loc[(T, 'refit', 'target'), 'rmse_mean']:.4f}",
                         f"    cross_fixed_source_scale: {c.loc[(S, 'fixed', 'source'), 'rmse_mean']:.4f}",
                         f"    cross_fixed_target_scale: {c.loc[(S, 'fixed', 'target'), 'rmse_mean']:.4f}",
                         f"    cross_refit_target_scale: {c.loc[(S, 'refit', 'target'), 'rmse_mean']:.4f}",
                         f"    mean_baseline: {c.loc[(T, 'mean', 'target'), 'rmse_mean']:.4f}"]
    head = [f"date: {today}", f"author: {AUTHOR}", "script: src/sr_cross_group.py",
            "nature: 사후 탐색적 검증. 정식 SR 결과를 본 뒤 실행, 사전확정 실험·test 평가 아님. folds_v1 오차",
            f"input: {args.selection.name}, {args.folds.name} (박종화, sr/results 브랜치 9fa27bd),"
            " data/processed/features_v1.csv + wafer_table.csv(MRR, train+val, is_outlier=0), folds_v1.csv",
            f"equations_1se: {{A-456: '{texts['A-456']}', B-456: '{texts['B-456']}'}}",
            "reused: sr_select.parse_equation / refit(max_nfev 200, 지수 −3~3) / predict / fold_masks, sr_run.load_frame(검산)",
            "scale: 변수 ÷ 그룹 train+val 전체 절댓값 중앙값 (sr_run 규칙). source = 식을 만든 그룹 척도, target = 적용 그룹 척도",
            "refit_init: 식의 PySR 상수 (상대 식이면 상대 그룹 상수)",
            f"repro_own_sr_max_abs_diff: {repro}",
            f"leakage: {leak}",
            f"runtime_sec: {elapsed:.0f}",
            "labels: train+val만. test·data/locked/ 사용 안 함. PySR 재실행 없음"]
    out.with_suffix(".yaml").write_text("\n".join(head + ["result_rmse_mean:"] + common_lines) + "\n",
                                        encoding="utf-8")
    out_d.with_suffix(".yaml").write_text("\n".join([
        f"date: {today}", f"author: {AUTHOR}", "script: src/sr_cross_group.py",
        "content: 식별 정의역 문제 웨이퍼 수. fixed = 전체 웨이퍼 1회, refit = fold별 재적합 상수로 평가 쪽 점검한 합을 5로 나눈 값"
        " (반복당 평균). missing = 상대 식 변수의 결측 웨이퍼 수",
        "columns: sqrt_neg, log_nonpos, div_zero, pow_neg_base(음수 밑 + 정수 아닌 지수), nonfinite(최종 예측 NaN/inf)", ""]),
        encoding="utf-8")
    print("저장:", out.name, out_s.name, out_d.name)


if __name__ == "__main__":
    main()

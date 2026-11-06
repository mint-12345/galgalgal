"""공동 SR 바깥 5-fold 결과의 사후 해석용 기술 통계 (2026-11-06).

저장된 결과 파일만 읽는다. PySR 탐색·모형 재적합·원자료(features_v1.csv, wafer_table.csv) 읽기는 하지 않는다.
기존 판정·분류(계획 2026-11-05 3~5절)는 바꾸지 않는다. 여기 숫자는 모두 사후 기술 통계이고 판정에 쓰지 않는다.

입력
- results/2026-11-05_전준혁_FV1_sr_joint_outer5{_oof,_selection,_inner,_structure}.csv
- results/diagnostics/2026-11-01_박종화_sr_candidates.csv (정식 SR 후보 순위, 전체 Train+Val 라벨로 계산된 것)
출력 (results/diagnostics/, 덮어쓰기 금지)
- 2026-11-06_전준혁_sr_joint_outer5_interpret.csv (+ .yaml): section, group, fold, item, value 긴 형식

실행: python src/sr_joint_outer5_interpret.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BASE = RES / "2026-11-05_전준혁_FV1_sr_joint_outer5"
CANDS = RES / "diagnostics" / "2026-11-01_박종화_sr_candidates.csv"
OUT = RES / "diagnostics" / "2026-11-06_전준혁_sr_joint_outer5_interpret"
SEED = 42
N_BOOT = 2000
DROP = (-4230160594, "B")               # 극단 B 웨이퍼 (사후 민감도만)
GROUPS = ["A-456", "B-456"]
JOINT, LR = "joint_sr", "lr_group_samevars"
INTEG = ["auc_MAIN_OUTER_s1", "auc_SLURRY_C_s1", "auc_PCHAMBER_s1"]

rows: list[dict] = []


def add(section, item, value, group="", fold="", note=""):
    rows.append({"section": section, "group": group, "fold": fold, "item": item,
                 "value": float(value) if value is not None else np.nan, "note": note})


def read(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# ---------------------------------------------------------------- 1. 예측 일치 (OOF)
def agreement(oof):
    w = oof.pivot_table(index=["WAFER_ID", "STAGE", "group", "fold", "mrr_true"], columns="model",
                        values="mrr_pred").reset_index()
    for g in GROUPS:
        for tag, d in (("all", w[w["group"] == g]),
                       ("drop_extreme", w[(w["group"] == g) & ~((w["WAFER_ID"] == DROP[0]) & (w["STAGE"] == DROP[1]))])):
            if tag == "drop_extreme" and len(d) == (w["group"] == g).sum():
                continue
            y, pj, pl = d["mrr_true"].to_numpy(), d[JOINT].to_numpy(), d[LR].to_numpy()
            s = f"agreement_{tag}"
            add(s, "n", len(d), g)
            add(s, "corr_pred_joint_lr", np.corrcoef(pj, pl)[0, 1], g)
            add(s, "rmse_between_preds", rmse(pj - pl), g, note="두 모형 예측끼리의 RMSE")
            add(s, "corr_true_joint", np.corrcoef(y, pj)[0, 1], g)
            add(s, "corr_true_lr", np.corrcoef(y, pl)[0, 1], g)
            add(s, "r2_oof_joint", 1 - np.mean((y - pj) ** 2) / np.var(y), g, note="1-MSE/var(y), OOF y 분산 기준(기술용)")
            add(s, "r2_oof_lr", 1 - np.mean((y - pl) ** 2) / np.var(y), g)
            add(s, "corr_errors_joint_lr", np.corrcoef(y - pj, y - pl)[0, 1], g, note="두 모형 오차의 상관")
            add(s, "frac_wafers_joint_abs_err_smaller", np.mean(np.abs(y - pj) < np.abs(y - pl)), g)
            add(s, "sd_mrr_true", np.std(y, ddof=1), g)
    return w


# ---------------------------------------------------------------- 2. 웨이퍼 부트스트랩 (OOF RMSE 차이)
def bootstrap(w):
    rng = np.random.default_rng(SEED)
    for g in GROUPS:
        for tag in ("all", "drop_extreme"):
            d = w[w["group"] == g]
            if tag == "drop_extreme":
                d = d[~((d["WAFER_ID"] == DROP[0]) & (d["STAGE"] == DROP[1]))]
                if len(d) == (w["group"] == g).sum():
                    continue
            y, pj, pl, f = (d[c].to_numpy() for c in ("mrr_true", JOINT, LR, "fold"))
            idx_by_fold = [np.flatnonzero(f == k) for k in np.unique(f)]
            obs = rmse(y - pj) - rmse(y - pl)
            bs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                i = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_fold])   # fold별 층화
                bs[b] = rmse(y[i] - pj[i]) - rmse(y[i] - pl[i])
            s = f"bootstrap_{tag}"
            note = "웨이퍼 재표집(fold 층화), 적합된 모형 고정. 모형 적합 변동은 반영 안 됨"
            add(s, "rmse_diff_obs", obs, g, note="OOF RMSE 공동 - LR")
            add(s, "rmse_diff_ci_lo", np.percentile(bs, 2.5), g, note=note)
            add(s, "rmse_diff_ci_hi", np.percentile(bs, 97.5), g, note=note)
            add(s, "rmse_diff_rel_obs", obs / rmse(y - pl), g, note="LR RMSE 대비 비율")
            add(s, "frac_boot_joint_better", np.mean(bs < 0), g)


# ---------------------------------------------------------------- 3. 계수 안정성
def coefs(sel, st, oof):
    full = st[st["fold"].astype(str) == "full"].iloc[0]
    add("coef", "ratio_aA_aB", full["a_A"] / full["a_B"], fold="full")
    rat = []
    for _, r in sel.iterrows():
        k = int(r["fold"])
        add("coef", "a_A", r["a_A"], fold=k)
        add("coef", "a_B", r["a_B"], fold=k)
        add("coef", "b_A", r["b_A"], fold=k)
        add("coef", "b_B", r["b_B"], fold=k)
        add("coef", "ratio_aA_aB", r["a_A"] / r["a_B"], fold=k, note="f 척도와 무관 (두 테이블이 같은 f를 공유)")
        add("coef", "same_sign_aA_aB", float(np.sign(r["a_A"]) == np.sign(r["a_B"])), fold=k)
        rat.append(r["a_A"] / r["a_B"])
    rat = np.array(rat)
    add("coef", "ratio_mean_5fold", rat.mean())
    add("coef", "ratio_sd_5fold", rat.std(ddof=1))
    add("coef", "ratio_cv_5fold", rat.std(ddof=1) / abs(rat.mean()))
    add("coef", "ratio_min_5fold", rat.min())
    add("coef", "ratio_max_5fold", rat.max())
    y = oof[oof["model"] == JOINT]
    sdA = y[y["group"] == "A-456"]["mrr_true"].std(ddof=1)
    sdB = y[y["group"] == "B-456"]["mrr_true"].std(ddof=1)
    add("coef", "ratio_sd_mrr_A_over_B", sdA / sdB, note="참고: 테이블 MRR 표준편차 비")
    # 같은 변수 LR의 테이블별 계수 (fold마다 변수 순서는 selection의 variables 열)
    for _, r in sel.iterrows():
        k = int(r["fold"])
        names = r["variables"].split()
        lr = json.loads(r["lr_coefs"])
        for g in GROUPS:
            for nm, c in zip(names, lr[g]["coef"]):
                add("lr_coef", nm, c, g, k, note="척도화 변수 기준 기울기")


# ---------------------------------------------------------------- 4. 거의 같은 점수의 후보 묶음 (안쪽 CV)
def rashomon(inner, sel, names):
    pat = {n: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])") for n in names}
    for k, d in inner.groupby("fold"):
        d = d[np.isfinite(d["mean"])].copy()
        s = sel[sel["fold"] == k].iloc[0]
        curve = d.sort_values(["complexity", "mean", "equation"]).drop_duplicates("complexity")
        best = curve.loc[curve["mean"].idxmin()]
        limit = best["mean"] + best["se"]
        near = d[d["mean"] <= limit]
        add("rashomon", "n_candidates", len(d), fold=k)
        add("rashomon", "best_inner_rmse", best["mean"], fold=k)
        add("rashomon", "limit_1se", limit, fold=k)
        add("rashomon", "selected_inner_rmse", s["inner_mean"], fold=k)
        add("rashomon", "n_within_1se", len(near), fold=k, note="안쪽 점수가 1-표준오차 한계 이하인 서로 다른 식 수")
        add("rashomon", "complexity_range_within_1se_lo", near["complexity"].min(), fold=k)
        add("rashomon", "complexity_range_within_1se_hi", near["complexity"].max(), fold=k)
        c1 = d[d["complexity"] == 1].sort_values("mean").iloc[0]
        add("rashomon", "best_single_var_inner_rmse", c1["mean"], fold=k, note=c1["equation"])
        for n in INTEG:
            has_all = d["equation"].map(lambda e: bool(pat[n].search(e)))
            add("rashomon", f"frac_all_cands_using:{n}", has_all.mean(), fold=k)
            add("rashomon", f"frac_within_1se_using:{n}", near["equation"].map(lambda e: bool(pat[n].search(e))).mean(), fold=k)
        all3 = near["equation"].map(lambda e: all(pat[n].search(e) for n in INTEG))
        add("rashomon", "frac_within_1se_using_all3_integrals", all3.mean(), fold=k)


# ---------------------------------------------------------------- 5. 변수 반복 대 사전 순위
def var_rank(st, cands):
    used = {c[4:]: int(st[st["fold"].astype(str) != "full"][c].sum()) for c in st.columns if c.startswith("use_")}
    for g in GROUPS:
        c = cands[cands["group"] == g].set_index("feature")
        for nm, k in used.items():
            if nm in c.index:
                add("var_rank", nm, k, g, note=f"rank={int(c.loc[nm, 'rank'])}; |spearman|={c.loc[nm, 'abs_spearman']:.3f}; "
                                               f"formal_selected={int(c.loc[nm, 'selected'] or 0)}")


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    t0 = time.time()
    out_csv, out_yaml = Path(f"{OUT}.csv"), Path(f"{OUT}.yaml")
    for p in (out_csv, out_yaml):
        assert not p.exists(), f"이미 있음 (덮어쓰지 않음): {p}"
    oof, sel, inner = read(f"{BASE}_oof.csv"), read(f"{BASE}_selection.csv"), read(f"{BASE}_inner.csv")
    st, cands = read(f"{BASE}_structure.csv"), read(CANDS)
    names = [c[4:] for c in st.columns if c.startswith("use_")]
    w = agreement(oof)
    bootstrap(w)
    coefs(sel, st, oof)
    rashomon(inner, sel, names)
    var_rank(st, cands)
    df = pd.DataFrame(rows)
    df.insert(0, "featureset", "features_v1")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    inputs = [f"{BASE}{s}.csv" for s in ("_oof", "_selection", "_inner", "_structure")] + [str(CANDS)]
    meta = {
        "nature": "사후 해석용 기술 통계. 저장된 결과만 읽음(탐색·재적합·원자료 읽기 없음). 계획 2026-11-05의 판정·분류는 바꾸지 않음",
        "date_run": "2026-11-06", "author": "전준혁",
        "inputs_sha256_16": {Path(p).name: sha16(p) for p in inputs},
        "seed": SEED, "n_boot": N_BOOT,
        "bootstrap": "테이블별, 바깥 fold 층화 웨이퍼 재표집. 적합된 예측을 고정하므로 탐색·적합 변동은 들어가지 않음",
        "extreme_wafer": list(DROP),
        "env": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "platform": platform.platform()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    lines = []
    for k, v in meta.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"  {kk}: {vv}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {v}")
    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_csv.name} ({len(df)}행), {meta['runtime_sec']}초")


if __name__ == "__main__":
    sys.exit(main())

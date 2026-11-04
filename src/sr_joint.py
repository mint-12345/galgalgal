"""공동 SR 후속 실험: MRR_g = a_g · f(x) + b_g (사후 후속 실험, 정식 SR 판정은 바꾸지 않음).

계획: docs/plans/2026-11-04_joint_sr_plan.md (결과 계산 전 커밋 3949cb4). 매뉴얼: 김혜연 공동SR_실험매뉴얼_1104.
- f의 형태·변수·내부 상수는 A-456·B-456 공통, a_g·b_g만 테이블별. 테이블 구분값은 f에 들어가지 않는다.
- 데이터: features_v1 + wafer_table의 MRR(train+val, is_outlier=0). 정식 SR(`sr_run.load_frame`)과 같은 라벨 출처.
  공통 후보 14개(정식 후보 합집합) 값이 모두 있는 웨이퍼만(A 940, B 951 = 정식 SR 웨이퍼).
- 척도: 변수 ÷ |중앙값|. 주 = 두 테이블 합산(train+val 전체), 민감도 = 테이블별.
- 재적합: `sr_select`의 파싱·예측·지수 범위(−3~3)를 쓰고, f 상수 + a_A, b_A, a_B, b_B를 fold 학습 쪽에서 한 번에 비선형 최소제곱.
- test 라벨과 data/locked/는 읽지 않는다.

명령:
  python src/sr_joint.py screen      1단계 (정식 파레토 식 걸러 보기, 새 SR 없음)
  python src/sr_joint.py baselines   기준선 (가)~(라)·정식 SR 재현
  (2단계 명령은 아래에 추가)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_select as ss  # noqa: E402
from cv_common import corrected_ttest  # noqa: E402
from sr_cross_group import domain_issues  # noqa: E402
from sr_run import CANDIDATES, GROUPS, MAX_ABS_EXPONENT, PROC, RESULTS  # noqa: E402

AUTHOR = "전준혁"
STAMP = "2026-11-04"                      # 계획·매뉴얼이 정한 파일 이름 날짜
KEYS = ["WAFER_ID", "STAGE"]
FEATURESET = "features_v1"
PLAN = "docs/plans/2026-11-04_joint_sr_plan.md"
PLAN_COMMIT = "3949cb4"
PARETO = RESULTS / "2026-11-01_박종화_FV1_sr_pareto.csv"
FORMAL_SELECTION = RESULTS / "diagnostics" / "2026-11-01_박종화_sr_selection.csv"
FORMAL_FOLDS = RESULTS / "2026-11-01_박종화_FV1_sr_folds.csv"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "galgalgal_joint_cache"
N_WORKERS = int(os.environ.get("SR_JOINT_WORKERS", "8"))
NATURE = "사후 후속 실험(계획 커밋 3949cb4 이후 실행). 정식 SR 판정은 바꾸지 않음. folds_v1 오차는 식 선택에 쓴 데이터라 낙관적"


# ---------------------------------------------------------------- 데이터
def candidates14() -> list[str]:
    c = pd.read_csv(CANDIDATES)
    sel = {g: c[(c["group"] == g) & (c["selected"] == 1)].sort_values("rank")["feature"].tolist() for g in GROUPS}
    names = list(dict.fromkeys(sel["A-456"] + sel["B-456"]))
    assert len(names) == 14, f"공통 후보 {len(names)}개 (계획 14개)"
    return names


def load_all() -> pd.DataFrame:
    """A-456·B-456 train+val(이상치 제외) 전체 웨이퍼, 척도 적용 전. MRR은 wafer_table(정식 SR과 같은 출처)."""
    feats = pd.read_csv(PROC / "features_v1.csv")
    table = pd.read_csv(PROC / "wafer_table.csv", usecols=KEYS + ["split", "MRR", "is_outlier"])
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    assert table["MRR"].notna().all()
    d = feats[feats["group"].isin(GROUPS)].merge(table, on=KEYS, validate="one_to_one")
    return d.reset_index(drop=True)


def joint_frame() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(공동 웨이퍼 표, 전체 표, 후보 14개). 공동 = 14개 값이 모두 있는 웨이퍼, 테이블 순서 A → B."""
    full = load_all()
    names = candidates14()
    d = full[full[names].notna().all(axis=1)].copy()
    d["gi"] = (d["group"] == "B-456").astype(int)
    d = d.sort_values(["gi"], kind="mergesort").merge(ss.head_reset_intervals(), on=KEYS, how="left",
                                                     validate="one_to_one").reset_index(drop=True)
    counts = d["group"].value_counts().to_dict()
    assert counts == {"A-456": 940, "B-456": 951}, f"공동 웨이퍼 {counts} (계획 A 940 / B 951)"
    return d, full, names


def scale_table(full: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
    """|중앙값| (그룹 train+val 전체, 결측 웨이퍼 포함 — sr_run.load_frame과 같은 규칙)."""
    out = {"pooled": full[cols].abs().median()}
    for g in GROUPS:
        out[g] = full.loc[full["group"] == g, cols].abs().median()
    assert all((s >= 1e-9).all() for s in out.values())
    return out


def design(d: pd.DataFrame, cols: list[str], scales: dict, mode: str) -> dict[str, np.ndarray]:
    """mode = pooled(주) / group(민감도). 행마다 자기 테이블 척도로 나눈다."""
    if mode == "pooled":
        return {c: (d[c] / scales["pooled"][c]).to_numpy(float) for c in cols}
    div = pd.DataFrame({g: scales[g] for g in GROUPS}).T          # 그룹 × 변수
    return {c: (d[c] / d["group"].map(div[c])).to_numpy(float) for c in cols}


# ---------------------------------------------------------------- 공동 재적합
def _init_ab(f, y, gi):
    a, b = np.ones(2), np.zeros(2)
    for g in (0, 1):
        m = (gi == g) & np.isfinite(f)
        if m.sum() >= 3 and np.var(f[m]) > 0:
            a[g], b[g] = np.polyfit(f[m], y[m], 1)
        else:
            b[g] = y[gi == g].mean()
    return a, b


def joint_refit(node, theta0, X, y, gi, max_nfev=200):
    """f 내부 상수(공통) + a_g·b_g(테이블별)를 함께 맞춘다. 학습 쪽 X·y·gi만 받는다."""
    ids = sorted(ss.const_ids(node))
    theta = np.array(theta0, dtype=float)
    n = len(y)
    a0, b0 = _init_ab(ss.predict(node, theta, X, n), y, gi)
    powers = ss.exponent_ids(node)
    lo = np.r_[[-MAX_ABS_EXPONENT if i in powers else -np.inf for i in ids], [-np.inf] * 4]
    hi = np.r_[[MAX_ABS_EXPONENT if i in powers else np.inf for i in ids], [np.inf] * 4]
    k = len(ids)

    def unpack(p):
        t = theta.copy()
        t[ids] = p[:k]
        return t, p[k:k + 2], p[k + 2:k + 4]

    def residual(p):
        t, a, b = unpack(p)
        r = a[gi] * ss.predict(node, t, X, n) + b[gi] - y
        r[np.isnan(r)] = 1e6
        return r

    x0 = np.r_[np.clip(theta[ids], lo[:k], hi[:k]), a0, b0]
    res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=max_nfev)
    t, a, b = unpack(res.x)
    return t, a, b, {"nfev": int(res.nfev), "status": int(res.status)}


def joint_predict(node, theta, a, b, X, gi):
    return a[gi] * ss.predict(node, theta, X, len(gi)) + b[gi]


def _metrics(p, y):
    ok = np.isfinite(p)
    e = p[ok] - y[ok]
    if ok.sum() == 0:
        return {"mse": np.nan, "mae": np.nan, "rmse": np.nan, "n_eval": 0, "n_eval_nan": int((~ok).sum())}
    return {"mse": float(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "n_eval": int(ok.sum()), "n_eval_nan": int((~ok).sum())}


def check_fold(d: pd.DataFrame, ev: np.ndarray) -> None:
    """학습·평가 쪽 키와 WAFER_ID가 겹치지 않는지 (A·B 공유 WAFER_ID 포함)."""
    tr_id, ev_id = set(d.loc[~ev, "WAFER_ID"]), set(d.loc[ev, "WAFER_ID"])
    assert tr_id.isdisjoint(ev_id), "학습·평가 쪽 WAFER_ID 겹침"


def eval_joint(node, theta0, X, y, gi, masks, max_nfev=200):
    """fold마다 학습 쪽으로 공동 재적합 → 테이블별 평가 지표. 평가 쪽 y는 채점에만 쓴다."""
    rows = []
    for r in sorted(masks):
        ev = masks[r]
        tr = ~ev
        t, a, b, info = joint_refit(node, theta0, ss._subset(X, tr), y[tr], gi[tr], max_nfev)
        n_tr_nan = int(np.isnan(ss.predict(node, t, ss._subset(X, tr), int(tr.sum()))).sum())
        p = joint_predict(node, t, a, b, ss._subset(X, ev), gi[ev])
        for g in (0, 1):
            m = gi[ev] == g
            rows.append({"fold": r, "group": GROUPS[g], **_metrics(p[m], y[ev][m]),
                         "n_train": int((tr & (gi == g)).sum()), "n_train_nan": n_tr_nan,
                         "a": float(a[g]), "b": float(b[g]), "nfev": info["nfev"], "ls_status": info["status"]})
    return rows


# ---------------------------------------------------------------- 1단계 작업자
_W: dict = {}


def _worker_init(mode: str, max_nfev: int):
    d, full, names = joint_frame()
    sc = scale_table(full, names)
    _W.update(d=d, X=design(d, names, sc, mode), y=d["MRR"].to_numpy(float), gi=d["gi"].to_numpy(),
              masks=ss.fold_masks(d[KEYS]), max_nfev=max_nfev)


def _screen_one(eq_text: str):
    eq = ss.parse_equation(eq_text)
    t0 = time.perf_counter()
    rows = eval_joint(eq.node, eq.theta, _W["X"], _W["y"], _W["gi"], _W["masks"], _W["max_nfev"])
    return eq_text, rows, time.perf_counter() - t0


def run_parallel(texts: list[str], mode: str, max_nfev: int, cache: Path, tag: str) -> pd.DataFrame:
    """식마다 결과를 캐시(jsonl)에 덧붙인다. 다시 실행하면 끝난 식은 건너뛴다."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["equation"]] = rec
    todo = [t for t in texts if t not in done]
    print(f"[{tag}] 전체 {len(texts)}, 끝남 {len(done)}, 남음 {len(todo)}, 작업자 {N_WORKERS}", flush=True)
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(N_WORKERS, initializer=_worker_init, initargs=(mode, max_nfev)) as ex, \
                cache.open("a", encoding="utf-8") as fh:
            for i, (text, rows, sec) in enumerate(ex.map(_screen_one, todo, chunksize=1), 1):
                rec = {"equation": text, "seconds": sec, "rows": rows}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[text] = rec
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} ({time.perf_counter() - t0:.0f}초)", flush=True)
    out = []
    for t in texts:
        for r in done[t]["rows"]:
            out.append({"equation": t, "seconds": done[t]["seconds"], **r})
    return pd.DataFrame(out)


def summarize_screen(res: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """식 × 테이블 평균. 평가 쪽 계산 불가가 있으면 실패."""
    g = res.groupby(["equation", "group"]).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", lambda x: x.std(ddof=1)),
                                               mse=("mse", "mean"), mae=("mae", "mean"),
                                               n_eval_nan=("n_eval_nan", "sum"), n_train_nan=("n_train_nan", "max"),
                                               nfev_max=("nfev", "max")).reset_index()
    w = g.pivot(index="equation", columns="group")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index().merge(meta, on="equation", how="left")
    w["failed"] = (w["n_eval_nan_A-456"] + w["n_eval_nan_B-456"]) > 0
    w["rmse_mean2"] = (w["rmse_A-456"] + w["rmse_B-456"]) / 2
    w.loc[w["failed"], "rmse_mean2"] = np.inf
    return w.sort_values(["rmse_mean2", "complexity", "equation"]).reset_index(drop=True)


# ---------------------------------------------------------------- 기준선
def baseline_rows(d, X, cols, masks, kinds=("lr_group", "lr_common", "mean")):
    """(가) 테이블별 선형회귀 / (나) 공통 기울기 + 테이블별 절편 / 학습 쪽 평균. 같은 웨이퍼·fold."""
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    M = np.column_stack([X[c] for c in cols]) if cols else np.zeros((len(y), 0))
    rows = []
    for r in sorted(masks):
        ev = masks[r]
        tr = ~ev
        for kind in kinds:
            p = np.empty(ev.sum())
            if kind == "lr_group":
                for g in (0, 1):
                    mt, me = tr & (gi == g), gi[ev] == g
                    p[me] = LinearRegression().fit(M[mt], y[mt]).predict(M[ev][me])
            elif kind == "lr_common":
                Z = np.column_stack([M, gi])
                p = LinearRegression().fit(Z[tr], y[tr]).predict(Z[ev])
            elif kind == "mean":
                for g in (0, 1):
                    me = gi[ev] == g
                    p[me] = y[tr & (gi == g)].mean()
            for g in (0, 1):
                m = gi[ev] == g
                rows.append({"fold": r, "group": GROUPS[g], "model": kind, **_metrics(p[m], y[ev][m]),
                             "n_train": int((tr & (gi == g)).sum())})
    return rows


def formal_sr_rows(d, full, masks_by_group=None):
    """(다) 정식 SR 테이블별 최종 식(1se), 그 테이블 척도(정식 후보 10개, sr_run 규칙), fold마다 상수 재적합."""
    sel = pd.read_csv(FORMAL_SELECTION, encoding="utf-8-sig")
    sel = sel[sel["rule"] == "1se"].set_index("group")
    rows = []
    c = pd.read_csv(CANDIDATES)
    for gidx, g in enumerate(GROUPS):
        names = c[(c["group"] == g) & (c["selected"] == 1)]["feature"].tolist()
        scale = full.loc[full["group"] == g, names].abs().median()
        sub = d[d["gi"] == gidx].reset_index(drop=True)
        eq = ss.parse_equation(sel.loc[g, "equation"])
        X = {n: (sub[n] / scale[n]).to_numpy(float) for n in names}
        y = sub["MRR"].to_numpy(float)
        masks = ss.fold_masks(sub[KEYS])
        for r in sorted(masks):
            ev = masks[r]
            th = ss.refit(eq.node, eq.theta, ss._subset(X, ~ev), y[~ev])
            p = ss.predict(eq.node, th, ss._subset(X, ev), int(ev.sum()))
            rows.append({"fold": r, "group": g, "model": "formal_sr", **_metrics(p, y[ev]), "n_train": int((~ev).sum())})
    return rows, sel


def rf_rows(d, masks):
    cols = ss.rf_columns()
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    R = d[cols].to_numpy(float)
    rows = []
    for r in sorted(masks):
        ev = masks[r]
        tr = ~ev
        for g in (0, 1):
            mt, me = tr & (gi == g), ev & (gi == g)
            p = RandomForestRegressor(max_features=1.0, random_state=ss.SEED, n_jobs=N_WORKERS).fit(R[mt], y[mt]).predict(R[me])
            rows.append({"fold": r, "group": GROUPS[g], "model": "rf", **_metrics(p, y[me]), "n_train": int(mt.sum())})
    return rows


# ---------------------------------------------------------------- 기록
def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def env_info() -> dict:
    import scipy
    import sklearn
    info = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(),
            "workers": N_WORKERS}
    try:
        import importlib.metadata as md
        info["pysr"] = md.version("pysr")
        info["juliacall"] = md.version("juliacall")
    except Exception:  # noqa: BLE001
        pass
    return info


def git_head() -> str:
    head = (ROOT / ".git").read_text().strip() if (ROOT / ".git").is_file() else ""
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip() + (" (+작업 중 변경)" if subprocess.run(
                                  ["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return head


def write_yaml(path: Path, items: dict) -> None:
    def fmt(v, ind=0):
        pad = "  " * ind
        if isinstance(v, dict):
            return "\n" + "\n".join(f"{pad}  {k}: {fmt(x, ind + 1)}" for k, x in v.items())
        return str(v)
    path.write_text("\n".join(f"{k}: {fmt(v)}" for k, v in items.items()) + "\n", encoding="utf-8")


def guard(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            sys.exit(f"이미 있어서 멈춘다 (덮어쓰지 않음): {p}")


def common_yaml(t0: float, extra: dict) -> dict:
    inputs = [PROC / "features_v1.csv", PROC / "wafer_table.csv", PROC / "folds_v1.csv", CANDIDATES, PARETO,
              FORMAL_SELECTION]
    return {"nature": NATURE, "date_run": date.today().isoformat(), "author": AUTHOR, "plan": f"{PLAN} ({PLAN_COMMIT})",
            "code_commit": git_head(), "labels": "wafer_table.csv MRR, train+val, is_outlier=0 (정식 SR과 같은 출처)."
            " data/raw가 이 PC에 없어 load_labels()는 쓰지 못함. test·data/locked/ 사용 안 함",
            "inputs_sha256_16": {p.name: sha16(p) for p in inputs}, "env": env_info(),
            "runtime_sec": round(time.perf_counter() - t0), **extra}


# ---------------------------------------------------------------- 명령: 1단계
def cmd_screen(args):
    t0 = time.perf_counter()
    out = RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_screen.csv"
    out_f = out.with_name(f"{out.stem}_folds.csv")
    guard(out, out_f, out.with_suffix(".yaml"))
    p = pd.read_csv(PARETO)
    assert len(p) == 281
    meta = (p.groupby("equation").agg(complexity=("complexity", "first"), source_group=("group", "first"),
                                      n_source=("group", "nunique"), seeds=("seed", lambda s: ",".join(map(str, sorted(s)))))
            .reset_index())
    meta["n_vars"] = meta["equation"].map(lambda t: len(ss.variables(ss.parse_equation(t).node)))
    no_var = meta[meta["n_vars"] == 0]["equation"].tolist()
    texts = meta[meta["n_vars"] > 0]["equation"].tolist()
    print(f"고유 식 {len(meta)}개, 변수 없는 식 {len(no_var)}개 제외 → {len(texts)}개")
    res = {}
    for mode in ("pooled", "group"):
        res[mode] = run_parallel(texts, mode, 200, CACHE / f"screen_{mode}.jsonl", f"screen {mode}").assign(scale=mode)
    summ = {m: summarize_screen(r, meta) for m, r in res.items()}
    top = summ["pooled"].head(5)["equation"].tolist()
    nfev = run_parallel(top, "pooled", 2000, CACHE / "screen_pooled_nfev2000.jsonl", "top5 nfev2000").assign(scale="pooled_nfev2000")
    summ["pooled_nfev2000"] = summarize_screen(nfev, meta)
    folds = pd.concat([res["pooled"], res["group"], nfev], ignore_index=True)
    folds.insert(0, "featureset", FEATURESET)
    folds.round(6).to_csv(out_f, index=False, encoding="utf-8-sig")
    allsum = pd.concat([s.assign(scale=m, rank=np.arange(1, len(s) + 1)) for m, s in summ.items()], ignore_index=True)
    allsum.insert(0, "featureset", FEATURESET)
    allsum.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    s = summ["pooled"]
    print(s.head(10)[["equation", "complexity", "source_group", "rmse_A-456", "rmse_B-456", "rmse_mean2"]].round(4).to_string())
    print("실패(평가 계산 불가) 식:", {m: int(x["failed"].sum()) for m, x in summ.items()})
    write_yaml(out.with_suffix(".yaml"), common_yaml(t0, {
        "stage": "1단계: 정식 SR 파레토 식을 공통 f로 두고 걸러 보기 (새 SR 없음). 선택 낙관성 있음, 판정에 쓰지 않음",
        "candidates": f"{PARETO.name} 281행 → 고유 식 {len(meta)}개, 변수 없는 식 {len(no_var)}개 제외 → {len(texts)}개",
        "model": "MRR_g = a_g·f(x) + b_g. f 상수 공통, a_g·b_g 테이블별, fold 학습 쪽에서 함께 비선형 최소제곱 (max_nfev 200, 지수 −3~3)",
        "scale": "pooled = 두 테이블 합산 |중앙값|(주), group = 테이블별 |중앙값|(민감도)",
        "wafers": "A-456 940 / B-456 951 (공통 후보 14개 값 있음 = 정식 SR 웨이퍼)",
        "folds": "folds_v1 25 fold, 학습 = 두 테이블 학습 조각 합집합, 평가는 테이블별",
        "rank": "테이블별 25 fold 평균 RMSE의 두 테이블 평균 (rmse_mean2)",
        "failed": {m: int(x["failed"].sum()) for m, x in summ.items()},
        "outputs": {"summary": out.name, "folds": out_f.name},
    }))
    print(f"저장: {out.name}, {out_f.name} ({time.perf_counter() - t0:.0f}초)")


def cmd_baselines(args):
    """기준선 (가)·(나)·평균은 변수 목록마다 다르므로 여기서는 (다)·(라)와 재현 점검만 저장한다."""
    t0 = time.perf_counter()
    out = RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_baselines.csv"
    guard(out, out.with_suffix(".yaml"))
    d, full, names = joint_frame()
    masks = ss.fold_masks(d[KEYS])
    for ev in masks.values():
        check_fold(d, ev)
    formal, sel = formal_sr_rows(d, full)
    rf = rf_rows(d, masks)
    sc = scale_table(full, names)
    X = design(d, names, sc, "pooled")
    b14 = [dict(r, model=f"{r['model']}_14") for r in baseline_rows(d, X, names, masks)]
    res = pd.DataFrame(formal + rf + b14)
    res.insert(0, "featureset", FEATURESET)
    # 재현 점검: 정식 SR 결과 파일의 sr_final·rf와 fold별 비교
    old = pd.read_csv(FORMAL_FOLDS, encoding="utf-8-sig")
    rep = {}
    for model_new, model_old in (("formal_sr", "sr_final"), ("rf", "rf")):
        a = res[res["model"] == model_new].set_index(["group", "fold"])["rmse"].sort_index()
        b = old[old["model"] == model_old].set_index(["group", "fold"])["rmse"].sort_index()
        rep[model_new] = float((a - b).abs().max())
    res.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(res.groupby(["model", "group"])["rmse"].agg(["mean", "std"]).round(4))
    print("정식 결과와 fold별 최대 차이:", rep)
    write_yaml(out.with_suffix(".yaml"), common_yaml(t0, {
        "stage": "기준선 (다) 정식 SR 테이블별 최종 식 재적합, (라) RF(F-free 112, since_POLISH_TABLE_w 제외), 참고: 14개 변수 선형회귀",
        "repro_max_abs_rmse_diff_vs_formal": rep,
        "formal_equations": {g: sel.loc[g, "equation"] for g in GROUPS},
        "note": "(가)·(나)·평균은 비교 대상 f의 변수로 따로 계산해 각 결과 파일에 넣는다",
    }))
    print(f"저장: {out.name} ({time.perf_counter() - t0:.0f}초)")


def compare_rows(d, X, eq_text, masks, label):
    """한 식의 공동 모형과 같은 변수의 (가)·(나)·평균을 같은 웨이퍼·fold로."""
    eq = ss.parse_equation(eq_text)
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    rows = [dict(r, model="joint") for r in eval_joint(eq.node, eq.theta, X, y, gi, masks)]
    rows += baseline_rows(d, X, sorted(ss.variables(eq.node)), masks)
    return [dict(r, candidate=label, equation=eq_text) for r in rows]


def paired_tests(res: pd.DataFrame, a: str, b: str, by: list[str]) -> list[dict]:
    """테이블마다 fold별 MSE 차이 a − b의 보정 t-검정 (비율 = 그 테이블 평가/학습 장수)."""
    out = []
    for key, g in res.groupby(by):
        pa = g[g["model"] == a].set_index("fold").sort_index()
        pb = g[g["model"] == b].set_index("fold").sort_index()
        if len(pa) != 25 or len(pb) != 25 or (pa["n_eval"] != pb["n_eval"]).any():
            continue
        t = corrected_ttest(pa["mse"] - pb["mse"], pa["n_train"], pa["n_eval"])
        out.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "a": a, "b": b,
                    "rmse_a": pa["rmse"].mean(), "rmse_b": pb["rmse"].mean(),
                    "rmse_diff": pa["rmse"].mean() - pb["rmse"].mean(), **t})
    return out


def cmd_screen_compare(args):
    t0 = time.perf_counter()
    out = RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_screen_compare.csv"
    out_t = out.with_name(f"{out.stem}_tests.csv")
    guard(out, out_t, out.with_suffix(".yaml"))
    s = pd.read_csv(RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_screen.csv", encoding="utf-8-sig")
    top = s[s["scale"] == "pooled"].head(5)
    d, full, names = joint_frame()
    X = design(d, names, scale_table(full, names), "pooled")
    masks = ss.fold_masks(d[KEYS])
    rows = []
    for i, e in enumerate(top["equation"], 1):
        rows += compare_rows(d, X, e, masks, f"top{i}")
    res = pd.DataFrame(rows)
    tests = pd.DataFrame(paired_tests(res, "joint", "lr_group", ["candidate", "group"])
                         + paired_tests(res, "joint", "lr_common", ["candidate", "group"]))
    res.insert(0, "featureset", FEATURESET)
    res.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    tests.round(6).to_csv(out_t, index=False, encoding="utf-8-sig")
    print(res.groupby(["candidate", "model", "group"])["rmse"].mean().unstack().round(4))
    print(tests[["candidate", "group", "a", "b", "rmse_a", "rmse_b", "rmse_diff", "t", "p"]].round(4).to_string(index=False))
    write_yaml(out.with_suffix(".yaml"), common_yaml(t0, {
        "stage": "1단계 상위 5개 식과 같은 변수의 기준선 (가) 테이블별 선형회귀, (나) 공통 기울기 + 테이블별 절편, 학습 쪽 평균."
                 " 선택 낙관성 있음, 판정에 쓰지 않음 (검정은 참고)",
        "scale": "pooled", "outputs": {"folds": out.name, "tests": out_t.name},
    }))
    print(f"저장: {out.name}, {out_t.name}")


def _nested_one(eq_text: str):
    """반복 0의 바깥 fold마다: 안쪽 4-fold(바깥 학습 쪽 안) 점수와 바깥 평가 점수."""
    eq = ss.parse_equation(eq_text)
    X, y, gi, masks = _W["X"], _W["y"], _W["gi"], _W["masks"]
    out = []
    for r in range(5):                                   # 반복 0: r = 0..4
        outer_ev = masks[r]
        inner = []
        for r2 in range(5):
            if r2 == r:
                continue
            ev2 = masks[r2]
            tr2 = ~outer_ev & ~ev2
            t, a, b, _ = joint_refit(eq.node, eq.theta, ss._subset(X, tr2), y[tr2], gi[tr2], _W["max_nfev"])
            p = joint_predict(eq.node, t, a, b, ss._subset(X, ev2), gi[ev2])
            rm = [_metrics(p[gi[ev2] == g], y[ev2][gi[ev2] == g]) for g in (0, 1)]
            inner.append(np.inf if any(m["n_eval_nan"] for m in rm) else (rm[0]["rmse"] + rm[1]["rmse"]) / 2)
        tr = ~outer_ev
        t, a, b, _ = joint_refit(eq.node, eq.theta, ss._subset(X, tr), y[tr], gi[tr], _W["max_nfev"])
        p = joint_predict(eq.node, t, a, b, ss._subset(X, outer_ev), gi[outer_ev])
        outer = {GROUPS[g]: _metrics(p[gi[outer_ev] == g], y[outer_ev][gi[outer_ev] == g]) for g in (0, 1)}
        out.append({"outer": r, "inner_score": float(np.mean(inner)),
                    **{f"outer_rmse_{g}": outer[g]["rmse"] for g in GROUPS},
                    **{f"outer_nan_{g}": outer[g]["n_eval_nan"] for g in GROUPS}})
    return eq_text, out, 0.0


def cmd_screen_nested(args):
    """부분 중첩 검증(계획 10절 6번, 1단계 후보). 파레토 탐색 자체는 전체 데이터를 써서 완전한 독립 검증은 아님."""
    t0 = time.perf_counter()
    out = RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_screen_nested.csv"
    guard(out, out.with_suffix(".yaml"))
    s = pd.read_csv(RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_screen.csv", encoding="utf-8-sig")
    texts = s[s["scale"] == "pooled"]["equation"].tolist()
    cache = CACHE / "screen_nested.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["equation"]] = rec["rows"]
    todo = [t for t in texts if t not in done]
    print(f"[nested] 남음 {len(todo)} / {len(texts)}", flush=True)
    if todo:
        with ProcessPoolExecutor(N_WORKERS, initializer=_worker_init, initargs=("pooled", 200)) as ex, \
                cache.open("a", encoding="utf-8") as fh:
            for i, (text, rows, _) in enumerate(ex.map(_nested_one, todo, chunksize=1), 1):
                fh.write(json.dumps({"equation": text, "rows": rows}, ensure_ascii=False) + "\n")
                fh.flush()
                done[text] = rows
                if i % 40 == 0:
                    print(f"  {i}/{len(todo)} ({time.perf_counter() - t0:.0f}초)", flush=True)
    long = pd.DataFrame([{"equation": t, **r} for t in texts for r in done[t]])
    # 바깥 fold마다 안쪽 점수 최소 식을 고르고 바깥 평가로 채점. 같은 바깥 fold에서 전체 25-fold 1위 식과 (가)도 같이 적는다.
    d, full, names = joint_frame()
    X = design(d, names, scale_table(full, names), "pooled")
    masks = ss.fold_masks(d[KEYS])
    picks = []
    for r, g in long.groupby("outer"):
        best = g.sort_values(["inner_score", "equation"]).iloc[0]
        glob = g[g["equation"] == texts[0]].iloc[0]
        lr = pd.DataFrame(baseline_rows(d, X, sorted(ss.variables(ss.parse_equation(best["equation"]).node)),
                                        {r: masks[r]}, kinds=("lr_group",))).set_index("group")["rmse"]
        picks.append({"outer": r, "picked": best["equation"], "picked_is_global_top1": best["equation"] == texts[0],
                      **{f"picked_rmse_{gg}": best[f"outer_rmse_{gg}"] for gg in GROUPS},
                      **{f"global_top1_rmse_{gg}": glob[f"outer_rmse_{gg}"] for gg in GROUPS},
                      **{f"lr_group_same_vars_rmse_{gg}": lr[gg] for gg in GROUPS}})
    picks = pd.DataFrame(picks)
    picks.insert(0, "featureset", FEATURESET)
    picks.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(picks.drop(columns=["picked"]).round(4).to_string(index=False))
    print(picks[[c for c in picks.columns if "rmse" in c]].mean().round(4))
    write_yaml(out.with_suffix(".yaml"), common_yaml(t0, {
        "stage": "1단계 부분 중첩 검증: folds_v1 반복 0의 바깥 fold 5개마다 바깥 학습 쪽 안의 4-fold로 241개 식 중 하나를 고르고"
                 " 바깥 평가 쪽에서 채점. 파레토 식은 전체 데이터로 찾은 것이라 완전한 독립 검증 아님",
        "selection_metric": "안쪽 4-fold의 (A RMSE + B RMSE)/2 평균",
        "compare": "같은 바깥 fold의 전체 25-fold 1위 식, 고른 식과 같은 변수의 (가) 테이블별 선형회귀",
    }))
    print(f"저장: {out.name} ({time.perf_counter() - t0:.0f}초)")


# ---------------------------------------------------------------- 2단계: 선택·판정·추가 보고
SEEDS = list(range(42, 52))
MULTIPLIERS = [1.00, 1.02, 1.05, 1.10]
ALPHA = 0.05


def load_search() -> tuple[pd.DataFrame, dict]:
    """프로세스별 탐색 캐시를 합친다. 정식 설정(반복 500, 시드 42~51)만."""
    recs = []
    for p in sorted(CACHE.glob("search_pareto_p*.jsonl")):
        recs += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    recs = [r for r in recs if r["niterations"] == 500]
    seeds = sorted(r["seed"] for r in recs)
    assert seeds == SEEDS, f"시드가 모자라거나 겹친다: {seeds}"
    rows = [{"seed": r["seed"], "fit_seconds": r["fit_seconds"], **e} for r in recs for e in r["equations"]]
    info = {"fit_seconds": {r["seed"]: r["fit_seconds"] for r in recs}, "n_rows": recs[0]["n_rows"]}
    return pd.DataFrame(rows), info


def one_se(curve: pd.DataFrame, col="mean", se_col="se"):
    best = curve.loc[curve[col].idxmin()]
    limit = best[col] + best[se_col]
    return curve[curve[col] <= limit].sort_values("complexity").iloc[0], float(limit)


def judge(tests_a: pd.DataFrame, rmse_joint: dict, rmse_c: dict, se_c: dict) -> dict:
    """계획 9절. tests_a: 테이블별 공동 − (가) 검정."""
    lower = {}
    for g in GROUPS:
        t = tests_a.set_index("group").loc[g]
        lower[g] = bool(t["mean_diff"] < 0 and t["p"] < ALPHA)
    within = {g: bool(rmse_joint[g] - rmse_c[g] <= se_c[g]) for g in GROUPS}
    if all(lower.values()) and all(within.values()):
        verdict = "공통 형태 있음"
    elif any(lower.values()):
        verdict = "부분적"
    else:
        verdict = "공통 형태 없음"
    border = []
    for g in GROUPS:
        p = float(tests_a.set_index("group").loc[g, "p"])
        if 0.04 <= p <= 0.06:
            border.append(f"{g} p = {p:.3f}")
        gap = rmse_joint[g] - rmse_c[g]
        if abs(gap - se_c[g]) <= 0.1 * se_c[g]:
            border.append(f"{g} RMSE 차이 {gap:.3f} 대 SE {se_c[g]:.3f}")
    return {"verdict": verdict + (" (경계: " + "; ".join(border) + ")" if border else ""),
            "lower_than_ga": lower, "within_1se_of_formal": within,
            "rmse_joint_minus_formal": {g: rmse_joint[g] - rmse_c[g] for g in GROUPS}, "se_formal": se_c}


def standardized_coefs(node, theta, a, b, X, gi):
    """f를 두 테이블 합친 데이터 기준 평균 0·표준편차 1로 바꾼 a_g·b_g (f 바깥 상수는 a·b와 겹침)."""
    f = ss.predict(node, theta, X, len(gi))
    mu, sd = float(np.nanmean(f)), float(np.nanstd(f))
    return {GROUPS[g]: {"a_std": float(a[g] * sd), "b_std": float(b[g] + a[g] * mu), "a_raw": float(a[g]),
                        "b_raw": float(b[g])} for g in (0, 1)}, mu, sd


def interval_stability(d, node, theta, X):
    """헤드 부품 리셋 구간마다 f 고정, a_g·b_g만 단순회귀. 최소 웨이퍼 max(10, 2 × (2 + 1)) = 10."""
    f = ss.predict(node, theta, X, len(d))
    y = d["MRR"].to_numpy(float)
    rows = []
    for (g, k), sub in d.groupby(["group", "interval"]):
        idx = sub.index.to_numpy()
        used = len(idx) >= 10
        row = {"group": g, "interval": int(k), "n": len(idx), "used": int(used)}
        if used:
            a, b = np.polyfit(f[idx], y[idx], 1)
            row.update(a=float(a), b=float(b))
        rows.append(row)
    st = pd.DataFrame(rows)
    summ = {}
    for g, s in st[st["used"] == 1].groupby("group"):
        summ[g] = {"n_intervals_used": len(s), "n_skipped": int((st[(st.group == g)]["used"] == 0).sum()),
                   "a_cv": float(s["a"].std(ddof=1) / abs(s["a"].mean())), "b_cv": float(s["b"].std(ddof=1) / abs(s["b"].mean())),
                   "a_sign_same": bool(len(set(np.sign(s["a"]))) == 1)}
    return st, summ


def eval_joint_fold_scale(d, names_used, node, theta0, masks, full_cols):
    """민감도: 척도를 fold 학습 쪽 공동 웨이퍼의 합산 |중앙값|으로 계산."""
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    rows = []
    for r in sorted(masks):
        ev = masks[r]
        tr = ~ev
        sc = d.loc[tr, full_cols].abs().median()
        X = {c: (d[c] / sc[c]).to_numpy(float) for c in names_used}
        t, a, b, _ = joint_refit(node, theta0, ss._subset(X, tr), y[tr], gi[tr])
        p = joint_predict(node, t, a, b, ss._subset(X, ev), gi[ev])
        for g in (0, 1):
            m = gi[ev] == g
            rows.append({"fold": r, "group": GROUPS[g], "model": "joint_foldscale", **_metrics(p[m], y[ev][m]),
                         "n_train": int((tr & (gi == g)).sum())})
    return rows


def _nested_inner(eq_text: str):
    """반복 0 바깥 fold마다 안쪽 4 fold 점수 목록과 바깥 평가 RMSE."""
    eq = ss.parse_equation(eq_text)
    X, y, gi, masks = _W["X"], _W["y"], _W["gi"], _W["masks"]
    out = []
    for r in range(5):
        outer_ev = masks[r]
        inner = []
        for r2 in range(5):
            if r2 == r:
                continue
            ev2 = masks[r2]
            tr2 = ~outer_ev & ~ev2
            t, a, b, _ = joint_refit(eq.node, eq.theta, ss._subset(X, tr2), y[tr2], gi[tr2], _W["max_nfev"])
            p = joint_predict(eq.node, t, a, b, ss._subset(X, ev2), gi[ev2])
            rm = [_metrics(p[gi[ev2] == g], y[ev2][gi[ev2] == g]) for g in (0, 1)]
            inner.append(np.inf if any(m["n_eval_nan"] for m in rm) else (rm[0]["rmse"] + rm[1]["rmse"]) / 2)
        tr = ~outer_ev
        t, a, b, _ = joint_refit(eq.node, eq.theta, ss._subset(X, tr), y[tr], gi[tr], _W["max_nfev"])
        p = joint_predict(eq.node, t, a, b, ss._subset(X, outer_ev), gi[outer_ev])
        o = {GROUPS[g]: _metrics(p[gi[outer_ev] == g], y[outer_ev][gi[outer_ev] == g]) for g in (0, 1)}
        out.append({"outer": r, "inner": inner, **{f"outer_rmse_{g}": o[g]["rmse"] for g in GROUPS},
                    **{f"outer_mse_{g}": o[g]["mse"] for g in GROUPS}})
    return eq_text, out, 0.0


def cmd_select(args):
    t0 = time.perf_counter()
    base = RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint"
    paths = {k: base.with_name(f"{base.name}_{k}.csv") for k in
             ("pareto", "curve", "selection", "compare", "tests", "judgment", "coefs", "stability", "sensitivity",
              "domain", "nested")}
    guard(*paths.values(), *[p.with_suffix(".yaml") for p in paths.values()])
    par, sinfo = load_search()
    n_bad = int((~par["exponent_ok"]).sum())
    par_ok = par[par["exponent_ok"]]
    uniq = (par_ok.sort_values(["complexity", "seed"]).groupby("f")
            .agg(complexity=("complexity", "min"), seeds=("seed", lambda s: ",".join(map(str, sorted(set(s))))),
                 loss=("loss", "min")).reset_index())
    print(f"파레토 {len(par)}행, 사후 지수 제외 {n_bad}, 고유 f {len(uniq)}개", flush=True)
    meta = uniq.rename(columns={"f": "equation"})
    res = run_parallel(meta["equation"].tolist(), "pooled", 200, CACHE / "select_pooled.jsonl", "select pooled")
    # fold 지표 = 두 테이블 RMSE 평균
    fm = res.pivot_table(index=["equation", "fold"], columns="group", values="rmse").reset_index()
    nan_ = res.groupby("equation")["n_eval_nan"].sum()
    fm["metric"] = (fm["A-456"] + fm["B-456"]) / 2
    sc = fm.groupby("equation")["metric"].agg(mean="mean", sd=lambda x: x.std(ddof=1)).reset_index()
    sc["se"] = sc["sd"] / np.sqrt(25)
    sc = sc.merge(meta, on="equation")
    sc["failed"] = sc["equation"].map(nan_) > 0
    ok = sc[~sc["failed"]]
    curve = ok.sort_values(["complexity", "mean", "equation"]).drop_duplicates("complexity").reset_index(drop=True)
    final, limit = one_se(curve)
    sel = [{"rule": "1se", "limit": limit, **final.to_dict()}]
    for m in MULTIPLIERS:
        lim = curve["mean"].min() * m
        pick = curve[curve["mean"] <= lim].sort_values("complexity").iloc[0]
        sel.append({"rule": f"x{m:.2f}", "limit": lim, **pick.to_dict()})
    sel = pd.DataFrame(sel)
    f_text = final["equation"]
    eq = ss.parse_equation(f_text)
    print("최종 f (1se):", f_text, "복잡도", final["complexity"], flush=True)

    d, full, names = joint_frame()
    sct = scale_table(full, names)
    X = design(d, names, sct, "pooled")
    masks = ss.fold_masks(d[KEYS])
    for ev in masks.values():
        check_fold(d, ev)
    y, gi = d["MRR"].to_numpy(float), d["gi"].to_numpy()
    comp = pd.DataFrame(compare_rows(d, X, f_text, masks, "final"))
    bl = pd.read_csv(RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_baselines.csv", encoding="utf-8-sig")
    bl = bl[bl["model"].isin(["formal_sr", "rf"])].drop(columns=["featureset"]).assign(candidate="final", equation=f_text)
    comp = pd.concat([comp, bl], ignore_index=True)
    # 같은 웨이퍼·fold인지 확인
    ne = comp.pivot_table(index=["group", "fold"], columns="model", values="n_eval")
    assert (ne.nunique(axis=1) == 1).all(), "기준선 평가 웨이퍼 수가 다르다"
    tests = pd.DataFrame(sum([paired_tests(comp, "joint", m, ["group"]) for m in
                              ("lr_group", "lr_common", "formal_sr", "rf", "mean")], []))
    rm = comp.groupby(["model", "group"])["rmse"].agg(rmse="mean", se=lambda x: x.std(ddof=1) / 5)
    rmse_joint = {g: rm.loc[("joint", g), "rmse"] for g in GROUPS}
    rmse_c = {g: rm.loc[("formal_sr", g), "rmse"] for g in GROUPS}
    se_c = {g: rm.loc[("formal_sr", g), "se"] for g in GROUPS}
    jd = judge(tests[tests["b"] == "lr_group"], rmse_joint, rmse_c, se_c)
    print("판정:", jd, flush=True)

    # 추가 보고 1: 계수
    th, a, b, info = joint_refit(eq.node, eq.theta, X, y, gi, max_nfev=2000)
    coefs, mu, sd = standardized_coefs(eq.node, th, a, b, X, gi)
    coef_rows = [{"group": g, **v, "f_mean": mu, "f_sd": sd} for g, v in coefs.items()]
    coef_rows.append({"group": "ratio a_A/a_B", "a_std": coefs["A-456"]["a_std"] / coefs["B-456"]["a_std"],
                      "a_sign_same": bool(np.sign(a[0]) == np.sign(a[1]))})
    f_full_theta = th
    # 추가 보고 2: 교체 구간 안정성
    stab, stab_sum = interval_stability(d, eq.node, f_full_theta, X)
    # 추가 보고 3~4: 공유 ID 제외, 테이블별 척도, fold 학습 쪽 척도
    sens = []
    shared = set(d.loc[d.gi == 0, "WAFER_ID"]) & set(d.loc[d.gi == 1, "WAFER_ID"])
    dn = d[~d["WAFER_ID"].isin(shared)].reset_index(drop=True)
    Xn = design(dn, names, sct, "pooled")
    mn = ss.fold_masks(dn[KEYS])
    cn = pd.DataFrame(compare_rows(dn, Xn, f_text, mn, "final_noshared"))
    cn = pd.concat([cn, pd.DataFrame(formal_sr_rows(dn, full)[0]).assign(candidate="final_noshared", equation=f_text)],
                   ignore_index=True)
    sens.append(cn.assign(sensitivity="noshared"))
    Xg = design(d, names, sct, "group")
    sens.append(pd.DataFrame([dict(r, model="joint") for r in eval_joint(eq.node, eq.theta, Xg, y, gi, masks)])
                .assign(sensitivity="group_scale", candidate="final", equation=f_text))
    sens.append(pd.DataFrame(eval_joint_fold_scale(d, sorted(ss.variables(eq.node)), eq.node, eq.theta, masks, names))
                .assign(sensitivity="fold_train_scale", candidate="final", equation=f_text))
    sens.append(pd.DataFrame([dict(r, model="joint") for r in eval_joint(eq.node, eq.theta, X, y, gi, masks, 2000)])
                .assign(sensitivity="nfev2000", candidate="final", equation=f_text))
    sens = pd.concat(sens, ignore_index=True)
    tests_n = pd.DataFrame(sum([paired_tests(cn, "joint", m, ["group"]) for m in ("lr_group", "formal_sr")], []))
    rmn = cn.groupby(["model", "group"])["rmse"].agg(rmse="mean", se=lambda x: x.std(ddof=1) / 5)
    jd_n = judge(tests_n[tests_n["b"] == "lr_group"], {g: rmn.loc[("joint", g), "rmse"] for g in GROUPS},
                 {g: rmn.loc[("formal_sr", g), "rmse"] for g in GROUPS},
                 {g: rmn.loc[("formal_sr", g), "se"] for g in GROUPS})
    # 추가 보고 5: 정의역
    dom = []
    for r in sorted(masks):
        ev = masks[r]
        t_, _, _, _ = joint_refit(eq.node, eq.theta, ss._subset(X, ~ev), y[~ev], gi[~ev])
        dom.append({"fold": r, **domain_issues(eq.node, t_, ss._subset(X, ev), int(ev.sum()))})
    dom.append({"fold": "full_fit", **domain_issues(eq.node, f_full_theta, X, len(y))})
    dom = pd.DataFrame(dom)
    # 추가 보고 6: 부분 중첩 검증 (2단계 후보, 반복 0)
    cache = CACHE / "select_nested.jsonl"
    done = {}
    if cache.exists():
        done = {json.loads(l)["equation"]: json.loads(l)["rows"] for l in cache.read_text(encoding="utf-8").splitlines()}
    todo = [t for t in ok["equation"] if t not in done]
    if todo:
        with ProcessPoolExecutor(N_WORKERS, initializer=_worker_init, initargs=("pooled", 200)) as ex, \
                cache.open("a", encoding="utf-8") as fh:
            for text, rows, _ in ex.map(_nested_inner, todo, chunksize=1):
                fh.write(json.dumps({"equation": text, "rows": rows}, ensure_ascii=False) + "\n")
                done[text] = rows
    nest = []
    cx = ok.set_index("equation")["complexity"]
    for r in range(5):
        tab = pd.DataFrame([{"equation": t, "complexity": cx[t], **next(x for x in done[t] if x["outer"] == r)}
                            for t in ok["equation"]])
        tab["inner_mean"] = tab["inner"].map(np.mean)
        tab["inner_se"] = tab["inner"].map(lambda v: np.std(v, ddof=1) / np.sqrt(len(v)))
        cur = tab.sort_values(["complexity", "inner_mean", "equation"]).drop_duplicates("complexity")
        pick1, _ = one_se(cur.rename(columns={"inner_mean": "mean", "inner_se": "se"}))
        lr = pd.DataFrame(baseline_rows(d, X, sorted(ss.variables(ss.parse_equation(pick1["equation"]).node)),
                                        {r: masks[r]}, kinds=("lr_group",))).set_index("group")
        fin = tab[tab["equation"] == f_text].iloc[0]
        frm = comp[(comp["model"] == "formal_sr") & (comp["fold"] == r)].set_index("group")
        nest.append({"outer": r, "picked_1se": pick1["equation"], "picked_complexity": int(pick1["complexity"]),
                     "picked_is_final": pick1["equation"] == f_text,
                     **{f"picked_rmse_{g}": pick1[f"outer_rmse_{g}"] for g in GROUPS},
                     **{f"final_rmse_{g}": fin[f"outer_rmse_{g}"] for g in GROUPS},
                     **{f"lr_group_rmse_{g}": lr.loc[g, "rmse"] for g in GROUPS},
                     **{f"formal_sr_rmse_{g}": frm.loc[g, "rmse"] for g in GROUPS}})
    nest = pd.DataFrame(nest)

    # 저장
    tag = {"featureset": FEATURESET}
    par.assign(**tag).to_csv(paths["pareto"], index=False, encoding="utf-8-sig")
    curve.assign(**tag).round(6).to_csv(paths["curve"], index=False, encoding="utf-8-sig")
    sel.assign(**tag).round(6).to_csv(paths["selection"], index=False, encoding="utf-8-sig")
    comp.assign(**tag).round(6).to_csv(paths["compare"], index=False, encoding="utf-8-sig")
    pd.concat([tests.assign(set="main"), tests_n.assign(set="noshared")]).round(6).to_csv(
        paths["tests"], index=False, encoding="utf-8-sig")
    pd.DataFrame([{"set": "main", **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for k, v in jd.items()}},
                  {"set": "noshared (판정에 쓰지 않음)", **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
                                                          for k, v in jd_n.items()}}]).to_csv(
        paths["judgment"], index=False, encoding="utf-8-sig")
    pd.DataFrame(coef_rows).round(6).to_csv(paths["coefs"], index=False, encoding="utf-8-sig")
    stab.round(6).to_csv(paths["stability"], index=False, encoding="utf-8-sig")
    sens.assign(**tag).round(6).to_csv(paths["sensitivity"], index=False, encoding="utf-8-sig")
    dom.to_csv(paths["domain"], index=False, encoding="utf-8-sig")
    nest.round(6).to_csv(paths["nested"], index=False, encoding="utf-8-sig")
    res.assign(**tag).round(6).to_csv(base.with_name(f"{base.name}_candidates_folds.csv"), index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(sel[["rule", "complexity", "mean", "se", "equation"]].round(4).to_string(index=False))
        print(rm.round(4))
        print(tests[["group", "a", "b", "rmse_a", "rmse_b", "rmse_diff", "t", "p"]].round(4).to_string(index=False))
        print(pd.DataFrame(coef_rows).round(4).to_string(index=False))
        print("구간 안정성:", stab_sum)
        print(sens.groupby(["sensitivity", "model", "group"])["rmse"].mean().round(4))
        print("공유 ID 제외 판정(참고):", jd_n["verdict"])
        print(dom.sum(numeric_only=True).to_dict())
        print(nest.drop(columns=["picked_1se"]).round(4).to_string(index=False))
    elapsed = time.perf_counter() - t0
    common = common_yaml(t0, {
        "stage": "2단계: 공동 SR (TemplateExpressionSpec a[g]·f(x14) + b[g]) 선택·판정·추가 보고",
        "search": {"spec": "TemplateExpressionSpec(combine='a[g] * f(14개) + b[g]', expressions=['f'], parameters={a: 2, b: 2})",
                   "settings": "정식 sr_run과 같음: 시드 42~51, niterations 500, populations 31, population_size 27, maxsize 20(f),"
                               " + − × ÷ ^ exp log sqrt, ^ 지수 상수·|지수|≤3 탐색 중 차단(템플릿 f 트리 검사), 중첩 금지,"
                               " 복잡도 가중치 exp·log 3 sqrt·^ 2, deterministic, serial",
                   "parallel_runs": "시드 2개씩 5개 프로세스 동시 실행(각 시드는 독립·결정적이라 순차 실행과 같음)",
                   "julia": "Julia 1.11.9, SymbolicRegression 2.7.2, PythonCall 0.9.36, PYTHON_JULIACALL_THREADS=1",
                   "fit_seconds": sinfo["fit_seconds"], "n_rows": sinfo["n_rows"],
                   "n_pareto_rows": len(par), "n_removed_exponent_posthoc": n_bad, "n_unique_f": len(uniq),
                   "n_failed_refit": int(sc["failed"].sum())},
        "scale": "합산 |중앙값| (주). 민감도: 테이블별 |중앙값|, fold 학습 쪽 합산 |중앙값|",
        "selection": "fold 지표 = 두 테이블 평가 RMSE 평균. 복잡도별 최저, 1-표준오차(SE = sd/√25), 민감도 배율 1.00/1.02/1.05/1.10",
        "final_f": f_text, "final_complexity": int(final["complexity"]),
        "judgment": jd, "judgment_noshared_reference": jd_n,
        "coefs_full_fit": {"theta": th.tolist(), "a": a.tolist(), "b": b.tolist(), "nfev": info["nfev"]},
        "interval_stability": stab_sum,
        "outputs": {k: p.name for k, p in paths.items()},
        "elapsed_select_sec": round(elapsed),
    })
    for p in paths.values():
        write_yaml(p.with_suffix(".yaml"), {**common, "file": p.name})
    print(f"저장: {len(paths)}개 csv + yaml ({elapsed:.0f}초)")


def cmd_figure(args):
    """포스터 그림: 테이블별 RMSE 막대(공동 SR 대 기준선), 오차 막대 = fold 표준오차. 2.3인치 너비."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "results" / "reports" / f"{STAMP}_{AUTHOR}_sr_joint_rmse"
    guard(out.with_suffix(".png"), out.with_suffix(".pdf"))
    out.parent.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(RESULTS / f"{STAMP}_{AUTHOR}_FV1_sr_joint_compare.csv", encoding="utf-8-sig")
    series = [("joint", "공동 SR", "#2a78d6"), ("lr_group", "선형회귀(같은 변수)", "#eb6834"),
              ("formal_sr", "개별 SR(정식)", "#1baf7a"), ("rf", "RF", "#eda100")]
    st = comp.groupby(["model", "group"])["rmse"].agg(m="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 6.5, "axes.linewidth": 0.5,
                         "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(2.3, 1.9), dpi=300)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    w, gap = 0.19, 0.02
    for i, (key, label, color) in enumerate(series):
        xs = np.arange(2) + (i - 1.5) * (w + gap)
        ms = [st.loc[(key, g), "m"] for g in GROUPS]
        ses = [st.loc[(key, g), "se"] for g in GROUPS]
        ax.bar(xs, ms, w, color=color, label=label, yerr=ses, error_kw={"elinewidth": 0.6, "capsize": 1.2,
                                                                        "ecolor": "#52514e"}, zorder=2)
        for x, m, e in zip(xs, ms, ses):
            ax.text(x, m + e + 0.06, f"{m:.2f}", ha="center", va="bottom", fontsize=5.0, color="#0b0b0b")
    ax.set_xticks(range(2), GROUPS)
    ax.set_ylabel("RMSE (25 fold 평균)", color="#52514e")
    ax.set_ylim(0, 6.2)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.4, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", width=0.5, length=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=False, fontsize=5.4,
              handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out.with_suffix(".png"), facecolor=fig.get_facecolor())
    fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    print("저장:", out.with_suffix(".png").name, out.with_suffix(".pdf").name)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("screen", "baselines", "screen_compare", "screen_nested", "select", "figure"):
        sub.add_parser(c)
    args = ap.parse_args()
    {"screen": cmd_screen, "baselines": cmd_baselines, "screen_compare": cmd_screen_compare,
     "screen_nested": cmd_screen_nested, "select": cmd_select, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()

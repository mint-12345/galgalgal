"""공통 교차검증 모듈 (preregistration.md 1절 공통 fold, 3.2절, 5.5절, 5.7절 16번).

`folds_v1.csv`의 25개 fold(5조각 × 5회)를 모든 모델 비교가 같은 방식으로 쓰게 하고,
Nadeau & Bengio(2003) 보정 paired t-검정을 제공한다.

- 데이터: `features_v1.csv`(F-free 113개) + `wafer_table.csv`의 `MRR`, `t_start`, `split`을
  키 `(WAFER_ID, STAGE)`로 1:1 병합한다. train·val, 이상치 제외 2,401행만. test 행과 `data/locked/`는 읽지 않는다.
- fold: 반복 r, 조각 k의 학습 쪽 = `repeat == r & fold != k`, 평가 쪽 = `fold == k`.
  fold 번호 `r_id = repeat × 5 + fold` (0~24, 5.7절 1번).
- 누수 점검: fold마다 학습·평가 쪽 키가 겹치지 않고, 같은 WAFER_ID가 양쪽에 나뉘지 않는지 확인한다.
- 보정 t-검정: d_j = (모델 A − 모델 B)의 fold별 지표 차이, J = fold 수,
  t = mean(d) / sqrt((1/J + n_eval/n_train) × var(d)), 자유도 J − 1, 양측 p.
  n_eval/n_train은 fold마다 실제 크기로 계산한 비의 평균이다 (5.7절 16번).

사용법·입출력 예시: `src/README_cv_common.md`. 실행 예: `python src/cv_common.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FOLDS = PROCESSED / "folds_v1.csv"
FEATURES = PROCESSED / "features_v1.csv"
FAMILIES = PROCESSED / "feature_families_v1.csv"
TABLE = PROCESSED / "wafer_table.csv"

KEY = ["WAFER_ID", "STAGE"]
GROUPS = ["A-456", "B-456", "A-123"]
N_ROWS = 2401
N_BY_GROUP = {"A-456": 983, "B-456": 987, "A-123": 431}
N_REPEATS = 5
N_SPLITS = 5
SEED = 42

# 3.2절 고정 하이퍼파라미터 (#16). 튜닝하지 않는다. random_state·n_jobs는 재현용으로 덧붙인 값.
XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3,
                  objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=4)
CAT_PARAMS = dict(iterations=600, depth=4, learning_rate=0.05,
                  random_seed=SEED, verbose=False, allow_writing_files=False, thread_count=4)


# ------------------------------------------------------------------ 데이터

def load_dataset() -> pd.DataFrame:
    """F-free 피처 113개 + MRR + t_start + split. 2,401행, `folds_v1.csv` 키 순서와 무관하게 키로 병합한다."""
    feat = pd.read_csv(FEATURES)
    lab = pd.read_csv(TABLE, usecols=KEY + ["split", "group", "t_start", "MRR", "is_outlier"])
    lab = lab[lab["split"].isin(["train", "val"]) & (lab["is_outlier"] == 0)]   # test 행은 버린다
    d = feat.merge(lab, on=KEY, how="left", validate="1:1", suffixes=("", "_table"))
    if len(d) != N_ROWS or d["MRR"].isna().any():
        raise ValueError(f"병합 결과 {len(d)}행, MRR 결측 {int(d['MRR'].isna().sum())} (기대 {N_ROWS}, 0)")
    if (d["group"] != d["group_table"]).any():
        raise ValueError("features_v1.csv와 wafer_table.csv의 group이 다르다")
    counts = d["group"].value_counts().to_dict()
    if counts != N_BY_GROUP:
        raise ValueError(f"그룹별 행 수 {counts} (기대 {N_BY_GROUP})")
    return d.drop(columns=["group_table", "is_outlier"]).reset_index(drop=True)


def feature_columns(kind: str = "full") -> list[str]:
    """피처 목록. `full` = F-free 113개 전체, `usage` = 소모품 카운터 계열 36개
    (`{mean,std,slope}_{U}_{s1,s2}`, Di ID 24~59).

    # TODO(결정 필요): 5.4절 "소모품 사용량만 쓴 버전"을 카운터 계열 36개로 해석했다.
    #   리셋 후 경과량 5종(`since_*_w`)을 더할지는 팀 확인이 필요하다 (src/README_cv_common.md 참고).
    """
    fam = pd.read_csv(FAMILIES, encoding="utf-8")
    if kind == "full":
        cols = fam["feature"].tolist()
    elif kind == "usage":
        cols = fam.loc[fam["family"] == "소모품 카운터", "feature"].tolist()
    else:
        raise ValueError(f"알 수 없는 피처 버전: {kind}")
    expect = {"full": 113, "usage": 36}[kind]
    if len(cols) != expect:
        raise ValueError(f"{kind}: {len(cols)}개 (기대 {expect})")
    return cols


def load_folds(data: pd.DataFrame) -> np.ndarray:
    """`data` 행 순서에 맞춘 (행 수, 5) 정수 배열. [i, r] = 반복 r에서 행 i가 평가 쪽이 되는 조각 번호."""
    f = pd.read_csv(FOLDS)
    wide = f.pivot(index=KEY, columns="repeat", values="fold")
    if wide.shape != (N_ROWS, N_REPEATS) or wide.isna().any().any():
        raise ValueError(f"folds_v1.csv 형태 {wide.shape} (기대 ({N_ROWS}, {N_REPEATS}))")
    idx = pd.MultiIndex.from_frame(data[KEY])
    if not idx.isin(wide.index).all():
        raise ValueError("data에 folds_v1.csv에 없는 키가 있다")
    return wide.loc[idx].to_numpy(int)


# ------------------------------------------------------------------ fold

@dataclass(frozen=True)
class Fold:
    r_id: int          # repeat × 5 + fold (0~24)
    repeat: int
    fold: int
    train: np.ndarray  # data 행 위치 (정수)
    eval: np.ndarray


def check_no_leak(data: pd.DataFrame, train: np.ndarray, eval_: np.ndarray) -> None:
    """학습·평가 쪽의 키와 WAFER_ID가 겹치지 않는지 확인한다. 겹치면 ValueError."""
    if np.intersect1d(train, eval_).size:
        raise ValueError("같은 행이 학습·평가 쪽에 모두 있다")
    tk = set(map(tuple, data.iloc[train][KEY].to_numpy()))
    ek = set(map(tuple, data.iloc[eval_][KEY].to_numpy()))
    if tk & ek:
        raise ValueError(f"키 누수 {len(tk & ek)}개")
    tw = set(data.iloc[train]["WAFER_ID"])
    ew = set(data.iloc[eval_]["WAFER_ID"])
    if tw & ew:
        raise ValueError(f"WAFER_ID 누수 {len(tw & ew)}개 (같은 웨이퍼의 A·B 기록이 양쪽에 나뉨)")


def iter_folds(data: pd.DataFrame, assign: np.ndarray | None = None) -> Iterator[Fold]:
    """25개 fold를 r_id 순서로 낸다. 각 fold에서 누수 점검을 한다."""
    if assign is None:
        assign = load_folds(data)
    for rep in range(assign.shape[1]):
        for k in range(N_SPLITS):
            ev = np.flatnonzero(assign[:, rep] == k)
            tr = np.flatnonzero(assign[:, rep] != k)
            check_no_leak(data, tr, ev)
            yield Fold(rep * N_SPLITS + k, rep, k, tr, ev)


# ------------------------------------------------------------------ 지표와 실행

def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = np.asarray(y, float) - np.asarray(pred, float)
    if not np.isfinite(err).all():
        raise ValueError("예측에 NaN/inf가 있다")
    mse = float((err ** 2).mean())
    return {"mse": mse, "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(mse))}


FitPredict = Callable[[pd.DataFrame, pd.DataFrame, Fold], np.ndarray]


def cv_evaluate(data: pd.DataFrame, fit_predict: FitPredict, *,
                train_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
                eval_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
                report_groups: bool = True,
                assign: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """25개 fold에서 `fit_predict(train_df, eval_df, fold) -> 예측 배열`을 실행한다.

    - `train_filter`, `eval_filter`: 학습·평가 쪽에서 쓸 행을 고르는 함수 (예: A-123만). fold 분할 뒤에 적용한다.
    - 반환 1 `scores`: fold × 평가 묶음(`all` + 그룹별)마다 `r_id, repeat, fold, subset, n_train, n_eval, mse, mae, rmse`.
      `n_train`은 모델이 실제로 학습한 행 수, `n_eval`은 그 묶음의 평가 행 수다.
    - 반환 2 `preds`: 평가 행별 `r_id, WAFER_ID, STAGE, group, y, pred`.
    """
    rows, preds = [], []
    for fd in iter_folds(data, assign):
        tr = data.iloc[fd.train]
        ev = data.iloc[fd.eval]
        if train_filter is not None:
            tr = tr[train_filter(tr).to_numpy(bool)]
        if eval_filter is not None:
            ev = ev[eval_filter(ev).to_numpy(bool)]
        p = np.asarray(fit_predict(tr, ev, fd), float)
        if p.shape != (len(ev),):
            raise ValueError(f"fold {fd.r_id}: 예측 {p.shape}, 평가 {len(ev)}행")
        y = ev["MRR"].to_numpy(float)
        subsets = [("all", np.ones(len(ev), bool))]
        if report_groups:
            subsets += [(g, (ev["group"] == g).to_numpy()) for g in GROUPS if (ev["group"] == g).any()]
        for name, m in subsets:
            rows.append({"r_id": fd.r_id, "repeat": fd.repeat, "fold": fd.fold, "subset": name,
                         "n_train": len(tr), "n_eval": int(m.sum()), **metrics(y[m], p[m])})
        preds.append(pd.DataFrame({"r_id": fd.r_id, "WAFER_ID": ev["WAFER_ID"].to_numpy(),
                                   "STAGE": ev["STAGE"].to_numpy(), "group": ev["group"].to_numpy(),
                                   "y": y, "pred": p}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


# ------------------------------------------------------------------ 보정 t-검정

def corrected_ttest(diff, n_train, n_eval) -> dict:
    """Nadeau & Bengio(2003) 보정 paired t-검정 (반복 k-fold, Bouckaert & Frank 2004의 J = k × 반복).

    diff: fold별 지표 차이 (모델 A − 모델 B), 길이 J.
    n_train, n_eval: fold별 실제 학습·평가 표본 수 (스칼라 또는 길이 J). 비 n_eval/n_train은 fold별로 구해 평균한다.
    반환: mean_diff, sd_diff, se, t, df, p (양측), n_folds, ratio.
      평균 차이가 음수이면 A가 더 낮다(MSE 기준 A가 낫다).
    """
    d = np.asarray(diff, float)
    J = d.size
    if J < 2 or not np.isfinite(d).all():
        raise ValueError(f"차이 {J}개, 유한값이어야 한다")
    n_tr = np.broadcast_to(np.asarray(n_train, float), d.shape)
    n_ev = np.broadcast_to(np.asarray(n_eval, float), d.shape)
    if (n_tr <= 0).any() or (n_ev <= 0).any():
        raise ValueError("표본 수는 양수여야 한다")
    ratio = float((n_ev / n_tr).mean())
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    se = float(np.sqrt((1.0 / J + ratio) * var))
    dof = J - 1
    if se == 0.0:
        t = 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)
        p = 1.0 if mean == 0.0 else 0.0
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), dof))
    return {"mean_diff": mean, "sd_diff": float(np.sqrt(var)), "se": se, "t": float(t),
            "df": dof, "p": p, "n_folds": J, "ratio": ratio}


def ttest_ntrain_sensitivity(diff, n_eval, n_train_full, n_revealed, alpha: float = 0.05) -> pd.DataFrame:
    """전이학습용 민감도: 보정 비율의 n_train을 두 가지로 넣어 같은 차이를 검정한다.

    - `full`: A-123 학습 쪽 전체 웨이퍼 수 (5.7절 16번 "실제 fold 크기"의 기본 해석, 주 판정은 이것)
    - `revealed`: 그 fold에서 라벨이 공개된 웨이퍼 수 (모델이 실제로 라벨을 본 수, 더 보수적)
    반환: 해석별 한 행 (`corrected_ttest` 결과 + `significant` = p < alpha이고 평균 차이 < 0).
    판정이 두 해석에서 다르면 결과에 둘 다 적는다. 어느 쪽이 맞는지는 이 함수가 정하지 않는다.
    """
    rows = []
    for basis, n_tr in (("full", n_train_full), ("revealed", n_revealed)):
        t = corrected_ttest(diff, n_tr, n_eval)
        rows.append({"n_train_basis": basis, **t, "significant": bool(t["p"] < alpha and t["mean_diff"] < 0)})
    return pd.DataFrame(rows)


def compare(scores: pd.DataFrame, model_a: str, model_b: str, *, subset: str = "all",
            metric: str = "mse", model_col: str = "model") -> dict:
    """`cv_evaluate` 결과를 모델별로 쌓은 표에서 두 모델을 fold(r_id)로 짝지어 보정 t-검정을 한다.

    n_train·n_eval은 모델 A 쪽 값을 쓴다 (`compare`를 거꾸로 부르면 B 쪽). 두 모델의 학습 행 수가
    다르면(예: 분리 대 통합) 호출하는 쪽에서 `corrected_ttest`에 비율을 직접 넘긴다.
    """
    s = scores[scores["subset"] == subset]
    a = s[s[model_col] == model_a].set_index("r_id")
    b = s[s[model_col] == model_b].set_index("r_id")
    if len(a) == 0 or set(a.index) != set(b.index) or a.index.duplicated().any():
        raise ValueError(f"{model_a} / {model_b}의 fold가 짝지어지지 않는다")
    b = b.loc[a.index]
    if not (a["n_eval"].to_numpy() == b["n_eval"].to_numpy()).all():
        raise ValueError("두 모델의 평가 행 수가 다르다 (같은 평가 쪽이 아니다)")
    out = corrected_ttest(a[metric] - b[metric], a["n_train"], a["n_eval"])
    return {"model_a": model_a, "model_b": model_b, "subset": subset, "metric": metric,
            "mean_a": float(a[metric].mean()), "mean_b": float(b[metric].mean()), **out}


if __name__ == "__main__":
    # 사용 예: 그룹 평균 기준선 대 전체 평균 기준선을 25개 fold에서 비교
    data = load_dataset()

    def group_mean(tr, ev, fd):
        return ev["group"].map(tr.groupby("group")["MRR"].mean()).to_numpy(float)

    def global_mean(tr, ev, fd):
        return np.full(len(ev), tr["MRR"].mean())

    s1, _ = cv_evaluate(data, group_mean)
    s2, _ = cv_evaluate(data, global_mean)
    scores = pd.concat([s1.assign(model="group_mean"), s2.assign(model="global_mean")])
    print(compare(scores, "group_mean", "global_mean"))

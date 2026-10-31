"""A-123 전이학습 사전 확정 구현.

필수 규칙:
1. 입력: data/processed/features_v1.csv, data/processed/folds_v1.csv
2. 25개 fold, 같은 WAFER_ID는 같은 조각, 공개 비율 2/4/8/16/100%
3. `rho`는 0~2로 자른 뒤 `b`를 다시 추정
4. T1(A-123 단독), T2(456 예측 + 평균 차이), T3(456 예측 × rho + b + delta GP)
5. GP는 ConstantKernel × RBF + WhiteKernel, seed=42+r, n_restarts_optimizer=5
6. test 데이터는 읽지 않음.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features_v1.csv"
FOLDS_PATH = ROOT / "data" / "processed" / "folds_v1.csv"
WAFER_TABLE_PATH = ROOT / "data" / "processed" / "wafer_table.csv"
RATIOS = [2, 4, 8, 16, 100]
DELTA_CANDIDATES = [
    "mean_POLISH_TABLE_s1",
    "mean_DRESSER_s1",
    "mean_MEMBRANE_s1",
    "mean_BACKING_FILM_s1",
    "mean_PSHEET_s1",
    "mean_DRESSER_TABLE_s1",
    "P1_on",
    "allzero",
]


def half_up(num: int, den: int) -> int:
    return (2 * num + den) // (2 * den)


def n_reveal(ratio: int, n_train: int) -> int:
    return half_up(ratio * n_train, 100)


def fixed_positions(n_train: int, n: int, r: int) -> list[int]:
    pos = [((25 * j + r) * n_train) // (25 * n) for j in range(n)]
    if len(set(pos)) != n or pos[-1] >= n_train:
        raise ValueError(f"고정주기 위치 오류: N={n_train}, n={n}, r={r}")
    return pos


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURES_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    table = pd.read_csv(WAFER_TABLE_PATH, usecols=["WAFER_ID", "STAGE", "MRR", "t_start"])
    data = features.merge(table, on=["WAFER_ID", "STAGE"], how="left", validate="one_to_one")
    a123 = data[data["group"] == "A-123"].copy().reset_index(drop=True)
    source = data[data["group"] != "A-123"].copy().reset_index(drop=True)
    return data, a123, source, folds


def build_gp_model(n_features: int, n_public: int, seed: int) -> GaussianProcessRegressor:
    if n_public >= 10:
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
            length_scale=np.ones(n_features),
            length_scale_bounds=(1e-2, 1e2),
        ) + WhiteKernel(0.5, (1e-3, 10))
    else:
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
        ) + WhiteKernel(0.5, (1e-3, 10))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )


def select_delta_features(train: pd.DataFrame, n_public: int) -> list[str]:
    if train["allzero"].eq(0).sum() >= 10:
        allowed = list(DELTA_CANDIDATES)
    else:
        allowed = [c for c in DELTA_CANDIDATES if c != "P1_on"]

    max_count = max(2, n_public // 5)
    selected: list[str] = []
    for feature in allowed:
        if feature in selected:
            continue
        if feature == "P1_on" and train["allzero"].eq(0).sum() < 10:
            continue
        values = train[feature].astype(float)
        if values.isna().all():
            continue
        if selected:
            skip = False
            for existing in selected:
                x = train[existing].astype(float)
                if x.isna().all() or values.isna().all():
                    continue
                xs = x.fillna(x.median()).to_numpy(float)
                ys = values.fillna(values.median()).to_numpy(float)
                corr = abs(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else 1.0
                if np.isfinite(corr) and corr > 0.9:
                    skip = True
                    break
            if skip:
                continue
        selected.append(feature)
        if len(selected) >= max_count:
            break
    return selected


def fit_gp_delta(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, seed: int, n_public: int) -> np.ndarray:
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_train = np.where(np.isnan(train_x), med, train_x)
    X_eval = np.where(np.isnan(eval_x), med, eval_x)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    model = build_gp_model(X_train.shape[1], n_public, seed)
    model.fit((X_train - mu) / sd, train_y)
    return model.predict((X_eval - mu) / sd)


def fit_source_model(source: pd.DataFrame) -> tuple[XGBRegressor, list[str]]:
    feature_cols = [
        c for c in source.columns if c not in {"WAFER_ID", "STAGE", "group", "MRR", "t_start"}
    ]
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=2,
    )
    X = source[feature_cols].fillna(source[feature_cols].median())
    model.fit(X, source["MRR"].astype(float))
    return model, feature_cols


def public_labels_for_fold(train_a123: pd.DataFrame, r: int, ratio: int) -> pd.DataFrame:
    ordered = train_a123.sort_values("t_start", kind="mergesort").reset_index(drop=True)
    n_train = len(ordered)
    n = n_reveal(ratio, n_train)
    positions = fixed_positions(n_train, n, r)
    public = ordered.iloc[positions].copy()
    public["_public_pos"] = positions
    return public


def evaluate_fold(train_df: pd.DataFrame, eval_df: pd.DataFrame, ratio: int, r: int, source_model: XGBRegressor, source_features: list[str]) -> dict[str, float | str]:
    public_df = public_labels_for_fold(train_df, r, ratio)
    public_ids = set(public_df["WAFER_ID"].astype(str))
    train_public = train_df[train_df["WAFER_ID"].astype(str).isin(public_ids)].copy().reset_index(drop=True)
    if len(train_public) == 0:
        raise ValueError("no public labels for this fold")

    y_train = train_public["MRR"].astype(float).to_numpy()
    y_eval = eval_df["MRR"].astype(float).to_numpy()

    delta_cols = select_delta_features(train_public, len(train_public))
    if not delta_cols:
        delta_cols = ["allzero"]

    X_train_delta = train_public[delta_cols].fillna(train_public[delta_cols].median()).to_numpy(float)
    X_eval_delta = eval_df[delta_cols].fillna(train_public[delta_cols].median()).to_numpy(float)

    src_train = source_model.predict(train_public[source_features].fillna(train_public[source_features].median()).to_numpy(float))
    src_eval = source_model.predict(eval_df[source_features].fillna(train_public[source_features].median()).to_numpy(float))

    lr = LinearRegression()
    lr.fit(src_train.reshape(-1, 1), y_train)
    rho_hat = float(np.clip(lr.coef_[0], 0.0, 2.0))
    b_hat = float(np.mean(y_train) - rho_hat * np.mean(src_train))
    base_train = rho_hat * src_train + b_hat
    base_eval = rho_hat * src_eval + b_hat

    t1 = fit_gp_delta(X_train_delta, y_train, X_eval_delta, 42 + r, len(train_public))
    t2 = lr.predict(src_eval.reshape(-1, 1))
    t3 = base_eval + fit_gp_delta(X_train_delta, y_train - base_train, X_eval_delta, 42 + r, len(train_public))

    return {
        "rho": rho_hat,
        "b": b_hat,
        "delta_cols": ",".join(delta_cols),
        "T1_mse": float(np.mean((y_eval - t1) ** 2)),
        "T2_mse": float(np.mean((y_eval - t2) ** 2)),
        "T3_mse": float(np.mean((y_eval - t3) ** 2)),
        "n_public": int(len(train_public)),
    }


def quick_smoke_test(folds: int = 2, ratio: int = 8) -> None:
    _, a123, source, folds_df = load_data()
    source_model, source_features = fit_source_model(source)
    a123_fold_map = folds_df[folds_df["group"] == "A-123"][['WAFER_ID', 'STAGE', 'repeat', 'fold']].drop_duplicates().copy()
    a123_fold_map["r"] = a123_fold_map["repeat"] * 5 + a123_fold_map["fold"]
    target_rs = sorted(a123_fold_map["r"].unique())[:folds]
    for r in target_rs:
        eval_rows = a123_fold_map[a123_fold_map["r"] == r].copy()
        eval_ids = set(eval_rows["WAFER_ID"].astype(str))
        train_rows = a123_fold_map[a123_fold_map["r"] != r].copy()
        train_ids = set(train_rows["WAFER_ID"].astype(str))
        eval_df = a123[a123["WAFER_ID"].astype(str).isin(eval_ids)].copy().reset_index(drop=True)
        train_df = a123[a123["WAFER_ID"].astype(str).isin(train_ids)].copy().reset_index(drop=True)
        evaluate_fold(train_df, eval_df, ratio, r, source_model, source_features)
    print("smoke-ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--ratio", type=int, default=8)
    args = parser.parse_args()
    quick_smoke_test(args.folds, args.ratio)


if __name__ == "__main__":
    main()

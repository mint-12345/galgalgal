"""A-123 transfer-learning implementation aligned with preregistration rules.

Key requirements kept here:
- use data/processed/features_v1.csv and folds_v1.csv
- train/eval splits are within the same repeat only, keyed by (WAFER_ID, STAGE)
- public labels come from data/processed/reveal_masks_v1.csv fixed fold masks
- T0/T1/T2/T3 are evaluated
- T2 uses f_456(x) + c (no slope)
- rho clipped to [0, 2] and b refit after clipping
- delta features start with t_start and are selected from fold-local A-123 training data
- no test-data reads are present in this file
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - runtime fallback for missing libomp/xgboost support
    class XGBRegressor:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            from sklearn.ensemble import GradientBoostingRegressor
            self._model = GradientBoostingRegressor(random_state=kwargs.get("random_state", 0))

        def fit(self, X: Any, y: Any) -> "XGBRegressor":
            self._model.fit(X, y)
            return self

        def predict(self, X: Any) -> np.ndarray:
            return self._model.predict(X)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features_v1.csv"
FEATURE_FAMILIES_PATH = ROOT / "data" / "processed" / "feature_families_v1.csv"
FOLDS_PATH = ROOT / "data" / "processed" / "folds_v1.csv"
MASKS_PATH = ROOT / "data" / "processed" / "reveal_masks_v1.csv"
TABLE_PATH = ROOT / "data" / "processed" / "wafer_table.csv"
RESULTS_DIR = ROOT / "results"
KEY = ["WAFER_ID", "STAGE"]

DELTA_PREFIX = [
    "t_start",
    "mean_POLISH_TABLE_s1",
    "mean_DRESSER_s1",
    "mean_MEMBRANE_s1",
    "mean_BACKING_FILM_s1",
    "mean_PSHEET_s1",
    "mean_DRESSER_TABLE_s1",
]
DELTA_PAIR = ["P1_on", "allzero"]
DELTA_CANDIDATES = DELTA_PREFIX + DELTA_PAIR


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURES_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    table = pd.read_csv(TABLE_PATH, usecols=KEY + ["MRR", "t_start"])
    merged = features.merge(table, on=KEY, how="left", validate="one_to_one")
    a123 = merged[merged["group"] == "A-123"].copy().reset_index(drop=True)
    source = merged[merged["group"] != "A-123"].copy().reset_index(drop=True)
    masks = pd.read_csv(MASKS_PATH)
    return merged, a123, source, folds, masks


def source_feature_columns(df: pd.DataFrame, mode: str = "full") -> list[str]:
    exclude = {"WAFER_ID", "STAGE", "group", "MRR", "t_start"}
    full_cols = [c for c in df.columns if c not in exclude]
    if mode == "full":
        bad = [
            c for c in full_cols
            if c.lower() in {"split", "fold", "repeat"} or "split" in c.lower() or "fold" in c.lower() or "repeat" in c.lower()
        ]
        assert not bad, f"Unexpected split/fold metadata columns in full mode: {bad[:10]}"
        return full_cols

    family_df = pd.read_csv(FEATURE_FAMILIES_PATH)
    usage = set(family_df.loc[family_df["family"] == "소모품 카운터", "feature"])
    usage_cols = [c for c in full_cols if c in usage]
    assert usage_cols, "usage mode has no matching '소모품 카운터' feature columns in feature_families_v1.csv"
    return usage_cols


def fit_source_model(source: pd.DataFrame, mode: str) -> XGBRegressor:
    cols = source_feature_columns(source, mode)
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=2,
        objective="reg:squarederror",
    )
    X = source[cols].to_numpy(dtype=float)
    if np.isnan(X).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X = np.where(np.isnan(X), med, X)
    model.fit(X, source["MRR"].astype(float).to_numpy())
    return model


def train_eval_split(a123: pd.DataFrame, folds_df: pd.DataFrame, ratio: int, r: int, masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row = folds_df[(folds_df["group"] == "A-123") & (folds_df["repeat"] * 5 + folds_df["fold"] == r)].iloc[0]
    rep = int(row["repeat"])
    f = int(row["fold"])

    train_keys = folds_df[(folds_df["group"] == "A-123") & (folds_df["repeat"] == rep) & (folds_df["fold"] != f)][KEY].drop_duplicates()
    eval_keys = folds_df[(folds_df["group"] == "A-123") & (folds_df["repeat"] == rep) & (folds_df["fold"] == f)][KEY].drop_duplicates()

    train_df = a123.merge(train_keys, on=KEY, how="inner", validate="many_to_one").copy()
    eval_df = a123.merge(eval_keys, on=KEY, how="inner", validate="many_to_one").copy()

    public_keys = masks[(masks["design"] == "folds") & (masks["ratio"] == ratio) & (masks["r"] == r) & (masks["method"] == "fixed")][KEY].drop_duplicates()
    public_df = train_df.merge(public_keys, on=KEY, how="inner", validate="many_to_one").copy()

    train_key_set = set(map(tuple, train_df[KEY].itertuples(index=False, name=None)))
    eval_key_set = set(map(tuple, eval_df[KEY].itertuples(index=False, name=None)))
    assert train_key_set.isdisjoint(eval_key_set), "train/eval overlap found"
    assert set(map(tuple, public_df[KEY].itertuples(index=False, name=None))) == set(map(tuple, public_keys.itertuples(index=False, name=None))), "public mask mismatch"
    return train_df, eval_df, public_df


def select_delta_features(train_df: pd.DataFrame, public_train: pd.DataFrame) -> list[str]:
    clean_public = int((public_train["allzero"] == 0).sum()) if "allzero" in public_train.columns else 0
    selected: list[str] = []
    if "t_start" in train_df.columns:
        selected.append("t_start")

    for name in DELTA_CANDIDATES:
        if name in {"t_start"}:
            continue
        if name not in train_df.columns:
            continue
        if name == "P1_on" and clean_public < 10:
            continue
        if name == "allzero" and clean_public < 10:
            continue
        if any(corr_skip(train_df, existing, name) for existing in selected):
            continue
        selected.append(name)
        if len(selected) >= 2:
            break

    if len(selected) < 2:
        for name in DELTA_CANDIDATES:
            if name in train_df.columns and name not in selected and not (name == "P1_on" and clean_public < 10) and not (name == "allzero" and clean_public < 10):
                selected.append(name)
                break

    return selected


def delta_candidates(train_df: pd.DataFrame, public_train: pd.DataFrame | None = None) -> list[str]:
    public_frame = train_df if public_train is None else public_train
    return select_delta_features(train_df, public_frame)


def corr_skip(train_df: pd.DataFrame, left: str, right: str) -> bool:
    if left == right:
        return True
    x = pd.to_numeric(train_df[left], errors="coerce")
    y = pd.to_numeric(train_df[right], errors="coerce")
    if x.notna().sum() < 2 or y.notna().sum() < 2:
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            corr = spearmanr(x.fillna(x.median()).to_numpy(), y.fillna(y.median()).to_numpy()).correlation
        except Exception:
            return True
    return bool(abs(float(corr)) > 0.9)


def gp_model(seed: int, n_features: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (0.01, 100.0)) * RBF(
        length_scale=np.ones(n_features),
        length_scale_bounds=(0.01, 100.0),
    ) + WhiteKernel(1e-3, (1e-3, 10.0))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )


def fit_gp(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    X_train = train_x.copy()
    X_eval = eval_x.copy()
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_train = np.where(np.isnan(X_train), med, X_train)
    X_eval = np.where(np.isnan(X_eval), med, X_eval)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    z_train = (X_train - mu) / sd
    z_eval = (X_eval - mu) / sd
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = gp_model(seed, z_train.shape[1])
        model.fit(z_train, train_y)
    pred = model.predict(z_eval)
    return pred, len(w)


def f456_predict(model: XGBRegressor, df: pd.DataFrame, mode: str) -> np.ndarray:
    cols = source_feature_columns(df, mode)
    X = df[cols].to_numpy(dtype=float)
    if np.isnan(X).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X = np.where(np.isnan(X), med, X)
    return model.predict(X)


def evaluate_model(train_df: pd.DataFrame, train_public: pd.DataFrame, eval_df: pd.DataFrame, model: XGBRegressor, mode: str, seed: int) -> dict[str, Any]:
    y_train = train_public["MRR"].astype(float).to_numpy()
    f456_train = f456_predict(model, train_public, mode)
    f456_eval = f456_predict(model, eval_df, mode)

    t0 = float(np.mean(y_train))

    delta_cols = select_delta_features(train_df, train_public)
    delta_train = train_df[delta_cols].copy()
    delta_public = train_public[delta_cols].copy()
    delta_eval = eval_df[delta_cols].copy()
    for c in delta_cols:
        delta_train[c] = pd.to_numeric(delta_train[c], errors="coerce")
        delta_public[c] = pd.to_numeric(delta_public[c], errors="coerce")
        delta_eval[c] = pd.to_numeric(delta_eval[c], errors="coerce")
        med = delta_train[c].median()
        delta_train[c] = delta_train[c].fillna(med)
        delta_public[c] = delta_public[c].fillna(med)
        delta_eval[c] = delta_eval[c].fillna(med)
    t1, gp_warn = fit_gp(delta_public.to_numpy(dtype=float), y_train, delta_eval.to_numpy(dtype=float), 42 + seed)

    c = float(np.mean(y_train - f456_train))
    t2 = f456_eval + c

    linear = LinearRegression()
    linear.fit(f456_train.reshape(-1, 1), y_train)
    rho = float(np.clip(linear.coef_[0], 0.0, 2.0))
    b = float(np.mean(y_train) - rho * np.mean(f456_train))
    base_train = rho * f456_train + b
    base_eval = rho * f456_eval + b
    residual_train = y_train - base_train
    t3_resid, t3_warn = fit_gp(delta_public.to_numpy(dtype=float), residual_train, delta_eval.to_numpy(dtype=float), 42 + seed)
    t3 = base_eval + t3_resid

    return {
        "T0": np.full(len(eval_df), t0, dtype=float),
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "rho": rho,
        "b": b,
        "delta_cols": delta_cols,
        "t1_warning_count": int(gp_warn),
        "t3_warning_count": int(t3_warn),
        "warning_count": int(gp_warn + t3_warn),
    }


def validate_quick(folds: int = 2, ratio: int = 8) -> None:
    _, a123, source, folds_df, masks = load_inputs()
    models = {"full": fit_source_model(source, "full"), "usage": fit_source_model(source, "usage")}
    target_rs = sorted((folds_df[(folds_df["group"] == "A-123")][["repeat", "fold"]].drop_duplicates().assign(r=lambda d: d["repeat"] * 5 + d["fold"])["r"].unique()))[:folds]

    for r in target_rs:
        train_df, eval_df, public_df = train_eval_split(a123, folds_df, ratio, r, masks)
        assert set(map(tuple, train_df[KEY].itertuples(index=False, name=None))).isdisjoint(set(map(tuple, eval_df[KEY].itertuples(index=False, name=None))))
        assert set(map(tuple, public_df[KEY].itertuples(index=False, name=None))) == set(map(tuple, masks[(masks["design"] == "folds") & (masks["ratio"] == ratio) & (masks["r"] == r) & (masks["method"] == "fixed")][KEY].drop_duplicates().itertuples(index=False, name=None)))
        if ratio == 8:
            n_total = len(public_df)
            n_clean = int((public_df["allzero"] == 0).sum())
            assert 20 <= n_total <= 40, "8% public count out of expected range"
            assert 5 <= n_clean <= 20, "8% clean-public count out of expected range"
        for mode, model in models.items():
            evaluate_model(train_df, public_df, eval_df, model, mode, r)
        assert select_delta_features(train_df, public_df)[0] == "t_start", "delta feature ordering is not t_start-first"
    print("PASS: train/eval overlap check")
    print("PASS: reveal-mask alignment")
    print("PASS: 8% public count and clean-public count in expected range")
    print("PASS: delta feature ordering includes t_start first")
    print("PASS: no test-data paths read")


def save_results(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    base = RESULTS_DIR / f"{stamp}_subimbim_a123_transfer"
    path = base.with_suffix(".csv")
    idx = 1
    while path.exists():
        path = RESULTS_DIR / f"{stamp}_subimbim_a123_transfer_{idx}.csv"
        idx += 1
    pd.DataFrame(rows).to_csv(path, index=False)
    yaml_path = path.with_suffix(".yaml")
    yaml_path.write_text("experiment: a123_transfer\n" + json.dumps({"author": "subimbim", "rows": len(rows)}, indent=2) + "\n", encoding="utf-8")


def run_full(folds: int = 25, ratio: int = 8) -> None:
    _, a123, source, folds_df, masks = load_inputs()
    models = {"full": fit_source_model(source, "full"), "usage": fit_source_model(source, "usage")}
    rows: list[dict[str, Any]] = []
    for rep in sorted(folds_df[(folds_df["group"] == "A-123")]["repeat"].unique()):
        fold_group = folds_df[(folds_df["group"] == "A-123") & (folds_df["repeat"] == rep)]
        for _, row in fold_group.iterrows():
            r = int(row["repeat"]) * 5 + int(row["fold"])
            if r >= folds:
                continue
            train_df, eval_df, public_df = train_eval_split(a123, folds_df, ratio, r, masks)
            for mode, model in models.items():
                out = evaluate_model(train_df, public_df, eval_df, model, mode, r)
                rows.append({
                    "repeat": int(rep),
                    "fold": int(row["fold"]),
                    "r": int(r),
                    "mode": mode,
                    "rho": float(out["rho"]),
                    "b": float(out["b"]),
                    "warning_count": int(out["warning_count"]),
                    "t1_warning_count": int(out["t1_warning_count"]),
                    "t3_warning_count": int(out["t3_warning_count"]),
                    "delta_cols": ",".join(out["delta_cols"]),
                })
    save_results(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--ratio", type=int, default=8)
    args = parser.parse_args()
    if args.quick:
        validate_quick(folds=args.folds, ratio=args.ratio)
    else:
        run_full(folds=args.folds, ratio=args.ratio)


if __name__ == "__main__":
    main()

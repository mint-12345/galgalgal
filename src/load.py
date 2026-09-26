"""원본 CSV 로드 (CLAUDE.md 4절·10절 1단계).

data/raw/만 읽는다. data/locked/는 어떤 경로로도 읽지 않는다 (CLAUDE.md 3절).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SPLITS = {"train": "CMP-training", "val": "CMP-validation", "test": "CMP-test"}
N_FILES = 185
N_COLS = 25
KEY = ["WAFER_ID", "STAGE"]


def load_sensors(split: str) -> pd.DataFrame:
    """split 하나의 센서 파일 185개를 읽어 합치고 `split` 컬럼을 붙인다."""
    prefix = SPLITS[split]
    files = sorted((RAW / split).glob(f"{prefix}-[0-9][0-9][0-9].csv"))
    if len(files) != N_FILES:
        raise ValueError(f"{split}: 파일 {len(files)}개 (기대 {N_FILES}개)")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df.shape[1] != N_COLS:
        raise ValueError(f"{split}: 컬럼 {df.shape[1]}개 (기대 {N_COLS}개)")
    df["split"] = split
    return df


def load_labels(split: str) -> pd.DataFrame:
    """train·val 라벨만 읽는다. test 라벨은 읽을 수 없다."""
    if split not in ("train", "val"):
        raise ValueError(f"{split} 라벨은 읽지 않는다 (CLAUDE.md 3절)")
    df = pd.read_csv(RAW / "labels" / f"{SPLITS[split]}-removalrate.csv")
    if df.duplicated(KEY).any():
        raise ValueError(f"{split} 라벨에 중복 키가 있다")
    return df

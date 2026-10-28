"""SR·전이학습 공통 피처 테이블 data/processed/features_v1.csv를 만든다.

- 행: folds_v1.csv와 같은 train+val 2,401행 (키 = (WAFER_ID, STAGE)). test 웨이퍼·data/locked/는 쓰지 않는다.
- 라벨: MRR은 읽지 않는다 (wafer_table.csv에서 MRR 열을 읽지 않음). 상관 계산·변수 선택도 하지 않는다.
- 열 (키 3 + 피처):
  1. wafer_table.csv의 라벨-프리 R 피처 (Di ID 22~125, 104열). 라벨 의존 피처(ID 1~21)는 집계표에 없다.
  2. since_{U}_w: 소모품 "리셋 후 경과량" (wafer_table.csv의 S3 열 그대로, 정의 확인만).
     DRESSER_TABLE은 리셋이 한 번도 없어 집계표에 열이 없다 → 만들지 않는다.
  3. P1_on: RETAINER_RING_PRESSURE 평균, "압력 6개 중 하나라도 > 0 이고 STAGE_ROTATION > 0"인 행
     (src/p1_on_val.py, hyp_r2r_retainer.py와 같은 정의).
  4. frac_hi: P1_on 행 중 RETAINER_RING_PRESSURE > 4,000인 행 비율 (7,135 단계 체류 비율).
  5. ch56_step: 챔버 5·6 단계 수행 여부 (0/1). 압력 6개 중 하나라도 > 0인 행 수 ≥ 경계(A-456 245 / B-456 255)이면 1.
     results/diagnostics/2026-10-28_김혜연_recipe_cluster.md의 press 기준 무리와 같은 정의. A-123은 NaN.
  6. allzero: 전체-0 판정 센서 12개(압력 6 + 슬러리 3 + 회전 3) 중 웨이퍼 전체가 0인 센서가 하나라도 있으면 1.
- 결측 처리: 웨이퍼 전체가 0인 센서에서 계산된 R 피처(mean·std·auc, 두 구간)를 0 → NaN.
  연마 중간에 꺼져 생긴 0은 그대로 둔다(웨이퍼 전체가 0인 경우만 바꿈). 소모품 카운터·DRESS_WATER는 바꾸지 않는다.
- 센서 기반 열(3~6, 전체-0 판정)은 train·val 원본 센서를 다시 읽어 계산한다 (완전 중복 행 제거 후).

실행: python src/make_features.py
출력: data/processed/features_v1.csv, feature_families_v1.csv, features_v1.md
출력 파일이 이미 있으면 멈춘다 (덮어쓰기 금지).
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import ALLZERO_SENSORS, FLOW, PRESSURE, ROTATION, USAGE, dedup, r_feature_names  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402

PROC = ROOT / "data" / "processed"
TABLE = PROC / "wafer_table.csv"
FOLDS = PROC / "folds_v1.csv"
OUT = PROC / "features_v1.csv"
OUT_FAM = PROC / "feature_families_v1.csv"
OUT_MD = PROC / "features_v1.md"
SPLITS = ("train", "val")
N_EXPECTED = 2401
RET = "RETAINER_RING_PRESSURE"
HI = 4000                                   # frac_hi: 7,135 단계 (1,450 단계와 가르는 값)
CH56_BOUND = {"A-456": 245, "B-456": 255}   # recipe_cluster.md 경계 (압력 > 0 행 수)
ABBR = {**PRESSURE, **FLOW, **ROTATION}     # 원본 컬럼 → 약어
SINCE = [f"since_{ab}_w" for ab in USAGE.values() if ab != "DRESSER_TABLE"]
ADDED = ["P1_on", "frac_hi", "ch56_step", "allzero"]

FAMILY_BY_ABBR = {
    **{ab: "소모품 카운터" for ab in USAGE.values()},
    "RETAINER": "리테이너 링 압력",
    **{ab: "에어백 압력" for ab in ("MAIN_OUTER", "CENTER", "RIPPLE", "EDGE")},
    "PCHAMBER": "확인 필요",                 # 가압 챔버 압력: 에어백 계열인지 불분명
    **{ab: "슬러리 유량" for ab in ("SLURRY_A", "SLURRY_B", "SLURRY_C")},
    "DRESS_WATER": "확인 필요",              # 드레싱 워터 상태(0/1): 슬러리가 아니고, 유량 4번째 변수라는 것도 추정
    **{ab: "회전" for ab in ROTATION.values()},
    "POLISH": "시간",
}


def split_name(feat: str) -> tuple[str, str, str]:
    """'{stat}_{abbr}_{seg}' → (stat, abbr, seg). 약어에 '_'가 들어 있을 수 있다."""
    stat, rest = feat.split("_", 1)
    abbr, seg = rest.rsplit("_", 1)
    return stat, abbr, seg


def sensor_columns(keys: pd.DataFrame) -> pd.DataFrame:
    """원본 센서(train·val)로 웨이퍼별 P1_on, frac_hi, 압력 > 0 행 수, 센서별 전체-0 여부."""
    parts = []
    for split in SPLITS:
        d, _ = dedup(load_sensors(split))
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.sort_values(KEY + ["TIMESTAMP"], kind="stable").reset_index(drop=True)
    by = [raw[k] for k in KEY]
    press = (raw[list(PRESSURE)] > 0).any(axis=1)
    on = press & (raw["STAGE_ROTATION"] > 0)
    ret_on = raw[RET].where(on)
    out = pd.concat([
        ret_on.groupby(by).mean().rename("P1_on"),
        (ret_on > HI).where(on).groupby(by).mean().rename("frac_hi"),
        press.groupby(by).sum().rename("n_press_rows"),
        *[(raw[s] == 0).groupby(by).all().rename(f"az_{ABBR[s]}") for s in ALLZERO_SENSORS],
    ], axis=1).reset_index()
    out = keys[KEY].merge(out, on=KEY, how="left", validate="1:1")
    if out["n_press_rows"].isna().any():
        raise SystemExit("원본 센서에 없는 키가 있다")
    # 기존 진단(hyp_r2r_retainer)의 7,135 단계 경계는 > 3,000이었다. 두 경계 사이 행 수를 기록한다.
    r = raw.loc[on, RET]
    out.attrs["n_on_rows"] = int(len(r))
    out.attrs["n_mid_rows"] = int(((r > 3000) & (r <= HI)).sum())
    return out


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(FOLDS)
    keys = folds.loc[folds["repeat"] == 0, KEY + ["group"]].sort_values(KEY, kind="mergesort").reset_index(drop=True)
    if len(keys) != N_EXPECTED:
        raise SystemExit(f"folds_v1 키 {len(keys)}행 (기대 {N_EXPECTED})")

    rfeat = r_feature_names()
    tab = pd.read_csv(TABLE, usecols=KEY + ["group", "split", "n_allzero_sensors"] + rfeat + SINCE)  # MRR 읽지 않음
    t = keys.merge(tab, on=KEY, how="left", suffixes=("", "_tab"), validate="1:1")
    if t["split"].isna().any() or (t["group"] != t["group_tab"]).any() or not t["split"].isin(SPLITS).all():
        raise SystemExit("folds_v1 키가 wafer_table의 train·val 행과 맞지 않는다")

    s = sensor_columns(keys)
    az_cols = [f"az_{ABBR[c]}" for c in ALLZERO_SENSORS]
    n_az = s[az_cols].sum(axis=1)
    if not (n_az.to_numpy() == t["n_allzero_sensors"].to_numpy()).all():
        raise SystemExit("전체-0 센서 수가 wafer_table.n_allzero_sensors와 다르다")

    f = t[KEY + ["group"] + rfeat + SINCE].copy()
    # 결측 처리: 웨이퍼 전체가 0인 센서의 R 피처 0 → NaN
    nan_rows = []
    for col in rfeat:
        stat, ab, seg = split_name(col)
        if f"az_{ab}" not in s.columns:
            continue
        m = s[f"az_{ab}"].to_numpy()
        vals = f.loc[m, col]
        if not (vals.isna() | (vals == 0)).all():
            raise SystemExit(f"{col}: 전체-0 센서인데 0이 아닌 값이 있다")
        changed = m & (f[col] == 0).to_numpy()
        if changed.any():
            nan_rows.append({"feature": col, "sensor": ab,
                             **{g: int((changed & (f["group"] == g).to_numpy()).sum()) for g in ("A-456", "B-456", "A-123")},
                             "total": int(changed.sum())})
            f.loc[changed, col] = np.nan

    f["P1_on"] = s["P1_on"]
    f["frac_hi"] = s["frac_hi"]
    bound = f["group"].map(CH56_BOUND)
    f["ch56_step"] = np.where(bound.isna(), np.nan, (s["n_press_rows"] >= bound).astype(float))
    f["allzero"] = (n_az > 0).astype(int)
    f.attrs.update(s.attrs)

    fam = pd.DataFrame({"feature": rfeat + SINCE + ADDED})
    fam["family"] = [
        FAMILY_BY_ABBR[split_name(c)[1]] if c in rfeat else
        "리셋 후 경과량" if c in SINCE else
        "리테이너 링 압력" if c in ("P1_on", "frac_hi") else "기타"
        for c in fam["feature"]
    ]
    return f, fam, pd.DataFrame(nan_rows)


def to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    out += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main() -> None:
    for p in (OUT, OUT_FAM, OUT_MD):
        if p.exists():
            raise SystemExit(f"{p} 가 이미 있다. 덮어쓰지 않는다 (CLAUDE.md 3절 3번).")
    f, fam, nan = build()
    print(f"features {f.shape}, NaN 처리 피처 {len(nan)}개", flush=True)

    # 검증
    folds = pd.read_csv(FOLDS)
    k_f = set(map(tuple, folds.loc[folds["repeat"] == 0, KEY].to_numpy()))
    k_t = set(map(tuple, f[KEY].to_numpy()))
    checks = [("행 수 2,401", len(f) == N_EXPECTED, f"{len(f):,}행"),
              ("키가 folds_v1.csv와 완전히 일치", k_f == k_t and not f.duplicated(KEY).any(),
               f"folds 키 {len(k_f):,} / features 키 {len(k_t):,}, 차집합 {len(k_f ^ k_t)}")]
    f2, fam2, _ = build()
    same = to_bytes(f) == to_bytes(f2) and to_bytes(fam) == to_bytes(fam2)
    checks.append(("같은 코드로 다시 만들면 같은 파일", same,
                   f"SHA-256 {hashlib.sha256(to_bytes(f)).hexdigest()[:16]}… 두 번 생성 일치" if same else "불일치"))
    for name, ok, note in checks:
        print(f"{'통과' if ok else '실패'}  {name}: {note}", flush=True)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("검증 실패. 파일을 쓰지 않는다.")

    OUT.write_bytes(to_bytes(f))
    OUT_FAM.write_bytes(to_bytes(fam))

    groups = ["A-456", "B-456", "A-123"]
    n_by = f.groupby("group").size().reindex(groups)
    nan_sum = (nan.groupby("sensor")[groups + ["total"]].agg(["count", "sum"])
               if len(nan) else pd.DataFrame())
    sens = nan.groupby("sensor").agg(n_features=("feature", "size"),
                                     **{f"{g} 웨이퍼": (g, "max") for g in groups}).reset_index() if len(nan) else nan
    added_nan = pd.DataFrame([{"열": c, **{g: int(f.loc[f.group == g, c].isna().sum()) for g in groups}}
                              for c in SINCE + ADDED])
    famc = fam["family"].value_counts().rename_axis("family").reset_index(name="피처 수")
    md = [
        "# features_v1 — SR·전이학습 공통 피처 테이블",
        "",
        f"- 생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 생성 코드: `src/make_features.py` (실행: `python src/make_features.py`)",
        f"- 파일: `data/processed/features_v1.csv` ({len(f):,}행 × {f.shape[1]}열), "
        f"SHA-256 `{hashlib.sha256(to_bytes(f)).hexdigest()}`",
        "- 계열 초안: `data/processed/feature_families_v1.csv` (열: feature, family). **팀이 최종 확인할 초안**이다.",
        f"- 행: `folds_v1.csv`와 같은 train+val 2,401행 (A-456 {n_by['A-456']} / B-456 {n_by['B-456']} / A-123 {n_by['A-123']}).",
        "- MRR은 읽지 않았다 (집계표에서 MRR 열 제외). 상관 계산·변수 선택은 하지 않았다. test 웨이퍼와 `data/locked/`는 쓰지 않았다.",
        "",
        "## 검증",
        "",
        md_table(pd.DataFrame([{"검증": n, "결과": "통과" if ok else "실패", "내용": note} for n, ok, note in checks])),
        "",
        "## 열 목록과 정의",
        "",
        "| 열 | 개수 | 정의 | 출처 |",
        "|---|---|---|---|",
        "| `WAFER_ID`, `STAGE`, `group` | 3 | 키와 그룹 | `folds_v1.csv` |",
        "| `dur_POLISH_{s1,s2}` | 2 | 구간별 연마 시간 (Di ID 22~23) | `wafer_table.csv` 그대로 |",
        "| `{mean,std,slope}_{U}_{s1,s2}` | 36 | 소모품 카운터 6개의 구간 통계 (ID 24~59) | 그대로 |",
        "| `{mean,std,auc}_{P}_{s1,s2}` | 36 | 압력 6개 (ID 60~95) | 그대로 + 결측 처리 |",
        "| `{mean,std,auc}_{F}_{s1,s2}` | 24 | 슬러리 A·B·C + DRESS_WATER (ID 96~119) | 그대로 + 결측 처리 (DRESS_WATER 제외) |",
        "| `mean_{R}_{s1,s2}` | 6 | 회전 3개 평균 (ID 120~125) | 그대로 + 결측 처리 |",
        f"| `since_{{U}}_w` | {len(SINCE)} | 소모품 리셋 후 경과량 = 시작값 − 가장 최근 리셋 웨이퍼의 시작값. 스테이지 첫 리셋 이전은 NaN | `wafer_table.csv` S3 열 그대로 (정의 확인) |",
        "| `P1_on` | 1 | `RETAINER_RING_PRESSURE` 평균, 압력 6개 중 하나라도 > 0 이고 `STAGE_ROTATION` > 0 인 행 | 원본 센서 (train·val) |",
        f"| `frac_hi` | 1 | `P1_on` 행 중 `RETAINER_RING_PRESSURE` > {HI:,}인 행 비율 (7,135 단계 체류 비율) | 원본 센서 |",
        "| `ch56_step` | 1 | 챔버 5·6 단계 수행 여부 (0/1): 압력 6개 중 하나라도 > 0인 행 수 ≥ 245 (A-456) / 255 (B-456). A-123은 NaN | 원본 센서, `results/diagnostics/2026-10-28_김혜연_recipe_cluster.md`와 같은 정의 |",
        "| `allzero` | 1 | 전체-0 판정 센서 12개 중 웨이퍼 전체가 0인 센서가 하나라도 있으면 1 | 원본 센서 (`n_allzero_sensors > 0`과 일치 확인) |",
        "",
        "- `U` = BACKING_FILM, DRESSER, POLISH_TABLE, DRESSER_TABLE, MEMBRANE, PSHEET / `P` = PCHAMBER, MAIN_OUTER, CENTER, RETAINER, RIPPLE, EDGE /"
        " `F` = SLURRY_A·B·C, DRESS_WATER / `R` = WAFER_ROT, STAGE_ROT, HEAD_ROT. 이름 규칙과 단위는 `COLUMNS.md`.",
        "- **`since_DRESSER_TABLE_w`는 없다.** DRESSER_TABLE은 리셋이 한 번도 감지되지 않아 집계표가 열을 만들지 않았다 (COLUMNS.md 4절). 6종 중 5종만 있다.",
        "- `since_*` 리셋 감지는 집계표대로 train·val·test 센서를 합쳐 한 것이다 (라벨 불필요, 임계값 0.20은 임시값).",
        "- 집계표의 다른 열(`t_start`, `chambers`, 플래그, `start_*`·`delta_*`·`reset_*`·`nreset_*` 등)은 요청 범위 밖이라 넣지 않았다.",
        "",
        "## 결측 처리 (0 → NaN)",
        "",
        "웨이퍼 전체가 0인 센서(전체-0 판정 12개)에서 계산된 R 피처만 NaN으로 바꿨다. 바꾸기 전 값이 모두 0(또는 이미 NaN)인지 확인했다.",
        "연마 중간에 꺼져 생긴 0, 소모품 카운터, DRESS_WATER는 바꾸지 않았다. 456 그룹에는 전체-0 센서가 없다.",
        "",
        "센서별 요약 (웨이퍼 수 = 그 센서의 피처에서 NaN으로 바뀐 웨이퍼 수):",
        "",
        md_table(sens),
        "",
        "피처별 (NaN으로 바꾼 값 수):",
        "",
        md_table(nan),
        "",
        "## 추가 열과 since 열의 NaN 수 (그룹별)",
        "",
        md_table(added_nan),
        "",
        "- `P1_on`·`frac_hi` NaN = 압력 > 0 이고 STAGE 회전 > 0인 행이 하나도 없는 웨이퍼. A-123은 대부분 기록 누락 웨이퍼이고,"
        " 456 그룹에도 있다(모두 `ch56_step` = 0). `ch56_step`의 A-123 NaN은 정의상.",
        f"- `allzero` = 1: A-123 {int(f.loc[f.group == 'A-123', 'allzero'].sum())}장, 456 그룹 0장.",
        f"- `frac_hi` 경계 {HI:,}: 기존 진단(`hyp_r2r_retainer.py`)의 7,135 단계 경계는 > 3,000이었다. 3,000 초과 {HI:,} 이하인 행은"
        f" P1_on 행의 {f.attrs['n_mid_rows']:,} / {f.attrs['n_on_rows']:,}행({f.attrs['n_mid_rows'] / f.attrs['n_on_rows'] * 100:.2f}%, 단계 전환 구간)이라 두 정의의 차이는 작다.",
        "- R 피처의 기존 NaN(챔버 4만 있는 2장의 `_s2` 등)은 그대로다.",
        "",
        "## 계열 초안 (feature_families_v1.csv)",
        "",
        md_table(famc),
        "",
        "- **확인 필요**: `PCHAMBER`(가압 챔버 압력, 에어백 계열인지 불분명), `DRESS_WATER`(드레싱 워터 상태 0/1, 슬러리가 아니며 유량 4번째 변수라는 것도 추정).",
        "- `frac_hi`는 리테이너 압력 단계의 체류 비율이라 \"시간\"으로 볼 수도 있으나, 리테이너 압력 수준으로 정의되므로 리테이너 링 압력으로 두었다.",
        "- `dur_POLISH_*`는 구간 체류 시간(마지막 − 첫 TIMESTAMP)이라 \"시간\"이다.",
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"저장: {OUT.name}, {OUT_FAM.name}, {OUT_MD.name}", flush=True)


if __name__ == "__main__":
    main()

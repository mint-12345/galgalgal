"""팀 공통 교차검증 fold 파일 data/processed/folds_v1.csv를 만든다.

SR, A-123 분리/통합 비교, 전이학습이 모두 이 파일 하나를 공유한다.

- 대상: wafer_table.csv의 train·val 웨이퍼 중 is_outlier = 0 (2,401행, 키 = (WAFER_ID, STAGE)).
  test 웨이퍼는 넣지 않는다. MRR 열은 읽지 않는다 (이상치는 is_outlier 열로 판단).
- 5조각 × 5회 = 25개 fold. 반복(repeat) 0~4에 시드 42~46.
- StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=시드),
  groups = WAFER_ID (같은 웨이퍼의 A·B 스테이지 기록은 같은 조각), 층화 = group (A-456 / B-456 / A-123).
- 출력 열: WAFER_ID, STAGE, group, repeat, fold (fold = 그 반복에서 이 행이 평가 쪽이 되는 조각 0~4).

검증: (1) 반복마다 모든 행이 정확히 한 번 평가 쪽 (2) 같은 WAFER_ID가 두 조각에 나뉘지 않음
      (3) 조각별 행 수와 그룹 비율 (4) 같은 시드로 다시 만들면 같은 파일.
하나라도 실패하면 파일을 쓰지 않는다. 출력 파일이 이미 있으면 멈춘다 (덮어쓰기 금지).

실행: python src/make_folds.py
출력: data/processed/folds_v1.csv, data/processed/folds_v1.md
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "wafer_table.csv"
OUT = ROOT / "data" / "processed" / "folds_v1.csv"
OUT_MD = ROOT / "data" / "processed" / "folds_v1.md"
KEY = ["WAFER_ID", "STAGE"]
SPLITS = ["train", "val"]
GROUPS = ["A-456", "B-456", "A-123"]
N_SPLITS = 5
SEEDS = [42, 43, 44, 45, 46]  # repeat 0~4
N_EXPECTED = 2401


def load_targets() -> pd.DataFrame:
    tab = pd.read_csv(TABLE, usecols=KEY + ["split", "group", "is_outlier"])
    t = tab[tab["split"].isin(SPLITS) & (tab["is_outlier"] == 0)]
    if t.duplicated(KEY).any():
        raise SystemExit("대상에 중복 키가 있다")
    if len(t) != N_EXPECTED:
        raise SystemExit(f"대상 {len(t)}행 (기대 {N_EXPECTED}행). 멈춘다.")
    return t[KEY + ["group"]].sort_values(KEY, kind="mergesort").reset_index(drop=True)


def build(t: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for rep, seed in enumerate(SEEDS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold = pd.Series(-1, index=t.index)
        for k, (_, ev) in enumerate(cv.split(t, y=t["group"], groups=t["WAFER_ID"])):
            fold.iloc[ev] = k
        parts.append(t.assign(repeat=rep, fold=fold.to_numpy()))
    return pd.concat(parts, ignore_index=True)[KEY + ["group", "repeat", "fold"]]


def to_bytes(f: pd.DataFrame) -> bytes:
    return f.to_csv(index=False, lineterminator="\n").encode("utf-8")


def check(t: pd.DataFrame, f: pd.DataFrame) -> tuple[list[tuple[str, bool, str]], pd.DataFrame]:
    res = []
    # 1. 반복마다 모든 행이 정확히 한 번 평가 쪽
    ok1 = True
    for rep in range(len(SEEDS)):
        r = f[f["repeat"] == rep]
        ok1 &= len(r) == len(t) and not r.duplicated(KEY).any() and r["fold"].between(0, N_SPLITS - 1).all() \
            and set(map(tuple, r[KEY].to_numpy())) == set(map(tuple, t[KEY].to_numpy()))
    res.append(("1. 반복마다 2,401행이 정확히 한 번씩 평가 쪽", bool(ok1),
                f"반복 {len(SEEDS)}개 × {len(t):,}행, 키 중복 0, fold 값 0~{N_SPLITS - 1}"))
    # 2. 같은 WAFER_ID가 두 조각에 나뉘지 않음
    split_ids = f.groupby(["repeat", "WAFER_ID"])["fold"].nunique()
    n_bad = int((split_ids > 1).sum())
    both = int((t.groupby("WAFER_ID").size() > 1).sum())
    res.append(("2. 같은 WAFER_ID가 두 조각에 나뉘지 않음", n_bad == 0,
                f"두 조각에 걸친 WAFER_ID {n_bad}개 (A·B 두 기록이 있는 WAFER_ID {both:,}개 포함, 전 반복)"))
    # 3. 조각별 행 수와 그룹 비율
    overall = t["group"].value_counts(normalize=True).reindex(GROUPS)
    cnt = f.groupby(["repeat", "fold"])["group"].value_counts().unstack(fill_value=0).reindex(columns=GROUPS)
    tab = cnt.copy()
    tab.insert(0, "n", cnt.sum(axis=1))
    for g in GROUPS:
        tab[f"{g} 비율"] = (cnt[g] / tab["n"]).round(3)
    dev = max((cnt[g] / tab["n"] - overall[g]).abs().max() for g in GROUPS)
    res.append(("3. 조각별 행 수·그룹 비율", True,
                f"행 수 {tab['n'].min()}~{tab['n'].max()}, 그룹 비율과 전체 비율의 최대 차이 {dev * 100:.2f}%p "
                f"(전체 A-456 {overall['A-456']:.3f} / B-456 {overall['B-456']:.3f} / A-123 {overall['A-123']:.3f})"))
    # 4. 같은 시드로 다시 만들면 같은 파일
    again = build(load_targets())
    h1, h2 = hashlib.sha256(to_bytes(f)).hexdigest(), hashlib.sha256(to_bytes(again)).hexdigest()
    res.append(("4. 같은 시드로 다시 만들면 같은 파일", h1 == h2, f"SHA-256 {h1[:16]}… 두 번 생성 일치"))
    return res, tab.reset_index()


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main() -> None:
    for p in (OUT, OUT_MD):
        if p.exists():
            raise SystemExit(f"{p} 가 이미 있다. 덮어쓰지 않는다 (CLAUDE.md 3절 3번).")
    t = load_targets()
    f = build(t)
    res, tab = check(t, f)
    for name, ok, note in res:
        print(f"{'통과' if ok else '실패'}  {name}: {note}", flush=True)
    if not all(ok for _, ok, _ in res):
        raise SystemExit("검증 실패. 파일을 쓰지 않는다.")

    OUT.write_bytes(to_bytes(f))
    if hashlib.sha256(OUT.read_bytes()).digest() != hashlib.sha256(to_bytes(build(t))).digest():
        raise SystemExit("저장한 파일이 재생성 결과와 다르다")
    n_by = t.groupby("group").size().reindex(GROUPS)
    md = [
        "# folds_v1 — 팀 공통 교차검증 fold",
        "",
        f"- 생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 생성 코드: `src/make_folds.py` (실행: `python src/make_folds.py`)",
        f"- 파일: `data/processed/folds_v1.csv` ({len(f):,}행 = {len(t):,}행 × 반복 {len(SEEDS)}회), "
        f"SHA-256 `{hashlib.sha256(OUT.read_bytes()).hexdigest()}`"
        " (LF 줄바꿈 기준. `core.autocrlf=true`인 Windows 체크아웃은 CRLF가 되어 해시가 다르다. 내용 비교는 pandas로 읽어서 한다)",
        f"- 환경: Python {sys.version.split()[0]}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}",
        "- 사용처: SR, A-123 분리/통합 비교, 전이학습이 모두 이 파일 하나를 공유한다.",
        "",
        "## 사양",
        "",
        "- 입력: `data/processed/wafer_table.csv`의 train·val 웨이퍼, `is_outlier = 0` (이상치 4장 = train A-123, MRR > 400 제외).",
        "  MRR 열은 읽지 않았다. test 웨이퍼는 넣지 않았고 `data/locked/`와 test 파일은 열지 않았다.",
        f"- 대상: {len(t):,}행, 키 `(WAFER_ID, STAGE)`. A-456 {n_by['A-456']} / B-456 {n_by['B-456']} / A-123 {n_by['A-123']}.",
        "- 5조각 × 5회 = 25개 fold. 반복(repeat) 0~4에 시드 42~46.",
        "- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=시드)`. groups = `WAFER_ID`"
        " (같은 웨이퍼의 A·B 스테이지 기록은 같은 조각), 층화 = `group`.",
        "- 열: `WAFER_ID, STAGE, group, repeat, fold`. `fold` = 그 반복에서 이 행이 평가 쪽이 되는 조각 번호(0~4).",
        "  반복 r, 조각 k의 학습 쪽 = `repeat == r & fold != k`, 평가 쪽 = `repeat == r & fold == k`.",
        "",
        "## 검증",
        "",
        md_table(pd.DataFrame([{"검증": n, "결과": "통과" if ok else "실패", "내용": note} for n, ok, note in res])),
        "",
        "### 조각별 행 수와 그룹 비율 (평가 쪽 기준)",
        "",
        md_table(tab),
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"저장: {OUT.relative_to(ROOT)} ({len(f):,}행), {OUT_MD.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

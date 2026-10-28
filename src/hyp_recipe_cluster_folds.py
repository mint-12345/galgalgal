"""recipe cluster 판정 조건 (3) 확인 (진단, CLAUDE.md 14.4).

(3) folds_v1.csv의 25개 fold(5회 × 5조각) 모두에서 학습 쪽(그 fold가 아닌 행)에 작은 무리가 1장 이상.

입력: results/diagnostics/2026-10-28_김혜연_recipe_cluster_wafer.csv (hyp_recipe_cluster.py 결과, MRR 없음)의
      long_press(압력 > 0 행 수 기준 무리), data/processed/folds_v1.csv.
MRR·라벨 파일·test·data/locked/는 읽지 않는다.

실행: python src/hyp_recipe_cluster_folds.py
출력: results/diagnostics/2026-10-28_김혜연_recipe_cluster_folds.csv (+ .yaml)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results" / "diagnostics"
WAFER = DIAG / "2026-10-28_김혜연_recipe_cluster_wafer.csv"
FOLDS = ROOT / "data" / "processed" / "folds_v1.csv"
OUT = DIAG / "2026-10-28_김혜연_recipe_cluster_folds.csv"
KEY = ["WAFER_ID", "STAGE"]
GROUPS = ["A-456", "B-456"]
N_FOLDS = 25


def main() -> None:
    for p in (OUT, OUT.with_suffix(".yaml")):
        if p.exists():
            raise SystemExit(f"{p} 가 이미 있다. 덮어쓰지 않는다 (CLAUDE.md 3절 3번).")
    w = pd.read_csv(WAFER, usecols=KEY + ["group", "long_press"])
    folds = pd.read_csv(FOLDS)
    f = folds[folds["group"].isin(GROUPS)].merge(w, on=KEY, how="left", suffixes=("", "_w"), validate="m:1")
    if f["long_press"].isna().any() or (f["group"] != f["group_w"]).any():
        raise SystemExit("folds_v1과 웨이퍼 표의 키·그룹이 맞지 않는다")

    rows = []
    for grp in GROUPS:
        g = f[f["group"] == grp]
        n_rows = g.groupby("repeat").size()
        if n_rows.nunique() != 1:
            raise SystemExit(f"{grp}: 반복마다 행 수가 다르다")
        short_all = int((g.loc[g["repeat"] == 0, "long_press"] == 0).sum())
        long_all = int(n_rows.iloc[0]) - short_all
        minor = 0 if short_all <= long_all else 1
        for (rep, k), _ in g.groupby(["repeat", "fold"]):
            r = g[g["repeat"] == rep]
            tr = r[r["fold"] != k]
            n_short = int((tr["long_press"] == 0).sum())
            n_long = int((tr["long_press"] == 1).sum())
            n_minor = n_short if minor == 0 else n_long
            rows.append({"group": grp, "repeat": rep, "fold": k, "n_train": len(tr),
                         "train_short": n_short, "train_long": n_long,
                         "minor": ["short", "long"][minor], "train_minor": n_minor,
                         "train_minor_frac": round(n_minor / len(tr), 3), "ok": n_minor >= 1})
    t = pd.DataFrame(rows)
    if t.groupby("group").size().ne(N_FOLDS).any():
        raise SystemExit("그룹별 fold 수가 25가 아니다")
    t.to_csv(OUT, index=False)
    OUT.with_suffix(".yaml").write_text("\n".join([
        "date: 2026-10-28", "author: 김혜연", "script: src/hyp_recipe_cluster_folds.py",
        "question: 판정 조건 (3) — folds_v1 25개 fold 모두 학습 쪽에 작은 무리 1장 이상",
        f"input: {WAFER.relative_to(ROOT).as_posix()} (long_press), {FOLDS.relative_to(ROOT).as_posix()}",
        "labels: MRR 사용 안 함",
        "train_side: 같은 repeat에서 fold != k 인 행",
        "note: 진단 결과 (14.4)",
    ]) + "\n", encoding="utf-8")

    for grp in GROUPS:
        g = t[t["group"] == grp]
        print(f"{grp}: 작은 무리 = {g['minor'].iloc[0]}, 학습 쪽 작은 무리 {g['train_minor'].min()}~{g['train_minor'].max()}장 "
              f"({g['train_minor_frac'].min():.3f}~{g['train_minor_frac'].max():.3f}), 충족 fold {int(g['ok'].sum())}/{len(g)}",
              flush=True)


if __name__ == "__main__":
    main()

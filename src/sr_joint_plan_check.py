"""공동 SR 계획(0단계)에 넣을 라벨 비의존 점검. MRR은 읽지 않는다.

1. 공통 후보 = A-456·B-456 정식 SR 후보 10개씩의 합집합 (sr_candidates 결과)
2. 그룹별 결측 수와 공동 사용 웨이퍼 수 (후보 전부 값 있음)
3. 그룹별·두 그룹 합산 절댓값 중앙값 (그룹 train+val 전체, sr_run.load_frame 규칙)
4. folds_v1에서 A·B 공유 WAFER_ID가 반복마다 같은 조각에 있는지

실행: python src/sr_joint_plan_check.py   출력: 화면 (계획 파일에 옮겨 적음)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_run import CANDIDATES, GROUPS, PROC  # noqa: E402

KEYS = ["WAFER_ID", "STAGE"]


def union_candidates():
    c = pd.read_csv(CANDIDATES)
    sel = {g: c[(c["group"] == g) & (c["selected"] == 1)].sort_values("rank")["feature"].tolist() for g in GROUPS}
    union = list(dict.fromkeys(sel["A-456"] + sel["B-456"]))
    return sel, union


def wafers():
    """train+val, is_outlier = 0인 A-456·B-456 웨이퍼의 피처 (MRR 열은 읽지 않음)."""
    feats = pd.read_csv(PROC / "features_v1.csv")
    table = pd.read_csv(PROC / "wafer_table.csv", usecols=KEYS + ["split", "is_outlier"])
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    d = feats[feats["group"].isin(GROUPS)].merge(table, on=KEYS, validate="one_to_one")
    return d.reset_index(drop=True)


def main():
    sel, union = union_candidates()
    d = wafers()
    print("후보 A:", sel["A-456"]); print("후보 B:", sel["B-456"])
    print(f"합집합 {len(union)}개:", union)
    print("공통:", [v for v in sel["A-456"] if v in sel["B-456"]])
    rows = []
    for v in union:
        r = {"variable": v, "in_A": int(v in sel["A-456"]), "in_B": int(v in sel["B-456"])}
        for g in GROUPS:
            x = d.loc[d["group"] == g, v]
            r[f"nan_{g}"] = int(x.isna().sum())
            r[f"absmed_{g}"] = float(x.abs().median())
        r["absmed_pooled"] = float(d[v].abs().median())
        r["ratio_A_over_B"] = r["absmed_A-456"] / r["absmed_B-456"]
        rows.append(r)
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.round(4).to_string(index=False))
    for g in GROUPS:
        x = d[d["group"] == g]
        own = x[sel[g]].notna().all(axis=1)
        joint = x[union].notna().all(axis=1)
        print(f"{g}: 전체 {len(x)}, 정식 SR 웨이퍼 {int(own.sum())}, 공동(14개 값 있음) {int(joint.sum())},"
              f" 정식에 있으나 공동에서 빠짐 {int((own & ~joint).sum())}")
    # fold 점검
    f = pd.read_csv(PROC / "folds_v1.csv")
    f = f.drop(columns=[c for c in ("group",) if c in f.columns]).merge(d[KEYS + ["group"]], on=KEYS, how="inner")
    shared = set(d.loc[d.group == "A-456", "WAFER_ID"]) & set(d.loc[d.group == "B-456", "WAFER_ID"])
    bad = 0
    for rep, p in f.groupby("repeat"):
        s = p[p["WAFER_ID"].isin(shared)].groupby("WAFER_ID")["fold"].nunique()
        bad += int((s > 1).sum())
    print(f"공유 WAFER_ID {len(shared)}개, 반복 5회 중 같은 WAFER_ID가 다른 조각에 있는 경우: {bad}")
    jw = d[d[union].notna().all(axis=1)]
    shared_joint = set(jw.loc[jw.group == "A-456", "WAFER_ID"]) & set(jw.loc[jw.group == "B-456", "WAFER_ID"])
    print(f"공동 웨이퍼 안 공유 WAFER_ID {len(shared_joint)}개")
    sizes = f.merge(jw[KEYS], on=KEYS).groupby(["repeat", "fold", "group"]).size().unstack()
    print("공동 웨이퍼 fold 크기 범위:", sizes.min().to_dict(), sizes.max().to_dict())


if __name__ == "__main__":
    main()

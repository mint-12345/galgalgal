"""SR 후보 변수를 preregistration.md 4.2절 + 4.8절 규칙으로 고른다 (A-456, B-456 각 k = 10).

- 입력: features_v1.csv(FV1), feature_families_v1.csv, wafer_table.csv의 MRR.
  train·val, is_outlier = 0만 쓴다. test 라벨과 data/locked/는 읽지 않는다.
- 후보 풀: sr_candidate = 1 → since_POLISH_TABLE_w 제외 (4.8 11번) → 그룹 안 NaN 5% 초과 제외 →
  절댓값 중앙값 1e-9 미만 제외 (4.8 2·3번).
- 선택: 대표 자리 보장 계열마다 |스피어만 상관|이 가장 큰 변수 1개 →
  남은 자리는 상관 순서로 채우되 이미 뽑힌 변수와 |스피어만 상관| > 0.9면 건너뜀.
- 계산 데이터: 그룹별 train+val 전체 1회 (4.8 1번).

실행: python src/sr_candidates.py
출력: results/diagnostics/{날짜}_박종화_sr_candidates.csv, 같은 이름의 .yaml
      이미 있으면 멈춘다 (덮어쓰지 않음).
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "diagnostics"
GROUPS = ["A-456", "B-456"]
EXPECTED_N = {"A-456": 983, "B-456": 987}
EXPECTED_POOL = 106
EXCLUDED_FEATURES = {"since_POLISH_TABLE_w": "missed_reset"}   # 4.8 11번
K = 10
MAX_NAN_FRAC = 0.05
MIN_ABS_MEDIAN = 1e-9
MAX_PAIR_CORR = 0.9


def load():
    feats = pd.read_csv(PROC / "features_v1.csv")
    fam = pd.read_csv(PROC / "feature_families_v1.csv")
    table = pd.read_csv(
        PROC / "wafer_table.csv",
        usecols=["WAFER_ID", "STAGE", "split", "MRR", "is_outlier"],
    )
    table = table[table["split"].isin(["train", "val"]) & (table["is_outlier"] == 0)]
    data = feats.merge(table, on=["WAFER_ID", "STAGE"], validate="one_to_one")
    assert len(data) == len(feats) == 2401, "대상 2,401행이 아니다"
    assert data["MRR"].notna().all(), "MRR 결측이 있다"
    return data, fam


def select_group(x, fam):
    """한 그룹의 전체 후보 표(순위·선택 여부 포함)를 돌려준다."""
    cands = fam[fam["sr_candidate"] == 1]
    rows = []
    for _, c in cands.iterrows():
        v = x[c["feature"]]
        rows.append({
            "feature": c["feature"], "family": c["family"],
            "guaranteed_slot": c["guaranteed_slot"],
            "nan_frac": v.isna().mean(), "abs_median": v.abs().median(),
        })
    t = pd.DataFrame(rows)
    t["excluded"] = t["feature"].map(EXCLUDED_FEATURES).fillna("")
    t.loc[t["nan_frac"] > MAX_NAN_FRAC, "excluded"] = "nan_over_5pct"
    t.loc[t["abs_median"] < MIN_ABS_MEDIAN, "excluded"] = "abs_median_under_1e-9"
    pool = t[t["excluded"] == ""].copy()
    assert len(pool) == EXPECTED_POOL, f"후보 풀이 {len(pool)}개다"

    pool["spearman"] = [x[f].corr(x["MRR"], method="spearman") for f in pool["feature"]]
    assert pool["spearman"].notna().all(), "상관을 계산할 수 없는 후보가 있다"
    pool["abs_spearman"] = pool["spearman"].abs()
    pool = pool.sort_values(["abs_spearman", "feature"], ascending=[False, True])

    picked, slot, skipped = [], {}, {}
    for family, sub in pool[pool["guaranteed_slot"] == 1].groupby("family", sort=False):
        f = sub.iloc[0]["feature"]
        picked.append(f)
        slot[f] = f"guaranteed:{family}"
    assert len(picked) == 7, f"보장 계열이 {len(picked)}개다"

    for f in pool["feature"]:
        if len(picked) == K:
            break
        if f in picked:
            continue
        pair = {p: abs(x[f].corr(x[p], method="spearman")) for p in picked}
        worst = max(pair, key=pair.get)
        if pair[worst] > MAX_PAIR_CORR:
            skipped[f] = f"corr_{pair[worst]:.3f}_with_{worst}"
            continue
        picked.append(f)
        slot[f] = "fill"
    assert len(picked) == K, f"{len(picked)}개만 뽑혔다"

    pool["rank"] = range(1, len(pool) + 1)
    pool["selected"] = pool["feature"].isin(picked).astype(int)
    pool["slot"] = pool["feature"].map(slot).fillna("")
    pool["skip_reason"] = pool["feature"].map(skipped).fillna("")
    out = pd.concat([pool, t[t["excluded"] != ""]], ignore_index=True)
    return out, picked


def main():
    stamp = date.today().isoformat()
    out_csv = OUT_DIR / f"{stamp}_박종화_sr_candidates.csv"
    out_yaml = out_csv.with_suffix(".yaml")
    if out_csv.exists() or out_yaml.exists():
        sys.exit(f"이미 있어서 멈춘다: {out_csv}")

    data, fam = load()
    tables, lines = [], []
    for g in GROUPS:
        x = data[data["group"] == g]
        assert len(x) == EXPECTED_N[g], f"{g} 행 수가 {len(x)}다"
        assert x["allzero"].sum() == 0, f"{g}에 전체-0 센서 웨이퍼가 있다"
        t, picked = select_group(x, fam)
        t.insert(0, "group", g)
        tables.append(t)
        n_complete = int(x[picked].notna().all(axis=1).sum())
        lines.append({"group": g, "n_wafers": len(x), "n_complete_on_selected": n_complete,
                      "n_dropped": len(x) - n_complete, "selected": picked})

        print(f"\n== {g}: 사용 {n_complete}장 / 제외 {len(x) - n_complete}장")
        cols = ["feature", "family", "spearman", "rank", "nan_frac", "slot"]
        sel = t[t["selected"] == 1].sort_values("rank")
        print(sel[cols].round(4).to_string(index=False))
        skips = t[t["skip_reason"].fillna("") != ""]
        print("건너뜀:", skips[["feature", "skip_reason"]].values.tolist())
        print("제외:", t[t["excluded"] != ""][["feature", "excluded"]].values.tolist())

    result = pd.concat(tables, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    yaml = [
        f"date: {stamp}",
        "author: 박종화",
        "script: src/sr_candidates.py",
        "input: data/processed/features_v1.csv + feature_families_v1.csv + wafer_table.csv(MRR)",
        "rule: preregistration.md 4.2, 4.8 1~3·11번",
        f"excluded_features: {list(EXCLUDED_FEATURES)}",
        f"k: {K}",
        f"max_nan_frac: {MAX_NAN_FRAC}",
        f"min_abs_median: {MIN_ABS_MEDIAN}",
        f"max_pair_corr: {MAX_PAIR_CORR}",
        "corr: spearman (train+val, 이상치 제외, 그룹별 1회)",
        "groups:",
    ]
    for r in lines:
        yaml += [f"  {r['group']}:", f"    n_wafers: {r['n_wafers']}",
                 f"    n_complete_on_selected: {r['n_complete_on_selected']}",
                 f"    n_dropped: {r['n_dropped']}", "    selected:"]
        yaml += [f"      - {f}" for f in r["selected"]]
    out_yaml.write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"\n저장: {out_csv.name}, {out_yaml.name}")


if __name__ == "__main__":
    main()

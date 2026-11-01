"""train+val에서 같은 WAFER_ID가 두 번 이상 나오는 경우 진단 (진단 CLAUDE.md 14.4, 라벨 불필요).

웨이퍼 키는 (WAFER_ID, STAGE)인데 WAFER_ID만으로는 중복된다(CLAUDE.md 5.2). 중복 ID마다 시각 순으로 이웃한 두 줄을 한 쌍으로 보고:
1. 그룹 조합 (먼저 → 나중, 순서 무관)
2. 간격 = 둘째 줄 t_start − 첫 줄 t_end (초). 음수면 시간이 겹침. 어느 STAGE가 먼저인지
3. 소모품 카운터 이어짐: 첫 줄 종료값(start + delta)과 둘째 줄 시작값의 차이 (6개).
   카운터는 장비 전체 공유값이라(PR #14) 같은 장비에서 시간이 흐른 만큼 늘어난다. 리셋이 끼면 음수가 될 수 있다.
4. 두 줄 사이(첫 줄 종료 ~ 둘째 줄 시작)에 시작한 다른 train+val 웨이퍼 수

입력: data/processed/wafer_table.csv의 키·시각·카운터 열만 (MRR 열은 읽지 않음). train·val 행만 쓰고 test 행·data/locked/는 쓰지 않는다.
이상치 4장(is_outlier)도 행으로는 포함하고 표시만 한다.
실행: python src/diagnostics/wafer_id_dup.py
출력: results/diagnostics/2026-11-01_김혜연_wafer_id_dup.{csv,yaml} (있으면 멈춤)
"""
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a123_transfer as at  # noqa: E402  (git_state·environment·write_yaml·sha만 사용)

DATE = "2026-11-01"
AUTHOR = "김혜연"
OUT = at.ROOT / "results" / "diagnostics" / f"{DATE}_{AUTHOR}_wafer_id_dup.csv"
P_TABLE = at.P_TABLE
COUNTERS = ["BACKING_FILM", "DRESSER", "POLISH_TABLE", "DRESSER_TABLE", "MEMBRANE", "PSHEET"]
SHORT_GAP_S = 600     # "수 분 이내" 기준 (10분)


def main():
    if OUT.exists() or OUT.with_suffix(".yaml").exists():
        sys.exit(f"이미 있음 (덮어쓰기 금지): {OUT.name}")
    t0 = time.time()
    started = dt.datetime.now()
    git = at.git_state()
    cols = ["WAFER_ID", "STAGE", "split", "group", "t_start", "t_end", "n_rows", "is_outlier"]
    cols += [f"{p}_{c}_w" for c in COUNTERS for p in ("start", "delta")]
    w = pd.read_csv(P_TABLE, usecols=cols)
    assert "MRR" not in w.columns
    w = w[w["split"].isin(["train", "val"])].reset_index(drop=True)
    assert len(w) == 2405, len(w)          # train 1,981 + val 424 (이상치 4장 포함)
    assert not w.duplicated(["WAFER_ID", "STAGE"]).any()
    starts = np.sort(w["t_start"].to_numpy())

    rows = []
    for wid, g in w.groupby("WAFER_ID"):
        if len(g) < 2:
            continue
        g = g.sort_values("t_start")
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            gap = b["t_start"] - a["t_end"]
            lo, hi = sorted([a["t_end"], b["t_start"]])
            n_between = int(np.searchsorted(starts, hi, "left") - np.searchsorted(starts, lo, "right"))
            rec = {"WAFER_ID": wid, "n_records": len(g), "pair_index": i,
                   "first_stage": a["STAGE"], "second_stage": b["STAGE"],
                   "first_group": a["group"], "second_group": b["group"],
                   "combo_ordered": f"{a['group']} → {b['group']}",
                   "combo": "+".join(sorted([a["group"], b["group"]])),
                   "first_split": a["split"], "second_split": b["split"],
                   "first_t_start": a["t_start"], "first_t_end": a["t_end"],
                   "second_t_start": b["t_start"], "second_t_end": b["t_end"],
                   "gap_s": gap, "overlap": int(gap < 0), "n_wafers_between": n_between,
                   "any_outlier": int(a["is_outlier"] or b["is_outlier"])}
            for c in COUNTERS:
                end_a = a[f"start_{c}_w"] + a[f"delta_{c}_w"]
                rec[f"cont_{c}"] = b[f"start_{c}_w"] - end_a   # 둘째 시작 − 첫째 종료
            rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(OUT, index=False, encoding="utf-8")

    # 요약
    ids = w.groupby("WAFER_ID").size()
    gap = D["gap_s"]

    def q(s):
        return {"median": float(s.median()), "min": float(s.min()), "max": float(s.max()),
                "p10": float(s.quantile(0.1)), "p90": float(s.quantile(0.9))}
    by_combo = {}
    for k, g in D.groupby("combo_ordered"):
        by_combo[k] = {"n": len(g), "gap_s": q(g["gap_s"]),
                       "gap_le_600s": int((g["gap_s"].between(0, SHORT_GAP_S)).sum()),
                       "overlap": int(g["overlap"].sum()),
                       "n_wafers_between_median": float(g["n_wafers_between"].median()),
                       "stage_order": g["first_stage"].str.cat(g["second_stage"], "→").value_counts().to_dict()}
    cont = {c: {"median": float(D[f"cont_{c}"].median()), "min": float(D[f"cont_{c}"].min()),
                "max": float(D[f"cont_{c}"].max()), "n_negative": int((D[f"cont_{c}"] < 0).sum())}
            for c in COUNTERS}
    # 리셋이 없는 DRESSER_TABLE: 두 줄 사이 증가량이 장비 전체의 시간당 증가율 × 간격과 맞는지
    rate = np.polyfit(w["t_start"], w["start_DRESSER_TABLE_w"], 1)[0]
    ratio = D["cont_DRESSER_TABLE"] / (rate * D["gap_s"])
    rho, p = spearmanr(D["gap_s"], D["cont_DRESSER_TABLE"])
    dt_check = {"machine_rate_per_hour": float(rate * 3600), "spearman_gap_vs_increase": float(rho),
                "spearman_p": float(p), "obs_over_expected_median": float(ratio.median()),
                "obs_over_expected_p25_p75": [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))]}
    info = {
        "experiment": "wafer_id_dup (train+val WAFER_ID 중복 진단, 14.4)",
        "author": AUTHOR,
        "run_started": f"{started:%Y-%m-%d %H:%M:%S}", "runtime_sec": round(time.time() - t0, 1),
        "git": git, "environment": at.environment(),
        "input": f"{P_TABLE.name} (train·val 행, 키·시각·카운터 열만, MRR 미사용)",
        "inputs_sha256_16": {P_TABLE.name: at.sha(P_TABLE)},
        "n_rows_train_val": len(w), "n_unique_wafer_id": int(ids.size),
        "n_dup_ids": int((ids >= 2).sum()), "records_per_dup_id": ids[ids >= 2].value_counts().to_dict(),
        "n_pairs": len(D),
        "combo_counts": D["combo"].value_counts().to_dict(),
        "combo_ordered_counts": D["combo_ordered"].value_counts().to_dict(),
        "split_pairs": (D["first_split"] + "→" + D["second_split"]).value_counts().to_dict(),
        "gap_s_all": q(gap),
        "gap_def": "둘째 줄 t_start − 첫 줄 t_end (초), 음수 = 겹침",
        "short_gap_rule_s": SHORT_GAP_S,
        "by_combo_ordered": by_combo,
        "counter_continuity": cont,
        "counter_def": "cont_{U} = 둘째 줄 start_{U}_w − (첫 줄 start_{U}_w + delta_{U}_w)",
        "gap_hours": {k: round(float(v), 2) for k, v in (gap / 3600).quantile([0, .1, .25, .5, .75, .9, 1]).items()},
        "dresser_table_vs_gap": dt_check,
        "output": OUT.name,
    }
    at.write_yaml(OUT.with_suffix(".yaml"), info)
    print(f"저장: {OUT.name} + .yaml ({len(D)}쌍, {time.time() - t0:.1f}초)", flush=True)


if __name__ == "__main__":
    main()

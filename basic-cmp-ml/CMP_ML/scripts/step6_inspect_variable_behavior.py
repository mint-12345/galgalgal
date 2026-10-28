"""Step 6: 공정 변수가 wafer+Stage 안에서 어떻게 변하는지 변수별로 확인 (읽기 전용).

- training 원데이터만 읽는다. test / validation 은 열지 않는다. raw 파일은 수정하지 않는다.
- 분석에서만 제외: (1) 같은 파일에서 바로 윗줄과 25개 컬럼이 모두 같은 1,819행
                   (2) training MRR 이 1,000 을 넘는 wafer+Stage 4개
- WAFER_ID, STAGE, CHAMBER, TIMESTAMP 는 구분 정보로만 쓰고 평가하지 않는다.
- feature 결정, 한 줄 변환, 모델 학습은 하지 않는다.
출력: 화면 출력, reports/step6_variable_behavior.csv
실행: python -B scripts/step6_inspect_variable_behavior.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 100)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET"
TRAIN_DIR = DS / "CMP-data" / "training"
MRR_FILE = DS / "CMP-training-removalrate.csv"
OUT_CSV = ROOT / "reports" / "step6_variable_behavior.csv"
KEY = ["WAFER_ID", "STAGE"]
ID_COLS = ["WAFER_ID", "STAGE", "CHAMBER", "TIMESTAMP"]
MRR_LIMIT = 1000

# 표현용 구분 기준. 실제 값에서 변수들이 크게 벌어지는 틈 안에 둔다 (문서에 근거 수치 기록).
CV_ALMOST_CONST = 0.10     # wafer 안 변동계수 중앙값: 관측값은 0.08 이하 또는 0.17 이상
WITHIN_NEGLIGIBLE = 0.01   # 전체 변동 중 wafer 안 변동 비율: 관측값은 0.001 이하 또는 0.33 이상
CHAMBER_BIG, CHAMBER_SOME = 0.50, 0.20  # wafer 안 변동 중 Chamber 가 설명하는 비율
WAFER_SAME, WAFER_SOME = 0.10, 0.30     # wafer 평균의 퍼짐 / 전체 값 범위: 관측값은 0.05 또는 0.15 이상
SHARE_ALL = 0.99


def load():
    frames = []
    for f in sorted(TRAIN_DIR.glob("CMP-training-*.csv")):
        df = pd.read_csv(f, dtype={"WAFER_ID": "int64", "STAGE": "string"})
        if df.empty:
            continue
        df["FILE"] = f.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    cols = [c for c in raw.columns if c != "FILE"]
    same_prev = raw[cols].eq(raw[cols].shift()).all(axis=1) & raw["FILE"].eq(raw["FILE"].shift())
    data = raw[~same_prev]
    mrr = pd.read_csv(MRR_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    big = mrr.loc[mrr["AVG_REMOVAL_RATE"] > MRR_LIMIT, KEY]
    drop = data.set_index(KEY).index.isin(big.set_index(KEY).index)
    data = data[~drop].sort_values(KEY + ["TIMESTAMP"], kind="stable")
    print(f"원본 행 {len(raw):,} → 연속 완전 중복 {int(same_prev.sum()):,}행 제외 → "
          f"MRR>{MRR_LIMIT:,} 묶음 {len(big)}개({int(drop.sum()):,}행) 제외 → 분석 행 {len(data):,}, "
          f"wafer+Stage {data.groupby(KEY).ngroups:,}개")
    print("제외한 MRR>1,000 묶음:", list(map(tuple, big.itertuples(index=False))))
    assert int(same_prev.sum()) == 1819 and len(big) == 4
    return data, cols


def measure(data, v):
    x = data[KEY + ["CHAMBER", v]]
    g = x.groupby(KEY)[v]
    mean, std, n_unique = g.mean(), g.std(ddof=0), g.nunique()

    # 변동 분해 (wafer+Stage 마다 같은 가중치):
    #   전체 = wafer 평균 사이 변동 + wafer 안 Chamber 평균 사이 변동 + 같은 Chamber 안 시간 변동
    between = mean.var(ddof=0)
    within = g.var(ddof=0).mean()
    gc = x.groupby(KEY + ["CHAMBER"])[v].agg(["mean", "size"])
    gc = gc.join(mean.rename("wafer_mean"), on=KEY)
    gc["w"] = gc["size"] / gc.groupby(level=[0, 1])["size"].transform("sum")
    chamber = (gc["w"] * (gc["mean"] - gc["wafer_mean"]) ** 2).groupby(level=[0, 1]).sum().mean()
    time = within - chamber
    total = between + within

    cv = np.where(mean.abs() > 0, std / mean.abs(), np.where(std == 0, 0.0, np.nan))
    nondecreasing = g.apply(lambda s: bool((s.diff().dropna() >= 0).all()))
    const_in_chamber = x.groupby(KEY + ["CHAMBER"])[v].nunique().eq(1).groupby(level=[0, 1]).all()
    row_range = x[v].quantile(0.99) - x[v].quantile(0.01)
    spread = (mean.quantile(0.9) - mean.quantile(0.1)) / row_range if row_range > 0 else 0.0

    return dict(
        variable=v,
        n_unique_all=int(x[v].nunique()),
        top_value_row_share=float(x[v].value_counts(normalize=True).iloc[0]),
        zero_row_share=float((x[v] == 0).mean()),
        const_in_wafer_share=float(n_unique.eq(1).mean()),
        const_in_each_chamber_share=float(const_in_chamber.mean()),
        within_wafer_cv_median=float(np.nanmedian(cv)),
        never_decreasing_share=float(nondecreasing.mean()),
        share_between_wafers=between / total if total > 0 else np.nan,
        share_between_chambers=chamber / total if total > 0 else np.nan,
        share_time_in_chamber=time / total if total > 0 else np.nan,
        within_share_of_total=within / total if total > 0 else np.nan,
        chamber_share_of_within=chamber / within if within > 0 else np.nan,
        wafer_mean_spread=float(spread),
    )


def label(r):
    if r["n_unique_all"] == 1:
        return "완전 고정", "해당 없음", "모든 wafer 같은 값", "거의 고정 (모든 행이 같은 값)"

    if r["const_in_each_chamber_share"] >= SHARE_ALL and r["const_in_wafer_share"] < SHARE_ALL:
        q1 = "Chamber 안에서는 고정"
    elif r["within_wafer_cv_median"] < CV_ALMOST_CONST:
        q1 = "거의 고정"
        if r["never_decreasing_share"] >= SHARE_ALL and r["const_in_wafer_share"] < 0.5:
            q1 += " (시간순으로 조금씩 증가)"
    else:
        q1 = "시간에 따라 크게 변화"

    if r["within_share_of_total"] < WITHIN_NEGLIGIBLE:
        q2 = "해당 없음 (wafer 안 변화가 거의 없음)"
    elif r["chamber_share_of_within"] >= CHAMBER_BIG:
        q2 = "Chamber에 따라 차이가 큼"
    elif r["chamber_share_of_within"] >= CHAMBER_SOME:
        q2 = "Chamber에 따라 일부 차이"
    else:
        q2 = "Chamber 영향 작음"

    if r["wafer_mean_spread"] < WAFER_SAME:
        q3 = "여러 wafer에서 거의 같음"
    elif r["wafer_mean_spread"] < WAFER_SOME:
        q3 = "wafer마다 조금 다름"
    else:
        q3 = "wafer마다 많이 다름"

    if q1.startswith("Chamber 안"):
        summary = "Chamber에 따라 차이가 큼 (Chamber 안에서는 고정)"
    elif q1.startswith("거의 고정"):
        if q2 == "Chamber에 따라 차이가 큼":
            summary = "대체로 고정, Chamber에 따라 차이 있음"
        elif q3 == "여러 wafer에서 거의 같음":
            summary = "거의 고정"
        else:
            summary = "wafer 안에서는 거의 고정, wafer마다 다름"
    else:
        summary = "시간에 따라 변화" + (" + Chamber에 따라 차이가 큼" if q2 == "Chamber에 따라 차이가 큼" else "")
    return q1, q2, q3, summary


def supporting_facts(data):
    print("\n" + "=" * 100 + "\n[참고 사실]\n" + "=" * 100)
    print("(a) MACHINE_DATA x CHAMBER 행 수:")
    print(pd.crosstab(data["MACHINE_DATA"], data["CHAMBER"]).to_string())

    u = ["USAGE_OF_BACKING_FILM", "USAGE_OF_MEMBRANE", "USAGE_OF_PRESSURIZED_SHEET"]
    print("\n(b) usage 변수 사이 상관계수:")
    print(data[[c for c in data.columns if c.startswith("USAGE")]].corr().round(4).to_string())
    for c in u[1:]:
        slope, intercept = np.polyfit(data[u[0]], data[c], 1)
        resid = (data[c] - (slope * data[u[0]] + intercept)).abs().max()
        print(f"    {c} ≈ {slope:.6f} × {u[0]} + {intercept:.6f} "
              f"(최대 오차 {resid:.4f}, 이 변수의 전체 범위 {data[c].max() - data[c].min():.2f})")

    g = data.groupby(KEY)
    zb = g["SLURRY_FLOW_LINE_B"].apply(lambda s: (s == 0).mean())
    print("\n(c) SLURRY_FLOW_LINE_B: 0 이 아닌 값 상위", data.loc[data["SLURRY_FLOW_LINE_B"] != 0, "SLURRY_FLOW_LINE_B"]
          .round(3).value_counts().head(3).to_dict())
    print("    wafer 안에서 0 인 행 비율의 분위수:", zb.quantile([0, .25, .5, .75, .9, 1]).round(3).to_dict())
    print("\n(d) HEAD_ROTATION wafer 별 중앙값 분포:", g["HEAD_ROTATION"].median().value_counts().to_dict())

    pattern = g["CHAMBER"].agg(lambda c: "1-2-3" if set(c) <= {1, 2, 3} else "4-5-6")
    varying = ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE",
               "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
               "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION", "DRESSING_WATER_STATUS"]
    print("\n(e) Chamber 패턴별 wafer 평균의 중앙값:")
    print(g[varying].mean().join(pattern.rename("pattern")).groupby("pattern").median().round(2).to_string())
    print("\n(f) Chamber 별로 본 'wafer+Stage x Chamber 평균'의 중앙값:")
    print(data.groupby(KEY + ["CHAMBER"])[varying].mean().groupby(level="CHAMBER").median().round(2).to_string())


def main():
    data, cols = load()
    variables = [c for c in cols if c not in ID_COLS]
    print(f"평가한 공정 변수 {len(variables)}개: {variables}")

    table = pd.DataFrame([measure(data, v) for v in variables]).set_index("variable")
    labels = table.apply(label, axis=1, result_type="expand")
    labels.columns = ["Q1_wafer_안_시간변화", "Q2_Chamber_차이", "Q3_여러_wafer_비교", "요약"]
    table = table.join(labels)

    print("\n" + "=" * 100 + "\n[측정값]\n" + "=" * 100)
    print(table.drop(columns=labels.columns).round(3).to_string())
    print("\n" + "=" * 100 + "\n[변수별 정리]\n" + "=" * 100)
    print(labels.to_string())

    OUT_CSV.parent.mkdir(exist_ok=True)
    table.round(4).to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV.relative_to(ROOT).as_posix()}")
    supporting_facts(data)


if __name__ == "__main__":
    main()

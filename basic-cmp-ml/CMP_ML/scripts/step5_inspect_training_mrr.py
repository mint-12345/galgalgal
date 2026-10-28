"""Step 5: training MRR 정답 값 확인 (읽기 전용).

- training MRR 파일(CMP-training-removalrate.csv)만 읽는다. 공정 데이터, test, validation 은 열지 않는다.
- 원본 파일을 수정하지 않고, feature 생성이나 모델 학습도 하지 않는다.
실행: python scripts/step5_inspect_training_mrr.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MRR_FILE = ROOT / "data" / "raw" / "PHM2016" / "2016 PHM DATA CHALLENGE CMP DATA SET" / "CMP-training-removalrate.csv"
LIMIT = 1000


def main():
    # 1) 빈 값 확인: 문자열 그대로 읽어 빈 칸, 공백, 숫자로 바꿀 수 없는 값을 모두 센다
    text = pd.read_csv(MRR_FILE, dtype=str, keep_default_na=False)
    print(f"행 수: {len(text):,} / 컬럼: {list(text.columns)}")
    for col in text.columns:
        blank = text[col].str.strip().eq("").sum()
        print(f"  {col}: 빈 칸 {blank}개")
    numeric = pd.to_numeric(text["AVG_REMOVAL_RATE"], errors="coerce")
    bad = text.loc[numeric.isna() & text["AVG_REMOVAL_RATE"].str.strip().ne(""), "AVG_REMOVAL_RATE"]
    print(f"  AVG_REMOVAL_RATE 숫자로 읽을 수 없는 값: {len(bad)}개 {bad.tolist()}")
    print(f"  pandas 기본 결측 처리 기준 NaN 개수: {pd.read_csv(MRR_FILE)['AVG_REMOVAL_RATE'].isna().sum()}")

    mrr = pd.read_csv(MRR_FILE, dtype={"WAFER_ID": "int64", "STAGE": "string"})
    y = mrr["AVG_REMOVAL_RATE"]

    # 2) 전체 범위
    print("\n전체 MRR 요약:")
    print(y.describe().round(3).to_string())
    print("분위수:", y.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3).to_dict())
    print("Stage 별 요약:")
    print(mrr.groupby("STAGE")["AVG_REMOVAL_RATE"].describe().round(3).to_string())

    # 3) 1,000 초과
    over = mrr[y > LIMIT].sort_values("AVG_REMOVAL_RATE", ascending=False)
    print(f"\nMRR > {LIMIT:,} 인 wafer+Stage: {len(over)}개")
    print(over.to_string(index=False))
    rest = y[y <= LIMIT]
    print(f"\nMRR <= {LIMIT:,} 인 {len(rest):,}개의 범위: {rest.min():.5f} ~ {rest.max():.5f}")
    print("그중 가장 큰 값 5개:", sorted(rest.nlargest(5).round(5).tolist(), reverse=True))


if __name__ == "__main__":
    main()

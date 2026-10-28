# Step 8. 모델 비교용 5-fold 평가 구성

- 입력: `data/processed/step7_wafer_stage_table_v1.csv` (training, 1,977행)
- 출력: `data/processed/step8_cv_folds_v1.csv` (열: WAFER_ID, STAGE, CHAMBER_PATH, FOLD)
- 코드: `scripts/step8_make_cv_folds.py`
- test / validation 은 사용하지 않았고, 모델은 학습하지 않았습니다.
- 이후 Model 1, 2, 3 비교는 모두 이 파일의 FOLD 를 그대로 사용합니다. Step 7 표와는 WAFER_ID + STAGE 로 연결합니다.

## 나눈 방법

- scikit-learn `StratifiedGroupKFold` (5개 fold)를 사용했습니다.
- group 은 WAFER_ID 이므로 같은 wafer 의 Stage A 와 B 는 항상 같은 fold 에 들어갑니다.
- 층화 기준은 STAGE 와 CHAMBER_PATH 를 합친 값입니다. 예: `A|4-5-6`
- random_state 0~99 를 모두 만들어 보고, fold 구성 비율이 전체와 가장 가까운 random_state=11 을 골랐습니다. 같은 코드를 다시 실행해도 같은 파일이 만들어지는 것을 확인했습니다.

## 결과

| FOLD | 행 | 고유 wafer | A 1-2-3 | A 4-5-6 | A other | B 4-5-6 | B other |
|---|---|---|---|---|---|---|---|
| 1 | 396 | 340 | 73 | 160 | 0 | 163 | 0 |
| 2 | 396 | 340 | 73 | 160 | 0 | 163 | 0 |
| 3 | 395 | 339 | 73 | 159 | 0 | 162 | 1 |
| 4 | 395 | 338 | 72 | 159 | 1 | 162 | 1 |
| 5 | 395 | 338 | 73 | 159 | 0 | 162 | 1 |
| 전체 | 1,977 | 1,695 | 364 | 797 | 1 | 812 | 3 |

- 모든 fold 에서 Stage A 비율은 0.587~0.588 이고, 전체는 0.588 입니다.
- 모든 fold 에서 CHAMBER_PATH 1-2-3 비율은 0.182~0.185 이고, 전체는 0.184 입니다.
- `other` 는 전체 4행뿐이라 fold 1, 2 에는 없습니다.

## wafer 분리 확인

- 두 fold 이상에 걸친 WAFER_ID 는 0개입니다.
- Stage A/B 가 모두 있는 wafer 282개는 모두 A 와 B 가 같은 fold 에 있습니다.
- 각 fold 를 평가용으로 뒀을 때 나머지 4개 fold 와 겹치는 WAFER_ID 는 모든 fold 에서 0개입니다.

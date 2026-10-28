# Step 9. Model 1 기준 모델: 1차 다중선형회귀

- 입력 표: `data/processed/step7_wafer_stage_table_v1.csv` (training, 1,977행)
- fold: `data/processed/step8_cv_folds_v1.csv` 를 그대로 사용
- 코드: `scripts/step9_model1_mlr_baseline.py`
- 결과: `reports/step9_model1_mlr/`
- test / validation 은 사용하지 않았습니다. 2차식, 트리 모델, feature 변경, 성능 개선은 하지 않았습니다.

## 모델 구성

- 입력: Step 7 표의 feature 32개. WAFER_ID 와 MRR 정답은 뺐습니다.
- STAGE, CHAMBER_PATH: one-hot 으로 바꿨습니다. 절편과 겹치지 않도록 첫 범주(STAGE A, CHAMBER_PATH 1-2-3)를 기준으로 두어 `STAGE_B`, `CHAMBER_PATH_4-5-6`, `CHAMBER_PATH_other` 3개 열이 됩니다.
- 숫자 30개: 평균 0, 표준편차 1 로 표준화했습니다.
- 모델 입력은 33개 열이고, 모델은 scikit-learn `LinearRegression` 입니다.
- 평가용 정보 차단: 변환과 회귀를 하나의 Pipeline 으로 묶어, 매 반복마다 학습용 4개 fold 로만 fit 했습니다. 평가용 fold 에는 transform 과 predict 만 적용했습니다. 평가용에만 있는 범주는 없었습니다.

## 5-fold 결과

| FOLD | 학습 행 | 평가 행 | R² | MAE | RMSE |
|---|---|---|---|---|---|
| 1 | 1,581 | 396 | 0.9729 | 3.8249 | 4.8747 |
| 2 | 1,581 | 396 | 0.9654 | 4.4597 | 5.5915 |
| 3 | 1,582 | 395 | 0.9656 | 4.4586 | 5.6503 |
| 4 | 1,582 | 395 | 0.9687 | 4.1663 | 5.3097 |
| 5 | 1,582 | 395 | 0.9711 | 4.0982 | 5.1040 |

| 지표 | 평균 | 표준편차 | 최소 | 최대 |
|---|---|---|---|---|
| R² | 0.9687 | 0.0033 | 0.9654 | 0.9729 |
| MAE | 4.2015 | 0.2676 | 3.8249 | 4.4597 |
| RMSE | 5.3060 | 0.3267 | 4.8747 | 5.6503 |

- 표준편차는 5개 fold 값의 표본 표준편차(ddof=1)입니다.
- 5개 fold 의 예측을 모두 모아 한 번에 계산하면 R² 0.9687, MAE 4.2015, RMSE 5.3140 입니다.

## 해석할 때 주의할 점

전체 R² 0.97 은 대부분 CHAMBER_PATH 1-2-3 과 4-5-6 의 MRR 수준 차이에서 나옵니다. 두 그룹의 MRR 평균은 약 151 과 약 77 로 크게 다릅니다.

| CHAMBER_PATH | 행 | R² | MAE | RMSE | 실제 MRR 평균 | 실제 MRR 표준편차 |
|---|---|---|---|---|---|---|
| 1-2-3 | 364 | -0.621 | 3.836 | 4.959 | 151.24 | 3.90 |
| 4-5-6 | 1,609 | 0.618 | 4.284 | 5.393 | 76.56 | 8.73 |
| other | 4 | 0.151 | 4.304 | 4.650 | 78.18 | 5.83 |

- 1-2-3 그룹 안에서는 R² 가 음수입니다. 이 그룹 안의 차이는 그룹 평균 하나로 예측하는 것보다 못하게 맞췄다는 뜻입니다.
- 4-5-6 그룹 안에서는 R² 0.62 입니다.
- 이후 모델을 비교할 때는 전체 R² 와 함께 이 그룹별 지표도 같이 보는 것이 좋습니다.
- STAGE x CHAMBER_PATH 별 전체 표: `reports/step9_model1_mlr/group_metrics_reference.csv`

## 저장 파일

| 파일 | 내용 |
|---|---|
| `fold_metrics.csv` | fold 별 R², MAE, RMSE, 학습/평가 행 수, one-hot 열 |
| `summary.csv` | 5-fold 평균, 표준편차, 최소, 최대, 모아서 계산한 값 |
| `oof_predictions.csv` | wafer + Stage 별 실제 MRR, 예측 MRR, 오차(예측-실제), FOLD. Step 7 표와 같은 행 순서 |
| `group_metrics_reference.csv` | STAGE x CHAMBER_PATH 별 참고 지표 |

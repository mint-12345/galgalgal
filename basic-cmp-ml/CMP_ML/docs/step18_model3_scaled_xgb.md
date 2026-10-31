# Step 18. Model 3: k_target 표준화로 XGBoost 상수 예측 문제 수정

- 대상: Step 17 과 같은 Chamber path 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold
- training 만 사용했습니다. test / validation 은 사용하지 않았고, raw 파일은 수정하지 않았습니다.
- Hybrid 구조 그대로입니다: MRR = k(x) × PV_proxy. PV_proxy 와 k_target 은 Step 16 값을 그대로 썼고, P/V 정의도 바꾸지 않았습니다.
- feature 구성 A, B, C 는 Step 17 과 같습니다 (13개 / 25개 / 31개). feature 를 더하거나 빼지 않았습니다.
- XGBoost 설정은 Step 10 기본값 그대로이고 튜닝하지 않았습니다.
- 코드: `scripts/step18_model3_scaled_xgb.py`, 결과: `reports/step18_model3_scaled_xgb/`

## 1. 이번에 바꾼 것

fold 마다 **학습 sample 만으로** k_target 의 평균과 표준편차를 구해 표준화한 뒤 학습했습니다.

- k_scaled = (k_target − training_mean) / training_std
- 예측은 k_pred = k_scaled_pred × training_std + training_mean 으로 되돌렸습니다.
- MRR_pred = k_pred × PV_proxy
- **평가 fold 의 k_target 평균이나 표준편차는 쓰지 않았습니다.** k_pred 에 clipping 이나 보정도 하지 않았습니다.

**fold 별 표준화 값 (C_full 기준, 다른 구성도 같은 값)**

| fold | training_mean | training_std | 평가 sample 수 | 서로 다른 k_pred 수 |
|---|---|---|---|---|
| 1 | 0.000283339 | 0.000032002 | 323 | 323 |
| 2 | 0.000284016 | 0.000031838 | 323 | 323 |
| 3 | 0.000284080 | 0.000031732 | 321 | 321 |
| 4 | 0.000283940 | 0.000032325 | 321 | 321 |
| 5 | 0.000283630 | 0.000031964 | 321 | 321 |

## 2. k 예측 성능 (k_target vs k_pred, 5-fold 평균 ± 표준편차)

| 조합 | R² | MAE | RMSE |
|---|---|---|---|
| A_state_only\|XGBoost_scaled | 0.689978 ± 0.022338 | 0.000013 ± 0.000000 | 0.000018 ± 0.000001 |
| B_physics_separated\|XGBoost_scaled | 0.713301 ± 0.014537 | 0.000013 ± 0.000001 | 0.000017 ± 0.000001 |
| C_full\|XGBoost_scaled | 0.769653 ± 0.017666 | 0.000012 ± 0.000001 | 0.000015 ± 0.000001 |

Step 17 의 원래 크기 XGBoost 는 세 구성 모두 k R² 가 -0.001873 이었습니다.

## 3. 최종 MRR 예측 성능

| 조합 | R² | MAE | RMSE |
|---|---|---|---|
| A_state_only\|XGBoost_scaled | 0.6895 ± 0.0218 | 3.6011 ± 0.1364 | 4.8496 ± 0.2188 |
| B_physics_separated\|XGBoost_scaled | 0.7200 ± 0.0238 | 3.4262 ± 0.1636 | 4.6052 ± 0.2550 |
| C_full\|XGBoost_scaled | 0.7732 ± 0.0217 | 3.1141 ± 0.1548 | 4.1419 ± 0.2063 |

**표준화 전후 비교 (MRR RMSE 평균)**

| feature 구성 | Step 17 원래 크기 | Step 18 표준화 | 차이 |
|---|---|---|---|
| A_state_only | 8.6539 | 4.8496 | -3.8043 |
| B_physics_separated | 8.6539 | 4.6052 | -4.0487 |
| C_full | 8.6539 | 4.1419 | -4.5120 |

## 4. 전체 비교 (같은 1,609개, 같은 5-fold, MRR RMSE 낮은 순)

| 출처 | 모델 | 조합 | R² | MAE | RMSE |
|---|---|---|---|---|---|
| Step 16 | Model 1 | Model1_XGB | 0.8296 ± 0.0248 | 2.7451 ± 0.1744 | 3.5826 ± 0.1942 |
| Step 16 | Model 1 | Model1_RF | 0.8246 ± 0.0220 | 2.7846 ± 0.1642 | 3.6401 ± 0.2167 |
| Step 18 | Model 3 (표준화 XGB) | C_full\|XGBoost_scaled | 0.7732 ± 0.0217 | 3.1141 ± 0.1548 | 4.1419 ± 0.2063 |
| Step 17 | Model 3 (Random Forest) | C_full\|RandomForest | 0.7468 ± 0.0247 | 3.2988 ± 0.1642 | 4.3770 ± 0.2277 |
| Step 18 | Model 3 (표준화 XGB) | B_physics_separated\|XGBoost_scaled | 0.7200 ± 0.0238 | 3.4262 ± 0.1636 | 4.6052 ± 0.2550 |
| Step 18 | Model 3 (표준화 XGB) | A_state_only\|XGBoost_scaled | 0.6895 ± 0.0218 | 3.6011 ± 0.1364 | 4.8496 ± 0.2188 |
| Step 16 | Model 2 | Model2_fixed_k | 0.0113 ± 0.0692 | 6.9583 ± 0.1837 | 8.6511 ± 0.3058 |
| Step 17 | Model 3 (원래 크기 XGB) | A_state_only\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |
| Step 17 | Model 3 (원래 크기 XGB) | C_full\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |
| Step 17 | Model 3 (원래 크기 XGB) | B_physics_separated\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |

**같은 fold 끼리 뺀 차이 (C_full 표준화 XGBoost 기준)**

| 비교 대상 | 평균 차이 | 표준화 XGB 가 더 좋은 fold |
|---|---|---|
| Step 17 C_full Random Forest | -0.2351 | 5 / 5 |
| Step 16 Model 1 XGBoost | +0.5593 | 0 / 5 |
| Step 16 Model 1 Random Forest | +0.5018 | 0 / 5 |
| Step 16 Model 2 고정 k | -4.5092 | 5 / 5 |

## 5. 확인 결과

- **더 이상 fold 별 상수만 예측하지 않습니다.** 모든 fold 에서 서로 다른 k_pred 개수가 그 fold 의 평가 sample 수와 같습니다 (323, 323, 321, 321, 321). 세 구성 모두 전체 1,609개가 서로 다른 값입니다.
- **음수 k_pred 는 없습니다.**
- **k_target 실제 범위를 벗어난 예측도 없습니다.** k_target 범위는 0.000185 ~ 0.000413 이고, k_pred 는 0.000199 ~ 0.000373 안에 들어옵니다.

| 조합 | k_pred 최소 | k_pred 최대 | 서로 다른 값 수 | 음수 | 범위 미만 | 범위 초과 |
|---|---|---|---|---|---|---|
| A_state_only\|XGBoost_scaled | 0.000200 | 0.000368 | 1609 | 0 | 0 | 0 |
| B_physics_separated\|XGBoost_scaled | 0.000204 | 0.000373 | 1609 | 0 | 0 | 0 |
| C_full\|XGBoost_scaled | 0.000199 | 0.000364 | 1609 | 0 | 0 | 0 |

- **Random Forest Hybrid 와의 차이:** C_full 기준으로 표준화 XGBoost 가 RMSE 4.1419, Random Forest 가 4.3770 입니다. 같은 fold 끼리 비교하면 평균 0.2351 낮고 5개 fold 전부에서 더 좋습니다.
- **Model 1 과는 여전히 차이가 있습니다.** Model 1 XGBoost 3.5826 대비 0.5593 높고, 5개 fold 모두에서 Model 1 이 더 좋습니다.
- **Model 2 고정 k 보다는 훨씬 낫습니다.** 평균 4.5092 낮고 5개 fold 전부에서 더 좋습니다.

## 6. Stage 별 최종 MRR 성능

Stage A 797개, Stage B 812개입니다.

| 조합 | A RMSE | B RMSE | A R² | B R² |
|---|---|---|---|---|
| A_state_only\|XGBoost_scaled | 4.2081 | 5.4122 | 0.5997 | 0.6504 |
| B_physics_separated\|XGBoost_scaled | 4.2241 | 4.9612 | 0.5967 | 0.7063 |
| C_full\|XGBoost_scaled | 3.8901 | 4.3829 | 0.6580 | 0.7708 |

- Stage B 의 RMSE 가 세 구성 모두 Stage A 보다 큽니다. 차이는 A_state_only 1.2041, B_physics_separated 0.7371, C_full 0.4929 로, feature 가 많아질수록 줄어듭니다.
- R² 는 Stage B 가 더 높습니다. Stage B 의 MRR 이 더 넓게 퍼져 있기 때문입니다.

## 7. 저장 파일

| 파일 | 내용 |
|---|---|
| `reports/step18_model3_scaled_xgb/oof_predictions.csv` | 3개 구성 x 1,609 sample 의 WAFER_ID, STAGE, fold, PV_proxy, k_target, k_pred, 실제 MRR, MRR_pred, 오차 |
| `reports/step18_model3_scaled_xgb/fold_metrics.csv` | 구성 x fold 별 k 성능과 MRR 성능 |
| `reports/step18_model3_scaled_xgb/summary.csv` | 구성 x 지표 별 5-fold 평균, 표준편차, 최소, 최대 |
| `reports/step18_model3_scaled_xgb/pooled_metrics.csv` | 5개 fold 예측을 모아 계산한 지표 |
| `reports/step18_model3_scaled_xgb/fold_scaling_values.csv` | fold 별 training_mean, training_std, 서로 다른 k_pred 수 |
| `reports/step18_model3_scaled_xgb/k_pred_checks.csv` | k_pred 최소/최대, 서로 다른 값 수, 음수 수, 범위 밖 수 |
| `reports/step18_model3_scaled_xgb/stage_metrics.csv` | Stage 별 k 성능과 MRR 성능 |
| `reports/step18_model3_scaled_xgb/comparison_all_models.csv` | Step 16, 17, 18 결과를 한 표로 정리 |

## 8. 확인한 내용

- 표준화에 쓴 평균과 표준편차가 학습 fold 의 k_target 으로만 계산되었고, 평가 fold 통계는 쓰이지 않았습니다.
- MRR_pred 가 k_pred × PV_proxy 와 일치하고, k_target 이 실제 MRR 을 PV_proxy 로 나눈 값과 일치합니다.
- 각 fold 에서 학습과 평가 사이에 겹치는 WAFER_ID 가 없습니다.
- 코드를 두 번 실행해 8개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

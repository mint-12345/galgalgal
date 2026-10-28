# Step 17. Model 3: Physics-Informed Hybrid (MRR = k(x) × PV_proxy)

- 대상: Step 16 에서 확정한 Chamber path 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold
- training 만 사용했습니다. test / validation 은 사용하지 않았고, raw 파일은 수정하지 않았습니다.
- PV_proxy 와 k_target 은 Step 16 표를 그대로 썼습니다. P = RETAINER_RING_PRESSURE, V = abs(WAFER_ROTATION - HEAD_ROTATION), 두 값이 모두 0 이 아닌 raw row 의 mean(P × V) 입니다. P/V 정의는 바꾸지 않았습니다.
- 코드: `scripts/step17_model3_hybrid.py`, 결과: `reports/step17_model3_hybrid/`
- **최종 조합은 선택하지 않았습니다.**

## 1. 계산 방식

- 학습 정답은 k_target = 실제 MRR / PV_proxy 입니다.
- 평가 sample 의 실제 MRR 과 k_target 은 입력에 쓰지 않았습니다. 전처리도 fold 마다 학습 sample 로만 맞췄습니다.
- 예측값 k_pred 에 PV_proxy 를 곱해 MRR_pred 를 만들었습니다.
- **k_pred 를 보정하거나 clipping 하지 않았습니다.**

## 2. feature 구성

| 구성 | 내용 | feature 수 |
|---|---|---|
| A State-only | STAGE, usage 평균 4개, SLURRY A/B/C 평균과 표준편차, DRESSING_WATER_STATUS 평균과 표준편차 | 13 |
| B Physics-separated | Baseline 에서 WAFER_ID, MRR, CHAMBER_PATH 와 PV 생성 feature 6개를 제외 | 25 |
| C Full | Baseline 에서 WAFER_ID, MRR, CHAMBER_PATH 만 제외 | 31 |

B 에서 뺀 PV 생성 feature 6개: RETAINER_RING_PRESSURE 평균과 표준편차, WAFER_ROTATION 평균과 표준편차, HEAD_ROTATION 평균과 표준편차

모델은 Linear Regression (Step 9 방식), Random Forest, XGBoost (둘 다 Step 10 기본 설정) 이고 튜닝하지 않았습니다.

## 3. k 예측 성능 (k_target vs k_pred, 5-fold 평균 ± 표준편차)

| 조합 | R² | MAE | RMSE |
|---|---|---|---|
| A_state_only\|LinearRegression | 0.499590 ± 0.036941 | 0.000018 ± 0.000001 | 0.000023 ± 0.000001 |
| A_state_only\|RandomForest | 0.671584 ± 0.021195 | 0.000014 ± 0.000001 | 0.000018 ± 0.000001 |
| A_state_only\|XGBoost | -0.001873 ± 0.001946 | 0.000026 ± 0.000001 | 0.000032 ± 0.000001 |
| B_physics_separated\|LinearRegression | 0.549597 ± 0.045264 | 0.000017 ± 0.000001 | 0.000021 ± 0.000001 |
| B_physics_separated\|RandomForest | 0.714895 ± 0.024991 | 0.000013 ± 0.000001 | 0.000017 ± 0.000001 |
| B_physics_separated\|XGBoost | -0.001873 ± 0.001946 | 0.000026 ± 0.000001 | 0.000032 ± 0.000001 |
| C_full\|LinearRegression | 0.597657 ± 0.039062 | 0.000016 ± 0.000001 | 0.000020 ± 0.000001 |
| C_full\|RandomForest | 0.739920 ± 0.025673 | 0.000012 ± 0.000001 | 0.000016 ± 0.000001 |
| C_full\|XGBoost | -0.001873 ± 0.001946 | 0.000026 ± 0.000001 | 0.000032 ± 0.000001 |

## 4. 최종 MRR 예측 성능 (실제 MRR vs k_pred × PV_proxy)

| 조합 | R² | MAE | RMSE |
|---|---|---|---|
| A_state_only\|LinearRegression | 0.5022 ± 0.0361 | 4.7863 ± 0.1796 | 6.1388 ± 0.2331 |
| A_state_only\|RandomForest | 0.6701 ± 0.0174 | 3.6682 ± 0.1840 | 5.0002 ± 0.2086 |
| A_state_only\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |
| B_physics_separated\|LinearRegression | 0.5643 ± 0.0410 | 4.5601 ± 0.2106 | 5.7428 ± 0.3248 |
| B_physics_separated\|RandomForest | 0.7228 ± 0.0244 | 3.4253 ± 0.1795 | 4.5804 ± 0.2259 |
| B_physics_separated\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |
| C_full\|LinearRegression | 0.6099 ± 0.0327 | 4.3376 ± 0.2067 | 5.4342 ± 0.2553 |
| C_full\|RandomForest | 0.7468 ± 0.0247 | 3.2988 ± 0.1642 | 4.3770 ± 0.2277 |
| C_full\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |

## 5. Step 16 결과와 함께 본 MRR 성능

같은 1,609개, 같은 5-fold 기준입니다. RMSE 가 낮은 순입니다.

| 출처 | 모델 | 조합 | R² | MAE | RMSE |
|---|---|---|---|---|---|
| Step 16 | Model 1 | Model1_XGB | 0.8296 ± 0.0248 | 2.7451 ± 0.1744 | 3.5826 ± 0.1942 |
| Step 16 | Model 1 | Model1_RF | 0.8246 ± 0.0220 | 2.7846 ± 0.1642 | 3.6401 ± 0.2167 |
| Step 17 | Model 3 | C_full\|RandomForest | 0.7468 ± 0.0247 | 3.2988 ± 0.1642 | 4.3770 ± 0.2277 |
| Step 17 | Model 3 | B_physics_separated\|RandomForest | 0.7228 ± 0.0244 | 3.4253 ± 0.1795 | 4.5804 ± 0.2259 |
| Step 17 | Model 3 | A_state_only\|RandomForest | 0.6701 ± 0.0174 | 3.6682 ± 0.1840 | 5.0002 ± 0.2086 |
| Step 17 | Model 3 | C_full\|LinearRegression | 0.6099 ± 0.0327 | 4.3376 ± 0.2067 | 5.4342 ± 0.2553 |
| Step 17 | Model 3 | B_physics_separated\|LinearRegression | 0.5643 ± 0.0410 | 4.5601 ± 0.2106 | 5.7428 ± 0.3248 |
| Step 17 | Model 3 | A_state_only\|LinearRegression | 0.5022 ± 0.0361 | 4.7863 ± 0.1796 | 6.1388 ± 0.2331 |
| Step 16 | Model 2 | Model2_fixed_k | 0.0113 ± 0.0692 | 6.9583 ± 0.1837 | 8.6511 ± 0.3058 |
| Step 17 | Model 3 | A_state_only\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |
| Step 17 | Model 3 | B_physics_separated\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |
| Step 17 | Model 3 | C_full\|XGBoost | 0.0107 ± 0.0688 | 6.9631 ± 0.1824 | 8.6539 ± 0.3081 |

## 6. XGBoost 세 구성의 결과가 같은 이유 (확인된 사실)

- **세 feature 구성의 k_pred 가 완전히 같습니다.** A, B, C 의 예측 배열이 서로 일치합니다.
- **XGBoost 는 fold 마다 상수 하나만 예측했습니다.** 각 fold 의 k_pred 는 그 fold 학습 데이터의 k_target 평균과 같습니다.

| fold | 학습 k_target 평균 | k_pred (모든 sample 동일) |
|---|---|---|
| 1 | 0.000283339 | 0.000283339 |
| 2 | 0.000284016 | 0.000284016 |
| 3 | 0.000284080 | 0.000284080 |
| 4 | 0.000283940 | 0.000283940 |
| 5 | 0.000283630 | 0.000283630 |

- 그래서 XGBoost 조합의 MRR 성능(RMSE 8.6539)이 Model 2 고정 k(8.6511)와 거의 같습니다. 두 방법 모두 사실상 하나의 k 를 쓰는 셈입니다.
- **원인은 k_target 의 크기입니다.** k_target 은 약 0.00028 수준이라 값의 흩어짐이 매우 작습니다. XGBoost 기본 설정의 L2 정규화(reg_lambda=1)가 이 크기에서는 잎 출력을 거의 0 으로 눌러, 모델이 시작값(학습 평균)에 머뭅니다.
- **진단으로 확인했습니다.** fold 1 에서 목표값에 상수만 곱해 다시 학습하면 결과가 달라집니다. 이 진단은 확인용이며 이번 결과표에는 반영하지 않았습니다.

| 목표값 배율 | k_pred 서로 다른 값 수 | k 예측 R² (fold 1) |
|---|---|---|
| ×1 (원래) | 1 | -0.0052 |
| ×1,000 | 323 | 0.7904 |
| ×10,000 | 323 | 0.7894 |
| ×1,000,000 | 323 | 0.7885 |

- 지시대로 튜닝을 하지 않았으므로 본 결과에는 원래 크기 그대로의 XGBoost 를 남겼습니다.

## 7. k_pred 확인

| 조합 | k_pred 최소 | k_pred 최대 | 음수 | k_target 최소 미만 | k_target 최대 초과 |
|---|---|---|---|---|---|
| A_state_only\|LinearRegression | 0.000224 | 0.000334 | 0 | 0 | 0 |
| A_state_only\|RandomForest | 0.000208 | 0.000345 | 0 | 0 | 0 |
| A_state_only\|XGBoost | 0.000283 | 0.000284 | 0 | 0 | 0 |
| B_physics_separated\|LinearRegression | 0.000200 | 0.000343 | 0 | 0 | 0 |
| B_physics_separated\|RandomForest | 0.000210 | 0.000346 | 0 | 0 | 0 |
| B_physics_separated\|XGBoost | 0.000283 | 0.000284 | 0 | 0 | 0 |
| C_full\|LinearRegression | 0.000203 | 0.000384 | 0 | 0 | 0 |
| C_full\|RandomForest | 0.000213 | 0.000352 | 0 | 0 | 0 |
| C_full\|XGBoost | 0.000283 | 0.000284 | 0 | 0 | 0 |

- **음수 k_pred 는 없습니다.**
- **k_target 실제 범위(0.000185 ~ 0.000413)를 벗어난 예측도 없습니다.** 9개 조합 모두 0개입니다.

## 8. Stage 별 최종 MRR 성능

Stage A 797개, Stage B 812개입니다.

| 조합 | A RMSE | B RMSE | A R² | B R² |
|---|---|---|---|---|
| A_state_only\|LinearRegression | 5.5697 | 6.6565 | 0.2988 | 0.4712 |
| A_state_only\|RandomForest | 4.3484 | 5.5720 | 0.5726 | 0.6295 |
| A_state_only\|XGBoost | 7.9760 | 9.2796 | -0.4379 | -0.0276 |
| B_physics_separated\|LinearRegression | 5.2947 | 6.1644 | 0.3664 | 0.5465 |
| B_physics_separated\|RandomForest | 4.2285 | 4.9097 | 0.5958 | 0.7123 |
| B_physics_separated\|XGBoost | 7.9760 | 9.2796 | -0.4379 | -0.0276 |
| C_full\|LinearRegression | 4.9744 | 5.8592 | 0.4407 | 0.5903 |
| C_full\|RandomForest | 4.0218 | 4.7085 | 0.6344 | 0.7354 |
| C_full\|XGBoost | 7.9760 | 9.2796 | -0.4379 | -0.0276 |

- **Stage B 의 RMSE 가 모든 조합에서 Stage A 보다 큽니다.** 반면 R² 는 Stage B 가 더 높습니다. Stage B 의 MRR 이 더 넓게 퍼져 있어 같은 오차라도 R² 가 높게 나옵니다.
- Stage 사이 RMSE 차이는 Random Forest 조합에서 0.68 ~ 1.22 입니다.

## 9. 결과 정리 (선택은 하지 않음)

- **feature 를 많이 줄수록 좋아집니다.** Linear Regression 과 Random Forest 모두 A < B < C 순서로 MRR RMSE 가 낮아집니다.
- **PV 생성 feature 를 다시 넣는 것(C)이 B 보다 낫습니다.** Random Forest 기준 4.5804 에서 4.3770 으로 0.2034 낮아집니다.
- **Model 3 의 가장 좋은 조합도 Model 1 보다는 오차가 큽니다.** C_full|RandomForest 4.3770 대 Model1_XGB 3.5826 입니다.
- **Model 2 고정 k 보다는 훨씬 낫습니다.** 8.6511 에서 4.3770 으로 줄어듭니다. k 를 sample 마다 예측하는 것이 하나의 k 를 쓰는 것보다 분명히 낫습니다.
- XGBoost 조합 세 개는 위에서 설명한 이유로 고정 k 와 같은 수준입니다.

## 10. 저장 파일

| 파일 | 내용 |
|---|---|
| `reports/step17_model3_hybrid/oof_predictions.csv` | 9개 조합 x 1,609 sample 의 WAFER_ID, STAGE, fold, PV_proxy, k_target, k_pred, 실제 MRR, MRR_pred, 오차 |
| `reports/step17_model3_hybrid/fold_metrics.csv` | 조합 x fold 별 k 성능과 MRR 성능 |
| `reports/step17_model3_hybrid/summary.csv` | 조합 x 지표 별 5-fold 평균, 표준편차, 최소, 최대 |
| `reports/step17_model3_hybrid/pooled_metrics.csv` | 5개 fold 예측을 모아 계산한 지표 |
| `reports/step17_model3_hybrid/comparison_with_step16.csv` | Step 16 Model 1, Model 2 와 함께 정리한 MRR 성능 |
| `reports/step17_model3_hybrid/k_pred_checks.csv` | k_pred 최소/최대, 음수 수, k_target 범위 밖 수 |
| `reports/step17_model3_hybrid/stage_metrics.csv` | Stage 별 k 성능과 MRR 성능 |

## 11. 확인한 내용

- k_target 이 실제 MRR 을 PV_proxy 로 나눈 값과 일치하고, 1,609개 모두 4-5-6 경로인 것을 확인했습니다.
- 각 fold 에서 학습과 평가 사이에 겹치는 WAFER_ID 가 없는 것을 확인했습니다.
- XGBoost 의 fold 별 k_pred 가 학습 fold 의 k_target 평균과 같은 것을 확인했습니다.
- 코드를 두 번 실행해 7개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

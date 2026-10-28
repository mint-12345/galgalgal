# Step 24. Pure ML XGBoost 에 Hybrid 와 같은 tuning 기회 주기

- 목적: Hybrid 는 Step 20 에서 54개 설정을 탐색했는데 Pure ML 은 기본 설정만 썼습니다. 비교를 공정하게 만들기 위해 Pure ML 에도 같은 크기의 탐색을 줍니다.
- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 training sample, Step 8 의 같은 5-fold
- training 만 사용했습니다. test / validation 은 사용하지 않았습니다.
- 새 feature 를 만들거나 지우지 않았습니다. **Hybrid 는 다시 튜닝하지 않았고, Step 20 에 저장된 결과를 읽어서 비교만 했습니다.**
- 코드: `scripts/step24_pure_ml_xgb_tuning.py`, 결과: `reports/step24_pure_ml_tuning/`

## 1. Pure ML 구조

공정 feature x → MRR 직접 예측입니다. Step 16 Model 1 의 Baseline feature 구성을 그대로 썼습니다.

- 입력 feature 31개 = STAGE(범주형 1개) + 숫자형 30개
- WAFER_ID 는 입력에서 제외했습니다.
- 실제 MRR 은 target 으로만 썼습니다.
- CHAMBER_PATH 는 전부 4-5-6 이므로 입력하지 않았습니다.
- 전처리는 Step 16 / 20 과 같습니다. STAGE 는 OneHotEncoder(drop="first"), 숫자형은 passthrough 입니다.

## 2. 탐색한 54개 설정

Step 20 과 같은 격자입니다.

| 항목 | 값 |
|---|---|
| n_estimators | 300, 600, 900 |
| max_depth | 2, 3, 4 |
| learning_rate | 0.03, 0.05, 0.1 |
| min_child_weight | 1, 3 |

모든 조합에서 subsample = 0.9, colsample_bytree = 0.9, random_state = 42, n_jobs = -1 로 고정했습니다. reg_alpha, reg_lambda, gamma 등 나머지는 기본값을 유지했고 탐색하지 않았습니다. 3 × 3 × 3 × 2 = 54개 설정을 5-fold 로 평가해 학습을 270회 했습니다.

## 3. Pure ML 상위 10개 설정

주 비교 기준은 5-fold 평균 RMSE 입니다.

| 순위 | 설정 | tree 수 | depth | learning_rate | min_child_weight | 평균 RMSE | RMSE 표준편차 | 평균 MAE | 평균 R² |
|---|---|---|---|---|---|---|---|---|---|
| 1 | n900_d4_lr0.05_mcw3 | 900 | 4 | 0.05 | 3 | 3.3163 | 0.1511 | 2.5530 | 0.8544 |
| 2 | n900_d4_lr0.05_mcw1 | 900 | 4 | 0.05 | 1 | 3.3202 | 0.1626 | 2.5527 | 0.8540 |
| 3 | n600_d4_lr0.05_mcw3 | 600 | 4 | 0.05 | 3 | 3.3212 | 0.1523 | 2.5613 | 0.8540 |
| 4 | n600_d4_lr0.05_mcw1 | 600 | 4 | 0.05 | 1 | 3.3340 | 0.1645 | 2.5650 | 0.8528 |
| 5 | n900_d4_lr0.03_mcw3 | 900 | 4 | 0.03 | 3 | 3.3346 | 0.1619 | 2.5681 | 0.8528 |
| 6 | n900_d4_lr0.03_mcw1 | 900 | 4 | 0.03 | 1 | 3.3453 | 0.1482 | 2.5722 | 0.8519 |
| 7 | n600_d4_lr0.03_mcw3 | 600 | 4 | 0.03 | 3 | 3.3579 | 0.1639 | 2.5849 | 0.8507 |
| 8 | n900_d4_lr0.1_mcw1 | 900 | 4 | 0.1 | 1 | 3.3621 | 0.1483 | 2.6043 | 0.8505 |
| 9 | n600_d4_lr0.1_mcw1 | 600 | 4 | 0.1 | 1 | 3.3653 | 0.1502 | 2.6064 | 0.8502 |
| 10 | n300_d4_lr0.05_mcw3 | 300 | 4 | 0.05 | 3 | 3.3701 | 0.1544 | 2.5982 | 0.8497 |

- 54개 전체의 평균 RMSE 는 3.3163 에서 4.0901 사이입니다. 가장 나쁜 설정은 n300_d2_lr0.03_mcw1 입니다.
- **상위 10개가 모두 max_depth = 4 입니다.** 이는 이번 격자에서 허용한 가장 깊은 값입니다. 격자를 넓히면 결과가 달라질 수 있지만, 이번 단계에서는 Step 20 과 같은 범위만 사용했고 범위를 넓히지 않았습니다.
- 상위 10개의 learning_rate 는 0.05 가 5개, 0.03 이 3개, 0.1 이 2개입니다. tree 수는 900 이 5개, 600 이 4개, 300 이 1개입니다.

## 4. 상위 설정 간 차이와 fold 변동

| 항목 | 값 |
|---|---|
| 상위 10개 평균 RMSE 최소 | 3.3163 |
| 상위 10개 평균 RMSE 최대 | 3.3701 |
| 상위 10개 평균 RMSE 폭 | 0.0538 |
| 상위 10개 설정의 fold 표준편차 평균 | 0.1557 |
| 폭 / fold 표준편차 평균 | 0.3453 |
| 1위와 2위의 평균 RMSE 차이 | 0.0039 |
| 상위 10개 중 가장 적은 tree 수 | 300 |
| 상위 10개 중 가장 얕은 depth | 4 |

- **상위 10개 전체의 폭(0.0538)이 fold 변동(평균 0.1557)의 3분의 1 수준입니다.** 1위와 2위의 차이는 0.0039 로 더 작습니다.
- **따라서 유일한 최적 설정이라고 말할 수 없습니다.** 상위 후보들은 사실상 같은 성능입니다.
- 더 단순한 설정을 고르고 싶다면 3위 n600_d4_lr0.05_mcw3 가 1위보다 0.0049 만 나쁘고 tree 수는 600 입니다. 10위 n300_d4_lr0.05_mcw3 은 tree 수 300 으로 1위보다 0.0538 나쁩니다. 두 차이 모두 fold 변동보다 작습니다.
- 아래 비교에서는 평균 RMSE 1위인 n900_d4_lr0.05_mcw3 을 Pure ML 대표로 썼습니다. 이는 계산을 위한 선택이고 유일한 최적이라는 뜻이 아닙니다.

## 5. Stage 별 성능

상위 5개 설정과 대표 Hybrid 입니다. 5-fold out-of-fold 예측을 Stage 로 나눠 계산했습니다.

| 설정 | Stage | sample 수 | R² | MAE | RMSE |
|---|---|---|---|---|---|
| n900_d4_lr0.05_mcw3 | A | 797 | 0.7952 | 2.3035 | 3.0097 |
| n900_d4_lr0.05_mcw3 | B | 812 | 0.8456 | 2.7981 | 3.5973 |
| n900_d4_lr0.05_mcw1 | A | 797 | 0.7909 | 2.3171 | 3.0419 |
| n900_d4_lr0.05_mcw1 | B | 812 | 0.8472 | 2.7842 | 3.5785 |
| n600_d4_lr0.05_mcw3 | A | 797 | 0.7935 | 2.3207 | 3.0224 |
| n600_d4_lr0.05_mcw3 | B | 812 | 0.8457 | 2.7977 | 3.5959 |
| n600_d4_lr0.05_mcw1 | A | 797 | 0.7887 | 2.3348 | 3.0573 |
| n600_d4_lr0.05_mcw1 | B | 812 | 0.8461 | 2.7912 | 3.5912 |
| n900_d4_lr0.03_mcw3 | A | 797 | 0.7903 | 2.3463 | 3.0462 |
| n900_d4_lr0.03_mcw3 | B | 812 | 0.8452 | 2.7861 | 3.6014 |
| Hybrid n900_d2_lr0.1_mcw1 | A | 797 | 0.7507 | 2.5762 | 3.3210 |
| Hybrid n900_d2_lr0.1_mcw1 | B | 812 | 0.8274 | 2.9489 | 3.8028 |

- **tuning 을 받은 Pure ML 은 Stage A 와 Stage B 모두에서 대표 Hybrid 보다 RMSE 가 낮습니다.**
- 두 모델 모두 Stage B 가 Stage A 보다 나쁩니다. 이 방향은 이전 단계들과 같습니다.

## 6. 기준 모델과의 비교

| 구분 | 설정 | 평균 R² | 평균 MAE | 평균 RMSE | RMSE 표준편차 |
|---|---|---|---|---|---|
| A. 기존 Pure ML XGBoost (Step 16 기본 설정) | XGBRegressor 기본값 (subsample / colsample 미설정) | 0.8296 | 2.7451 | 3.5826 | 0.1942 |
| B1. Pure ML tuning 1위 | n900_d4_lr0.05_mcw3 | 0.8544 | 2.5530 | 3.3163 | 0.1511 |
| B2. Pure ML tuning 2위 | n900_d4_lr0.05_mcw1 | 0.8540 | 2.5527 | 3.3202 | 0.1626 |
| B3. Pure ML tuning 3위 | n600_d4_lr0.05_mcw3 | 0.8540 | 2.5613 | 3.3212 | 0.1523 |
| B4. Pure ML tuning 4위 | n600_d4_lr0.05_mcw1 | 0.8528 | 2.5650 | 3.3340 | 0.1645 |
| B5. Pure ML tuning 5위 | n900_d4_lr0.03_mcw3 | 0.8528 | 2.5681 | 3.3346 | 0.1619 |
| B6. Pure ML tuning 6위 | n900_d4_lr0.03_mcw1 | 0.8519 | 2.5722 | 3.3453 | 0.1482 |
| B7. Pure ML tuning 7위 | n600_d4_lr0.03_mcw3 | 0.8507 | 2.5849 | 3.3579 | 0.1639 |
| B8. Pure ML tuning 8위 | n900_d4_lr0.1_mcw1 | 0.8505 | 2.6043 | 3.3621 | 0.1483 |
| B9. Pure ML tuning 9위 | n600_d4_lr0.1_mcw1 | 0.8502 | 2.6064 | 3.3653 | 0.1502 |
| B10. Pure ML tuning 10위 | n300_d4_lr0.05_mcw3 | 0.8497 | 2.5982 | 3.3701 | 0.1544 |
| C. Step 20 대표 Hybrid | n900_d2_lr0.1_mcw1, PV = P9 x V3, standardized k | 0.8314 | 2.7641 | 3.5693 | 0.1573 |
| D. fixed-k Preston | 절편 없는 fold 별 고정 k | 0.0113 | 6.9583 | 8.6511 | 0.3058 |

- tuning 으로 Pure ML 의 평균 RMSE 가 3.5826 에서 3.3163 으로 **0.2663 줄었습니다.**
- 54개 중 42개 설정이 기존 Pure ML(A) 보다 낮고, 같은 42개가 대표 Hybrid(C) 보다도 낮습니다.
- fixed-k Preston 은 여전히 크게 나쁩니다. tuning 한 Pure ML 보다 5.3348 높습니다.

## 7. Pure ML 최상위와 대표 Hybrid 의 fold 단위 비교

같은 fold 끼리 직접 비교했습니다.

| fold | Pure ML 최상위 RMSE | 대표 Hybrid RMSE | 차이 (Hybrid − Pure ML) | 더 낮은 쪽 |
|---|---|---|---|---|
| 1 | 3.2280 | 3.4961 | +0.2681 | Pure ML |
| 2 | 3.5608 | 3.8118 | +0.2509 | Pure ML |
| 3 | 3.3318 | 3.6423 | +0.3105 | Pure ML |
| 4 | 3.1643 | 3.4394 | +0.2751 | Pure ML |
| 5 | 3.2966 | 3.4571 | +0.1604 | Pure ML |

| 항목 | 값 |
|---|---|
| 비교한 Pure ML 설정 | n900_d4_lr0.05_mcw3 |
| 비교한 Hybrid 설정 | n900_d2_lr0.1_mcw1 (P9 x V3) |
| Hybrid 가 더 낮은 fold 수 | 0 |
| Pure ML 이 더 낮은 fold 수 | 5 |
| 평균 RMSE (Pure ML 최상위) | 3.3163 |
| 평균 RMSE (대표 Hybrid) | 3.5693 |
| 평균 RMSE 차이 (Hybrid - Pure ML) | +0.2530 |
| fold 별 차이의 표준편차 | 0.0561 |

- **5개 fold 모두에서 Pure ML 이 더 낮습니다.** fold 별 차이는 0.1604 에서 0.3105 사이이고 모두 같은 방향입니다.
- 평균 차이 0.2530 은 fold 별 차이의 표준편차 0.0561 보다 훨씬 큽니다. 상위 설정끼리의 차이(0.0538)보다도 큽니다.

## 8. 해석

**1) 같은 tuning 기회를 주면 순서가 바뀝니다**

tuning 전에는 대표 Hybrid(3.5693)가 기존 Pure ML(3.5826)보다 0.0133 낮아 근소하게 앞섰습니다. 같은 54개 격자를 Pure ML 에도 적용하자 Pure ML 이 3.3163 으로 내려가 **Hybrid 보다 0.2530 낮아졌고, 5개 fold 모두에서 앞섭니다.**

**앞선 단계들에서 보였던 Hybrid 의 근소한 우위는 상당 부분 tuning 기회의 차이에서 온 것으로 보입니다.** Hybrid 는 Step 20 에서 54개를 탐색했고 Pure ML 은 기본 설정 하나만 썼기 때문입니다.

**2) 유일한 최적 설정은 없습니다**

상위 10개의 평균 RMSE 폭은 0.0538 로 fold 변동(0.1557)보다 작습니다. 1위와 2위 차이는 0.0039 입니다. 상위 후보들은 사실상 같은 성능이므로, 더 단순한 설정을 대표로 삼아도 됩니다. 3위 설정은 tree 수가 600 이고 1위와 0.0049 차이입니다.

**3) Stage 별로도 같은 방향입니다**

tuning 한 Pure ML 이 Stage A(3.0097 대 3.3210)와 Stage B(3.5973 대 3.8028) 모두에서 Hybrid 보다 낮습니다. 한쪽 Stage 만 희생해서 평균이 좋아진 것이 아닙니다.

**4) 물리 구조의 가치는 정확도가 아닙니다**

fixed-k Preston 은 8.6511 로 여전히 크게 나쁩니다. Hybrid 는 그 격차를 대부분 메우지만, 같은 tuning 을 받은 Pure ML 을 넘지는 못합니다. **이 데이터에서 Hybrid 의 값어치는 정확도보다 해석 가능성(Step 21)에 있습니다.**

**5) 한계 — 앞 단계 비교에 미치는 영향**

**Step 22 와 Step 23 의 비교는 기본 설정 Pure ML 을 사용했습니다.** 그 단계들에서 본 Hybrid 의 robustness 우위는 tuning 한 Pure ML 을 상대로는 다시 확인되지 않았습니다. 이번 단계로 training 데이터에서의 tuning 을 종료하기로 했으므로 Step 22 와 Step 23 을 다시 계산하지는 않았습니다. 그 결과들을 읽을 때 이 점을 함께 고려해야 합니다.

그 밖의 한계는 다음과 같습니다.

- 상위 10개가 모두 이번 격자의 최대 depth 인 4 를 씁니다. 격자 밖은 확인하지 않았습니다.
- 모든 수치는 training 데이터의 random 5-fold 결과입니다. test 와 validation 은 사용하지 않았습니다.

## 9. 저장 파일

| 파일 | 내용 |
|---|---|
| `config_results.csv` | 54개 설정의 R², MAE, RMSE 평균과 표준편차, RMSE 순위 |
| `top10_configs.csv` | 평균 RMSE 상위 10개 설정 |
| `fold_metrics.csv` | 54개 설정 × 5 fold 의 지표 (270행) |
| `summary_long.csv` | 설정별 지표의 평균, 표준편차, 최소, 최대 |
| `stage_metrics.csv` | 54개 설정의 Stage A / B 별 지표 |
| `stage_metrics_top10.csv` | 상위 10개 설정의 Stage 별 지표 |
| `hybrid_stage_metrics_reference.csv` | 대표 Hybrid 의 Stage 별 지표 (Step 20 에서 읽음) |
| `comparison_with_reference_models.csv` | A / B1~B10 / C / D 비교표 |
| `paired_fold_comparison_vs_hybrid.csv` | Pure ML 최상위와 대표 Hybrid 의 fold 단위 비교 |
| `head_to_head_summary.csv` | fold 승패와 평균 차이 요약 |
| `top_config_spread.csv` | 상위 설정 간 차이와 fold 변동 비교 |
| `oof_all_configs.csv` | 54개 설정 전체의 sample 별 out-of-fold 예측 (재집계용) |
| `oof_top/oof_<설정>.csv` | 상위 5개 설정의 sample 별 예측 (WAFER_ID, STAGE, FOLD, 실제 MRR, MRR_PRED) |
| `consistency_checks.csv` | 아래 확인 항목의 결과 |

## 10. 확인한 내용

- sample 이 1,609개이고 feature 가 31개이며 WAFER_ID 와 CHAMBER_PATH 가 입력에서 빠졌습니다.
- FOLD 가 Step 8 배정과 같습니다.
- 설정이 54개이고 설정마다 1,609개 예측이 있으며, 각 sample 은 정확히 한 fold 에서만 평가됩니다.
- 기존 Pure ML 평균 RMSE 3.582600 과 대표 Hybrid 평균 RMSE 3.569330 을 Step 16 과 Step 20 저장값에서 그대로 읽었습니다.
- Hybrid 의 fold 별 RMSE 도 Step 20 저장값을 그대로 사용했습니다.
- 코드를 두 번 실행해 18개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

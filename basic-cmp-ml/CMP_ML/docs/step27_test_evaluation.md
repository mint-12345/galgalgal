# Step 27. 최종 external test 평가

- 이 연구의 **마지막 외부 평가**입니다. test 데이터는 이번 단계에서 처음 사용했습니다.
- **결과를 보고 feature, P/V, preprocessing, 모델 구조, hyperparameter 를 전혀 수정하지 않았습니다.**
- 학습 데이터는 Step 26 과 완전히 같습니다. training 4-5-6 의 1,609개 sample 뿐이고, **validation 을 학습에 추가하지 않았습니다.** validation 과 test 를 같은 모델로 평가하기 위해서입니다.
- 코드: `scripts/step27_test_evaluation.py`, 결과: `reports/step27_test/`

## 1. 평가한 세 모델

| 모델 | 구성 |
|---|---|
| Model 1 Pure ML | XGBoost(900, depth 4, lr 0.05, mcw 3, subsample 0.9, colsample 0.9, seed 42) 로 MRR 직접 예측 |
| Model 2 Fixed-k Preston | MRR = k × PV_proxy, 절편 없이 k = Σ(PV·MRR) / Σ(PV²) |
| Model 3 Hybrid | MRR = k(x) × PV_proxy, k(x) = XGBoost(900, depth 2, lr 0.1, mcw 1, subsample 0.9, colsample 0.9, seed 42) |

- PV_proxy = P9 × V3 입니다. P9 은 pressure 6개의 raw row 별 평균, V3 은 abs(HEAD_ROTATION) 이며, 두 값이 모두 0 이 아닌 raw row 에서만 P × V 를 평균했습니다.
- **fixed k = 0.001290789418** 로 Step 26 과 같은 값이며 test 에서 다시 맞추지 않았습니다.
- k 표준화 평균과 표준편차도 Step 26 과 같은 training 값을 그대로 적용했습니다.
- **test 의 MRR 과 k_target 은 학습, scaling, preprocessing 에 전혀 쓰지 않았습니다.** 평가에만 썼습니다.

## 2. test 전처리와 데이터 구조

Step 26 의 validation 처리와 완전히 같은 처리만 적용했습니다. 같은 wafer + Stage 단위, 같은 연속 완전중복 제거, 같은 feature 계산, 같은 P9 × V3 정의입니다. 검증을 위해 **validation 표를 같은 코드로 다시 만들어 Step 26 저장본과 일치하는 것을 확인했고, validation 성능도 Step 26 결과와 같습니다.**

| 데이터 | CHAMBER_PATH | sample 수 | wafer 수 | Stage A | Stage B | PV 계산 가능 | MRR 평균 | MRR 범위 |
|---|---|---|---|---|---|---|---|---|
| validation | 1-2-3 | 66 | 66 | 66 | 0 | 33 | 151.1365 | 141.1039 ~ 161.0377 |
| validation | 4-5-6 | 357 | 349 | 185 | 172 | 357 | 75.8591 | 53.4765 ~ 95.7897 |
| validation | other | 1 | 1 | 1 | 0 | 1 | 145.6347 | 145.6347 ~ 145.6347 |
| test | 1-2-3 | 73 | 73 | 73 | 0 | 34 | 152.2189 | 145.3953 ~ 163.7971 |
| test | 4-5-6 | 351 | 342 | 165 | 186 | 351 | 76.9822 | 54.5025 ~ 98.7765 |

- 세 모델의 직접 비교는 **test 4-5-6 의 351개**에서만 했습니다. 351개 모두 PV_proxy 를 계산할 수 있습니다.
- **1-2-3 의 73개는 삭제하지 않고 구조만 보고합니다.** 전부 Stage A 이고 절반도 안 되는 34개만 PV_proxy 를 계산할 수 있으며, MRR 평균이 152.22 로 4-5-6 의 76.98 과 다른 영역입니다. Preston 과 Hybrid 구조가 적용되지 않는 sample 에 억지로 예측값을 만들지 않았습니다.
- test 에는 other 로 분류된 sample 이 없습니다.

## 3. 최종 test 성능 (4-5-6, 351개)

| 모델 | sample 수 | R² | MAE | RMSE |
|---|---|---|---|---|
| Pure ML | 351 | 0.8699 | 2.4387 | 3.1856 |
| Fixed-k Preston | 351 | -0.0700 | 7.3425 | 9.1367 |
| Hybrid | 351 | 0.8461 | 2.6473 | 3.4650 |

### Stage 별

| 모델 | Stage | sample 수 | R² | MAE | RMSE |
|---|---|---|---|---|---|
| Pure ML | A | 165 | 0.7962 | 2.1858 | 2.8988 |
| Pure ML | B | 186 | 0.8393 | 2.6630 | 3.4200 |
| Fixed-k Preston | A | 165 | -0.6231 | 6.2476 | 8.1798 |
| Fixed-k Preston | B | 186 | -0.3492 | 8.3138 | 9.9085 |
| Hybrid | A | 165 | 0.7555 | 2.4334 | 3.1746 |
| Hybrid | B | 186 | 0.8115 | 2.8369 | 3.7035 |

두 ML 모델 모두 test 에서는 Stage A 가 Stage B 보다 RMSE 가 낮습니다. Pure ML 은 2.8988 대 3.4200, Hybrid 는 3.1746 대 3.7035 입니다.

## 4. Training / Validation / Test 비교

| 모델 | 데이터 | sample 수 | R² | MAE | RMSE |
|---|---|---|---|---|---|
| Pure ML | training | 1609 | 0.8552 | 2.5531 | 3.3193 |
| Pure ML | validation | 357 | 0.8684 | 2.4275 | 3.2560 |
| Pure ML | test | 351 | 0.8699 | 2.4387 | 3.1856 |
| Fixed-k Preston | training | 1609 | -0.0082 | 7.0386 | 8.7597 |
| Fixed-k Preston | validation | 357 | -0.0312 | 7.4405 | 9.1130 |
| Fixed-k Preston | test | 351 | -0.0700 | 7.3425 | 9.1367 |
| Hybrid | training | 1609 | 0.8323 | 2.7643 | 3.5723 |
| Hybrid | validation | 357 | 0.8553 | 2.6134 | 3.4142 |
| Hybrid | test | 351 | 0.8461 | 2.6473 | 3.4650 |

training 수치는 5-fold out-of-fold 결과입니다.

| 모델 | training OOF RMSE | validation RMSE | test RMSE | validation − training | test − training | test − validation |
|---|---|---|---|---|---|---|
| Pure ML | 3.3193 | 3.2560 | 3.1856 | -0.0633 | -0.1336 | -0.0703 |
| Fixed-k Preston | 8.7597 | 9.1130 | 9.1367 | +0.3533 | +0.3770 | +0.0237 |
| Hybrid | 3.5723 | 3.4142 | 3.4650 | -0.1581 | -0.1073 | +0.0508 |

- **세 데이터에서 모델 순위가 완전히 같습니다.** Pure ML, Hybrid, Fixed-k Preston 순입니다.
- 외부에서 나빠진 모델은 Fixed-k Preston 뿐입니다. 두 ML 모델은 training OOF 보다 오히려 조금 좋습니다.
- 이것을 일반화가 더 잘 된다고 읽으면 안 됩니다. training 수치는 80% 로 학습한 모델들의 5-fold 평균이고, validation 과 test 는 100% 로 학습한 모델의 결과입니다. 또한 351개와 357개는 1,609개보다 표본 변동이 큽니다.

## 5. 데이터 독립성

| 비교 | 왼쪽 sample 수 | 오른쪽 sample 수 | wafer+Stage 키 겹침 | 공유 WAFER_ID 수 |
|---|---|---|---|---|
| training vs validation | 1609 | 424 | 0 | 115 |
| training vs test | 1609 | 424 | 0 | 113 |
| validation vs test | 424 | 424 | 0 | 34 |

- **같은 wafer+Stage 기록이 두 데이터에 동시에 들어간 경우는 세 쌍 모두 0건입니다.** 평가 대상은 학습에 쓰이지 않은 기록입니다.
- 다만 **WAFER_ID 는 공유됩니다.** training 과 test 사이 113개, validation 과 test 사이 34개입니다. 같은 wafer 가 한쪽에서는 Stage A, 다른 쪽에서는 Stage B 로 나타납니다.
- **따라서 완전한 독립이 아닙니다. 이 점을 최종 결과의 한계로 기록합니다.** 같은 wafer 의 소모품 사용량이나 장비 상태가 두 데이터에 부분적으로 공유될 수 있습니다.

## 6. 분포 비교 (세 데이터의 4-5-6)

| 데이터 | sample 수 | Stage A | Stage B | Stage B 비율 |
|---|---|---|---|---|
| training | 1609 | 797 | 812 | 0.5047 |
| validation | 357 | 185 | 172 | 0.4818 |
| test | 351 | 165 | 186 | 0.5299 |

| 변수 | training 평균 | validation 평균 | test 평균 | training 표준편차 | validation 표준편차 | test 표준편차 | validation 표준화차 | test 표준화차 |
|---|---|---|---|---|---|---|---|---|
| USAGE_OF_DRESSER_MEAN | 393.6721 | 398.4838 | 372.7374 | 220.1822 | 215.0427 | 223.2330 | +0.0221 | -0.0944 |
| PV_PROXY | 59247.6392 | 59537.9232 | 59479.1386 | 2703.1853 | 2833.1898 | 2855.2326 | +0.1048 | +0.0833 |
| RETAINER_RING_PRESSURE_MEAN | 1372.0449 | 1367.2698 | 1400.2412 | 349.4993 | 363.6808 | 346.8453 | -0.0134 | +0.0810 |
| PRESSURIZED_CHAMBER_PRESSURE_MEAN | 56.1856 | 55.7695 | 57.3763 | 15.0416 | 15.4601 | 15.1055 | -0.0273 | +0.0790 |
| USAGE_OF_POLISHING_TABLE_MEAN | 175.9128 | 178.0989 | 169.0011 | 94.3200 | 94.8854 | 89.6136 | +0.0231 | -0.0751 |
| WAFER_ROTATION_MEAN | 14.3659 | 14.1354 | 14.2360 | 1.7741 | 2.2115 | 2.0014 | -0.1150 | -0.0687 |
| USAGE_OF_BACKING_FILM_MEAN | 4951.3312 | 4813.8676 | 5097.6976 | 2903.0103 | 2868.9744 | 2869.0871 | -0.0476 | +0.0507 |
| AVG_REMOVAL_RATE | 76.5595 | 75.8591 | 76.9822 | 8.7241 | 8.9739 | 8.8326 | -0.0791 | +0.0482 |
| HEAD_ROTATION_MEAN | 159.9614 | 160.1362 | 159.7428 | 4.0362 | 0.3196 | 8.2778 | +0.0611 | -0.0336 |
| USAGE_OF_DRESSER_TABLE_MEAN | 3508.5131 | 3472.4125 | 3517.0404 | 486.9446 | 500.5086 | 480.6499 | -0.0731 | +0.0176 |
| STAGE_ROTATION_MEAN | 58.0553 | 57.4933 | 57.9648 | 5.7833 | 6.1309 | 7.6736 | -0.0943 | -0.0133 |

표준화차는 두 집단의 합동 표준편차로 나눈 평균 차이입니다.

- **평균 기준으로 크게 다른 항목은 없습니다.** test 의 표준화차는 모두 절대값 0.10 미만이고, 가장 큰 것이 USAGE_OF_DRESSER_MEAN 의 -0.0944 입니다. MRR 분포도 +0.0482 로 거의 같습니다.
- **다만 HEAD_ROTATION_MEAN 의 퍼짐이 세 데이터에서 크게 다릅니다.** 표준편차가 training 4.0362, validation 0.3196, test 8.2778 입니다. 평균은 거의 같은데 퍼짐만 다릅니다. V3 이 PV_proxy 의 한 축이므로 Preston 과 Hybrid 를 읽을 때 참고가 필요합니다.
- Stage B 비율은 training 0.5047, validation 0.4818, test 0.5299 로 test 가 Stage B 쪽이 조금 많습니다.
- **이 분포 차이를 보고 모델을 수정하지 않았습니다.**

## 7. 최종 연구 질문에 대한 답

**1) Pure ML 이 최종 test 에서도 가장 정확한가**

그렇습니다. RMSE 3.1856, R² 0.8699 로 세 모델 중 가장 좋습니다. training, validation, test 세 곳 모두에서 1위입니다.

**2) Fixed-k Preston 의 낮은 설명력이 test 에서도 반복되는가**

반복됩니다. test R² 가 **-0.0700** 으로 음수입니다. **평가 구간의 평균으로 예측하는 것보다도 나쁩니다.** training -0.0082, validation -0.0312 에 이어 세 번 모두 음수이고, test 에서 가장 낮습니다. 상수 k 하나로는 이 데이터의 MRR 변동을 설명하지 못합니다.

**3) k 를 k(x) 로 바꾼 Hybrid 의 성능 회복이 test 에서도 반복되는가**

반복됩니다. Hybrid 는 fixed-k 대비 RMSE 를 **5.6717 줄입니다**(9.1367 → 3.4650). fixed-k RMSE 대비 **62.1%** 를 줄인 것이고, training 59.2%, validation 62.5% 와 같은 수준입니다. **이 연구에서 가장 크고 가장 일관된 효과입니다.**

**4) Hybrid 가 Pure ML 보다 지불하는 정확도 trade-off 는 어느 정도인가**

test 에서 RMSE 0.2793, MAE 0.2086 입니다. 상대적으로는 Pure ML 대비 RMSE 가 약 8.8% 높습니다. training 에서는 0.2530, validation 에서는 0.1582 였습니다. **세 데이터 모두에서 같은 방향이고 크기도 0.16 에서 0.28 사이로 비슷합니다.**

**5) Hybrid 의 장점으로 주장할 수 있는 것과 없는 것**

주장할 수 있는 것입니다.

- 상수 k 를 쓰는 Preston 대비 **압도적인 성능 개선**입니다. 세 데이터 모두에서 RMSE 를 약 60% 줄입니다.
- **해석 가능한 구조**입니다. PV 가 압력과 회전을 담고 k(x) 가 소모품 사용량과 Stage 차이를 담는다는 Step 21 의 분해는 모델을 바꾸지 않았으므로 그대로 유효합니다.
- Pure ML 과 **큰 차이 없는 정확도**입니다. 외부 데이터에서 RMSE 차이가 0.28 수준입니다.

주장할 수 없는 것입니다.

- **Pure ML 보다 정확하다고 말할 수 없습니다.** 세 데이터 모두에서 뒤집니다.
- **조건 변화에 더 강하다고 말할 수 없습니다.** Step 25 에서 tuning 을 맞춘 뒤에는 17개 조건 그룹 전부와 shift 7개 중 6개에서 Pure ML 이 더 좋았습니다.
- **외부 일반화가 더 낫다고 말할 수 없습니다.** test − training 차이가 Pure ML -0.1336, Hybrid -0.1073 으로 비슷합니다.

**6) training 에서 만든 결론이 validation 과 test 에서 어느 정도 재현되는가**

**핵심 결론은 모두 재현됩니다.**

- 모델 순위가 세 데이터에서 동일합니다.
- fixed-k 의 음수 R² 가 세 번 모두 반복됩니다.
- Hybrid 의 회복 폭이 59.2%, 62.5%, 62.1% 로 거의 같습니다.
- Pure ML 과 Hybrid 의 격차가 0.2530, 0.1582, 0.2793 으로 같은 범위입니다.

**한계**

- 세 데이터가 wafer 수준에서 완전히 독립적이지 않습니다. wafer+Stage 키 겹침은 0건이지만 WAFER_ID 는 training 과 test 사이 113개, validation 과 test 사이 34개가 공유됩니다.
- 비교는 CHAMBER_PATH = 4-5-6 에 한정됩니다. 1-2-3 은 MRR 수준이 다르고 절반 이상이 PV_proxy 를 계산할 수 없어 세 모델을 같은 조건에서 비교할 수 없습니다.
- test 4-5-6 은 351개로 표본 변동이 있습니다.
- HEAD_ROTATION_MEAN 의 퍼짐이 세 데이터에서 다릅니다.
- **이번 평가 결과를 보고 어떤 수정도 하지 않았고, 추가 tuning 이나 모델 변경을 제안하지 않습니다.**

## 8. 그림

| 파일 | 내용 |
|---|---|
| `figures/fig1_train_val_test_rmse.png` | 세 모델의 training / validation / test RMSE |
| `figures/fig2_test_metrics.png` | test 에서 세 모델의 R² 와 RMSE |
| `figures/fig3_test_actual_vs_predicted.png` | test 실제 MRR 대 예측 MRR (Pure ML, Hybrid) |
| `figures/fig4_fixedk_hybrid_pure.png` | fixed-k → Hybrid → Pure ML 의 성능 차이 |
| `figures/fig5_distribution_comparison.png` | 세 데이터의 주요 분포 비교 |

## 9. 저장 파일

| 파일 | 내용 |
|---|---|
| `test_metrics.csv` | test 에서 세 모델의 R², MAE, RMSE |
| `test_stage_metrics.csv` | test 의 Stage A / B 별 결과 |
| `training_oof_reference.csv` | 같은 세 모델의 training 5-fold OOF 기준값 |
| `three_way_comparison.csv` | training / validation / test 비교와 세 가지 차이 |
| `dataset_structure.csv` | validation 과 test 의 CHAMBER_PATH 별 구조 |
| `distribution_comparison.csv` | 세 데이터의 주요 변수 분포 비교 |
| `stage_ratio_comparison.csv` | 세 데이터의 Stage 비율 |
| `data_independence.csv` | 세 쌍의 wafer+Stage 키 겹침과 WAFER_ID 공유 수 |
| `test_predictions.csv` | test 4-5-6 sample 별 실제값과 세 모델 예측 |
| `consistency_checks.csv` | 아래 확인 항목의 결과 |
| `data/processed/step27/step27_test_wafer_stage_table.csv` | test wafer + Stage 표 (1-2-3 포함) |

## 10. 확인한 내용

| 항목 | 값 |
|---|---|
| test 원본 파일 수 (폴더 전체) | 185 |
| test 원본 파일 수 (내용이 있는 파일) | 155 |
| test 원본 행 수 | 156,262 |
| 연속 완전중복으로 제외한 행 수 | 98 |
| test wafer+Stage 수 | 424 |
| test 정답 행 수 | 424 |
| MRR 이 비어 있는 test wafer+Stage 수 | 0 |
| test 4-5-6 sample 수 | 351 |
| 4-5-6 중 PV_proxy 를 계산할 수 없는 sample 수 | 0 |
| 4-5-6 중 MRR > 1,000 인 sample 수 (제외하지 않음) | 0 |
| 두 test 정답 파일이 동일한가 (데이터셋 루트 / 정답 폴더) | True |
| validation 표를 다시 만들어 Step 26 저장본과 같은가 | True |
| fixed k (training 4-5-6 전체) | 0.001290789418 |
| fixed k 가 Step 26 값과 같은가 | True |
| k 표준화 평균/표준편차가 Step 26 값과 같은가 | True |
| validation 성능이 Step 26 결과와 같은가 | True |
| 세 데이터 사이에 겹치는 wafer+Stage 키가 있는가 | False |
| test 예측 = k_pred x PV_proxy 인가 | True |

- 폴더의 185개 test 파일 중 30개는 내용이 없어 155개만 읽혔습니다.
- training 의 "MRR > 1,000 제외" 규칙은 정답을 보고 표본을 고르는 규칙이라 test 에도 적용하지 않았습니다. 해당 sample 은 0개였습니다.
- 코드를 두 번 실행해 15개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

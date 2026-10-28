# Step 26. 고정된 세 모델의 PHM validation 외부 평가

- 이번 단계에서 **처음으로 validation 데이터를 사용했습니다.** 지금까지 모델 선택과 feature 선택에 한 번도 쓰지 않은 데이터입니다.
- **validation 결과를 보고 feature, P/V 정의, preprocessing, 모델 설정, hyperparameter 를 바꾸지 않았습니다.** 평가만 했습니다.
- 세 모델은 training 4-5-6 의 1,609개 sample 전체로 한 번 학습했습니다.
- 코드: `scripts/step26_validation_evaluation.py`, 결과: `reports/step26_validation/`

## 1. 평가한 세 모델

| 모델 | 구성 | 학습에 쓴 것 |
|---|---|---|
| Model 1 Pure ML | Step 24 대표. XGBoost(900, depth 4, lr 0.05, mcw 3, subsample 0.9, colsample 0.9, seed 42) 로 MRR 직접 예측 | training 1,609개의 feature 와 MRR |
| Model 2 Fixed-k Preston | MRR = k × PV_proxy, 절편 없이 k = Σ(PV·MRR) / Σ(PV²) | training 1,609개로 k 한 번 계산 |
| Model 3 Hybrid | MRR = k(x) × PV_proxy, k(x) = XGBoost(900, depth 2, lr 0.1, mcw 1, subsample 0.9, colsample 0.9, seed 42) | training k_target 과 그 평균·표준편차 |

- PV_proxy = P9 × V3 입니다. P9 은 pressure 6개의 raw row 별 평균, V3 은 abs(HEAD_ROTATION) 이고, 두 값이 모두 0 이 아닌 raw row 에서만 P × V 를 평균했습니다.
- **fixed k = 0.001290789418** 이고 validation 에서 다시 맞추지 않았습니다.
- k 표준화에 쓴 평균 0.001293403292 와 표준편차 0.0001469023049 는 모두 training 에서만 계산해 validation 에 그대로 적용했습니다.

## 2. validation 전처리와 구조

training 에서 확정한 것과 같은 처리만 적용했습니다. 같은 wafer + Stage 단위, 같은 raw feature 계산 방식, 같은 feature 정의, 같은 P9 × V3 정의, 같은 연속 완전중복 행 제거 규칙입니다.

**training 의 "MRR > 1,000 인 wafer+Stage 제외" 규칙은 validation 에 적용하지 않았습니다.** 그 규칙은 정답을 보고 표본을 걸러내는 것이라 평가 대상에 쓰면 정답을 본 뒤 표본을 고르는 셈이 됩니다. 해당하는 validation sample 이 있는지는 개수만 확인했고, 0개였습니다.

| CHAMBER_PATH | sample 수 | wafer 수 | Stage A | Stage B | PV 계산 가능 | MRR 평균 | MRR 범위 |
|---|---|---|---|---|---|---|---|
| 1-2-3 | 66 | 66 | 66 | 0 | 33 | 151.1365 | 141.1039 ~ 161.0377 |
| 4-5-6 | 357 | 349 | 185 | 172 | 357 | 75.8591 | 53.4765 ~ 95.7897 |
| other | 1 | 1 | 1 | 0 | 1 | 145.6347 | 145.6347 ~ 145.6347 |

- 세 모델의 직접 비교는 **4-5-6 의 357개** 에서만 했습니다. 이 357개는 모두 PV_proxy 를 계산할 수 있습니다.
- **1-2-3 과 other 는 삭제하지 않았습니다.** 표에 구조만 보고하고 비교에서 제외했습니다. 1-2-3 은 전부 Stage A 이고 66개 중 33개만 PV_proxy 를 계산할 수 있습니다. MRR 평균이 151.14 로 4-5-6 의 75.86 과 크게 다른 영역입니다. Hybrid 나 Preston 을 억지로 적용하지 않았습니다.

## 3. validation 성능 (4-5-6, 357개)

| 모델 | sample 수 | R² | MAE | RMSE |
|---|---|---|---|---|
| Pure ML | 357 | 0.8684 | 2.4275 | 3.2560 |
| Fixed-k Preston | 357 | -0.0312 | 7.4405 | 9.1130 |
| Hybrid | 357 | 0.8553 | 2.6134 | 3.4142 |

**순위는 training 과 같습니다.** Pure ML, Hybrid, Fixed-k Preston 순입니다.

### Stage 별

| 모델 | Stage | sample 수 | R² | MAE | RMSE |
|---|---|---|---|---|---|
| Pure ML | A | 185 | 0.7735 | 2.3676 | 3.2282 |
| Pure ML | B | 172 | 0.8803 | 2.4919 | 3.2855 |
| Fixed-k Preston | A | 185 | -0.5376 | 6.6059 | 8.4118 |
| Fixed-k Preston | B | 172 | -0.0676 | 8.3382 | 9.8113 |
| Hybrid | A | 185 | 0.7490 | 2.5934 | 3.3989 |
| Hybrid | B | 172 | 0.8695 | 2.6350 | 3.4305 |

- 두 ML 모델 모두 Stage A 와 B 의 RMSE 차이가 0.06 이하로 작습니다. training 에서는 Stage B 가 뚜렷하게 나빴는데, validation 에서는 그 차이가 거의 사라졌습니다.
- R² 는 Stage A 가 낮게 보이지만, 이는 Stage A 의 MRR 분산이 더 작기 때문입니다. RMSE 로 보면 두 Stage 가 비슷합니다.

## 4. Generalization gap

| 모델 | training OOF RMSE | validation RMSE | RMSE 차이 | 비율 | MAE 차이 | R² 차이 |
|---|---|---|---|---|---|---|
| Pure ML | 3.3193 | 3.2560 | -0.0633 | 0.9809 | -0.1256 | +0.0131 |
| Fixed-k Preston | 8.7597 | 9.1130 | +0.3533 | 1.0403 | +0.4020 | -0.0230 |
| Hybrid | 3.5723 | 3.4142 | -0.1581 | 0.9557 | -0.1508 | +0.0229 |

- **외부 성능이 나빠진 모델은 Fixed-k Preston 뿐입니다** (+0.3533, 비율 1.0403).
- Pure ML 과 Hybrid 는 validation 에서 오히려 조금 좋아졌습니다.
- **이것을 "일반화가 더 잘 된다" 고 읽으면 안 됩니다.** training 수치는 5-fold OOF 라 매번 80% 로 학습한 모델의 평균이고, validation 수치는 100% 로 학습한 모델의 결과입니다. 또한 357개는 1,609개보다 표본 변동이 큽니다. 차이의 크기(0.06~0.16)는 이 두 요인만으로도 설명될 수 있는 범위입니다.
- 마찬가지로 이 차이를 과적합의 증거로 보기도 어렵습니다. 아래 분포 비교를 함께 봐야 합니다.

## 5. training 과 validation 의 분포 차이

| 데이터 | sample 수 | Stage A | Stage B | Stage B 비율 |
|---|---|---|---|---|
| training 4-5-6 | 1609 | 797 | 812 | 0.5047 |
| validation 4-5-6 | 357 | 185 | 172 | 0.4818 |

Stage B 비율 차이는 -0.0229 로 작습니다.

| 변수 | training 평균 | validation 평균 | training 표준편차 | validation 표준편차 | 표준화 평균차 |
|---|---|---|---|---|---|
| WAFER_ROTATION_MEAN | 14.3659 | 14.1354 | 1.7741 | 2.2115 | -0.1150 |
| PV_PROXY | 59247.6392 | 59537.9232 | 2703.1853 | 2833.1898 | +0.1048 |
| STAGE_ROTATION_MEAN | 58.0553 | 57.4933 | 5.7833 | 6.1309 | -0.0943 |
| AVG_REMOVAL_RATE | 76.5595 | 75.8591 | 8.7241 | 8.9739 | -0.0791 |
| USAGE_OF_DRESSER_TABLE_MEAN | 3508.5131 | 3472.4125 | 486.9446 | 500.5086 | -0.0731 |
| HEAD_ROTATION_MEAN | 159.9614 | 160.1362 | 4.0362 | 0.3196 | +0.0611 |
| USAGE_OF_BACKING_FILM_MEAN | 4951.3312 | 4813.8676 | 2903.0103 | 2868.9744 | -0.0476 |
| PRESSURIZED_CHAMBER_PRESSURE_MEAN | 56.1856 | 55.7695 | 15.0416 | 15.4601 | -0.0273 |
| USAGE_OF_POLISHING_TABLE_MEAN | 175.9128 | 178.0989 | 94.3200 | 94.8854 | +0.0231 |
| USAGE_OF_DRESSER_MEAN | 393.6721 | 398.4838 | 220.1822 | 215.0427 | +0.0221 |
| RETAINER_RING_PRESSURE_MEAN | 1372.0449 | 1367.2698 | 349.4993 | 363.6808 | -0.0134 |

표준화 평균차는 두 집단의 합동 표준편차로 나눈 값입니다.

- **평균 기준으로 크게 다른 항목은 없습니다.** 모든 변수의 표준화 평균차가 0.12 미만입니다. MRR 분포도 평균 차이가 -0.0791 로 작습니다.
- 다만 **HEAD_ROTATION_MEAN 은 퍼짐이 크게 다릅니다.** 평균은 거의 같은데 표준편차가 training 4.0362, validation 0.3196 으로 validation 이 약 8% 수준입니다. validation 의 head rotation 이 거의 한 값에 몰려 있다는 뜻이고, V3 이 PV_proxy 의 한 축이므로 참고가 필요합니다.
- validation 분포를 보고 모델을 다시 조정하지 않았습니다.

## 6. 해석

**1) training 에서 가장 좋았던 Pure ML 이 validation 에서도 가장 좋은가**

그렇습니다. RMSE 3.2560 으로 세 모델 중 가장 낮고 R² 도 0.8684 로 가장 높습니다. 세 모델의 순위가 training 과 동일합니다.

**2) Hybrid 가 Pure ML 과 어느 정도 차이를 유지하는가**

**차이가 유지되지만 조금 줄었습니다.** training OOF 에서 RMSE 차이가 0.2530 이었는데 validation 에서는 0.1582 입니다. 순위는 그대로이고, 줄어든 폭(0.09)은 두 모델의 차이 자체보다 작습니다.

**3) Fixed-k Preston 대비 Hybrid 의 성능 회복이 외부 데이터에서도 유지되는가**

유지됩니다. Hybrid 는 fixed-k 대비 RMSE 를 training 에서 5.1873, validation 에서 5.6988 줄입니다. fixed-k RMSE 대비 비율로는 59.2% 에서 62.5% 로, 외부 데이터에서도 같은 수준입니다. **k 를 상수로 두는 것과 k(x) 로 두는 것의 차이는 외부에서도 그대로 큽니다.**

**4) Hybrid 의 해석 가능한 구조와 성능 trade-off 가 validation 에서도 유지되는가**

유지됩니다. Hybrid 는 여전히 Pure ML 보다 RMSE 가 0.1582 높습니다. 구조를 나눈 대가를 외부에서도 지불하지만, 그 크기는 training 에서 본 것보다 작습니다. Step 21 에서 확인한 해석 구조는 모델을 바꾸지 않았으므로 그대로 유효합니다.

**5) training 과 validation 사이에 성능이 크게 달라지는 모델이 있는가**

**Fixed-k Preston 만 나빠집니다** (+0.3533). 두 ML 모델은 오히려 조금 좋아집니다. 어느 모델도 급격히 무너지지 않습니다. 세 모델 모두 순위와 규모가 training 에서 본 것과 일관됩니다.

**한계**

- **validation 은 wafer 수준에서 완전히 독립적이지 않습니다.** 같은 wafer+Stage 기록이 양쪽에 들어간 경우는 0건이지만, 같은 WAFER_ID 가 115개 공유됩니다. validation 에서는 Stage A, training 에서는 Stage B 처럼 서로 다른 Stage 로 나타납니다.
- validation 4-5-6 은 357개로 training 1,609개보다 작아 표본 변동이 큽니다.
- 비교는 4-5-6 에 한정됩니다. 1-2-3 은 MRR 수준 자체가 다른 영역이고 절반은 PV_proxy 를 계산할 수 없어 세 모델을 같은 조건에서 비교할 수 없습니다.
- 이 단계에서는 어떤 수정도 하지 않았습니다. 추가 tuning 이나 모델 변경을 제안하지 않습니다.

## 7. 그림

| 파일 | 내용 |
|---|---|
| `figures/fig1_training_vs_validation_rmse.png` | 세 모델의 training OOF 대 validation RMSE |
| `figures/fig2_validation_metrics.png` | validation 에서 세 모델의 R² 와 RMSE |
| `figures/fig3_actual_vs_predicted.png` | validation 실제 MRR 대 예측 MRR (Pure ML, Hybrid) |
| `figures/fig4_distribution_comparison.png` | training 과 validation 의 주요 분포 비교 |

## 8. 저장 파일

| 파일 | 내용 |
|---|---|
| `validation_structure.csv` | CHAMBER_PATH 별 sample 수, Stage 구성, PV 계산 가능 수, MRR 범위 |
| `validation_metrics.csv` | 세 모델의 validation R², MAE, RMSE |
| `validation_stage_metrics.csv` | Stage A / B 별 결과 |
| `training_oof_reference.csv` | 같은 세 모델의 training 5-fold OOF 기준값 |
| `generalization_gap.csv` | validation 과 training OOF 의 차이와 비율 |
| `distribution_comparison.csv` | 주요 변수의 training 대 validation 분포 비교 |
| `stage_ratio_comparison.csv` | Stage A / B 비율 비교 |
| `validation_predictions.csv` | validation 4-5-6 sample 별 실제값과 세 모델 예측 |
| `consistency_checks.csv` | 아래 확인 항목의 결과 |
| `data/processed/step26/step26_validation_wafer_stage_table.csv` | validation wafer + Stage 표 (1-2-3 과 other 포함) |

## 9. 확인한 내용

| 항목 | 값 |
|---|---|
| validation 원본 파일 수 (폴더 전체) | 185 |
| validation 원본 파일 수 (내용이 있는 파일) | 158 |
| validation 원본 행 수 | 144,148 |
| 연속 완전중복으로 제외한 행 수 | 434 |
| validation wafer+Stage 수 | 424 |
| 정답 파일 행 수 | 424 |
| MRR 이 비어 있는 wafer+Stage 수 | 0 |
| validation 4-5-6 sample 수 | 357 |
| 4-5-6 중 MRR > 1,000 인 sample 수 (제외하지 않음) | 0 |
| 4-5-6 중 PV_proxy 를 계산할 수 없는 sample 수 | 0 |
| validation feature 열이 training 과 같은가 | True |
| fixed k (training 4-5-6 전체) | 0.001290789418 |
| k 표준화 평균 (training) | 0.001293403292 |
| k 표준화 표준편차 (training) | 0.0001469023049 |
| Hybrid 예측 = k_pred x PV_proxy 인가 | True |
| validation 과 training 이 공유하는 wafer+Stage 키 수 | 0 |
| validation 과 training 이 공유하는 WAFER_ID 수 (Stage 는 서로 다름) | 115 |

- 폴더의 185개 파일 중 27개는 내용이 없어 158개만 읽혔습니다.
- 코드를 두 번 실행해 13개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

# Step 22. Pure ML 과 Hybrid 의 robustness 비교

- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold
- training 만 사용했습니다. test / validation 은 사용하지 않았습니다.
- 새 feature, 모델 구조 변경, 추가 튜닝을 하지 않았습니다. **새로 학습하지도 않았습니다.**
  이미 저장된 5-fold 예측을 조건 그룹별로 다시 집계한 분석입니다.
- 코드: `scripts/step22_robustness_comparison.py`, 결과: `reports/step22_robustness/`

## 1. 비교한 두 모델

| 모델 | 내용 | 예측을 가져온 곳 |
|---|---|---|
| Pure ML | Step 16 의 Model 1 XGBoost. 공정 feature x 로 MRR 을 직접 예측합니다. | `reports/step16_model3_prep/oof_predictions.csv` |
| Hybrid | Step 20~21 의 대표 Hybrid. MRR = k(x) × PV_proxy, PV_proxy = P9 × V3 | `reports/step20_hybrid_xgb_tuning/oof_top/oof_n900_d2_lr0.1_mcw1.csv` |

- P9 = 6개 pressure 의 raw row 별 평균, V3 = abs(HEAD_ROTATION), 두 값이 모두 0 이 아닌 row 의 평균
- k(x): XGBoost (n_estimators 900, max_depth 2, learning_rate 0.1, min_child_weight 1, subsample 0.9, colsample_bytree 0.9, random_state 42)
- k 표준화는 Step 18~21 과 같이 training fold 통계만 사용한 값이 그대로 반영되어 있습니다.

## 2. 전체 성능 재확인

| 모델 | R² | MAE | RMSE |
|---|---|---|---|
| Pure ML | 0.8309 | 2.7454 | 3.5872 |
| Hybrid | 0.8323 | 2.7643 | 3.5723 |

- Hybrid 가 RMSE 는 0.0149 낮고 MAE 는 0.0188 높습니다. **평균 성능은 실질적으로 비슷합니다.**
- Pure ML 의 전체 지표는 Step 16 저장값과, Hybrid 의 Stage 별 지표는 Step 20 저장값과 완전히 같습니다.
- 두 모델의 예측값 모두 Step 21 에 저장된 값과 완전히 같습니다.

## 3. Stage 별 robustness

| Stage | sample 수 | Pure ML R² | Pure ML MAE | Pure ML RMSE | Hybrid R² | Hybrid MAE | Hybrid RMSE |
|---|---|---|---|---|---|---|---|
| stage = A | 797 | 0.7769 | 2.4045 | 3.1414 | 0.7507 | 2.5762 | 3.3210 |
| stage = B | 812 | 0.8113 | 3.0801 | 3.9765 | 0.8274 | 2.9489 | 3.8028 |

- **Pure ML 은 Stage A 에서 더 좋고 Stage B 에서 더 나쁩니다.** 두 Stage 의 RMSE 차이가 0.8351 입니다.
- **Hybrid 는 두 Stage 의 차이가 0.4818 로 더 작습니다.** Stage A 를 조금 손해 보고 Stage B 를 그만큼 얻는 형태입니다.
- 한쪽 Stage 에서만 성능이 크게 무너지는 모델은 없습니다. 두 모델 모두 Stage B 가 더 나쁩니다.

## 4. usage 수준별 robustness

구간 경계는 전체 1,609개의 33% / 67% 분위수입니다. 경계값 이하가 low, 67% 분위수 이하가 mid, 그보다 크면 high 입니다.

| usage 변수 | 33% 경계 | 67% 경계 | low | mid | high | 최소 | 최대 |
|---|---|---|---|---|---|---|---|
| USAGE_OF_DRESSER_MEAN | 257.3338 | 542.4459 | 531 | 547 | 531 | 5.3432 | 771.6568 |
| USAGE_OF_POLISHING_TABLE_MEAN | 116.8934 | 233.6182 | 531 | 547 | 531 | 14.7605 | 355.3130 |
| USAGE_OF_DRESSER_TABLE_MEAN | 3287.7711 | 3821.2994 | 531 | 547 | 531 | 2664.8638 | 4305.3651 |
| USAGE_OF_BACKING_FILM_MEAN | 3262.8203 | 6506.1563 | 531 | 547 | 531 | 24.7321 | 10525.9456 |

| usage 구간 | sample 수 | Pure ML R² | Pure ML MAE | Pure ML RMSE | Hybrid R² | Hybrid MAE | Hybrid RMSE |
|---|---|---|---|---|---|---|---|
| USAGE_OF_DRESSER_MEAN = low | 531 | 0.7316 | 2.7904 | 3.5671 | 0.7237 | 2.8290 | 3.6188 |
| USAGE_OF_DRESSER_MEAN = mid | 547 | 0.7833 | 2.7359 | 3.6567 | 0.7853 | 2.7959 | 3.6392 |
| USAGE_OF_DRESSER_MEAN = high | 531 | 0.8531 | 2.7103 | 3.5346 | 0.8597 | 2.6669 | 3.4540 |
| USAGE_OF_POLISHING_TABLE_MEAN = low | 531 | 0.8285 | 2.8375 | 3.6999 | 0.8387 | 2.7358 | 3.5878 |
| USAGE_OF_POLISHING_TABLE_MEAN = mid | 547 | 0.8213 | 2.7910 | 3.6478 | 0.8145 | 2.8789 | 3.7167 |
| USAGE_OF_POLISHING_TABLE_MEAN = high | 531 | 0.8432 | 2.6065 | 3.4052 | 0.8436 | 2.6747 | 3.4010 |
| USAGE_OF_DRESSER_TABLE_MEAN = low | 531 | 0.7687 | 2.8188 | 3.6574 | 0.7775 | 2.7637 | 3.5870 |
| USAGE_OF_DRESSER_TABLE_MEAN = mid | 547 | 0.7955 | 2.5820 | 3.3421 | 0.7873 | 2.6775 | 3.4080 |
| USAGE_OF_DRESSER_TABLE_MEAN = high | 531 | 0.7233 | 2.8404 | 3.7563 | 0.7287 | 2.8543 | 3.7199 |
| USAGE_OF_BACKING_FILM_MEAN = low | 531 | 0.8736 | 2.7842 | 3.5585 | 0.8642 | 2.8991 | 3.6885 |
| USAGE_OF_BACKING_FILM_MEAN = mid | 547 | 0.7659 | 2.8313 | 3.8027 | 0.7797 | 2.8674 | 3.6888 |
| USAGE_OF_BACKING_FILM_MEAN = high | 531 | 0.8254 | 2.6183 | 3.3813 | 0.8313 | 2.5232 | 3.3237 |

- **usage 가 높은 구간 4개 모두에서 Hybrid 의 RMSE 가 Pure ML 보다 낮습니다.** 차이는 −0.0042 에서 −0.0806 사이입니다.
- usage 가 높다고 두 모델이 항상 나빠지지는 않습니다. USAGE_OF_DRESSER_TABLE_MEAN 만 high 구간이 전체보다 나쁘고(Pure ML +0.1690, Hybrid +0.1476), 나머지 세 변수는 high 구간이 전체보다 좋습니다.
- 즉 **usage 가 높은 쪽에서 성능이 더 많이 나빠지는 모델은 Pure ML 쪽이며, 그 차이는 크지 않습니다.**
- R² 는 그 구간 안의 MRR 분산으로 계산되므로 구간끼리 직접 비교하기 어렵습니다. 구간 비교에는 RMSE 와 MAE 를 사용했습니다.

## 5. 시간 순서별 robustness

Step 7 과 같은 제외 규칙을 적용해 각 sample 의 첫 기록 시각을 구하고, 그 순서로 1,609개를 3등분했습니다.

| 구간 | sample 수 | 첫 기록 시각 최소 | 첫 기록 시각 최대 |
|---|---|---|---|
| early | 537 | 481634409.667 | 483795098.000 |
| middle | 536 | 483816441.667 | 485502690.667 |
| late | 536 | 485503020.333 | 487267851.333 |

| 시간 구간 | sample 수 | Pure ML R² | Pure ML MAE | Pure ML RMSE | Hybrid R² | Hybrid MAE | Hybrid RMSE |
|---|---|---|---|---|---|---|---|
| time = early | 537 | 0.7696 | 2.8145 | 3.6501 | 0.7784 | 2.7618 | 3.5796 |
| time = middle | 536 | 0.7862 | 2.5998 | 3.3603 | 0.7781 | 2.6848 | 3.4235 |
| time = late | 536 | 0.7251 | 2.8219 | 3.7402 | 0.7298 | 2.8462 | 3.7081 |

- 두 모델 모두 late 구간이 가장 나쁘고 middle 구간이 가장 좋습니다.
- **late 구간에서 Hybrid 가 0.0321 낮고, early 구간에서도 0.0705 낮습니다.** middle 구간에서만 Pure ML 이 0.0633 낮습니다.
- 이 분석은 기존 5-fold 예측을 시간 구간별로 다시 집계한 것입니다. 시간 기반 모델 학습은 하지 않았습니다.

## 6. 조건별 성능 저하량 (delta_RMSE)

delta_RMSE = 그 그룹의 RMSE − 그 모델의 전체 RMSE 입니다. 양수는 전체보다 나쁘다는 뜻입니다.

| 조건 그룹 | sample 수 | Pure ML delta_RMSE | Hybrid delta_RMSE | Hybrid − Pure ML RMSE |
|---|---|---|---|---|
| stage = A | 797 | -0.4458 | -0.2513 | +0.1797 |
| stage = B | 812 | +0.3893 | +0.2305 | -0.1737 |
| time = early | 537 | +0.0629 | +0.0073 | -0.0705 |
| time = late | 536 | +0.1530 | +0.1358 | -0.0321 |
| time = middle | 536 | -0.2270 | -0.1488 | +0.0633 |
| USAGE_OF_BACKING_FILM_MEAN = high | 531 | -0.2059 | -0.2486 | -0.0576 |
| USAGE_OF_BACKING_FILM_MEAN = low | 531 | -0.0288 | +0.1161 | +0.1300 |
| USAGE_OF_BACKING_FILM_MEAN = mid | 547 | +0.2155 | +0.1164 | -0.1139 |
| USAGE_OF_DRESSER_MEAN = high | 531 | -0.0526 | -0.1184 | -0.0806 |
| USAGE_OF_DRESSER_MEAN = low | 531 | -0.0201 | +0.0465 | +0.0517 |
| USAGE_OF_DRESSER_MEAN = mid | 547 | +0.0695 | +0.0669 | -0.0175 |
| USAGE_OF_DRESSER_TABLE_MEAN = high | 531 | +0.1690 | +0.1476 | -0.0364 |
| USAGE_OF_DRESSER_TABLE_MEAN = low | 531 | +0.0702 | +0.0147 | -0.0704 |
| USAGE_OF_DRESSER_TABLE_MEAN = mid | 547 | -0.2451 | -0.1643 | +0.0659 |
| USAGE_OF_POLISHING_TABLE_MEAN = high | 531 | -0.1820 | -0.1713 | -0.0042 |
| USAGE_OF_POLISHING_TABLE_MEAN = low | 531 | +0.1127 | +0.0154 | -0.1121 |
| USAGE_OF_POLISHING_TABLE_MEAN = mid | 547 | +0.0606 | +0.1444 | +0.0689 |

전체 17개 그룹에 대한 흔들림 요약입니다.

| 모델 | 전체 RMSE | 그룹 최소 | 그룹 최대 | 최대 − 최소 | 그룹 RMSE 표준편차 | 평균 절대 delta_RMSE | 최대 delta_RMSE |
|---|---|---|---|---|---|---|---|
| Pure ML | 3.5872 | 3.1414 | 3.9765 | 0.8351 | 0.1984 | 0.1594 | +0.3893 |
| Hybrid | 3.5723 | 3.3210 | 3.8028 | 0.4818 | 0.1466 | 0.1261 | +0.2305 |

- **세 가지 흔들림 지표 모두 Hybrid 가 더 작습니다.** 폭은 Pure ML 의 57.7%, 표준편차는 73.9%, 평균 절대 delta_RMSE 는 79.1% 입니다.
- 17개 그룹 중 11개에서 Hybrid 의 RMSE 가 더 낮습니다.

## 7. Worst-group 분석

각 모델에서 Stage, usage 구간, 시간 구간을 합친 17개 그룹 중 RMSE 가 가장 높은 그룹입니다.

| 모델 | 순위 | 조건 그룹 | sample 수 | RMSE | 전체 RMSE와 차이 | 다른 모델 RMSE | 차이 |
|---|---|---|---|---|---|---|---|
| Pure ML | 1 | stage = B | 812 | 3.9765 | +0.3893 | 3.8028 | +0.1737 |
| Pure ML | 2 | USAGE_OF_BACKING_FILM_MEAN = mid | 547 | 3.8027 | +0.2155 | 3.6888 | +0.1139 |
| Pure ML | 3 | USAGE_OF_DRESSER_TABLE_MEAN = high | 531 | 3.7563 | +0.1690 | 3.7199 | +0.0364 |
| Pure ML | 4 | time = late | 536 | 3.7402 | +0.1530 | 3.7081 | +0.0321 |
| Pure ML | 5 | USAGE_OF_POLISHING_TABLE_MEAN = low | 531 | 3.6999 | +0.1127 | 3.5878 | +0.1121 |
| Hybrid | 1 | stage = B | 812 | 3.8028 | +0.2305 | 3.9765 | -0.1737 |
| Hybrid | 2 | USAGE_OF_DRESSER_TABLE_MEAN = high | 531 | 3.7199 | +0.1476 | 3.7563 | -0.0364 |
| Hybrid | 3 | USAGE_OF_POLISHING_TABLE_MEAN = mid | 547 | 3.7167 | +0.1444 | 3.6478 | +0.0689 |
| Hybrid | 4 | time = late | 536 | 3.7081 | +0.1358 | 3.7402 | -0.0321 |
| Hybrid | 5 | USAGE_OF_BACKING_FILM_MEAN = mid | 547 | 3.6888 | +0.1164 | 3.8027 | -0.1139 |

- **두 모델의 worst group 은 stage = B 로 같습니다** (sample 812개).
- worst-group RMSE 는 Pure ML 3.9765, Hybrid 3.8028 로 **Hybrid 가 0.1737 낮습니다.**
- 전체 RMSE 와의 차이도 Pure ML +0.3893, Hybrid +0.2305 로 Hybrid 가 작습니다.
- 상위 5개 중 4개 그룹(stage = B, dresser table high, time late, backing film mid)이 두 모델에 공통으로 나타납니다.

## 8. 해석

**1) 조건 변화에 더 안정적인 모델이 있는가**

Hybrid 입니다. 17개 조건 그룹의 RMSE 폭이 0.4818 로 Pure ML 의 0.8351 보다 작고, 표준편차와 평균 절대 delta_RMSE 도 더 작습니다. worst group 의 RMSE 도 0.1737 낮습니다. 다만 **전체 평균 성능은 RMSE 차이 0.0149 로 실질적으로 비슷합니다.** 즉 차이는 평균이 아니라 조건별 편차에서 나타납니다.

**2) Hybrid 가 usage 나 시간 조건에서도 안정적인가**

대체로 그렇습니다. usage 가 높은 구간 4개 모두에서 Hybrid 의 RMSE 가 더 낮고, 시간 early 와 late 구간에서도 더 낮습니다. 예외는 시간 middle 구간(+0.0633)과 usage 가 낮거나 중간인 일부 구간입니다. **모든 조건에서 Hybrid 가 낫다는 뜻은 아니고, 17개 중 11개에서 낫습니다.**

**3) 특정 조건에서 유리하거나 불리한 패턴**

패턴이 뚜렷합니다. **Pure ML 이 잘하는 그룹에서는 Hybrid 가 조금 못하고, Pure ML 이 못하는 그룹에서는 Hybrid 가 낫습니다.**

- Hybrid 가 가장 불리한 그룹: stage = A (+0.1797), backing film low (+0.1300), polishing table mid (+0.0689). 이 중 stage = A 와 backing film low 는 Pure ML 의 성능이 전체보다 좋은 구간입니다.
- Hybrid 가 가장 유리한 그룹: stage = B (−0.1737), backing film mid (−0.1139), polishing table low (−0.1121). 모두 Pure ML 의 성능이 전체보다 나쁜 구간입니다.
- polishing table high 처럼 차이가 −0.0042 인 그룹은 **두 모델이 사실상 같습니다.**

**4) 평균 성능 차이보다 robustness 차이가 더 큰 조건**

거의 모든 조건이 그렇습니다. 전체 RMSE 차이는 0.0149 인데, **17개 그룹 중 16개에서 두 모델의 RMSE 차이가 그보다 큽니다.** 가장 큰 곳은 Stage 로, Stage A 에서 +0.1797, Stage B 에서 −0.1737 입니다. 평균만 보면 두 모델이 같아 보이지만 조건을 나누면 서로 다른 곳에서 강하고 약합니다.

**한계**

- 이 결과는 하나의 5-fold 분할에 대한 out-of-fold 예측을 다시 집계한 것입니다. 구간 경계(33% / 67% 분위수, 시간 3등분)를 바꾸면 수치가 달라질 수 있습니다.
- 그룹별 R² 는 그 그룹의 MRR 분산에 따라 달라지므로 그룹끼리 비교하지 않았습니다.
- 모든 서술은 인과가 아니라 조건별 예측 오차의 차이입니다.

## 9. 그림

| 파일 | 내용 |
|---|---|
| `figures/fig1_stage_rmse.png` | Stage 별 RMSE: Pure ML 대 Hybrid |
| `figures/fig2_usage_rmse.png` | usage 변수 4개의 low / mid / high 구간 RMSE 비교 |
| `figures/fig3_time_rmse.png` | 시간 early / middle / late 구간 RMSE 비교 |
| `figures/fig4_overall_vs_worst_group.png` | 전체 RMSE 와 worst-group RMSE 비교 |

## 10. 저장 파일

| 파일 | 내용 |
|---|---|
| `overall_metrics.csv` | 두 모델의 전체 R², MAE, RMSE |
| `stage_metrics.csv` | Stage 별 지표와 delta_RMSE |
| `usage_group_metrics.csv` | usage 구간별 지표와 delta_RMSE |
| `time_group_metrics.csv` | 시간 구간별 지표와 delta_RMSE |
| `group_metrics_all.csv` | 17개 그룹 × 2개 모델의 지표 전체 |
| `model_difference_by_group.csv` | 그룹별 두 모델의 지표와 차이 |
| `robustness_summary.csv` | 모델별 흔들림 요약과 worst / best group |
| `worst_groups.csv` | 모델별 RMSE 상위 5개 그룹과 상대 모델 비교 |
| `usage_bin_edges.csv` | usage 구간 경계와 구간별 sample 수 |
| `time_bin_info.csv` | 시간 구간별 sample 수와 첫 기록 시각 범위 |
| `sample_group_assignment.csv` | sample 별 그룹 배정, 실제 MRR, 두 모델의 예측 |
| `consistency_checks.csv` | 아래 확인 항목의 결과 |
| `data/processed/step22/step22_first_timestamp.csv` | wafer+Stage 별 첫 기록 시각 |

## 11. 확인한 내용

아홉 항목 모두 True 입니다.

- sample 이 1,609개이고 CHAMBER_PATH 가 모두 4-5-6 입니다.
- FOLD 가 Step 8 배정과 같습니다.
- 두 모델의 예측이 Step 21 저장값과 같습니다.
- Hybrid 예측이 k_pred × PV_proxy 와 같습니다.
- Pure ML 전체 지표가 Step 16 저장값과 같습니다.
- Hybrid 의 Stage 별 지표가 Step 20 저장값과 같습니다.
- 첫 기록 시각으로 정렬한 순서가 Step 7 표의 행 순서와 같습니다.
- 코드를 두 번 실행해 16개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

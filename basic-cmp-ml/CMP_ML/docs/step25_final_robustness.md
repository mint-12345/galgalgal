# Step 25. tuning 이 끝난 최종 후보끼리의 robustness 재비교

- 목적: Step 22 와 Step 23 의 robustness 비교는 **기본 설정 Pure ML** 을 사용했습니다. Step 24 에서 Pure ML 을 tuning 했으므로, 확정된 두 설정으로 같은 비교를 다시 합니다.
- 대상: CHAMBER_PATH = 4-5-6 의 training 1,609개, Step 8 의 같은 5-fold
- training 만 사용했습니다. test / validation 은 사용하지 않았습니다.
- **새 feature 생성, P/V 변경, hyperparameter tuning 을 하지 않았습니다.** 두 설정 모두 이전 단계에서 확정된 값입니다.
- 코드: `scripts/step25_final_robustness.py`, 결과: `reports/step25_final_robustness/`

## 1. 비교한 두 모델

| 모델 | 구성 |
|---|---|
| Pure ML | Step 24 tuning 1위. `XGBRegressor(n_estimators=900, max_depth=4, learning_rate=0.05, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, random_state=42)` 로 MRR 직접 예측 |
| Hybrid | Step 20~23 대표. MRR = k(x) × PV_proxy, PV_proxy = P9 × V3, k(x) = XGBoost(900, depth 2, lr 0.1, mcw 1, subsample 0.9, colsample 0.9, random_state 42) |

- feature 는 두 모델 모두 같은 31개입니다. Hybrid 의 k 표준화는 학습 데이터 통계만 사용했습니다.
- 5-fold 성능과 조건 그룹 분석은 **이미 저장된 out-of-fold 예측을 재집계**했습니다. 다시 학습하지 않았습니다.
- distribution-shift 시험만 split 마다 두 모델을 새로 학습했습니다.

## 2. 기본 5-fold 성능

| 모델 | pooled R² | pooled MAE | pooled RMSE | 5-fold 평균 RMSE | 5-fold RMSE 표준편차 |
|---|---|---|---|---|---|
| Pure ML | 0.8552 | 2.5531 | 3.3193 | 3.3163 | 0.1511 |
| Hybrid | 0.8323 | 2.7643 | 3.5723 | 3.5693 | 0.1573 |

- Pure ML 의 5-fold 평균 RMSE 는 Step 24 저장값과, Hybrid 는 Step 20 저장값과 같습니다. Hybrid 의 pooled RMSE 는 Step 22 저장값과 같습니다.
- 두 모델의 nominal 차이는 pooled 기준 0.2530 입니다.

## 3. Step 22 방식 조건 그룹 재평가 (17개 그룹)

구간 정의는 Step 22 와 완전히 같습니다. Step 22 가 저장한 구간 배정 파일을 그대로 사용했고, 17개 그룹의 sample 수가 Step 22 저장값과 일치합니다.

| 조건 그룹 | sample 수 | Pure ML R² | Pure ML MAE | Pure ML RMSE | Hybrid R² | Hybrid MAE | Hybrid RMSE | Pure ML delta | Hybrid delta | 차이 | 더 낮은 쪽 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage = A | 797 | 0.7952 | 2.3035 | 3.0097 | 0.7507 | 2.5762 | 3.3210 | -0.3095 | -0.2513 | +0.3113 | Pure ML |
| stage = B | 812 | 0.8456 | 2.7981 | 3.5973 | 0.8274 | 2.9489 | 3.8028 | +0.2780 | +0.2305 | +0.2056 | Pure ML |
| USAGE_OF_DRESSER_MEAN = low | 531 | 0.7730 | 2.5386 | 3.2800 | 0.7237 | 2.8290 | 3.6188 | -0.0393 | +0.0465 | +0.3388 | Pure ML |
| USAGE_OF_DRESSER_MEAN = mid | 547 | 0.8013 | 2.6850 | 3.5013 | 0.7853 | 2.7959 | 3.6392 | +0.1821 | +0.0669 | +0.1379 | Pure ML |
| USAGE_OF_DRESSER_MEAN = high | 531 | 0.8824 | 2.4317 | 3.1619 | 0.8597 | 2.6669 | 3.4540 | -0.1574 | -0.1184 | +0.2921 | Pure ML |
| USAGE_OF_POLISHING_TABLE_MEAN = low | 531 | 0.8526 | 2.6224 | 3.4293 | 0.8387 | 2.7358 | 3.5878 | +0.1101 | +0.0154 | +0.1584 | Pure ML |
| USAGE_OF_POLISHING_TABLE_MEAN = mid | 547 | 0.8472 | 2.5852 | 3.3726 | 0.8145 | 2.8789 | 3.7167 | +0.0534 | +0.1444 | +0.3441 | Pure ML |
| USAGE_OF_POLISHING_TABLE_MEAN = high | 531 | 0.8660 | 2.4508 | 3.1475 | 0.8436 | 2.6747 | 3.4010 | -0.1717 | -0.1713 | +0.2535 | Pure ML |
| USAGE_OF_DRESSER_TABLE_MEAN = low | 531 | 0.8008 | 2.5687 | 3.3942 | 0.7775 | 2.7637 | 3.5870 | +0.0750 | +0.0147 | +0.1928 | Pure ML |
| USAGE_OF_DRESSER_TABLE_MEAN = mid | 547 | 0.8324 | 2.3962 | 3.0250 | 0.7873 | 2.6775 | 3.4080 | -0.2942 | -0.1643 | +0.3830 | Pure ML |
| USAGE_OF_DRESSER_TABLE_MEAN = high | 531 | 0.7561 | 2.6991 | 3.5267 | 0.7287 | 2.8543 | 3.7199 | +0.2074 | +0.1476 | +0.1932 | Pure ML |
| USAGE_OF_BACKING_FILM_MEAN = low | 531 | 0.8907 | 2.5995 | 3.3082 | 0.8642 | 2.8991 | 3.6885 | -0.0110 | +0.1161 | +0.3802 | Pure ML |
| USAGE_OF_BACKING_FILM_MEAN = mid | 547 | 0.7941 | 2.6974 | 3.5663 | 0.7797 | 2.8674 | 3.6888 | +0.2470 | +0.1164 | +0.1225 | Pure ML |
| USAGE_OF_BACKING_FILM_MEAN = high | 531 | 0.8574 | 2.3581 | 3.0559 | 0.8313 | 2.5232 | 3.3237 | -0.2634 | -0.2486 | +0.2678 | Pure ML |
| time = early | 537 | 0.8022 | 2.5591 | 3.3815 | 0.7784 | 2.7618 | 3.5796 | +0.0622 | +0.0073 | +0.1981 | Pure ML |
| time = middle | 536 | 0.8241 | 2.4206 | 3.0482 | 0.7781 | 2.6848 | 3.4235 | -0.2710 | -0.1488 | +0.3753 | Pure ML |
| time = late | 536 | 0.7577 | 2.6797 | 3.5108 | 0.7298 | 2.8462 | 3.7081 | +0.1915 | +0.1358 | +0.1973 | Pure ML |

**17개 그룹 전부에서 Pure ML 의 RMSE 가 더 낮습니다.** 차이는 +0.1225 에서 +0.3830 사이입니다.

| 모델 | 그룹 최소 RMSE | 그룹 최대 RMSE | 범위 | 그룹 RMSE 표준편차 | 평균 절대 delta_RMSE | worst group | worst-group RMSE |
|---|---|---|---|---|---|---|---|
| Pure ML | 3.0097 | 3.5973 | 0.5875 | 0.1964 | 0.1720 | stage = B | 3.5973 |
| Hybrid | 3.3210 | 3.8028 | 0.4818 | 0.1466 | 0.1261 | stage = B | 3.8028 |

- **흔들림 지표 세 가지는 Hybrid 가 더 작습니다.** 범위 0.4818 대 0.5875, 표준편차 0.1466 대 0.1964, 평균 절대 delta_RMSE 0.1261 대 0.1720 입니다.
- 그러나 **Hybrid 는 17개 그룹 모두에서 더 높은 RMSE 를 가진 채로 평탄합니다.** 편차가 작다는 것이 성능이 좋다는 뜻은 아닙니다.
- worst-group 은 두 모델 모두 stage = B 이고, 그 값은 Pure ML 이 0.2056 낮습니다.

## 4. Step 23 방식 distribution-shift 재평가 (7개 test)

split 정의는 Step 23 과 같은 규칙으로 다시 만들었고, 7개 split 의 평가 sample 이 Step 23 과 완전히 같음을 확인했습니다. Hybrid 를 다시 학습한 결과도 Step 23 저장값과 같습니다. **따라서 이번 단계에서 바뀐 것은 Pure ML 설정뿐입니다.**

| test | 학습 sample | 평가 sample | Pure ML R² | Pure ML MAE | Pure ML RMSE | Hybrid R² | Hybrid MAE | Hybrid RMSE | 차이 | 더 낮은 쪽 |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 655 | 313 | -2.1422 | 9.6760 | 11.1933 | -1.3774 | 8.4138 | 9.7362 | -1.4571 | Hybrid |
| T2 | 968 | 317 | 0.5521 | 3.8978 | 4.8325 | 0.4805 | 4.2241 | 5.2045 | +0.3720 | Pure ML |
| T3 | 1285 | 324 | 0.1487 | 5.2213 | 6.4139 | 0.0752 | 5.5758 | 6.6850 | +0.2711 | Pure ML |
| U1 | 1067 | 542 | 0.6013 | 4.5345 | 5.7928 | 0.5749 | 4.8249 | 5.9815 | +0.1887 | Pure ML |
| U2 | 1089 | 520 | 0.8209 | 2.8723 | 3.6229 | 0.7799 | 3.1949 | 4.0159 | +0.3930 | Pure ML |
| U3 | 1070 | 539 | 0.3008 | 4.6489 | 5.9533 | 0.2932 | 4.7516 | 5.9855 | +0.0322 | Pure ML |
| U4 | 1066 | 543 | 0.6481 | 3.6267 | 4.5613 | 0.6096 | 3.7705 | 4.8044 | +0.2432 | Pure ML |

- T1 = block 1-2 학습 → block 3 평가, T2 = block 1-3 → block 4, T3 = block 1-4 → block 5
- U1~U4 = 각 usage 변수에서 낮은 약 2/3 wafer 학습 → 높은 약 1/3 wafer 평가
- **Pure ML 이 7개 중 6개에서 낮고, Hybrid 는 T1 에서만 낮습니다.** 다만 T1 의 차이(1.4571)는 나머지 여섯 개의 차이(+0.0322 ~ +0.3930)보다 훨씬 큽니다.
- U3 의 차이는 +0.0322 로 **두 모델이 사실상 같습니다.**
- T1 은 두 모델 모두 R² 가 음수입니다. 학습 sample 이 655개로 가장 적고 분포 이동이 가장 큰 split 입니다.

## 5. 핵심 비교

| 항목 | Pure ML | Hybrid |
|---|---|---|
| nominal pooled RMSE | 3.3193 | 3.5723 |
| nominal 5-fold 평균 RMSE | 3.3163 | 3.5693 |
| 17개 그룹 중 더 낮은 그룹 수 | 17 | 0 |
| 그룹 RMSE 범위 | 0.5875 | 0.4818 |
| 그룹 RMSE 표준편차 | 0.1964 | 0.1466 |
| 평균 절대 delta_RMSE | 0.1720 | 0.1261 |
| worst-group RMSE | 3.5973 | 3.8028 |
| 7개 shift test 중 더 낮은 test 수 | 6 | 1 |
| shift RMSE 평균 | 6.0529 | 6.0590 |
| shift RMSE 최소 | 3.6229 | 4.0159 |
| shift RMSE 최대 | 11.1933 | 9.7362 |
| nominal 대비 shift 평균 증가폭 | +2.7336 | +2.4867 |
| nominal 대비 shift 최대 증가폭 | +7.8740 | +6.1639 |

### A 와 B 중 어느 쪽인가

- **A. Pure ML 이 평균 정확도는 높지만 Hybrid 가 조건 변화에 더 안정적이다**
- **B. Pure ML 이 평균 정확도와 robustness 모두 더 좋거나 비슷하다**

**결과는 B 에 가깝습니다. 다만 한 가지 예외가 분명합니다.**

B 를 지지하는 근거입니다.

- within-distribution 17개 그룹 **전부**에서 Pure ML 이 낮습니다.
- shift 7개 중 6개에서 Pure ML 이 낮습니다.
- **shift RMSE 평균은 6.0529 대 6.0590 으로 차이가 0.0061 입니다. 이 정도면 우열이 아니라 동률로 읽어야 합니다.**
- within-distribution worst-group RMSE 도 Pure ML 이 0.2056 낮습니다.

A 를 지지하는 근거, 즉 예외입니다.

- **최악의 shift 상황에서는 Hybrid 가 낫습니다.** T1 에서 9.7362 대 11.1933 으로 1.4571 낮습니다.
- nominal 대비 증가폭도 Hybrid 가 작습니다. 평균 +2.4867 대 +2.7336, 최대 +6.1639 대 +7.8740 입니다. 배율로 보면 Hybrid 1.696배, Pure ML 1.824배입니다.
- 조건 그룹별 흔들림 지표 세 가지가 모두 Hybrid 가 작습니다. 다만 이는 **더 나쁜 수준에서 평탄한 것**입니다.

## 6. 해석

**Pure ML 의 장점**

정확도입니다. nominal 에서 0.2530 낮고, 17개 조건 그룹 전부와 shift 6개에서 낮습니다. 같은 크기의 tuning 을 받은 뒤에는 어느 조건에서도 크게 뒤지지 않습니다.

**Hybrid 의 장점**

두 가지입니다. 첫째, 구조가 해석을 제공합니다. PV 가 압력과 회전을 담고 k(x) 가 소모품 사용량과 Stage 차이를 담는다는 Step 21 의 결과는 그대로 유효합니다. 둘째, **학습 분포에서 가장 멀리 벗어난 상황에서 덜 무너집니다.** T1 이 그 사례입니다. 물리 구조가 예측을 PV 라는 알려진 양에 묶어 두기 때문으로 보이지만, 이는 한 개의 split 에서 관찰된 것입니다.

**Hybrid 가 지불하는 예측성능 trade-off**

nominal RMSE 0.2530, MAE 0.2112 를 더 씁니다. 조건 그룹 17개 전부에서 뒤지고, 그 차이는 +0.1225 에서 +0.3830 사이입니다. 일상적인 조건에서는 일관되게 손해입니다.

**Hybrid 의 robustness 장점이 실제로 남아 있는가**

**부분적으로만 남아 있습니다.** Step 22 와 Step 23 에서 보였던 우위의 상당 부분은 상대가 tuning 되지 않은 Pure ML 이었기 때문입니다. tuning 후에는 그룹 승수가 0/17, shift 승수가 1/7 로 바뀝니다. 남아 있는 것은 **최악의 경우 방어**와 **조건에 따른 편차의 작음**이며, 후자는 더 높은 오차 수준에서의 평탄함입니다. shift 평균으로 보면 두 모델은 사실상 같습니다.

**한계**

- 최악의 경우 우위는 사실상 T1 하나에 의존합니다. T1 은 학습 sample 이 655개로 가장 적고 분포 이동이 가장 큰 split 입니다. 하나의 split 으로 일반화하기 어렵습니다.
- 모든 수치는 하나의 5-fold 분할과 고정된 7개 split 에 대한 결과입니다.
- 그룹별 R² 는 그 그룹의 MRR 분산에 따라 달라지므로 그룹끼리 비교하지 않았습니다.
- 두 모델 모두 학습 분포를 벗어나면 RMSE 가 1.7배에서 1.8배로 커집니다. 어느 쪽도 shift 에 강하다고 말할 수 없습니다.

**이번 단계로 training 데이터에서의 모델 개발과 tuning 을 종료합니다.**

## 7. 그림

| 파일 | 내용 |
|---|---|
| `figures/fig1_nominal_rmse.png` | 최종 두 모델의 nominal 5-fold RMSE |
| `figures/fig2_group_rmse.png` | 17개 조건 그룹별 RMSE |
| `figures/fig3_shift_rmse.png` | distribution-shift 7개 test RMSE |
| `figures/fig4_nominal_vs_shift.png` | nominal, 평균 shift, 최악 shift RMSE 비교 |

## 8. 저장 파일

| 파일 | 내용 |
|---|---|
| `overall_metrics.csv` | 두 모델의 pooled 지표 |
| `fold_metrics.csv`, `fold_summary.csv` | fold 별 지표와 평균, 표준편차 |
| `group_metrics_all.csv`, `group_comparison.csv` | 17개 그룹 지표와 두 모델 비교 |
| `group_summary.csv` | 모델별 그룹 흔들림 요약과 worst group |
| `shift_test_results.csv`, `shift_comparison.csv` | 7개 shift test 결과와 비교 |
| `shift_eval_predictions.csv` | shift test 평가 sample 의 실제값과 두 모델 예측 |
| `headline_comparison.csv`, `key_questions.csv` | 핵심 비교 지표와 질문별 답 |
| `oof_predictions_both_models.csv` | 5-fold sample 별 예측과 조건 그룹 배정 |
| `consistency_checks.csv` | 아래 확인 항목의 결과 |

## 9. 확인한 내용

아홉 항목 모두 True 입니다.

- sample 이 1,609개입니다.
- Pure ML 5-fold 평균 RMSE 가 Step 24 저장값과, Hybrid 가 Step 20 저장값과 같습니다.
- Hybrid pooled RMSE 가 Step 22 저장값과 같습니다.
- Hybrid 예측이 k_pred × PV_proxy 와 같습니다.
- 모든 split 에서 학습과 평가의 wafer 가 겹치지 않습니다.
- 7개 split 의 평가 sample 이 Step 23 과 같습니다.
- **Hybrid 재학습 결과가 Step 23 저장값과 같습니다.** 바뀐 것은 Pure ML 설정뿐입니다.
- 17개 그룹의 sample 수가 Step 22 저장값과 같습니다.
- 코드를 두 번 실행해 17개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

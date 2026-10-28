# Step 23. 학습하지 않은 조건으로 넘어갔을 때의 성능 (distribution-shift stress test)

- 대상: CHAMBER_PATH = 4-5-6 의 training 1,609개 sample (wafer 1,329개)
- training 만 사용했습니다. test / validation 은 사용하지 않았습니다.
- 모델 구조, feature, P/V, hyperparameter 를 바꾸거나 튜닝하지 않았습니다.
- 평가 데이터의 실제 MRR 이나 k_target 으로 예측을 보정하거나 clipping 하지 않았습니다.
- 코드: `scripts/step23_distribution_shift.py`, 결과: `reports/step23_distribution_shift/`

## 1. Step 22 와 무엇이 다른가

| 단계 | 방법 | 성격 |
|---|---|---|
| Step 22 | 기존 random 5-fold 의 out-of-fold 예측을 Stage / usage / 시간 조건별로 나누어 집계 | within-distribution subgroup robustness. 학습 데이터에 그 조건이 이미 들어 있습니다. |
| Step 23 | 과거 → 미래, 낮은 usage → 높은 usage 로 학습과 평가를 실제로 분리하고 새로 학습 | distribution-shift stress test. 학습 때 충분히 보지 못한 조건으로 넘어갑니다. |

**두 결과를 같은 것으로 읽으면 안 됩니다.** Step 22 의 RMSE 는 3.5 대이고 Step 23 의 RMSE 는 3.8 에서 11.6 사이입니다. 조건을 실제로 분리하면 두 모델 모두 성능이 크게 떨어집니다.

모든 split 은 wafer 단위이고, 같은 WAFER_ID 가 학습과 평가에 동시에 들어가지 않습니다.

## 2. 비교한 두 모델 (고정)

| 모델 | 구성 |
|---|---|
| Pure ML | Step 16 의 Model 1 XGBoost. `XGBRegressor(random_state=42, n_jobs=-1)` 기본 설정으로 MRR 을 직접 예측합니다. |
| Hybrid | Step 20~22 의 대표 Hybrid. MRR = k(x) × PV_proxy, PV_proxy = P9 × V3 |

- P9 = 6개 pressure 의 raw row 별 평균, V3 = abs(HEAD_ROTATION), 두 값이 모두 0 이 아닌 row 의 평균
- k(x) = XGBoost (n_estimators 900, max_depth 2, learning_rate 0.1, min_child_weight 1, subsample 0.9, colsample_bytree 0.9, random_state 42), feature 는 C_full 31개
- k_target 표준화는 각 split 의 **학습 데이터 평균과 표준편차만** 사용했습니다.
- 두 모델의 feature 는 같은 31개이고 전처리도 같습니다.

## 3. 시간 변화 stress test

각 wafer 의 가장 이른 기록 시각(여기서 쓰는 4-5-6 sample 기준)으로 wafer 1,329개를 정렬하고 연속된 5개 block 으로 나눴습니다.

| block | wafer 수 | sample 수 | 첫 기록 시각 최소 | 첫 기록 시각 최대 | Stage B 비율 |
|---|---|---|---|---|---|
| 1 | 266 | 342 | 481634409.667 | 482506188.667 | 0.4912 |
| 2 | 266 | 313 | 482506789.000 | 484143419.667 | 0.5016 |
| 3 | 266 | 313 | 484143605.667 | 485222224.000 | 0.5016 |
| 4 | 266 | 317 | 485223505.667 | 486063724.333 | 0.5552 |
| 5 | 265 | 324 | 486063918.333 | 487267851.333 | 0.4753 |

- T1 = block 1-2 학습 → block 3 평가
- T2 = block 1-3 학습 → block 4 평가
- T3 = block 1-4 학습 → block 5 평가

## 4. usage 변화 stress test

각 wafer 의 usage 대표값은 그 wafer 에 속한 4-5-6 sample 들의 평균입니다. wafer 를 usage 가 낮은 순서로 정렬해 하위 2/3 를 학습, 상위 1/3 을 평가로 썼습니다.

| test | usage 변수 | 학습 wafer | 평가 wafer | 학습 usage 최소 | 학습 usage 최대 | 평가 usage 최소 | 평가 usage 최대 |
|---|---|---|---|---|---|---|---|
| U1 | USAGE_OF_DRESSER_MEAN | 886 | 443 | 5.3432 | 533.7037 | 534.4444 | 771.6568 |
| U2 | USAGE_OF_POLISHING_TABLE_MEAN | 886 | 443 | 14.7605 | 221.5630 | 221.8944 | 349.5009 |
| U3 | USAGE_OF_DRESSER_TABLE_MEAN | 886 | 443 | 2665.1112 | 3810.8678 | 3811.0000 | 4305.3651 |
| U4 | USAGE_OF_BACKING_FILM_MEAN | 886 | 443 | 24.7321 | 6395.5563 | 6402.6204 | 10521.3639 |

usage 범위는 wafer 대표값 기준입니다. 네 test 모두 평가 wafer 의 usage 최소값이 학습 wafer 의 최대값보다 큽니다. 즉 겹치지 않습니다.

## 5. 일곱 test 의 결과

| test | 학습 sample | 평가 wafer | 평가 sample | Pure ML R² | Pure ML MAE | Pure ML RMSE | Hybrid R² | Hybrid MAE | Hybrid RMSE | Hybrid − Pure ML RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 655 | 266 | 313 | -2.3517 | 10.0284 | 11.5604 | -1.3774 | 8.4138 | 9.7362 | -1.8242 |
| T2 | 968 | 266 | 317 | 0.2521 | 5.0696 | 6.2443 | 0.4805 | 4.2241 | 5.2045 | -1.0398 |
| T3 | 1285 | 265 | 324 | 0.2702 | 4.6779 | 5.9386 | 0.0752 | 5.5758 | 6.6850 | +0.7464 |
| U1 | 1067 | 443 | 542 | 0.5967 | 4.5941 | 5.8262 | 0.5749 | 4.8249 | 5.9815 | +0.1553 |
| U2 | 1089 | 443 | 520 | 0.7997 | 3.0121 | 3.8312 | 0.7799 | 3.1949 | 4.0159 | +0.1847 |
| U3 | 1070 | 443 | 539 | 0.1600 | 5.1036 | 6.5254 | 0.2932 | 4.7516 | 5.9855 | -0.5399 |
| U4 | 1066 | 443 | 543 | 0.6046 | 3.8425 | 4.8352 | 0.6096 | 3.7705 | 4.8044 | -0.0308 |

- 음수 값은 Hybrid 의 RMSE 가 더 낮다는 뜻입니다.
- **시간 test 3개 중 2개(T1, T2)에서 Hybrid 가 더 낮고, T3 에서는 Pure ML 이 더 낮습니다.**
- **usage test 4개 중 2개(U3, U4)에서 Hybrid 가 더 낮습니다.** U4 의 차이는 −0.0308 로 사실상 같습니다.
- T1 은 두 모델 모두 R² 가 음수입니다. 평가 구간의 평균으로 예측하는 것보다도 나쁘다는 뜻입니다.
- 각 test 의 평가 집합이 서로 다르고 크기도 다르므로, test 끼리 RMSE 를 직접 비교할 때는 주의가 필요합니다.

## 6. Stage 구성 확인

| test | 학습 Stage B 비율 | 평가 Stage B 비율 | 차이 |
|---|---|---|---|
| T1 | 0.4962 | 0.5016 | +0.0054 |
| T2 | 0.4979 | 0.5552 | +0.0573 |
| T3 | 0.5121 | 0.4753 | -0.0368 |
| U1 | 0.5080 | 0.4982 | -0.0098 |
| U2 | 0.4913 | 0.5327 | +0.0414 |
| U3 | 0.5028 | 0.5083 | +0.0055 |
| U4 | 0.4765 | 0.5599 | +0.0833 |

- 차이가 가장 큰 split 은 U4 로 0.0833 입니다. 나머지는 모두 0.06 이하이고 전부 0.09 미만입니다.
- **Stage 구성 차이만으로 결과가 설명된다고 보기는 어렵습니다.** 다만 U4 와 T2 는 평가 쪽 Stage B 비율이 더 높고, Stage B 는 대체로 오차가 큽니다. 그래서 이 두 split 의 절대 RMSE 에는 Stage 구성이 일부 기여했을 수 있습니다.
- 중요한 점은 **두 모델이 같은 평가 집합을 쓴다는 것**입니다. 따라서 Stage 구성 차이는 두 모델의 비교값(RMSE 차이)에는 영향을 주지 않습니다.
- Stage 비율을 맞추려고 데이터를 삭제하거나 재배치하지 않았습니다.

## 7. 학습 구간과 평가 구간의 MRR 분포 차이

모델과 무관한 기술 통계입니다. 왜 특정 test 만 유독 나쁜지 보기 위해 기록했습니다.

| test | 학습 MRR 평균 | 학습 MRR 표준편차 | 평가 MRR 평균 | 평가 MRR 표준편차 | 평균 차이 |
|---|---|---|---|---|---|
| T1 | 70.4880 | 7.4965 | 81.2983 | 6.3145 | +10.8104 |
| T2 | 73.9835 | 8.7457 | 80.8697 | 7.2206 | +6.8862 |
| T3 | 75.6822 | 8.9046 | 80.0387 | 6.9517 | +4.3565 |
| U1 | 78.9196 | 7.4482 | 71.9132 | 9.1747 | -7.0064 |
| U2 | 76.5134 | 8.8009 | 76.6561 | 8.5601 | +0.1427 |
| U3 | 74.1849 | 8.4900 | 81.2734 | 7.1198 | +7.0885 |
| U4 | 76.4785 | 9.2054 | 76.7185 | 7.6896 | +0.2399 |

- **T1 의 차이가 +10.8104 로 가장 큽니다.** 학습 구간에서 본 적 없는 높은 MRR 영역을 예측해야 합니다.
- MRR 평균 차이의 절대값과 두 모델 평균 RMSE 의 상관계수는 0.8392 입니다. **분포가 많이 옮겨간 split 일수록 성능이 나쁩니다.**
- U2 와 U4 는 평균 차이가 0.25 이하로 거의 없고, RMSE 도 random 5-fold 에 가장 가깝습니다.

## 8. random 5-fold 와 비교

| 구분 | 평가 sample 수 | Pure ML RMSE | Hybrid RMSE | Hybrid − Pure ML | Pure ML 증가 | Hybrid 증가 |
|---|---|---|---|---|---|---|
| Random 5-fold (Step 22) | 1609 | 3.5872 | 3.5723 | -0.0149 | +0.0000 | +0.0000 |
| T1 | 313 | 11.5604 | 9.7362 | -1.8242 | +7.9732 | +6.1639 |
| T2 | 317 | 6.2443 | 5.2045 | -1.0398 | +2.6571 | +1.6322 |
| T3 | 324 | 5.9386 | 6.6850 | +0.7464 | +2.3514 | +3.1127 |
| U1 | 542 | 5.8262 | 5.9815 | +0.1553 | +2.2390 | +2.4092 |
| U2 | 520 | 3.8312 | 4.0159 | +0.1847 | +0.2440 | +0.4436 |
| U3 | 539 | 6.5254 | 5.9855 | -0.5399 | +2.9382 | +2.4132 |
| U4 | 543 | 4.8352 | 4.8044 | -0.0308 | +1.2480 | +1.2321 |

**일곱 test 모두에서 두 모델의 RMSE 가 random 5-fold 보다 높습니다.** 가장 좋은 U2 조차 Pure ML 이 +0.2440, Hybrid 가 +0.4436 늘어납니다.

## 9. Stage 별 평가 성능

각 test 의 평가 집합을 Stage 로 나눈 RMSE 입니다.

| test / Stage | sample 수 | Pure ML RMSE | Hybrid RMSE | 차이 |
|---|---|---|---|---|
| T1 A | 156 | 8.7761 | 7.7410 | -1.0350 |
| T1 B | 157 | 13.7806 | 11.3772 | -2.4033 |
| T2 A | 141 | 4.8845 | 4.7521 | -0.1324 |
| T2 B | 176 | 7.1495 | 5.5404 | -1.6092 |
| T3 A | 170 | 6.0785 | 7.1749 | +1.0964 |
| T3 B | 154 | 5.7803 | 6.0987 | +0.3184 |
| U1 A | 272 | 4.7683 | 5.1100 | +0.3417 |
| U1 B | 270 | 6.7258 | 6.7466 | +0.0208 |
| U2 A | 243 | 3.4142 | 3.9466 | +0.5324 |
| U2 B | 277 | 4.1628 | 4.0757 | -0.0871 |
| U3 A | 265 | 5.0834 | 5.8149 | +0.7315 |
| U3 B | 274 | 7.6662 | 6.1460 | -1.5202 |
| U4 A | 239 | 4.0211 | 4.2063 | +0.1852 |
| U4 B | 304 | 5.3896 | 5.2269 | -0.1627 |

- **Stage B 에서는 7개 test 중 5개에서 Hybrid 가 더 낮습니다.** Stage A 에서는 2개뿐입니다.
- 이 방향은 Step 22 에서 본 패턴(Hybrid 가 Stage B 에 유리하고 Stage A 에 불리)과 같습니다.

## 10. 요약

| 항목 | 값 |
|---|---|
| temporal test 중 Hybrid 의 RMSE 가 더 낮은 경우 | 2 / 3 |
| usage test 중 Hybrid 의 RMSE 가 더 낮은 경우 | 2 / 4 |
| 전체 shift test 중 Hybrid 의 RMSE 가 더 낮은 경우 | 4 / 7 |
| Pure ML 의 shift RMSE 평균 | 6.3945 |
| Hybrid 의 shift RMSE 평균 | 6.0590 |
| Pure ML 의 shift RMSE 최대 | 11.5604 |
| Hybrid 의 shift RMSE 최대 | 9.7362 |
| Pure ML 의 random 5-fold 대비 평균 증가 | +2.8073 |
| Hybrid 의 random 5-fold 대비 평균 증가 | +2.4867 |
| RMSE 차이(Hybrid - Pure ML) 최소 / 최대 | -1.8242 / +0.7464 |
| Stage B 비율 차이(평가 - 학습)의 최대 절대값 | 0.0833 |

## 11. 해석

**1) 시간 forward test 3개 중 Hybrid 가 더 낮은 RMSE 를 보인 경우**

2개입니다. T1 에서 −1.8242, T2 에서 −1.0398 으로 Hybrid 가 낮고, T3 에서는 +0.7464 로 Pure ML 이 낮습니다.

**2) usage stress test 4개 중 Hybrid 가 더 낮은 RMSE 를 보인 경우**

2개입니다. U3 에서 −0.5399, U4 에서 −0.0308 입니다. U4 의 차이는 사실상 없습니다. U1 과 U2 에서는 Pure ML 이 낮지만 차이는 0.16 과 0.18 로 작습니다.

**3) 조건이 바뀔 때 성능 저하가 더 작은 모델**

평균과 최악의 경우 기준으로는 Hybrid 입니다. shift RMSE 평균이 6.0590 대 6.3945 이고, 최대가 9.7362 대 11.5604, random 5-fold 대비 평균 증가가 +2.4867 대 +2.8073 입니다. **다만 7개 중 4개에서만 낫고 T3 에서는 뚜렷하게 나쁘므로 모든 조건에서 저하가 작다고 말할 수는 없습니다.**

**4) 특정 test 에서만 결과가 크게 달라지는가**

그렇습니다. T1 이 압도적으로 나쁩니다. 두 모델 모두 R² 가 음수이고 RMSE 가 random 5-fold 의 세 배 수준입니다. T1 은 학습 sample 이 655개로 가장 적고, 평가 구간의 MRR 평균이 학습보다 +10.8104 높아 분포 차이도 가장 큽니다. 반대로 U2 와 U4 는 분포 차이가 거의 없고 결과도 random 5-fold 에 가장 가깝습니다. **MRR 평균 차이의 절대값과 평균 RMSE 의 상관계수가 0.8392 로, 성능 저하의 크기는 모델 종류보다 분포가 얼마나 옮겨갔는지와 더 강하게 연관됩니다.**

**5) Hybrid 의 robustness 우위가 실제 distribution shift 에서도 반복되는가**

부분적으로만 반복됩니다.

- Step 22 에서는 17개 조건 그룹 중 11개에서 Hybrid 가 낮고, 흔들림 지표 3개가 모두 작았습니다.
- Step 23 에서는 7개 중 4개에서만 낮습니다. 평균과 최악의 경우에서는 Hybrid 가 낫지만 **개별 test 결과는 엇갈립니다.**
- 반복되는 것은 Stage 방향입니다. Stage B 에서 Hybrid 가 유리하고 Stage A 에서 불리한 패턴이 Step 22 와 Step 23 에서 같게 나타납니다.
- **따라서 "Hybrid 가 distribution shift 에서 일관되게 더 강하다"고 단정할 수 없습니다.** 평균적으로 조금 낫다는 정도가 이 데이터가 뒷받침하는 표현입니다.

**한계**

- 일곱 test 는 평가 집합이 서로 다르고 크기도 달라서 test 끼리 RMSE 를 직접 비교하기 어렵습니다. 모델 비교는 같은 평가 집합 안에서만 했습니다.
- block 개수(5개)와 usage 분할 비율(2/3)을 바꾸면 수치가 달라질 수 있습니다.
- 모든 서술은 인과가 아니라 예측 오차의 차이입니다.

## 12. 그림

| 파일 | 내용 |
|---|---|
| `figures/fig1_temporal_forward_rmse.png` | 세 시간 forward test 의 Pure ML 대 Hybrid RMSE |
| `figures/fig2_usage_shift_rmse.png` | 네 usage stress test 의 Pure ML 대 Hybrid RMSE |
| `figures/fig3_random_vs_shift_rmse.png` | random 5-fold RMSE 와 distribution-shift RMSE 비교 |

## 13. 저장 파일

| 파일 | 내용 |
|---|---|
| `temporal_block_info.csv` | 시간 block 5개의 wafer 수, sample 수, 시각 범위, Stage 구성 |
| `temporal_test_results.csv` | 시간 forward test 3개의 모델별 지표 |
| `usage_split_info.csv` | usage test 4개의 split 정보와 usage 범위 |
| `usage_test_results.csv` | usage stress test 4개의 모델별 지표 |
| `model_comparison_by_test.csv` | 7개 test 의 두 모델 지표와 차이 |
| `stage_composition.csv` | split 별 학습/평가 Stage 구성 |
| `split_distribution_info.csv` | split 별 학습/평가 MRR, k_target, PV_proxy 분포 |
| `stagewise_eval_metrics.csv` | test × Stage 별 평가 성능 |
| `shift_vs_random_comparison.csv` | random 5-fold 와 shift test 의 RMSE 비교 |
| `eval_predictions.csv` | test 별 평가 sample 의 실제값과 두 모델 예측 |
| `summary_counts.csv` | 요약 수치 |
| `consistency_checks.csv` | 아래 확인 항목의 결과 |

## 14. 확인한 내용

아홉 항목 모두 True 입니다.

- sample 이 1,609개이고 feature 가 31개입니다.
- PV_PROXY 가 Step 20 저장값과 같고, k_target = MRR / PV_proxy 입니다.
- 모든 split 에서 학습과 평가의 wafer 가 겹치지 않습니다.
- 시간 block 의 sample 합이 1,609 이고 block 이 시간순으로 겹치지 않습니다.
- usage test 에서 평가 wafer 의 usage 가 학습 wafer 보다 높습니다.
- Hybrid 예측이 k_pred × PV_proxy 와 같습니다.
- 코드를 두 번 실행해 15개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

# Step 21. 대표 Hybrid 모델이 k(x) 예측에 사용하는 공정정보 해석

- 대상: CHAMBER_PATH = 4-5-6 의 1,609개 sample, Step 8 의 같은 5-fold
- training 만 사용했습니다. test / validation 은 사용하지 않았고, 새 feature 를 만들거나 성능을 개선하지 않았습니다.
- 코드: `scripts/step21_hybrid_interpretation.py`, 결과: `reports/step21_interpretation/`
- **모든 해석은 인과관계가 아니라 모델이 예측에 사용한 연관(association)입니다.**

## 1. 대표 Hybrid 모델

- 구조: MRR = k(x) × PV_proxy, PV_proxy = P9 × V3 (P9 = pressure 6개의 raw row 별 평균, V3 = abs(HEAD_ROTATION), 두 값이 모두 0 이 아닌 row 의 평균)
- k(x): XGBoost (n_estimators 900, max_depth 2, learning_rate 0.1, min_child_weight 1, subsample 0.9, colsample_bytree 0.9, random_state 42)
- k 는 fold 마다 학습 sample 로만 표준화해 학습하고 예측 후 원래 단위로 되돌립니다.
- **이 설정은 Step 20 상위 후보 중 하나일 뿐이며 유일한 최적 설정이 아닙니다.** 1위와 평균 RMSE 가 사실상 같고, 더 얕은 트리를 쓰며 fold 변동이 더 작아 대표로 사용했습니다.
- k(x) 입력은 Step 20 의 C_full 31개 feature 그대로입니다.

**두 모델의 성능 (5개 fold 예측을 모아 계산)**

| 모델 | R² | MAE | RMSE |
|---|---|---|---|
| 대표 Hybrid | 0.8323 | 2.7643 | 3.5723 |
| Pure ML (Model 1 XGBoost) | 0.8309 | 2.7454 | 3.5872 |

대표 Hybrid 의 예측은 Step 20 에 저장된 같은 설정 결과와, Pure ML 예측은 Step 16 의 Model 1 XGBoost 결과와 완전히 같습니다.

## 2. 기여도 계산 방법

- 각 fold 의 학습 모델에서 XGBoost 의 `pred_contribs`(SHAP 기여도)를 **평가 fold sample 에만** 계산하고 5개 fold 를 합쳤습니다.
- Hybrid 는 표준화된 k 를 학습하므로, 기여도에 그 fold 의 표준편차를 곱해 **k 단위**로 되돌렸습니다.
- Pure ML 과 비교할 때는 k 기여도에 그 sample 의 PV_proxy 를 곱해 **MRR 단위**로 환산했습니다. MRR = k × PV 이므로 k 가 δ 만큼 달라지면 MRR 은 δ × PV 만큼 달라진다는 관계를 이용한 환산입니다.

## 3. Hybrid k(x) 상위 15 feature

| 순위 | feature | 그룹 | 평균 절대 기여도 (k 단위) | MRR 단위 환산 | Stage A 순위 | Stage B 순위 | Pure ML 순위 |
|---|---|---|---|---|---|---|---|
| 1 | STAGE_B | STAGE | 0.000068 | 4.0488 | 1 | 1 | 2 |
| 2 | USAGE_OF_DRESSER_TABLE_MEAN | usage | 0.000053 | 3.1551 | 2 | 2 | 1 |
| 3 | USAGE_OF_DRESSER_MEAN | usage | 0.000035 | 2.0783 | 3 | 3 | 3 |
| 4 | RETAINER_RING_PRESSURE_MEAN | pressure | 0.000016 | 0.9809 | 4 | 4 | 30 |
| 5 | USAGE_OF_BACKING_FILM_MEAN | usage | 0.000010 | 0.6104 | 5 | 11 | 8 |
| 6 | STAGE_ROTATION_MEAN | STAGE | 0.000010 | 0.5989 | 8 | 5 | 9 |
| 7 | RETAINER_RING_PRESSURE_STD | pressure | 0.000010 | 0.5873 | 7 | 7 | 7 |
| 8 | CENTER_AIR_BAG_PRESSURE_STD | pressure | 0.000010 | 0.5890 | 10 | 6 | 16 |
| 9 | SLURRY_FLOW_LINE_A_MEAN | slurry | 0.000010 | 0.5705 | 6 | 9 | 5 |
| 10 | EDGE_AIR_BAG_PRESSURE_MEAN | pressure | 0.000009 | 0.5445 | 9 | 10 | 28 |
| 11 | WAFER_ROTATION_MEAN | rotation | 0.000008 | 0.4980 | 13 | 8 | 11 |
| 12 | USAGE_OF_POLISHING_TABLE_MEAN | usage | 0.000008 | 0.4922 | 11 | 12 | 4 |
| 13 | PRESSURIZED_CHAMBER_PRESSURE_STD | pressure | 0.000008 | 0.4730 | 12 | 13 | 21 |
| 14 | WAFER_ROTATION_STD | rotation | 0.000006 | 0.3472 | 14 | 15 | 12 |
| 15 | SLURRY_FLOW_LINE_C_STD | slurry | 0.000006 | 0.3369 | 15 | 14 | 19 |

- STAGE_ROTATION_MEAN 은 이름이 STAGE 로 시작해 그룹 분류에서 STAGE 로 묶였습니다. 실제로는 회전 관련 변수이므로 아래 그룹 비중을 읽을 때 참고가 필요합니다.
- Stage A 와 Stage B 의 상위 4개 순위는 동일합니다. 5위부터 조금 달라집니다.

## 4. Pure ML 상위 15 feature

| 순위 | feature | 그룹 | 평균 절대 기여도 (MRR 단위) | Hybrid 순위 |
|---|---|---|---|---|
| 1 | USAGE_OF_DRESSER_TABLE_MEAN | usage | 4.3205 | 2 |
| 2 | STAGE_B | STAGE | 3.2649 | 1 |
| 3 | USAGE_OF_DRESSER_MEAN | usage | 2.5152 | 3 |
| 4 | USAGE_OF_POLISHING_TABLE_MEAN | usage | 0.3842 | 12 |
| 5 | SLURRY_FLOW_LINE_A_MEAN | slurry | 0.3177 | 9 |
| 6 | EDGE_AIR_BAG_PRESSURE_STD | pressure | 0.3121 | 29 |
| 7 | RETAINER_RING_PRESSURE_STD | pressure | 0.3090 | 7 |
| 8 | USAGE_OF_BACKING_FILM_MEAN | usage | 0.3081 | 5 |
| 9 | STAGE_ROTATION_MEAN | STAGE | 0.2827 | 6 |
| 10 | SLURRY_FLOW_LINE_A_STD | slurry | 0.2641 | 20 |
| 11 | WAFER_ROTATION_MEAN | rotation | 0.2432 | 11 |
| 12 | WAFER_ROTATION_STD | rotation | 0.2180 | 14 |
| 13 | MAIN_OUTER_AIR_BAG_PRESSURE_MEAN | pressure | 0.2084 | 25 |
| 14 | SLURRY_FLOW_LINE_C_MEAN | slurry | 0.1898 | 16 |
| 15 | CENTER_AIR_BAG_PRESSURE_MEAN | pressure | 0.1710 | 17 |

**공통 / 한쪽에서만 중요한 feature**

| 구분 | feature |
|---|---|
| 두 모델 상위 15 공통 (10개) | STAGE_B, USAGE_OF_DRESSER_TABLE_MEAN, USAGE_OF_DRESSER_MEAN, USAGE_OF_BACKING_FILM_MEAN, STAGE_ROTATION_MEAN, RETAINER_RING_PRESSURE_STD, SLURRY_FLOW_LINE_A_MEAN, WAFER_ROTATION_MEAN, USAGE_OF_POLISHING_TABLE_MEAN, WAFER_ROTATION_STD |
| Hybrid 에서만 상위 15 (5개) | RETAINER_RING_PRESSURE_MEAN, CENTER_AIR_BAG_PRESSURE_STD, EDGE_AIR_BAG_PRESSURE_MEAN, PRESSURIZED_CHAMBER_PRESSURE_STD, SLURRY_FLOW_LINE_C_STD |
| Pure ML 에서만 상위 15 (5개) | SLURRY_FLOW_LINE_C_MEAN, CENTER_AIR_BAG_PRESSURE_MEAN, SLURRY_FLOW_LINE_A_STD, MAIN_OUTER_AIR_BAG_PRESSURE_MEAN, EDGE_AIR_BAG_PRESSURE_STD |

**순위 차이가 가장 큰 feature**

| feature | Hybrid 순위 | Pure ML 순위 | 차이 |
|---|---|---|---|
| RETAINER_RING_PRESSURE_MEAN | 4 | 30 | 26 |
| EDGE_AIR_BAG_PRESSURE_STD | 29 | 6 | -23 |
| EDGE_AIR_BAG_PRESSURE_MEAN | 10 | 28 | 18 |
| MAIN_OUTER_AIR_BAG_PRESSURE_MEAN | 25 | 13 | -12 |
| SLURRY_FLOW_LINE_A_STD | 20 | 10 | -10 |

## 5. feature 그룹별 기여 비중

MRR 단위로 환산한 평균 절대 기여도의 합계 비중입니다.

| 그룹 | feature 수 | Hybrid 비중 | Pure ML 비중 | Hybrid 합계 | Pure ML 합계 |
|---|---|---|---|---|---|
| usage | 4 | 33.2% | 49.2% | 6.3360 | 7.5280 |
| STAGE | 3 | 25.4% | 24.0% | 4.8469 | 3.6706 |
| pressure | 12 | 24.2% | 13.4% | 4.6250 | 2.0443 |
| slurry | 6 | 8.9% | 7.0% | 1.6990 | 1.0646 |
| rotation | 4 | 7.0% | 5.0% | 1.3273 | 0.7705 |
| dressing water | 2 | 1.4% | 1.4% | 0.2620 | 0.2122 |

- **Hybrid 는 pressure 비중이 Pure ML 의 약 1.8배입니다** (24.2% 대 13.4%).
- **Pure ML 은 usage 비중이 더 큽니다** (49.2% 대 33.2%).
- usage, slurry, dressing water 를 합치면 Hybrid 43.4%, Pure ML 57.6% 입니다.

## 6. 상위 feature 의 영향 방향

k 단위 기여도 기준입니다. Spearman 은 feature 값과 기여도의 순위 상관이고, 낮은 절반과 높은 절반은 feature 값을 중앙값으로 나눈 두 집단의 평균 기여도입니다 (STAGE_B 는 0/1 값별).

| feature | Spearman | 낮은 값 쪽 평균 기여도 | 높은 값 쪽 평균 기여도 | 경향 |
|---|---|---|---|---|
| STAGE_B | +0.87 | -6.93e-05 | +6.72e-05 | Stage B 에서 k 를 높이는 방향 |
| USAGE_OF_DRESSER_TABLE_MEAN | +0.77 | -4.54e-05 | +4.78e-05 | 값이 클수록 k 를 높이는 방향 |
| USAGE_OF_DRESSER_MEAN | -0.92 | +3.15e-05 | -3.41e-05 | 값이 클수록 k 를 낮추는 방향 |
| RETAINER_RING_PRESSURE_MEAN | -0.90 | +1.51e-05 | -1.52e-05 | 값이 클수록 k 를 낮추는 방향 |
| USAGE_OF_BACKING_FILM_MEAN | +0.08 | +5.59e-07 | -3.20e-07 | 단조 경향이 거의 없음 |
| STAGE_ROTATION_MEAN | +0.91 | -9.38e-06 | +9.76e-06 | 값이 클수록 k 를 높이는 방향 |
| RETAINER_RING_PRESSURE_STD | -0.87 | +9.80e-06 | -8.74e-06 | 값이 클수록 k 를 낮추는 방향 |
| CENTER_AIR_BAG_PRESSURE_STD | -0.85 | +8.71e-06 | -8.38e-06 | 값이 클수록 k 를 낮추는 방향 |
| SLURRY_FLOW_LINE_A_MEAN | -0.70 | +8.71e-06 | -8.68e-06 | 값이 클수록 k 를 낮추는 방향 |
| EDGE_AIR_BAG_PRESSURE_MEAN | +0.65 | -5.44e-06 | +7.00e-06 | 값이 클수록 k 를 높이는 방향 |
| WAFER_ROTATION_MEAN | +0.74 | -6.44e-06 | +7.11e-06 | 값이 클수록 k 를 높이는 방향 |
| USAGE_OF_POLISHING_TABLE_MEAN | -0.70 | +6.77e-06 | -6.63e-06 | 값이 클수록 k 를 낮추는 방향 |
| PRESSURIZED_CHAMBER_PRESSURE_STD | -0.22 | +3.85e-06 | -3.36e-06 | 단조 경향이 약함 |
| WAFER_ROTATION_STD | +0.62 | -4.20e-06 | +4.05e-06 | 값이 클수록 k 를 높이는 방향 |
| SLURRY_FLOW_LINE_C_STD | -0.45 | +2.00e-06 | -2.26e-06 | 값이 클수록 k 를 낮추는 방향 |

- **비선형 모델이므로 이 경향은 단순한 인과가 아닙니다.** 그림 4 를 보면 USAGE_OF_BACKING_FILM_MEAN 처럼 구간마다 방향이 바뀌는 feature 도 있고, RETAINER_RING_PRESSURE_MEAN 처럼 중간 구간은 평평하다가 높은 값에서 급격히 내려가는 형태도 있습니다.
- 값과 기여도의 관계는 `top_feature_contributions_long.csv` 와 그림 4 에 저장했습니다.

## 7. 해석

**1) k(x) 의 변화를 가장 많이 설명하는 공정정보**

STAGE 구분(4.05), dresser table 사용량(3.16), dresser 사용량(2.08) 세 가지가 가장 큽니다. MRR 단위로 환산한 기여도 기준이며, 4위 retainer ring 압력 평균(0.98)과 격차가 큽니다. 즉 k 의 변화는 **Stage 종류와 소모품 사용량**이 대부분 설명합니다.

**2) Hybrid 가 PV 에 쓴 pressure/rotation 을 다시 학습하는 것인가**

부분적으로만 그렇습니다.

- PV_proxy 를 직접 구성하는 4개 feature(retainer ring 압력 평균과 표준편차, head rotation 평균과 표준편차)의 비중은 Hybrid 에서 10.7%, Pure ML 에서 4.7% 입니다. Hybrid 가 더 많이 쓰지만 전체의 일부입니다.
- 나머지 89.3% 는 PV 에 들어가지 않은 정보입니다. usage, slurry, dressing water 만 합쳐도 43.4% 입니다.
- **따라서 Hybrid 는 PV 를 단순히 재현하는 것이 아니라, PV 가 담지 못한 공정상태(주로 소모품 사용량과 Stage)를 k 에 담고 있습니다.**
- 다만 retainer ring 압력 평균이 Hybrid 에서 4위, Pure ML 에서 30위인 점은 Hybrid 가 PV 로 나눈 뒤 남은 압력 정보를 보정하는 데 사용한다는 뜻으로 볼 수 있습니다.

**3) Pure ML 과 Hybrid 가 보는 정보의 차이**

상위 15개 중 10개가 공통입니다. 두 모델 모두 Stage 와 dresser 계열 사용량을 가장 크게 봅니다. 차이는 다음과 같습니다.

- Hybrid 는 pressure 비중이 더 큽니다 (24.2% 대 13.4%). 특히 retainer ring 압력 평균과 edge air-bag 압력 평균의 순위가 크게 올라갑니다.
- Pure ML 은 usage 비중이 더 큽니다 (49.2% 대 33.2%).
- 즉 **두 모델이 보는 정보는 크게 다르지 않고, 강조점이 이동한 정도**입니다.

**4) Hybrid 구조를 물리적으로 해석할 근거**

부분적인 근거가 생깁니다.

- PV 는 압력과 회전 정보를 담고, k(x) 는 주로 소모품 사용량과 Stage 차이를 담습니다. 이는 Preston 식에서 k 를 "공정 조건 외의 상태에 따라 달라지는 계수"로 보는 관점과 방향이 맞습니다.
- dresser 사용량이 커질수록 k 를 낮추는 방향, dresser table 사용량이 커질수록 k 를 높이는 방향으로 나타납니다. 다만 두 사용량 변수는 서로 다른 소모품의 누적 사용량이고, 이 데이터만으로 마모와 MRR 의 인과를 확인할 수는 없습니다.
- 성능 면에서는 Hybrid 와 Pure ML 이 거의 같습니다 (RMSE 3.5723 대 3.5872). **구조를 나눈 이점은 정확도 향상보다 해석 가능성에 있습니다.**

## 8. 그림

| 파일 | 내용 |
|---|---|
| `figures/fig1_hybrid_k_importance.png` | Hybrid k(x) 기여도 상위 15개 |
| `figures/fig2_pure_ml_mrr_importance.png` | Pure ML MRR 기여도 상위 15개 |
| `figures/fig3_group_share_comparison.png` | 그룹별 기여 비중 비교 (Hybrid 대 Pure ML) |
| `figures/fig4_value_vs_contribution.png` | 상위 6개 feature 의 값 대 기여도 관계 |

## 9. 저장 파일

| 파일 | 내용 |
|---|---|
| `feature_importance.csv` | 31개 feature 의 Hybrid / Pure ML 기여도, 순위, Stage 별 기여도와 순위, 비중, 순위 차이 |
| `group_contribution.csv` | 그룹별 기여 합계와 비중 |
| `top_feature_direction.csv` | 상위 15개 feature 의 Spearman 과 값 구간별 평균 기여도 |
| `top_feature_contributions_long.csv` | 상위 15개 feature 의 sample 별 값과 기여도 (k 단위, MRR 단위) |
| `fold_scaling_and_bias.csv` | fold 별 k 표준화 값과 두 모델의 bias |
| `oof_predictions_both_models.csv` | sample 별 PV_proxy, k_target, k_pred, 실제 MRR, 두 모델의 MRR 예측 |

## 10. 확인한 내용

- 대표 Hybrid 예측이 Step 20 의 같은 설정 결과와, Pure ML 예측이 Step 16 의 Model 1 XGBoost 결과와 완전히 같습니다.
- PV_proxy 가 Step 19 의 P9 × V3 값과 같습니다.
- 기여도는 각 fold 의 평가 sample 에만 계산했고, 표준화는 학습 fold 통계만 사용했습니다.
- 코드를 두 번 실행해 10개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

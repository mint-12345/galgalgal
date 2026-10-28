# 최종 연구 결과 정리 (Step 1~27 통합)

- 이 문서는 **새로운 학습, tuning, feature engineering, P/V 변경 없이** Step 1~27 에서 저장한 결과만 모아 작성했습니다.
- 모든 수치는 기존 결과 CSV 에서 읽었습니다. 집계 코드는 `scripts/step28_final_summary.py`, 집계 결과는 `reports/final_summary/` 입니다.
- 데이터: PHM 2016 CMP Data Challenge. training 으로만 모델을 개발했고, validation 은 Step 26 에서, test 는 Step 27 에서 각각 한 번씩만 사용했습니다.

## 1. 최종 연구 질문

1. 고정된 Preston coefficient k 를 쓰는 단순 physics model 이 이 데이터의 MRR 을 얼마나 설명하는가?
2. k 를 공정상태에 따라 변하는 k(x) 로 확장한 Hybrid 가 fixed-k 의 한계를 얼마나 회복하는가?
3. Hybrid 의 예측성능은 fully data-driven Pure ML 과 비교해 어느 정도인가?
4. k(x) 는 어떤 공정정보와 연관되어 있는가?
5. Hybrid 의 해석성, robustness, 예측 정확도 사이에는 어떤 trade-off 가 있는가?

## 2. 최종 모델 정의

| 모델 | 구조 | 설정 |
|---|---|---|
| Model 1 Pure ML | x → MRR | XGBoost (n_estimators 900, max_depth 4, learning_rate 0.05, min_child_weight 3, subsample 0.9, colsample_bytree 0.9, random_state 42) |
| Model 2 Fixed-k Preston | MRR = k × PV | 절편 없이 k = Σ(PV·MRR) / Σ(PV²), training 에서 한 번 계산 |
| Model 3 Hybrid | MRR = k(x) × PV | k(x) = XGBoost (900, depth 2, lr 0.1, mcw 1, subsample 0.9, colsample_bytree 0.9, random_state 42), k 표준화는 training 통계만 사용 |

**PV 의 정의와 그 한계**

- PV_proxy = **P9 × V3** 입니다.
- P9 = 6개 pressure 채널의 **raw row 별 평균**, V3 = **abs(HEAD_ROTATION)** 입니다.
- P 와 V 가 모두 0 이 아닌 raw row 에서만 P × V 를 평균했습니다.
- **P9 × V3 는 sensor 기반 proxy 입니다.** 실제 물리적 polishing pressure 나 actual relative linear velocity 가 아닙니다. PHM 변수는 스케일이 가려져 있어 물리 단위로 환산할 수 없습니다. 이 문서에서 k 를 부를 때도 물리 상수가 아니라 **effective k** 의 의미로 씁니다.
- Pure ML 과 Hybrid 는 같은 31개 feature 를 씁니다. 세 모델 모두 training 4-5-6 의 1,609개 sample 로만 학습했습니다.

## 3. 최종 핵심 성능표 (master table)

training 은 5-fold out-of-fold 결과이고, validation 과 test 는 1,609개 전체로 학습한 모델의 외부 평가입니다.

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

### 요청서 기대값과의 대조

| 데이터 | 모델 | 요청서 기대값 | 저장값 | 반올림 | 일치 |
|---|---|---|---|---|---|
| training | Pure ML | 3.32 | 3.319269 | 3.32 | True |
| training | Hybrid | 3.57 | 3.572315 | 3.57 | True |
| training | Fixed-k Preston | 8.65 | 8.759663 | 8.76 | False |
| validation | Pure ML | 3.256 | 3.255967 | 3.256 | True |
| validation | Hybrid | 3.4142 | 3.414173 | 3.4142 | True |
| validation | Fixed-k Preston | 9.113 | 9.112972 | 9.113 | True |
| test | Pure ML | 3.1856 | 3.185649 | 3.1856 | True |
| test | Hybrid | 3.465 | 3.464982 | 3.465 | True |
| test | Fixed-k Preston | 9.1367 | 9.136695 | 9.1367 | True |

**9개 중 8개가 일치하고, training 의 Fixed-k 하나만 다릅니다. 임의로 고치지 않았습니다.** 원인은 PV 정의의 차이입니다.

| 항목 | 값 |
|---|---|
| training fixed-k RMSE, PV = P9 x V3 (Step 26/27 에서 사용, master table 값) | 8.7597 |
| training fixed-k RMSE, Step 16 당시의 PV 조합 (5-fold 평균) | 8.6511 |

요청서의 8.65 는 Step 16 에서 쓰던 PV 조합으로 계산한 값입니다. Step 26 과 Step 27 은 validation, test 와 같은 조건으로 비교하기 위해 Hybrid 가 쓰는 **P9 × V3** 로 fixed-k 를 다시 계산했고 그 값이 8.7597 입니다. master table 은 세 데이터를 같은 PV 정의로 비교하기 위해 8.7597 을 씁니다. **두 값 모두 fixed-k 의 결론(R² 가 0 근처이거나 음수)을 바꾸지 않습니다.**

## 4. Fixed-k → Hybrid 회복 효과

| 데이터 | sample 수 | Fixed-k RMSE | Hybrid RMSE | 절대 감소 | 상대 감소 (%) | Fixed-k R² | Hybrid R² |
|---|---|---|---|---|---|---|---|
| training | 1609 | 8.7597 | 3.5723 | 5.1873 | 59.2186 | -0.0082 | 0.8323 |
| validation | 357 | 9.1130 | 3.4142 | 5.6988 | 62.5350 | -0.0312 | 0.8553 |
| test | 351 | 9.1367 | 3.4650 | 5.6717 | 62.0762 | -0.0700 | 0.8461 |

- **k 를 상수에서 k(x) 로 바꾼 것만으로 RMSE 가 세 데이터 모두에서 약 60% 줄어듭니다.** 상대 감소율이 59.22%, 62.54%, 62.08% 로 거의 같습니다.
- R² 는 음수에서 0.83~0.86 으로 올라갑니다. **이 연구에서 가장 크고 가장 일관된 효과입니다.**
- 구조(MRR = k × PV)와 PV 정의는 두 모델이 같습니다. 달라진 것은 k 가 상수인지 x 의 함수인지뿐입니다.

## 5. Pure ML 과 Hybrid 의 trade-off

| 데이터 | sample 수 | Pure ML RMSE | Hybrid RMSE | Hybrid − Pure ML | 상대 차이 (%) | MAE 차이 | 더 낮은 쪽 |
|---|---|---|---|---|---|---|---|
| training | 1609 | 3.3193 | 3.5723 | +0.2530 | 7.6235 | +0.2112 | Pure ML |
| validation | 357 | 3.2560 | 3.4142 | +0.1582 | 4.8589 | +0.1859 | Pure ML |
| test | 351 | 3.1856 | 3.4650 | +0.2793 | 8.7685 | +0.2086 | Pure ML |

- **세 데이터 모두에서 Pure ML 의 RMSE 가 더 낮습니다.** Hybrid 가 더 정확한 데이터는 하나도 없습니다.
- 차이는 절대값 0.16~0.28, 상대값 4.9%~8.8% 범위입니다. 방향과 크기가 세 데이터에서 일관됩니다.
- Pure ML 은 Step 24 에서 Hybrid 와 같은 54개 격자로 tuning 했습니다. 1위 설정은 n900_d4_lr0.05_mcw3 이고 5-fold 평균 RMSE 3.3163 입니다. 상위 10개의 평균 RMSE 폭 0.0538 이 fold 변동 0.1557 보다 작아 **유일한 최적 설정이라고 말하지 않습니다.**

## 6. k(x) 해석 결과 (Step 21)

기여도는 평가 fold sample 에 대한 SHAP 기여도이며, **인과가 아니라 모델이 예측에 사용한 연관(association)** 입니다.

| 순위 | feature | 그룹 | 평균 절대 기여도 (k 단위) | Pure ML 순위 |
|---|---|---|---|---|
| 1 | STAGE_B | STAGE | 0.000068 | 2 |
| 2 | USAGE_OF_DRESSER_TABLE_MEAN | usage | 0.000053 | 1 |
| 3 | USAGE_OF_DRESSER_MEAN | usage | 0.000035 | 3 |
| 4 | RETAINER_RING_PRESSURE_MEAN | pressure | 0.000016 | 30 |
| 5 | USAGE_OF_BACKING_FILM_MEAN | usage | 0.000010 | 8 |
| 6 | STAGE_ROTATION_MEAN | STAGE | 0.000010 | 9 |
| 7 | RETAINER_RING_PRESSURE_STD | pressure | 0.000010 | 7 |
| 8 | CENTER_AIR_BAG_PRESSURE_STD | pressure | 0.000010 | 16 |
| 9 | SLURRY_FLOW_LINE_A_MEAN | slurry | 0.000010 | 5 |
| 10 | EDGE_AIR_BAG_PRESSURE_MEAN | pressure | 0.000009 | 28 |

| feature 그룹 | feature 수 | Hybrid k(x) 비중 | Pure ML 비중 |
|---|---|---|---|
| usage | 4 | 33.2% | 49.2% |
| STAGE | 3 | 25.4% | 24.0% |
| pressure | 12 | 24.2% | 13.4% |
| slurry | 6 | 8.9% | 7.0% |
| rotation | 4 | 7.0% | 5.0% |
| dressing water | 2 | 1.4% | 1.4% |

- **Stage 구분과 dresser 계열 사용량이 k(x) 변화의 대부분을 설명합니다.** 상위 3개가 나머지와 격차가 큽니다.
- **pressure 는 Hybrid 에서 24.2%, Pure ML 에서 13.4% 로 Hybrid 쪽 비중이 약 1.8배입니다.** retainer ring 압력 평균이 Hybrid 에서 4위, Pure ML 에서 30위로 순위 차이가 가장 큽니다. PV 로 나눈 뒤 남은 압력 정보를 k(x) 가 보정하는 형태로 읽을 수 있습니다.
- slurry 8.9%, rotation 7.0%, dressing water 1.4% 로 나머지 공정상태도 사용됩니다.
- **Hybrid 가 PV 에 들어간 정보만 다시 학습하는 것은 아닙니다.** PV 를 직접 구성하는 4개 feature(retainer ring 압력 평균과 표준편차, head rotation 평균과 표준편차)의 비중은 Hybrid 10.7%, Pure ML 4.7% 입니다. 즉 **k(x) 기여도의 89.3% 는 PV 밖의 공정상태에서 나옵니다.**
- STAGE_ROTATION_MEAN 은 이름 때문에 STAGE 그룹으로 분류되었지만 실제로는 회전 관련 변수입니다. 그룹 비중을 읽을 때 참고가 필요합니다.

## 7. Robustness 결과 통합 (Step 22, 23, 25)

| 분석 | 비교 대상 | Hybrid 가 더 좋은 경우 | 전체 |
|---|---|---|---|
| Step 22 (within-distribution, 기본 설정 Pure ML) | Hybrid vs 기본 Pure ML | 11 | 17 |
| Step 23 (distribution shift, 기본 설정 Pure ML) | Hybrid vs 기본 Pure ML | 4 | 7 |
| Step 25 (within-distribution, tuned Pure ML) | Hybrid vs tuned Pure ML | 0 | 17 |
| Step 25 (distribution shift, tuned Pure ML) | Hybrid vs tuned Pure ML | 1 | 7 |

| 지표 | Pure ML | Hybrid | Hybrid − Pure ML |
|---|---|---|---|
| shift RMSE 평균 | 6.0529 | 6.0590 | +0.0061 |
| shift RMSE 최대 | 11.1933 | 9.7362 | -1.4571 |
| worst-group RMSE | 3.5973 | 3.8028 | +0.2056 |
| T1 RMSE | 11.1933 | 9.7362 | -1.4571 |

**최종 정리는 다음과 같습니다.**

- **초기 분석(Step 22, 23)에서는 Hybrid 가 더 안정적으로 보이는 부분이 있었습니다.** 17개 조건 그룹 중 11개, shift 7개 중 4개에서 Hybrid 의 RMSE 가 더 낮았습니다.
- **그러나 그 비교의 상대는 tuning 을 받지 않은 기본 설정 Pure ML 이었습니다.** Step 24 에서 Pure ML 에 Hybrid 와 같은 54개 격자를 준 뒤 Step 25 에서 다시 비교하자 결과가 바뀝니다.
- **Step 25 에서는 17개 조건 그룹 전부와 shift 7개 중 6개에서 Pure ML 의 RMSE 가 더 낮습니다.**
- **shift RMSE 평균은 6.0529 대 6.0590 으로 차이가 0.0061 입니다. 사실상 동률입니다.**
- **극단적인 T1(가장 적은 학습 데이터, 가장 큰 분포 이동)에서는 Hybrid 가 1.4571 낮습니다.** 조건별 흔들림 지표도 Hybrid 가 작지만, 이는 17개 그룹 모두에서 더 높은 오차 수준을 유지한 채 평탄한 것입니다.
- **따라서 Hybrid 가 일반적으로 더 robust 하다고 주장할 수 없습니다.** Step 22 결과만 인용해 우위를 주장하는 것은 공정한 비교가 아닙니다.

## 8. 데이터 독립성

| 비교 | wafer+Stage 키 겹침 | 공유 WAFER_ID 수 |
|---|---|---|
| training vs validation | 0 | 115 |
| training vs test | 0 | 113 |
| validation vs test | 0 | 34 |

같은 wafer+Stage 기록이 두 데이터에 동시에 들어간 경우는 없습니다. 다만 같은 WAFER_ID 가 서로 다른 Stage 로 양쪽에 나타납니다. **완전한 독립이 아닙니다.**

## 9. 최종 그림

| 최종 그림 | 출처 |
|---|---|
| fig1_model_concept.png | Step 28 에서 새로 그린 구조 개념도 |
| fig2_train_val_test_rmse.png | reports/step27_test/figures/fig1_train_val_test_rmse.png |
| fig3_fixedk_hybrid_pure.png | reports/step27_test/figures/fig4_fixedk_hybrid_pure.png |
| fig4_test_actual_vs_predicted.png | reports/step27_test/figures/fig3_test_actual_vs_predicted.png |
| fig5_hybrid_k_importance.png | reports/step21_interpretation/figures/fig1_hybrid_k_importance.png |
| fig6_feature_group_share.png | reports/step21_interpretation/figures/fig3_group_share_comparison.png |
| fig7_distribution_shift.png | reports/step25_final_robustness/figures/fig3_shift_rmse.png |

개념도 한 장만 Step 28 에서 새로 그렸습니다. 나머지 여섯 장은 기존 단계에서 저장한 그림을 그대로 복사했습니다. 새 분석은 하지 않았습니다.

## 10. 연구 스토리 (근거 Step 표시)

| 단계 | 내용 | 근거 |
|---|---|---|
| Background | CMP 공정의 MRR 예측은 장비 상태와 소모품 마모에 따라 달라진다. 고전적 설명은 Preston 식 MRR = k·P·V 이다. | Step 1, 참고문헌 |
| Problem | PHM 2016 데이터는 변수 스케일이 가려져 있어 물리 P 와 V 를 직접 복원할 수 없다. 센서 기반 proxy 로 Preston 구조를 검증할 수 있는가. | Step 2~6 |
| Fixed Preston baseline | P 후보 12개와 V 후보 6개, 요약 방식 3가지를 비교해 PV proxy 를 정하고, 절편 없이 fold 별 k 를 맞췄다. | Step 14, 14b, 15 |
| Why fixed-k fails | 상수 k 로는 R² 가 0 근처이거나 음수다. 평균 예측보다도 나쁘다. | Step 16, 27 |
| k(x) Hybrid formulation | k_target = MRR / PV_proxy 를 정의하고 fold 별 training 통계로만 표준화해 XGBoost 로 k(x) 를 학습했다. 표준화 없이는 XGBoost 가 상수 예측에 머물렀다. | Step 16~19 |
| Performance recovery | 세 데이터 모두에서 fixed-k 대비 RMSE 약 60% 감소. | Step 20, 26, 27 |
| Pure ML comparison | Pure ML 에도 같은 54개 격자를 주어 공정하게 비교했다. 세 데이터 모두에서 Pure ML 이 더 낮다. | Step 24, 26, 27 |
| Interpretability | k(x) 기여도는 Stage 와 dresser 사용량이 주도하고, pressure 비중이 Pure ML 보다 크며, 기여의 89.3% 가 PV 밖 정보에서 나온다. | Step 21 |
| Robustness | tuning 을 맞춘 뒤에는 Pure ML 이 대부분 조건에서 더 낮다. shift 평균은 동률, 극단 조건 T1 에서만 Hybrid 가 낫다. | Step 22, 23, 25 |
| External validation/test | validation 과 test 를 각각 한 번만 사용했고 결과를 보고 모델을 수정하지 않았다. 모델 순위가 세 데이터에서 동일하다. | Step 26, 27 |
| Limitations | 물리 단위 부재, proxy 의 성격, 4-5-6 한정, wafer 부분 공유, 단일 공개 데이터셋. | `final_claims_and_limitations.md` |
| Conclusion | 구조를 나눈 이점은 정확도가 아니라 해석 가능성과 fixed-k 대비 회복에 있다. | Step 21, 25, 27 |

## 11. 저장 파일

| 파일 | 내용 |
|---|---|
| `master_metrics.csv` | 세 모델 × 세 데이터의 R², MAE, RMSE |
| `recovery_analysis.csv` | fixed-k → Hybrid 의 절대 및 상대 감소 |
| `pureml_hybrid_tradeoff.csv` | Pure ML 과 Hybrid 의 차이와 상대 차이 |
| `consistency_with_request.csv` | 요청서 기대값과 저장값 대조 |
| `fixed_k_training_variants.csv` | training fixed-k 의 두 가지 PV 정의별 값 |
| `robustness_summary_across_steps.csv` | Step 22, 23, 25 의 승패 집계 |
| `shift_summary_step25.csv` | Step 25 의 shift 요약 지표 |
| `interpretation_group_shares.csv` | Step 21 의 feature 그룹별 비중 |
| `interpretation_top15_features.csv` | Step 21 의 상위 15개 feature |
| `figure_index.csv` | 최종 그림과 출처 |

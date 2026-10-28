# 연구 포스터 구성안

- Step 1~27 의 저장된 결과만 사용했습니다. 새로운 분석이나 학습은 없습니다.
- 그림은 모두 `reports/final_summary/figures/` 에 있습니다. 개념도 한 장만 Step 28 에서 새로 그렸고 나머지는 기존 단계의 그림입니다.

## 1. 포스터에 쓸 핵심 수치

| 항목 | 값 |
|---|---|
| test 에서 Pure ML RMSE | 3.1856 |
| test 에서 Hybrid RMSE | 3.4650 |
| test 에서 Fixed-k Preston RMSE | 9.1367 |
| test 에서 Fixed-k Preston R² | -0.0700 |
| test 에서 Hybrid 의 상대 감소율 (%) | 62.0762 |
| test 에서 Hybrid 의 상대 차이 (%) | 8.7685 |
| Step 25 에서 17개 조건 그룹 중 Hybrid 가 더 좋은 그룹 수 | 0 |
| Step 25 의 shift RMSE 평균 차이 (Hybrid − Pure ML) | +0.0061 |
| Hybrid k(x) 에서 PV 를 구성하는 4개 feature 의 비중 (%) | 10.7 |

## 2. 섹션별 구성

| 섹션 | 사용할 그림 | 핵심 문장 |
|---|---|---|
| Background | fig1_model_concept.png | CMP 의 재료 제거율은 고전적으로 Preston 식 MRR = k·P·V 로 설명되지만, 실제 장비 데이터에서 k 가 상수인지는 확인된 적이 없습니다. |
| Problem | fig1_model_concept.png | PHM 2016 데이터는 변수 스케일이 가려져 있어 물리 P 와 V 를 복원할 수 없으므로, 센서 기반 proxy 로 Preston 구조를 검증합니다. |
| Fixed Preston baseline | fig1_model_concept.png | P 후보 12개와 V 후보 6개, 요약 방식 3가지를 비교해 PV_proxy = P9 × V3 를 정하고, 절편 없이 하나의 k 를 맞췄습니다. |
| Why fixed-k fails | fig3_fixedk_hybrid_pure.png | 상수 k 를 쓰는 Preston baseline 은 test 에서 RMSE 9.1367, R² -0.0700 으로 평균 예측보다도 나쁩니다. |
| k(x) Hybrid formulation | fig1_model_concept.png | 구조와 PV 를 그대로 둔 채 k 만 공정상태의 함수 k(x) 로 바꾸고, k 의 표준화에는 training 통계만 사용했습니다. |
| Performance recovery | fig3_fixedk_hybrid_pure.png | k 를 k(x) 로 바꾸는 것만으로 test RMSE 가 62.0762% 줄어듭니다. 이 회복은 training, validation, test 에서 모두 약 60% 로 반복됩니다. |
| Pure ML comparison | fig2_train_val_test_rmse.png | 같은 크기의 tuning 을 받은 순수 데이터 기반 모델이 세 데이터 모두에서 더 정확하며, test 기준 차이는 8.7685% 입니다. |
| Interpretability | fig5_hybrid_k_importance.png | k(x) 기여는 Stage 구분과 dresser 계열 사용량이 주도하며, PV 를 구성하는 4개 feature 의 비중은 10.7% 에 그칩니다. |
| Interpretability (보조) | fig6_feature_group_share.png | Hybrid 는 PV 에 들어간 정보만 다시 학습하는 것이 아니라 소모품 사용량과 slurry, rotation 상태도 함께 사용합니다. |
| Robustness | fig7_distribution_shift.png | tuning 을 맞춘 뒤 다시 비교하면 17개 조건 그룹 중 Hybrid 가 더 좋은 그룹은 0개이고, shift 평균 차이는 +0.0061 로 사실상 같습니다. |
| External validation/test | fig2_train_val_test_rmse.png | validation 과 test 를 각각 한 번만 사용했고 결과를 보고 모델을 수정하지 않았으며, 모델 순위가 세 데이터에서 동일합니다. |
| Prediction quality | fig4_test_actual_vs_predicted.png | test 에서 두 모델 모두 실제 MRR 을 따라가며, Pure ML 의 RMSE 는 3.1856, Hybrid 는 3.4650 입니다. |
| Limitations | 그림 없음 | 변수에 물리 단위가 없고, PV 는 proxy 이며, 비교는 4-5-6 경로에 한정되고, 세 데이터가 wafer 수준에서 완전히 독립적이지는 않습니다. |
| Conclusion | fig3_fixedk_hybrid_pure.png | 구조를 나눈 이점은 정확도가 아니라 해석 가능성과 fixed-k 대비 회복에 있습니다. |

## 3. 사용할 그림

| 그림 | 설명 |
|---|---|
| fig1_model_concept.png | 세 모델 구조 개념도. PV 가 센서 기반 proxy 임을 그림 안에 명시했습니다. |
| fig2_train_val_test_rmse.png | 세 모델의 training, validation, test RMSE 비교 |
| fig3_fixedk_hybrid_pure.png | fixed-k → Hybrid → Pure ML 의 성능 변화 |
| fig4_test_actual_vs_predicted.png | test 의 실제 MRR 대 예측 MRR |
| fig5_hybrid_k_importance.png | Hybrid k(x) 의 feature 기여도 상위 15개 |
| fig6_feature_group_share.png | feature 그룹별 기여 비중, Hybrid 대 Pure ML |
| fig7_distribution_shift.png | distribution-shift 7개 시험 결과 |

## 4. 포스터에서 피해야 할 표현

- "Hybrid 가 더 정확하다" 또는 "Hybrid 가 더 robust 하다" 로 쓰지 않습니다. 세 데이터 모두에서 Pure ML 이 더 정확하고, tuning 을 맞춘 robustness 비교에서도 Pure ML 이 우세합니다.
- P9 과 V3 를 실제 polishing pressure 나 relative linear velocity 로 부르지 않습니다. 센서 기반 proxy 로 표기합니다.
- k 를 물리 상수로 부르지 않습니다. effective k 로 표기합니다.
- SHAP 기여도를 인과로 설명하지 않습니다. 모델이 사용한 연관으로 설명합니다.

## 5. 한 줄 메시지

**상수 k 의 Preston 은 이 데이터에서 실패하고, k(x) 로 바꾸면 약 60% 회복되지만, 순수 데이터 기반 모델이 여전히 더 정확합니다. Hybrid 의 값어치는 정확도가 아니라 구조적 해석 가능성입니다.**

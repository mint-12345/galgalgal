# Step 16. Model 3 준비: 공통 데이터, k_target, 같은 subset 기준 Model 1 / Model 2 성능

- training 만 사용했습니다. test / validation 은 사용하지 않았고, raw 파일은 수정하지 않았습니다.
- 제외 기준은 앞 단계와 같습니다: 연속 완전 중복 1,819행, MRR > 1,000 인 wafer + Stage 4개.
- fold 는 Step 8 번호를 그대로 사용했습니다. 새로 만들지 않았습니다.
- 코드: `scripts/step16_model3_preparation.py`
- 결과: `reports/step16_model3_prep/`, Model 3 준비용 표: `data/processed/step16/step16_model3_prep_table.csv`
- **Model 3 (k(x) 예측) 는 만들지 않았고, 어떤 모델이 최종적으로 좋은지도 결정하지 않았습니다.**

## 1. 비교 대상

- training 1,977개 중 CHAMBER_PATH = 4-5-6 인 **1,609개** wafer + Stage 만 사용했습니다.
- 이유는 Step 15 에서 현재 P/V proxy 를 계산할 수 없는 sample 이 모두 1-2-3 경로에서 나왔기 때문입니다.
- **1-2-3 경로 364개와 other 경로 4개를 오류로 보거나 삭제하지 않았습니다.** 프로젝트의 다른 표에는 그대로 남아 있고, 이번 strict Preston / Hybrid 비교에서만 빠집니다.
- fold 별 sample 수: fold 1 323, fold 2 323, fold 3 321, fold 4 321, fold 5 321

## 2. 주 PV proxy

- P = RETAINER_RING_PRESSURE, V = abs(WAFER_ROTATION - HEAD_ROTATION)
- 각 raw row 에서 PV(t) = P(t) × V(t) 를 계산한 뒤, **P 와 V 가 모두 0 이 아닌 row 만** 평균했습니다.
- 이 정의는 Step 15 의 P4 × V5 nonzero 방식(B, C)과 같습니다. 실제로 저장된 Step 15 값과 완전히 일치하는 것을 확인했습니다.
- **1,609개 모두 PV_proxy 를 계산할 수 있습니다.** 계산 불가 0개, 값이 0 인 sample 0개입니다.
- PV 계산에 쓰인 row 수는 sample 당 평균 259.1개, 최소 24개, 중앙값 296개, 최대 1,951개입니다.

| 값 | 평균 | 표준편차 | 최소 | 중앙값 | 최대 |
|---|---|---|---|---|---|
| PV_proxy | 270,144.08 | 15,277.34 | 210,683.31 | 271,999.83 | 337,907.65 |
| 실제 MRR | 76.559 | 8.727 | 53.427 | 76.593 | 101.465 |

- 1,609개에서 PV_proxy 와 실제 MRR 의 Pearson 상관은 0.2643 입니다.

## 3. k_target 분포

k_target = 실제 MRR / PV_proxy 입니다. **실제 단위의 Preston coefficient 가 아니라, 현재 센서 기반 PV proxy 에 대응하는 effective k target 입니다.**

| 구분 | n | 최소 | 1% | 5% | 중앙값 | 평균 | 95% | 99% | 최대 | 표준편차 |
|---|---|---|---|---|---|---|---|---|---|---|
| 전체 | 1,609 | 0.000185 | 0.000209 | 0.000231 | 0.000283 | 0.000284 | 0.000335 | 0.000353 | 0.000413 | 0.000032 |
| Stage A | 797 | 0.000185 | 0.000205 | 0.000222 | 0.000269 | 0.000268 | 0.000303 | 0.000323 | 0.000353 | 0.000024 |
| Stage B | 812 | 0.000198 | 0.000215 | 0.000243 | 0.000304 | 0.000300 | 0.000344 | 0.000359 | 0.000413 | 0.000031 |

**비정상적으로 크거나 작은 값이 있는가**
- 음수나 0 인 k_target 은 없습니다.
- 최대값은 중앙값의 1.46배, 최소값은 중앙값의 0.656배입니다. 몇 자릿수씩 벗어나는 값은 없습니다.
- 1% 지점 아래 또는 99% 지점 위인 sample 은 34개입니다. 목록은 `k_target_tail_samples.csv` 에 저장했고, **이상값을 삭제하지 않았습니다.**

**Stage 별 차이**
- Stage B 의 중앙값이 Stage A 의 1.13배입니다 (0.000304 대 0.000269).
- 두 분포는 겹칩니다. Stage A 중 Stage B 중앙값보다 큰 sample 은 4.5%, Stage B 중 Stage A 중앙값보다 작은 sample 은 18.5% 입니다.

## 4. 같은 1,609개 기준 Model 1 / Model 2 성능

각 fold 에서 **학습 데이터도 4-5-6 sample 만** 사용해 다시 학습했습니다. 이전에 전체 1,977개로 학습한 모델의 4-5-6 결과를 가져오지 않았습니다.

| 모델 | 구성 |
|---|---|
| Model1_RF | RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_leaf=1, max_features=1.0, random_state=42, n_jobs=1), Step 7 Baseline feature |
| Model1_XGB | XGBRegressor(random_state=42, n_jobs=-1), Step 10 기본 설정, Step 7 Baseline feature |
| Model2_fixed_k | MRR = k × PV_proxy, fold 마다 학습 sample 로만 절편 없는 최소제곱 k 하나 |

모델 입력 열은 31개입니다. 이 subset 에서는 CHAMBER_PATH 값이 하나뿐이라 one-hot 열이 생기지 않습니다.

**5-fold 평균 ± 표준편차**

| 모델 | R² | MAE | RMSE |
|---|---|---|---|
| Model1_RF | 0.8246 ± 0.0220 | 2.7846 ± 0.1642 | 3.6401 ± 0.2167 |
| Model1_XGB | 0.8296 ± 0.0248 | 2.7451 ± 0.1744 | 3.5826 ± 0.1942 |
| Model2_fixed_k | 0.0113 ± 0.0692 | 6.9583 ± 0.1837 | 8.6511 ± 0.3058 |

**5개 fold 예측을 모아 계산한 값**

| 모델 | R² | MAE | RMSE |
|---|---|---|---|
| Model1_RF | 0.8254 | 2.7847 | 3.6455 |
| Model1_XGB | 0.8309 | 2.7454 | 3.5872 |
| Model2_fixed_k | 0.0156 | 6.9585 | 8.6556 |

**fold 별 RMSE**

| 모델 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
|---|---|---|---|---|---|
| Model1_RF | 3.5127 | 3.9440 | 3.7952 | 3.4656 | 3.4831 |
| Model1_XGB | 3.5736 | 3.9191 | 3.5074 | 3.4627 | 3.4502 |
| Model2_fixed_k | 8.5365 | 8.8908 | 8.8364 | 8.1636 | 8.8282 |

**Model 2 의 fold 별 effective k**

5개 fold 의 k 는 0.000282535 ~ 0.000283264 이고, 평균 0.000282980, 표준편차 3.047e-07, 변동계수 0.00108 입니다.

## 5. 이전 단계 값과의 비교 (참고)

이번에는 4-5-6 만으로 다시 학습했고, 이전에는 전체 1,977개로 학습한 뒤 4-5-6 부분만 본 값입니다.

| 모델 | 이전 (전체로 학습) | 이번 (4-5-6 만으로 학습) |
|---|---|---|
| RF 같은 설정, RMSE | 3.6481 (Step 13) | 3.6401 |
| RF 같은 설정, R² | 0.8238 | 0.8246 |
| XGB 기본 설정, RMSE | 3.5831 (Step 13) | 3.5826 |
| XGB 기본 설정, R² | 0.8296 | 0.8296 |
| 고정 k Preston, RMSE | 10.3650 (Step 15, 모든 경로로 k 추정) | 8.6511 |
| 고정 k Preston, R² | -0.4193 | 0.0113 |

- Model 1 두 모델은 학습 범위를 4-5-6 으로 줄여도 성능이 거의 같습니다.
- Model 2 는 4-5-6 만으로 k 를 추정하자 RMSE 가 10.3650 에서 8.6511 로 줄었습니다. k 도 0.000304 에서 0.000283 으로 바뀌었습니다. 다만 R² 는 0.0113 으로, 여전히 평가 fold 평균으로 예측하는 수준과 큰 차이가 없습니다.

## 6. 저장 파일

| 파일 | 내용 |
|---|---|
| `data/processed/step16/step16_model3_prep_table.csv` | Model 3 준비용 표: WAFER_ID, STAGE, CHAMBER_PATH, FOLD, PV_PROXY, AVG_REMOVAL_RATE, K_TARGET (1,609행) |
| `reports/step16_model3_prep/oof_predictions.csv` | 세 모델의 OOF 예측과 오차, PV_proxy, k_target |
| `reports/step16_model3_prep/fold_metrics.csv` | 모델 x fold 별 R², MAE, RMSE, 학습/평가 sample 수 |
| `reports/step16_model3_prep/summary.csv` | 모델 x 지표 별 5-fold 평균, 표준편차, 최소, 최대 |
| `reports/step16_model3_prep/pooled_metrics.csv` | 5개 fold 예측을 모아 계산한 지표 |
| `reports/step16_model3_prep/k_target_stats.csv` | 전체 / Stage 별 k_target 분포 |
| `reports/step16_model3_prep/k_target_tail_samples.csv` | 1% 아래 또는 99% 위인 34개 sample |
| `reports/step16_model3_prep/model2_effective_k_by_fold.csv` | Model 2 의 fold 별 effective k |

## 7. 확인한 내용

- PV_proxy 가 Step 15 의 P4 × V5 nonzero 값과 완전히 같은 것을 확인했습니다.
- 1,609개 모두 PV_proxy 가 계산 가능하고 0 이 아닌 것을 확인했습니다.
- 각 fold 의 학습 데이터와 평가 데이터가 모두 4-5-6 이고, 학습과 평가 사이에 겹치는 WAFER_ID 가 없는 것을 확인했습니다.
- 코드를 두 번 실행해 8개 출력 파일이 완전히 같게 만들어지는 것을 확인했습니다.

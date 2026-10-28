# Step 12. Model 1 두 번째 feature 후보 비교

- 기준: Step 7 feature 표(Baseline), Step 8 의 같은 5-fold
- training 만 사용했습니다. test / validation 은 사용하지 않았고, raw 파일은 수정하지 않았습니다.
- 코드: `scripts/step12_build_candidate_tables.py` (후보 표), `scripts/step12_compare_feature_candidates.py` (비교)
- 후보 표: `data/processed/step12/`
- 결과: `reports/step12_feature_candidates/`
- 이번 단계는 비교만 했습니다. feature 구성이나 모델을 최종 결정하지 않았습니다.

## 1. feature 구성

대상 변수 10개: PRESSURIZED_CHAMBER_PRESSURE, MAIN_OUTER_AIR_BAG_PRESSURE, CENTER_AIR_BAG_PRESSURE, RETAINER_RING_PRESSURE, RIPPLE_AIR_BAG_PRESSURE, EDGE_AIR_BAG_PRESSURE, SLURRY_FLOW_LINE_A, SLURRY_FLOW_LINE_C, WAFER_ROTATION, STAGE_ROTATION

| 구성 | 추가한 값 | feature 수 |
|---|---|---|
| Baseline | Step 7 표 그대로 | 32 |
| C5_nonzero | 변수마다 0 이 아닌 행의 비율, 0 이 아닌 행만의 평균 (20열) | 52 |
| C6_distribution | 변수마다 25% 지점, 중앙값, 75% 지점 (30열) | 62 |
| C7_combined | C5 + C6 추가 열 모두 (50열) | 82 |

- 후보 표는 Step 7 과 같은 제외 기준으로 raw 에서 계산했고, 여러 CSV 에 걸친 wafer + Stage 25개는 모든 행을 합쳐 계산했습니다.
- 같은 데이터로 Baseline 숫자 30열을 다시 계산해 Step 7 표와 같은 것을 확인했습니다.
- C5 의 값은 "값이 0 이 아니었던 비율"과 "0 이 아닐 때의 평균"일 뿐이며, 공정 상태나 연마로 해석하지 않았습니다.
- 분위수는 pandas 기본 방식(선형 보간)으로 계산했습니다.

**C5 의 빈 값**
- 0 이 아닌 행이 하나도 없어 NONZERO_MEAN 이 빈 값인 칸은 1,873개입니다.
- 빈 값이 하나 이상 있는 행은 225개이고 모두 CHAMBER_PATH 1-2-3 입니다. 그중 130행은 10개 변수 모두 빈 값입니다.
- RF 와 XGB 는 빈 값을 직접 처리하므로 채우지 않았습니다.

**C6 에서 확인된 값**
- STAGE_ROTATION_P25 는 1,977행 모두 0 입니다.
- WAFER_ROTATION_P25 는 99.9%, STAGE_ROTATION_MEDIAN 은 98.9% 의 행이 같은 값입니다.

## 2. 모델

Step 10 과 같은 설정을 썼고 튜닝하지 않았습니다.

- RF: one-hot(STAGE, CHAMBER_PATH) + 숫자 그대로 → RandomForestRegressor(random_state=42, n_jobs=1)
- XGB: one-hot + 숫자 그대로 → XGBRegressor(random_state=42, n_jobs=-1)
- 전처리는 학습용 4개 fold 로만 fit 했습니다.
- Baseline 의 RF, XGB 예측은 Step 10 에 저장된 예측과 완전히 같습니다.

## 3. 5-fold 결과 (평균 ± 표준편차)

표준편차는 5개 fold 값의 표본 표준편차입니다.

**전체 1,977행**

| 구성 | 모델 | R² | MAE | RMSE |
|---|---|---|---|---|
| Baseline | RF | 0.9850 ± 0.0026 | 2.7023 ± 0.1516 | 3.6667 ± 0.3206 |
| Baseline | XGB | 0.9836 ± 0.0062 | 2.6927 ± 0.1137 | 3.7941 ± 0.6680 |
| C5_nonzero | RF | 0.9842 ± 0.0029 | 2.7795 ± 0.1356 | 3.7618 ± 0.3459 |
| C5_nonzero | XGB | 0.9835 ± 0.0061 | 2.7533 ± 0.1193 | 3.8196 ± 0.6551 |
| C6_distribution | RF | 0.9849 ± 0.0029 | 2.7149 ± 0.1457 | 3.6802 ± 0.3511 |
| C6_distribution | XGB | 0.9834 ± 0.0064 | 2.7517 ± 0.1345 | 3.8184 ± 0.6846 |
| C7_combined | RF | 0.9842 ± 0.0028 | 2.7811 ± 0.1417 | 3.7640 ± 0.3419 |
| C7_combined | XGB | 0.9835 ± 0.0066 | 2.7355 ± 0.1464 | 3.8134 ± 0.7028 |

**일반 경로 1-2-3 과 4-5-6, 1,973행**

| 구성 | 모델 | R² | MAE | RMSE |
|---|---|---|---|---|
| Baseline | RF | 0.9863 ± 0.0014 | 2.6708 ± 0.1553 | 3.5145 ± 0.2001 |
| Baseline | XGB | 0.9868 ± 0.0015 | 2.6453 ± 0.1541 | 3.4472 ± 0.1868 |
| C5_nonzero | RF | 0.9856 ± 0.0014 | 2.7463 ± 0.1365 | 3.6037 ± 0.1880 |
| C5_nonzero | XGB | 0.9864 ± 0.0011 | 2.7067 ± 0.1088 | 3.5024 ± 0.1535 |
| C6_distribution | RF | 0.9863 ± 0.0015 | 2.6816 ± 0.1468 | 3.5106 ± 0.2011 |
| C6_distribution | XGB | 0.9865 ± 0.0016 | 2.7030 ± 0.1540 | 3.4858 ± 0.1991 |
| C7_combined | RF | 0.9855 ± 0.0015 | 2.7495 ± 0.1391 | 3.6151 ± 0.2020 |
| C7_combined | XGB | 0.9865 ± 0.0013 | 2.6890 ± 0.1126 | 3.4914 ± 0.1822 |

**Chamber 경로 1-2-3, 364행**

| 구성 | 모델 | R² | MAE | RMSE |
|---|---|---|---|---|
| Baseline | RF | 0.5202 ± 0.1305 | 2.0035 ± 0.1933 | 2.6330 ± 0.2341 |
| Baseline | XGB | 0.4758 ± 0.1504 | 2.1804 ± 0.2408 | 2.7506 ± 0.3134 |
| C5_nonzero | RF | 0.5044 ± 0.1274 | 2.0351 ± 0.1834 | 2.6793 ± 0.2147 |
| C5_nonzero | XGB | 0.4014 ± 0.1577 | 2.3003 ± 0.2103 | 2.9420 ± 0.2920 |
| C6_distribution | RF | 0.5204 ± 0.1251 | 2.0044 ± 0.1837 | 2.6360 ± 0.2219 |
| C6_distribution | XGB | 0.4148 ± 0.1544 | 2.2790 ± 0.2394 | 2.9151 ± 0.3082 |
| C7_combined | RF | 0.5023 ± 0.1303 | 2.0388 ± 0.1856 | 2.6841 ± 0.2260 |
| C7_combined | XGB | 0.4257 ± 0.1501 | 2.2807 ± 0.1872 | 2.8827 ± 0.2775 |

**Chamber 경로 4-5-6, 1,609행**

| 구성 | 모델 | R² | MAE | RMSE |
|---|---|---|---|---|
| Baseline | RF | 0.8205 ± 0.0220 | 2.8220 ± 0.1739 | 3.6833 ± 0.2268 |
| Baseline | XGB | 0.8296 ± 0.0259 | 2.7505 ± 0.1786 | 3.5831 ± 0.2198 |
| C5_nonzero | RF | 0.8109 ± 0.0226 | 2.9073 ± 0.1575 | 3.7804 ± 0.2136 |
| C5_nonzero | XGB | 0.8269 ± 0.0208 | 2.7986 ± 0.1136 | 3.6148 ± 0.1766 |
| C6_distribution | RF | 0.8209 ± 0.0229 | 2.8350 ± 0.1661 | 3.6782 ± 0.2318 |
| C6_distribution | XGB | 0.8279 ± 0.0272 | 2.7991 ± 0.1669 | 3.5997 ± 0.2291 |
| C7_combined | RF | 0.8095 ± 0.0245 | 2.9104 ± 0.1577 | 3.7929 ± 0.2263 |
| C7_combined | XGB | 0.8271 ± 0.0219 | 2.7814 ± 0.1216 | 3.6131 ± 0.2007 |

## 4. Baseline 과의 같은 fold 차이 (RMSE)

"후보 - Baseline" 의 5-fold 평균 ± 표준편차와, 후보 RMSE 가 Baseline 보다 큰 fold 수입니다. R² 와 MAE 차이는 `paired_difference_vs_baseline.csv` 에 있습니다.

| 구성 | 모델 | 전체 | 일반 경로 1,973행 | 1-2-3 | 4-5-6 |
|---|---|---|---|---|---|
| C5_nonzero | RF | +0.0950 ± 0.0567 (5/5) | +0.0892 ± 0.0525 (5/5) | +0.0463 ± 0.0269 (5/5) | +0.0971 ± 0.0624 (5/5) |
| C5_nonzero | XGB | +0.0255 ± 0.0640 (3/5) | +0.0552 ± 0.0831 (3/5) | +0.1913 ± 0.1167 (5/5) | +0.0317 ± 0.0940 (3/5) |
| C6_distribution | RF | +0.0134 ± 0.0379 (2/5) | -0.0039 ± 0.0093 (1/5) | +0.0030 ± 0.0349 (2/5) | -0.0051 ± 0.0127 (1/5) |
| C6_distribution | XGB | +0.0243 ± 0.0289 (4/5) | +0.0386 ± 0.0550 (4/5) | +0.1644 ± 0.0826 (5/5) | +0.0166 ± 0.0657 (3/5) |
| C7_combined | RF | +0.0972 ± 0.0486 (5/5) | +0.1006 ± 0.0541 (5/5) | +0.0511 ± 0.0151 (5/5) | +0.1096 ± 0.0650 (5/5) |
| C7_combined | XGB | +0.0193 ± 0.0842 (3/5) | +0.0442 ± 0.1137 (3/5) | +0.1320 ± 0.0405 (5/5) | +0.0300 ± 0.1341 (3/5) |

## 5. CHAMBER_PATH other 4행의 절대오차

R² 는 계산하지 않았습니다. 실제값과 예측값은 `other_path_rows.csv` 에 있습니다.

| 구성 | 모델 | -4019511766 B (실제 74.96) | 2939014292 B (실제 86.31) | 3015014228 B (실제 73.15) | 2665292924 A (실제 78.28) |
|---|---|---|---|---|---|
| Baseline | RF | 7.877 | 2.702 | 18.811 | 43.397 |
| Baseline | XGB | 11.740 | 10.691 | 8.353 | 73.032 |
| C5_nonzero | RF | 8.709 | 3.886 | 18.818 | 45.139 |
| C5_nonzero | XGB | 12.743 | 8.862 | 11.200 | 70.070 |
| C6_distribution | RF | 7.911 | 2.844 | 19.330 | 46.286 |
| C6_distribution | XGB | 10.936 | 11.972 | 12.204 | 71.485 |
| C7_combined | RF | 8.593 | 2.213 | 18.810 | 43.687 |
| C7_combined | XGB | 12.073 | 8.535 | 10.613 | 71.351 |

## 6. 결과를 읽을 때 확인된 사실

- **어느 후보도 네 평가 범위의 RMSE 평균을 Baseline 보다 뚜렷하게 낮추지 않았습니다.** 평균 RMSE 가 Baseline 보다 낮은 경우는 RF + C6 의 일반 경로(-0.0039)와 4-5-6(-0.0051) 두 곳이고, 그 차이는 fold 간 차이의 표준편차보다 작습니다.
- **RF 에 C5 또는 C7 을 넣으면 네 범위 모두에서 5개 fold 전부 RMSE 가 Baseline 보다 컸습니다.** 전체 RMSE 평균 차이는 C5 +0.0950, C7 +0.0972 입니다.
- **XGB 는 세 후보 모두 Chamber 경로 1-2-3 안에서 5개 fold 전부 RMSE 가 Baseline 보다 컸습니다.** 평균 차이는 C5 +0.1913, C6 +0.1644, C7 +0.1320 입니다.
- **RF + C6 은 Baseline 과 가장 가깝습니다.** 일반 경로 1,973행 기준 RMSE 가 Baseline 보다 큰 fold 는 5개 중 1개입니다.
- **other 경로 2665292924 A 의 큰 오차는 후보를 넣어도 비슷하게 남았습니다.** RF 는 43.397 에서 43.687~46.286, XGB 는 73.032 에서 70.070~71.485 입니다.

## 7. 저장 파일

| 파일 | 내용 |
|---|---|
| `data/processed/step12/step12_candidate5_nonzero.csv` 외 2개 | 후보별 feature 표 (WAFER_ID, 모든 feature, MRR 정답) |
| `data/processed/step12/step12_feature_columns.csv` | 구성별 Baseline 열과 추가 열 목록 |
| `reports/step12_feature_candidates/fold_metrics.csv` | 구성 x 모델 x fold x 범위(4개) 별 R², MAE, RMSE |
| `reports/step12_feature_candidates/summary.csv` | 평균, 표준편차, 최소, 최대, 5개 fold 예측을 모아 계산한 값 |
| `reports/step12_feature_candidates/paired_difference_vs_baseline.csv` | 같은 fold 기준 후보 - Baseline 차이 요약 |
| `reports/step12_feature_candidates/other_path_rows.csv` | other 4행의 실제값, 예측값, 절대오차 |
| `reports/step12_feature_candidates/oof/oof_{구성}_{모델}.csv` | 8개 조합의 wafer + Stage 별 실제 MRR, 예측 MRR, 오차 |
| `reports/step12_feature_candidates/feature_set_info.csv` | 구성별 feature 수와 빈 값 수 |
| `reports/step12_feature_candidates/run_time_reference.csv` | fold 별 실행 시간 (실행마다 달라짐) |

## 8. 재현 확인

- 후보 표 생성 코드를 다시 실행해 세 후보 표가 완전히 같게 만들어지는 것을 확인했습니다.
- C7_combined 의 fold 4 를 RF, XGB 로 다시 학습해 예측값이 저장된 예측과 완전히 같은 것을 확인했습니다.

# Step 6. 공정 변수가 wafer + Stage 안에서 변하는 방식

- 대상: training 공정 데이터와 training MRR. test / validation 은 열지 않았습니다.
- 분석에서만 제외했습니다. raw 파일은 수정하지 않았습니다.
  - 같은 파일에서 바로 윗줄과 완전히 같은 1,819행
  - MRR 이 1,000 을 넘는 4개 묶음: 1834206730 A, 1834206944 A, 1834206972 A, 2058207580 A (1,051행)
- 분석 행 669,874, wafer + Stage 1,977개입니다.
- WAFER_ID, STAGE, CHAMBER, TIMESTAMP 는 구분 정보로만 썼고, 나머지 21개 변수를 평가했습니다.
- feature 결정, 한 줄 변환, 모델 학습은 하지 않았습니다.
- 확인 코드: `scripts/step6_inspect_variable_behavior.py`
- 변수별 전체 측정값: `reports/step6_variable_behavior.csv`

## 1. 변수별 정리

| 변수 | 요약 | wafer 안 시간 변화 | Chamber 차이 | 여러 wafer 비교 |
|---|---|---|---|---|
| MACHINE_ID | 거의 고정 (모든 행이 같은 값) | 완전 고정 | 해당 없음 | 모든 wafer 같은 값 |
| MACHINE_DATA | Chamber에 따라 차이가 큼 | Chamber 안에서는 고정 | 큼 | 많이 다름 |
| USAGE_OF_BACKING_FILM | wafer 안에서는 거의 고정, wafer마다 다름 | 거의 고정 (조금씩 증가) | 해당 없음 | 많이 다름 |
| USAGE_OF_MEMBRANE | wafer 안에서는 거의 고정, wafer마다 다름 | 거의 고정 (조금씩 증가) | 해당 없음 | 많이 다름 |
| USAGE_OF_PRESSURIZED_SHEET | wafer 안에서는 거의 고정, wafer마다 다름 | 거의 고정 (조금씩 증가) | 해당 없음 | 많이 다름 |
| USAGE_OF_POLISHING_TABLE | wafer 안에서는 거의 고정, wafer마다 다름 | 거의 고정 (조금씩 증가) | 해당 없음 | 많이 다름 |
| USAGE_OF_DRESSER | wafer 안에서는 거의 고정, wafer마다 다름 | 거의 고정 (조금씩 증가) | 해당 없음 | 많이 다름 |
| USAGE_OF_DRESSER_TABLE | wafer 안에서는 거의 고정, wafer마다 다름 | 거의 고정 (조금씩 증가) | 해당 없음 | 많이 다름 |
| PRESSURIZED_CHAMBER_PRESSURE | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 많이 다름 |
| MAIN_OUTER_AIR_BAG_PRESSURE | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 많이 다름 |
| CENTER_AIR_BAG_PRESSURE | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 많이 다름 |
| RIPPLE_AIR_BAG_PRESSURE | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 많이 다름 |
| EDGE_AIR_BAG_PRESSURE | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 많이 다름 |
| RETAINER_RING_PRESSURE | 시간에 따라 변화 | 크게 변화 | 영향 작음 | 조금 다름 |
| SLURRY_FLOW_LINE_A | 시간에 따라 변화 | 크게 변화 | 영향 작음 | 조금 다름 |
| SLURRY_FLOW_LINE_B | 대체로 고정, Chamber에 따라 차이 있음 | 거의 고정 | 큼 | 많이 다름 |
| SLURRY_FLOW_LINE_C | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 많이 다름 |
| WAFER_ROTATION | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 많이 다름 |
| STAGE_ROTATION | 시간에 따라 변화 | 크게 변화 | 일부 차이 | 조금 다름 |
| HEAD_ROTATION | 거의 고정 | 거의 고정 | 일부 차이 | 거의 같음 |
| DRESSING_WATER_STATUS | 시간에 따라 변화 | 크게 변화 (0/1 전환) | 일부 차이 | 많이 다름 |

## 2. 판단에 쓴 값과 기준

wafer + Stage 마다 같은 가중치를 주고, 각 변수의 변동을 세 부분으로 나눴습니다.

- wafer 사이: wafer 평균끼리 다른 정도
- Chamber 사이: 같은 wafer 안에서 Chamber 평균끼리 다른 정도
- 시간: 같은 wafer, 같은 Chamber 안에서 시간에 따라 변하는 정도

| 질문 | 쓴 값 | 표현 기준 | 실제 값이 벌어진 틈 |
|---|---|---|---|
| wafer 안 시간 변화 | wafer 안 변동계수(표준편차/평균)의 중앙값 | 0.1 미만이면 거의 고정 | 0.08 이하 변수들과 0.17 이상 변수들로 나뉨 |
| Chamber 안에서는 고정 | 모든 Chamber 안에서 값이 하나뿐인 wafer 비율 | 99% 이상 | MACHINE_DATA 100%, 나머지 50% 이하 |
| Chamber 차이 | wafer 안 변동 중 Chamber 사이 변동 비율 | 0.5 이상 큼, 0.2~0.5 일부, 0.2 미만 작음 | 뚜렷한 틈 없음 |
| Chamber 차이 해당 없음 | 전체 변동 중 wafer 안 변동 비율 | 0.01 미만 | usage 0.001 이하, 나머지 0.33 이상 |
| 여러 wafer 비교 | wafer 평균의 10~90% 분위 폭 / 전체 행의 1~99% 분위 폭 | 0.1 미만 거의 같음, 0.3 미만 조금 다름 | HEAD_ROTATION 0.05, 그다음 0.15 |

- 경계에 가까운 변수가 있습니다. SLURRY_FLOW_LINE_B 의 Chamber 비율은 0.51, DRESSING_WATER_STATUS 는 0.45 이고, WAFER_ROTATION 의 wafer 비교 값은 0.33 입니다. 이 변수들은 표현이 기준에 따라 달라질 수 있습니다.
- 모든 수치는 `reports/step6_variable_behavior.csv` 에 있습니다.

## 3. 함께 확인된 사실

- **MACHINE_DATA 는 CHAMBER 와 1:1 로 같습니다.** 값 1~6 이 Chamber 1~6 과 정확히 일치합니다.
- **USAGE_OF_BACKING_FILM, USAGE_OF_MEMBRANE, USAGE_OF_PRESSURIZED_SHEET 는 같은 값을 비율만 바꾼 것입니다.** 상관계수가 1.0000 이고, MEMBRANE ≈ 0.01186 × BACKING_FILM, PRESSURIZED_SHEET ≈ 0.3 × BACKING_FILM 입니다.
- **usage 6개는 wafer 안에서 시간순으로 줄어들지 않습니다.** 99.8% 이상의 wafer 에서 값이 그대로이거나 조금씩 커집니다.
- **SLURRY_FLOW_LINE_B 는 행의 79% 가 0.909 입니다.** wafer 절반은 0 인 시간이 1% 미만이고, 상위 25% wafer 는 0 인 시간이 45% 이상입니다.
- **HEAD_ROTATION 의 wafer 별 중앙값은 대부분 160 또는 156.8 입니다.** 중앙값이 0 인 wafer 도 4개 있습니다.
- **Chamber 1→2→3 wafer 는 압력, slurry C, rotation 의 Chamber 별 중앙값이 모두 0 입니다.** 이 차이가 "여러 wafer 비교"에서 wafer 마다 다르게 나온 이유 중 하나입니다.
- **Chamber 4~6 안에서 rotation 은 Chamber 별로 다르게 켜집니다.** Chamber 별 중앙값으로 WAFER_ROTATION 은 Chamber 5 에서만 크고, STAGE_ROTATION 은 Chamber 4 와 6 에서만 큽니다. 다만 wafer 안 전체 변동에서는 시간 변화 비율이 더 큽니다.

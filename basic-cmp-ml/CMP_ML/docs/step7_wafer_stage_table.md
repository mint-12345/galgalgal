# Step 7. wafer + Stage 한 행 표 (v1)

- 파일: `data/processed/step7_wafer_stage_table_v1.csv`
- 만든 코드: `scripts/step7_build_wafer_stage_table.py`
- training 데이터만 사용했습니다. raw 파일은 수정하지 않았고, 모델은 만들지 않았습니다.
- 제외: 연속 완전 중복 1,819행, MRR 이 1,000 을 넘는 4개 wafer + Stage
- 사용하지 않은 변수: MACHINE_ID, MACHINE_DATA, USAGE_OF_MEMBRANE, USAGE_OF_PRESSURIZED_SHEET

## 규모

| 항목 | 값 |
|---|---|
| 행 수 (wafer + Stage) | 1,977 |
| 전체 열 | 34 |
| feature | 32 (구분형 2 + 숫자형 30) |
| 빈 값 | 없음 |

## 열 정의

| 열 | 내용 |
|---|---|
| WAFER_ID | wafer 확인용. 머신러닝 입력에는 쓰지 않음 |
| STAGE | A 또는 B |
| CHAMBER_PATH | 시간순 Chamber 기록에서 연속된 같은 번호를 하나로 줄인 경로. `1-2-3`, `4-5-6`, 그 외는 `other` |
| USAGE_OF_DRESSER_MEAN 외 usage 3개 | wafer + Stage 안의 평균 |
| 공정 변수 13개의 `_MEAN`, `_STD` | wafer + Stage 안의 평균과 표준편차 |
| AVG_REMOVAL_RATE | training MRR 정답 |

- 평균과 표준편차를 계산한 13개 변수: 압력 6개, SLURRY_FLOW_LINE_A/B/C, WAFER/STAGE/HEAD_ROTATION, DRESSING_WATER_STATUS
- 표준편차는 해당 wafer + Stage 에 기록된 행 전체의 모표준편차(ddof=0)입니다.
- 행 순서는 각 wafer + Stage 의 첫 기록 시각 순입니다. 시각은 열로 넣지 않았습니다.

## 구성

- STAGE: A 1,162행, B 815행
- CHAMBER_PATH: 4-5-6 1,609행, 1-2-3 364행, other 4행
- other 4개: -4019511766 B, 2939014292 B, 3015014228 B, 2665292924 A

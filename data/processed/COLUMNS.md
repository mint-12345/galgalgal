# wafer_table.csv 컬럼 설명

- 생성: `python src/aggregate.py` (CLAUDE.md 10절 순서). 입력은 `data/raw/`만 쓴다.
- 행: 웨이퍼 1장 = 1행, 키 `(WAFER_ID, STAGE)`. train 1,981 + val 424 + test 424 = **2,829행**. 행은 지우지 않고 플래그만 붙인다.
- 열: 메타·플래그 14 + R 피처 104 (ID 22~125) + S3 35 = **153열**.
- 전처리: split별 완전 중복 행 제거(train 1,819 / val 434 / test 98) → `(WAFER_ID, STAGE, TIMESTAMP)` 순으로 안정 정렬.

## 단위

원자료 설명에 단위가 없다 (CLAUDE.md 8절). 이 표에서 "원 단위"는 원본 CSV 값의 단위를 그대로 쓴다는 뜻이다.

- `TIMESTAMP`는 **초로 추정한다**. 웨이퍼 안의 연속 행 간격이 대부분 1.0이고, 챔버 체류 시간(수십~수백)이 공정 시간으로 자연스럽다.
- 파생 단위: 기울기 = 원 단위/초, AUC = 원 단위 × 초.

## 1. 메타·플래그 (14열)

| 이름 | 의미 | 단위 | 계산 방법 |
|---|---|---|---|
| `WAFER_ID` | 웨이퍼 ID | — | 원본 그대로. 단독으로는 고유하지 않다 (5.2절) |
| `STAGE` | 스테이지 (A/B) | — | 원본 그대로 |
| `split` | train / val / test | — | 읽어 온 폴더 |
| `group` | `A-456`, `B-456`, `A-123` | — | `STAGE` + 거친 챔버 집합이 {1,2,3}이면 `123`, {4,5,6}이면 `456` |
| `chambers` | 거친 챔버 목록 | — | 고유 `CHAMBER` 값을 정렬해 `-`로 연결 (예: `4-5-6`, 불완전이면 `4`, `4-6`) |
| `t_start` | 첫 행 시각 | 초(추정) | 웨이퍼 내 `TIMESTAMP` 최솟값 |
| `t_end` | 마지막 행 시각 | 초(추정) | 웨이퍼 내 `TIMESTAMP` 최댓값 |
| `n_rows` | 행 수 | 행 | 중복 제거 후 행 수 |
| `MRR` | 평균 제거율 (타깃) | 원 단위 | 라벨 파일의 `AVG_REMOVAL_RATE`. **train·val만 채우고 test는 비움** |
| `is_outlier` | 이상치 | 0/1 | `split == train` 이고 `MRR > 400` |
| `flag_incomplete_chamber` | 챔버 불완전 | 0/1 | 거친 챔버가 3개 미만 |
| `flag_long_trace` | 긴 트레이스 | 0/1 | `n_rows >= 2000` |
| `n_allzero_sensors` | 웨이퍼 전체에서 0인 센서 수 | 개 (0~12) | 압력 6 + 슬러리 3 + 회전 3, 총 12개 중 웨이퍼의 모든 행이 0인 센서 수. `DRESSING_WATER_STATUS`는 제외 (5.4절) |
| `n_dup_ts_rows` | 같은 시각에 값이 다른 행 수 | 행 | 웨이퍼 안에서 `TIMESTAMP`가 겹치는 행 수 (쌍이면 2). **처리 방식 미결정(13절). 이 행들은 지우지 않고 피처 계산에 그대로 들어간다** |

## 2. 피처 이름 규칙과 약어

`{통계}_{변수 약어}_{구간}` (CLAUDE.md 6.1절)

- 구간: `s1` = 첫 챔버 (챔버 4 또는 1), `s2` = 나머지 두 챔버, `w` = 웨이퍼 전체 (S3 전용)
- 통계: `mean` 평균, `std` 표준편차(표본, ddof=1), `slope` 선형회귀 기울기, `auc` 사다리꼴 적분, `dur` 지속 시간

| 분류 | 원본 컬럼 | 약어 |
|---|---|---|
| 사용량 | USAGE_OF_BACKING_FILM | `BACKING_FILM` |
| 사용량 | USAGE_OF_DRESSER | `DRESSER` |
| 사용량 | USAGE_OF_POLISHING_TABLE | `POLISH_TABLE` |
| 사용량 | USAGE_OF_DRESSER_TABLE | `DRESSER_TABLE` |
| 사용량 | USAGE_OF_MEMBRANE | `MEMBRANE` |
| 사용량 | USAGE_OF_PRESSURIZED_SHEET | `PSHEET` |
| 압력 | PRESSURIZED_CHAMBER_PRESSURE | `PCHAMBER` |
| 압력 | MAIN_OUTER_AIR_BAG_PRESSURE | `MAIN_OUTER` |
| 압력 | CENTER_AIR_BAG_PRESSURE | `CENTER` |
| 압력 | RETAINER_RING_PRESSURE | `RETAINER` |
| 압력 | RIPPLE_AIR_BAG_PRESSURE | `RIPPLE` |
| 압력 | EDGE_AIR_BAG_PRESSURE | `EDGE` |
| 유량 | SLURRY_FLOW_LINE_A / B / C | `SLURRY_A` / `SLURRY_B` / `SLURRY_C` |
| 유량 | DRESSING_WATER_STATUS | `DRESS_WATER` |
| 회전 | WAFER_ROTATION / STAGE_ROTATION / HEAD_ROTATION | `WAFER_ROT` / `STAGE_ROT` / `HEAD_ROT` |

## 3. R 피처 (Di et al., ID 22~125, 104열)

열 순서는 ID 블록 순서다. 블록 안에서는 변수 → 통계 → 구간 순이다. Di 원문은 블록 안의 개별 ID 순서를 밝히지 않았다.

| ID | 이름 | 개수 | 의미 | 단위 | 계산 방법 |
|---|---|---|---|---|---|
| 22–23 | `dur_POLISH_{s1,s2}` | 2 | 구간별 연마 시간 | 초(추정) | 구간 마지막 − 첫 `TIMESTAMP` |
| 24–59 | `mean_{U}_{s}` | 12 | 사용량 평균 | 원 단위 | 구간 내 평균 |
| | `std_{U}_{s}` | 12 | 사용량 표준편차 | 원 단위 | 구간 내 표본 표준편차 (행 2개 미만이면 NaN) |
| | `slope_{U}_{s}` | 12 | 사용량 감소율 | 원 단위/초 | 구간 내 `TIMESTAMP`에 대한 1차 선형회귀 기울기 (고유 시각 2개 미만이면 NaN) |
| 60–95 | `mean_{P}_{s}` | 12 | 압력 평균 | 원 단위 | 구간 내 평균 |
| | `std_{P}_{s}` | 12 | 압력 표준편차 | 원 단위 | 표본 표준편차 |
| | `auc_{P}_{s}` | 12 | 압력 AUC | 원 단위 × 초 | `TIMESTAMP` 기준 사다리꼴 적분 (`numpy.trapezoid`) |
| 96–119 | `mean_{F}_{s}` | 8 | 유량 평균 | 원 단위 (`DRESS_WATER`는 0/1 → 켜진 비율) | 구간 내 평균 |
| | `std_{F}_{s}` | 8 | 유량 표준편차 | 원 단위 | 표본 표준편차 |
| | `auc_{F}_{s}` | 8 | 유량 AUC | 원 단위 × 초 (`DRESS_WATER`는 켜진 시간, 초) | 사다리꼴 적분 |
| 120–125 | `mean_{R}_{s}` | 6 | 회전 평균 | 원 단위 | 구간 내 평균 (0인 행 포함) |

- `U` = 사용량 6개, `P` = 압력 6개, `F` = SLURRY_A·B·C + DRESS_WATER, `R` = 회전 3개, `s` = s1, s2.
- 유량 4번째 변수를 `DRESSING_WATER_STATUS`로 본 것은 개수로 역산한 추정이다 (6.1절).
- **NaN**: 챔버 4만 있는 train 웨이퍼 2장은 `s2` 구간이 비어 있어서 `_s2` 피처 52개가 모두 NaN이다.

## 4. S3 소모품 상태 (35열)

사용량 6개(`U`) × 아래 6종 = 36개에서 `since_DRESSER_TABLE_w`를 뺀 35개다. 모두 구간 `w`(웨이퍼 전체)이고, 라벨은 쓰지 않는다.

| 이름 | 의미 | 단위 | 계산 방법 |
|---|---|---|---|
| `start_{U}_w` | 시작값 | 원 단위 | 웨이퍼 첫 행의 카운터 값 |
| `delta_{U}_w` | 웨이퍼 내 증가분 | 원 단위 | 마지막 행 − 첫 행 |
| `slope_{U}_w` | 웨이퍼 내 감소율 | 원 단위/초 | 웨이퍼 전체의 `TIMESTAMP`에 대한 선형회귀 기울기 |
| `reset_{U}_w` | 리셋(교체) 플래그 | 0/1 | 아래 "리셋 감지" 참고 |
| `nreset_{U}_w` | 교체 횟수 | 회 | 같은 스테이지에서 이 웨이퍼까지 (포함) 누적한 `reset` 합 |
| `since_{U}_w` | 마지막 교체 이후 누적 사용량 | 원 단위 | `start` − (가장 최근 리셋 웨이퍼의 `start`). 스테이지 첫 리셋 이전 웨이퍼는 NaN. **DRESSER_TABLE은 리셋이 없어 이 컬럼을 만들지 않는다** |

**리셋 감지**
- **정렬:** 스테이지별로 train·val·test 웨이퍼를 합쳐 `t_start` 순으로 늘어놓는다. 센서만 쓰고 라벨은 쓰지 않는다.
- **판정:** 시작값이 직전 웨이퍼의 시작값보다 `0.20 × (스테이지 내 변수 범위)` 이상 줄면 리셋으로 본다.
- **공유 카운터:** 스테이지 A의 A-123과 A-456은 카운터를 공유한다. 시작 시각 순으로 늘어놓으면 두 그룹의 값이 한 줄로 이어지므로 합쳐서 감지한다.
- **비교 기준:** 웨이퍼끼리 시간이 겹치는 경우가 많다(A 1,655쌍 중 903쌍). 그래서 직전 웨이퍼의 마지막 값이 아니라 시작값과 비교한다.
- **웨이퍼 내부:** 웨이퍼 안에서는 리셋(큰 감소)이 관측되지 않았다.
- **임계값 0.20은 임시값이다.** 5.5절 사전 확인치를 그대로 썼고, 확정은 13절 미결정 항목이다.

감지 결과(세 split 합산):

| 변수 | Stage A | Stage B |
|---|---|---|
| POLISH_TABLE | 100 | 87 |
| DRESSER | 3 | 3 |
| BACKING_FILM, MEMBRANE, PSHEET | 각 5 (세 변수의 리셋 웨이퍼가 모두 같음) | 각 5 |
| DRESSER_TABLE | 0 | 0 |

- **NaN**: `since_*`는 첫 리셋 이전 웨이퍼가 NaN이다 (BACKING_FILM·MEMBRANE·PSHEET 87, DRESSER 332, POLISH_TABLE 12).

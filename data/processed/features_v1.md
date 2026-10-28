# features_v1 — SR·전이학습 공통 피처 테이블

- 생성 일시: 2026-10-28 13:54:41
- 생성 코드: `src/make_features.py` (실행: `python src/make_features.py`)
- 파일: `data/processed/features_v1.csv` (2,401행 × 116열), SHA-256 `31aefa89f84aa209535a03fe12190f7cd4fbd43177be11ba51b22acbde7fc4f8`
- 계열 구분: `data/processed/feature_families_v1.csv` (열: feature, family, guaranteed_slot, sr_candidate). 2026-10-28 팀 회의 결정표 #6 기준(채택 후 확정).
- 행: `folds_v1.csv`와 같은 train+val 2,401행 (A-456 983 / B-456 987 / A-123 431).
- MRR은 읽지 않았다 (집계표에서 MRR 열 제외). 상관 계산·변수 선택은 하지 않았다. test 웨이퍼와 `data/locked/`는 쓰지 않았다.

## 검증

| 검증 | 결과 | 내용 |
|---|---|---|
| 행 수 2,401 | 통과 | 2,401행 |
| 키가 folds_v1.csv와 완전히 일치 | 통과 | folds 키 2,401 / features 키 2,401, 차집합 0 |
| 같은 코드로 다시 만들면 같은 파일 | 통과 | SHA-256 31aefa89f84aa209… 두 번 생성 일치 |

## 열 목록과 정의

| 열 | 개수 | 정의 | 출처 |
|---|---|---|---|
| `WAFER_ID`, `STAGE`, `group` | 3 | 키와 그룹 | `folds_v1.csv` |
| `dur_POLISH_{s1,s2}` | 2 | 구간별 연마 시간 (Di ID 22~23) | `wafer_table.csv` 그대로 |
| `{mean,std,slope}_{U}_{s1,s2}` | 36 | 소모품 카운터 6개의 구간 통계 (ID 24~59) | 그대로 |
| `{mean,std,auc}_{P}_{s1,s2}` | 36 | 압력 6개 (ID 60~95) | 그대로 + 결측 처리 |
| `{mean,std,auc}_{F}_{s1,s2}` | 24 | 슬러리 A·B·C + DRESS_WATER (ID 96~119) | 그대로 + 결측 처리 (DRESS_WATER 제외) |
| `mean_{R}_{s1,s2}` | 6 | 회전 3개 평균 (ID 120~125) | 그대로 + 결측 처리 |
| `since_{U}_w` | 5 | 소모품 리셋 후 경과량 = 시작값 − 가장 최근 리셋 웨이퍼의 시작값. 스테이지 첫 리셋 이전은 NaN | `wafer_table.csv` S3 열 그대로 (정의 확인) |
| `P1_on` | 1 | `RETAINER_RING_PRESSURE` 평균, 압력 6개 중 하나라도 > 0 이고 `STAGE_ROTATION` > 0 인 행 | 원본 센서 (train·val) |
| `frac_hi` | 1 | `P1_on` 행 중 `RETAINER_RING_PRESSURE` > 4,000인 행 비율 (7,135 단계 체류 비율) | 원본 센서 |
| `ch56_step` | 1 | 챔버 5·6 단계 수행 여부 (0/1): 압력 6개 중 하나라도 > 0인 행 수 ≥ 245 (A-456) / 255 (B-456). A-123은 NaN | 원본 센서, `results/diagnostics/2026-10-28_김혜연_recipe_cluster.md`와 같은 정의 |
| `allzero` | 1 | 전체-0 판정 센서 12개 중 웨이퍼 전체가 0인 센서가 하나라도 있으면 1 | 원본 센서 (`n_allzero_sensors > 0`과 일치 확인) |

- `U` = BACKING_FILM, DRESSER, POLISH_TABLE, DRESSER_TABLE, MEMBRANE, PSHEET / `P` = PCHAMBER, MAIN_OUTER, CENTER, RETAINER, RIPPLE, EDGE / `F` = SLURRY_A·B·C, DRESS_WATER / `R` = WAFER_ROT, STAGE_ROT, HEAD_ROT. 이름 규칙과 단위는 `COLUMNS.md`.
- **리셋 후 경과량은 5종. DRESSER_TABLE은 데이터 기간 중 교체(리셋)가 없어 카운터 값 자체가 경과량이므로 별도 열을 만들지 않음.** (집계표에도 `since_DRESSER_TABLE_w`가 없다, COLUMNS.md 4절)
- `since_*` 리셋 감지는 집계표대로 train·val·test 센서를 합쳐 한 것이다 (라벨 불필요, 임계값 0.20은 임시값).
- 집계표의 다른 열(`t_start`, `chambers`, 플래그, `start_*`·`delta_*`·`reset_*`·`nreset_*` 등)은 요청 범위 밖이라 넣지 않았다.

## 결측 처리 (0 → NaN)

웨이퍼 전체가 0인 센서(전체-0 판정 12개)에서 계산된 R 피처만 NaN으로 바꿨다. 바꾸기 전 값이 모두 0(또는 이미 NaN)인지 확인했다.
연마 중간에 꺼져 생긴 0, 소모품 카운터, DRESS_WATER는 바꾸지 않았다. 456 그룹에는 전체-0 센서가 없다.

센서별 요약 (웨이퍼 수 = 그 센서의 피처에서 NaN으로 바뀐 웨이퍼 수):

| sensor | n_features | A-456 웨이퍼 | B-456 웨이퍼 | A-123 웨이퍼 |
|---|---|---|---|---|
| CENTER | 6 | 0 | 0 | 233 |
| EDGE | 6 | 0 | 0 | 233 |
| MAIN_OUTER | 6 | 0 | 0 | 233 |
| PCHAMBER | 6 | 0 | 0 | 232 |
| RETAINER | 6 | 0 | 0 | 232 |
| RIPPLE | 6 | 0 | 0 | 229 |
| SLURRY_A | 6 | 0 | 0 | 183 |
| SLURRY_B | 6 | 0 | 0 | 183 |
| SLURRY_C | 6 | 0 | 0 | 196 |
| STAGE_ROT | 2 | 0 | 0 | 214 |
| WAFER_ROT | 2 | 0 | 0 | 215 |

피처별 (NaN으로 바꾼 값 수):

| feature | sensor | A-456 | B-456 | A-123 | total |
|---|---|---|---|---|---|
| mean_PCHAMBER_s1 | PCHAMBER | 0 | 0 | 232 | 232 |
| mean_PCHAMBER_s2 | PCHAMBER | 0 | 0 | 232 | 232 |
| std_PCHAMBER_s1 | PCHAMBER | 0 | 0 | 232 | 232 |
| std_PCHAMBER_s2 | PCHAMBER | 0 | 0 | 232 | 232 |
| auc_PCHAMBER_s1 | PCHAMBER | 0 | 0 | 232 | 232 |
| auc_PCHAMBER_s2 | PCHAMBER | 0 | 0 | 232 | 232 |
| mean_MAIN_OUTER_s1 | MAIN_OUTER | 0 | 0 | 233 | 233 |
| mean_MAIN_OUTER_s2 | MAIN_OUTER | 0 | 0 | 233 | 233 |
| std_MAIN_OUTER_s1 | MAIN_OUTER | 0 | 0 | 233 | 233 |
| std_MAIN_OUTER_s2 | MAIN_OUTER | 0 | 0 | 233 | 233 |
| auc_MAIN_OUTER_s1 | MAIN_OUTER | 0 | 0 | 233 | 233 |
| auc_MAIN_OUTER_s2 | MAIN_OUTER | 0 | 0 | 233 | 233 |
| mean_CENTER_s1 | CENTER | 0 | 0 | 233 | 233 |
| mean_CENTER_s2 | CENTER | 0 | 0 | 233 | 233 |
| std_CENTER_s1 | CENTER | 0 | 0 | 233 | 233 |
| std_CENTER_s2 | CENTER | 0 | 0 | 233 | 233 |
| auc_CENTER_s1 | CENTER | 0 | 0 | 233 | 233 |
| auc_CENTER_s2 | CENTER | 0 | 0 | 233 | 233 |
| mean_RETAINER_s1 | RETAINER | 0 | 0 | 232 | 232 |
| mean_RETAINER_s2 | RETAINER | 0 | 0 | 232 | 232 |
| std_RETAINER_s1 | RETAINER | 0 | 0 | 232 | 232 |
| std_RETAINER_s2 | RETAINER | 0 | 0 | 232 | 232 |
| auc_RETAINER_s1 | RETAINER | 0 | 0 | 232 | 232 |
| auc_RETAINER_s2 | RETAINER | 0 | 0 | 232 | 232 |
| mean_RIPPLE_s1 | RIPPLE | 0 | 0 | 229 | 229 |
| mean_RIPPLE_s2 | RIPPLE | 0 | 0 | 229 | 229 |
| std_RIPPLE_s1 | RIPPLE | 0 | 0 | 229 | 229 |
| std_RIPPLE_s2 | RIPPLE | 0 | 0 | 229 | 229 |
| auc_RIPPLE_s1 | RIPPLE | 0 | 0 | 229 | 229 |
| auc_RIPPLE_s2 | RIPPLE | 0 | 0 | 229 | 229 |
| mean_EDGE_s1 | EDGE | 0 | 0 | 233 | 233 |
| mean_EDGE_s2 | EDGE | 0 | 0 | 233 | 233 |
| std_EDGE_s1 | EDGE | 0 | 0 | 233 | 233 |
| std_EDGE_s2 | EDGE | 0 | 0 | 233 | 233 |
| auc_EDGE_s1 | EDGE | 0 | 0 | 233 | 233 |
| auc_EDGE_s2 | EDGE | 0 | 0 | 233 | 233 |
| mean_SLURRY_A_s1 | SLURRY_A | 0 | 0 | 183 | 183 |
| mean_SLURRY_A_s2 | SLURRY_A | 0 | 0 | 183 | 183 |
| std_SLURRY_A_s1 | SLURRY_A | 0 | 0 | 183 | 183 |
| std_SLURRY_A_s2 | SLURRY_A | 0 | 0 | 183 | 183 |
| auc_SLURRY_A_s1 | SLURRY_A | 0 | 0 | 183 | 183 |
| auc_SLURRY_A_s2 | SLURRY_A | 0 | 0 | 183 | 183 |
| mean_SLURRY_B_s1 | SLURRY_B | 0 | 0 | 183 | 183 |
| mean_SLURRY_B_s2 | SLURRY_B | 0 | 0 | 183 | 183 |
| std_SLURRY_B_s1 | SLURRY_B | 0 | 0 | 183 | 183 |
| std_SLURRY_B_s2 | SLURRY_B | 0 | 0 | 183 | 183 |
| auc_SLURRY_B_s1 | SLURRY_B | 0 | 0 | 183 | 183 |
| auc_SLURRY_B_s2 | SLURRY_B | 0 | 0 | 183 | 183 |
| mean_SLURRY_C_s1 | SLURRY_C | 0 | 0 | 196 | 196 |
| mean_SLURRY_C_s2 | SLURRY_C | 0 | 0 | 196 | 196 |
| std_SLURRY_C_s1 | SLURRY_C | 0 | 0 | 196 | 196 |
| std_SLURRY_C_s2 | SLURRY_C | 0 | 0 | 196 | 196 |
| auc_SLURRY_C_s1 | SLURRY_C | 0 | 0 | 196 | 196 |
| auc_SLURRY_C_s2 | SLURRY_C | 0 | 0 | 196 | 196 |
| mean_WAFER_ROT_s1 | WAFER_ROT | 0 | 0 | 215 | 215 |
| mean_WAFER_ROT_s2 | WAFER_ROT | 0 | 0 | 215 | 215 |
| mean_STAGE_ROT_s1 | STAGE_ROT | 0 | 0 | 214 | 214 |
| mean_STAGE_ROT_s2 | STAGE_ROT | 0 | 0 | 214 | 214 |

## 추가 열과 since 열의 NaN 수 (그룹별)

| 열 | A-456 | B-456 | A-123 |
|---|---|---|---|
| since_BACKING_FILM_w | 35 | 28 | 11 |
| since_DRESSER_w | 122 | 113 | 57 |
| since_POLISH_TABLE_w | 3 | 4 | 3 |
| since_MEMBRANE_w | 35 | 28 | 11 |
| since_PSHEET_w | 35 | 28 | 11 |
| P1_on | 7 | 7 | 276 |
| frac_hi | 7 | 7 | 276 |
| ch56_step | 0 | 0 | 431 |
| allzero | 0 | 0 | 0 |

- `P1_on`·`frac_hi` NaN = 압력 > 0 이고 STAGE 회전 > 0인 행이 하나도 없는 웨이퍼. A-123은 대부분 기록 누락 웨이퍼이고, 456 그룹에도 있다(모두 `ch56_step` = 0). `ch56_step`의 A-123 NaN은 정의상.
- `allzero` = 1: A-123 265장, 456 그룹 0장.
- `frac_hi` 경계 4,000: 기존 진단(`hyp_r2r_retainer.py`)의 7,135 단계 경계는 > 3,000이었다. 3,000 초과 4,000 이하인 행은 P1_on 행의 506 / 198,139행(0.26%, 단계 전환 구간)이라 두 정의의 차이는 작다.
- R 피처의 기존 NaN(챔버 4만 있는 2장의 `_s2` 등)은 그대로다.

## 계열 구분 (feature_families_v1.csv)

계열 구분은 2026-10-28 팀 회의 결정표 #6 기준(채택 후 확정).

열: `feature, family, guaranteed_slot, sr_candidate`. `feature` 목록은 `features_v1.csv`의 피처 열(키 3개 제외 113개)과 순서까지 정확히 일치한다.

| family | 피처 수 | guaranteed_slot | 포함 피처 |
|---|---|---|---|
| 소모품 카운터 | 36 | 1 | `{mean,std,slope}_{U}_{s1,s2}` |
| 헤드 압력 | 30 | 1 | `{mean,std,auc}_{MAIN_OUTER,CENTER,RIPPLE,EDGE,PCHAMBER}_{s1,s2}` |
| 슬러리 유량 | 18 | 1 | `{mean,std,auc}_{SLURRY_A,SLURRY_B,SLURRY_C}_{s1,s2}` |
| 리테이너 링 압력 | 8 | 1 | `{mean,std,auc}_RETAINER_{s1,s2}`, `P1_on`, `frac_hi` |
| 회전 | 6 | 1 | `mean_{WAFER_ROT,STAGE_ROT,HEAD_ROT}_{s1,s2}` |
| 리셋 후 경과량 | 5 | 1 | `since_{BACKING_FILM,DRESSER,POLISH_TABLE,MEMBRANE,PSHEET}_w` |
| 시간 | 2 | 1 | `dur_POLISH_{s1,s2}` |
| 기타 | 8 | 0 | `{mean,std,auc}_DRESS_WATER_{s1,s2}`, `ch56_step`, `allzero` |

- **헤드 압력**: 에어백 4개(MAIN_OUTER, CENTER, RIPPLE, EDGE)와 가압 챔버(PCHAMBER). 가압 챔버는 멤브레인 뒤에서 웨이퍼를 누르는 헤드 압력 계통이다. (초안의 "에어백 압력"에서 이름 변경)
- **기타**: `DRESS_WATER`는 드레싱 워터 0/1 상태라 유량과 성격이 달라 기타로 둔다. `ch56_step`, `allzero`도 기타.
- `frac_hi`는 리테이너 압력 단계의 체류 비율이고, 리테이너 압력 수준으로 정의되므로 리테이너 링 압력이다.
- `dur_POLISH_*`는 구간 체류 시간(마지막 − 첫 TIMESTAMP)이라 시간이다.

**추가 열의 의미**
- `guaranteed_slot` (0/1): SR 후보 선택에서 계열 대표 자리를 보장받는 계열이면 1. 1인 계열은 소모품 카운터, 리셋 후 경과량, 리테이너 링 압력, 헤드 압력, 슬러리 유량, 회전, 시간(7개)이고, 기타는 0이다.
- `sr_candidate` (0/1): SR 후보 풀에 들어가는지. `allzero`만 0이다 (SR은 456 정상 웨이퍼만 쓰므로 값이 전부 0). 나머지 112개는 1.

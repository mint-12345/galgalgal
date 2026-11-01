# CLAUDE.md — 갈갈갈 CMP 가상계측 프로젝트

> 저장소 루트의 이 파일은 Claude Code가 작업을 시작할 때 자동으로 읽는다.
> 팀원 4명의 Claude가 같은 규칙으로 일하게 하는 **단일 기준 문서**다.
> 규칙을 바꾸려면 이 파일을 고치는 PR을 올리고 팀원 1명 이상의 승인을 받는다.
> 작성자 계정과 승인자가 같을 수밖에 없을 때(예: 팀 공용 계정으로 PR을 올리는 경우)는, 팀 회의 또는 팀 합의 내용을 PR 코멘트로 남겨 승인을 갈음한다. (2026-10-29 팀 합의)
> 상세 피처 정의의 원본은 `docs/Drift.k 피처 설계서.docx`(v1)이고, 이 파일은 그중 코드가 지켜야 할 부분만 옮긴 것이다.

## 1. 프로젝트 한 줄 요약

PHM 2016 CMP 데이터로 웨이퍼 MRR을 예측하는 가상계측 모델을 만들고,
**일부 웨이퍼만 실측할 수 있는 조건**에서 샘플링 정책(어떤 웨이퍼를 실측할지)을 비교한다.

## 2. 작업 원칙 (Claude 포함 모든 팀원)

- **코드를 쓰기 전에 데이터를 열어 확인한다.** 컬럼명·행 수·값 범위를 추측하지 않는다. 5절의 검증된 사실과 다르면 멈추고 보고한다.
- 한 번에 한 가지 작업만 한다. 끝나면 무엇을 했고 어떻게 검산했는지 요약한다.
- 결과를 만드는 코드는 `.py`로 쓴다. 노트북은 탐색용으로만 쓰고 커밋하지 않는다.
- 랜덤 시드는 `SEED = 42`.
- 확실하지 않은 결정은 임의로 하지 말고 `# TODO(결정 필요):`로 남기고 보고한다.
- 그룹은 번호가 아니라 **`A-456`, `B-456`, `A-123`** 으로 부른다 (논문마다 번호가 달라서 혼동됨. 11절 참고).
- 논문 원문 확인이 필요하면 먼저 docs/paper_notes/를 보고, 부족할 때만 docs/papers/의 PDF에서 필요한 페이지만 읽는다.

## 3. 절대 금지

1. **test MRR 라벨을 읽지 않는다.** 원본 저장소에는 test 정답이 두 곳에 들어 있다:
   `CMP-test-removalrate.csv`(424장 전부 값 있음), `PHM16TestValidationAnswers/`.
   둘 다 `data/locked/`로 옮기고 코드에서 읽지 않는다. 게이트 확인 1회는 팀장만 하고 `preregistration.md`에 기록한다.
   - PHM16TestValidationAnswers/ 안의 orig_CMP-validation-removalrate.csv는
     data/raw/labels/CMP-validation-removalrate.csv와 내용이 완전히 같다 (2026-09-22 확인).
     따라서 validation 라벨은 data/raw/labels/의 파일만 쓰고, data/locked/는 어떤 파일도 열지 않는다.
2. `data/raw/` 원본을 수정하거나 커밋하지 않는다.
3. `results/`의 기존 파일을 덮어쓰지 않는다.
4. `preregistration.md`, `src/harness.py`, `src/plans.py`는 팀 합의 없이 수정하지 않는다.
5. 모델 파일 안에서 피처나 타깃을 새로 만들지 않는다. 새 피처는 `src/features.py`에만 추가한다.
6. 라벨로 계산한 값(S9 소프트센서, 잔차, MAPE)을 예측 입력이나 샘플링 트리거로 쓰지 않는다.

## 4. 데이터 위치와 폴더 구조

원본: https://github.com/akangel0307/PHM-Data-Challenge
(`data/2016 PHM Data Challenge/` 아래, 약 188MB)

```
data/raw/train/       CMP-training-000~184.csv   (원본 .../CMP DATA SET/CMP-data/training/)
data/raw/val/         CMP-validation-000~184.csv (원본 .../CMP VALIDATION DATA SET/validation/)
data/raw/test/        CMP-test-000~184.csv       (원본 .../CMP DATA SET/CMP-data/test/)
data/raw/labels/      CMP-training-removalrate.csv, CMP-validation-removalrate.csv
data/locked/          CMP-test-removalrate.csv, PHM16TestValidationAnswers/  ← 읽기 금지
data/processed/       웨이퍼 단위 집계표 (커밋함)
docs/                 피처 설계서, 연구설계 문서
src/load.py  src/aggregate.py  src/features.py  src/models/  src/harness.py
results/              모델 실험 결과 csv + 설정 yaml (14.3)
results/diagnostics/  진단·점검 결과 (14.4)
```

`.gitignore`에 `data/raw/`, `data/locked/`를 넣는다.

## 5. 검증된 데이터 사실 (2026-09-22 원본 확인)

코드 결과가 아래와 다르면 코드를 의심한다.

### 5.1 파일 구조
- 센서 파일 컬럼(25개): `MACHINE_ID, MACHINE_DATA, TIMESTAMP, WAFER_ID, STAGE, CHAMBER, USAGE_OF_BACKING_FILM, USAGE_OF_DRESSER, USAGE_OF_POLISHING_TABLE, USAGE_OF_DRESSER_TABLE, PRESSURIZED_CHAMBER_PRESSURE, MAIN_OUTER_AIR_BAG_PRESSURE, CENTER_AIR_BAG_PRESSURE, RETAINER_RING_PRESSURE, RIPPLE_AIR_BAG_PRESSURE, USAGE_OF_MEMBRANE, USAGE_OF_PRESSURIZED_SHEET, SLURRY_FLOW_LINE_A, SLURRY_FLOW_LINE_B, SLURRY_FLOW_LINE_C, WAFER_ROTATION, STAGE_ROTATION, HEAD_ROTATION, DRESSING_WATER_STATUS, EDGE_AIR_BAG_PRESSURE`
- 라벨 파일 컬럼: `WAFER_ID, STAGE, AVG_REMOVAL_RATE`
- `MACHINE_ID`는 항상 2, `MACHINE_DATA`는 `CHAMBER`와 1:1로 같다 → 둘 다 피처에서 제외.
- `DRESSING_WATER_STATUS`는 0/1 값.

### 5.2 웨이퍼 키와 개수
- **웨이퍼 키는 `(WAFER_ID, STAGE)`** 이다. `WAFER_ID`만으로는 중복된다 (train 1,981장인데 고유 WAFER_ID는 1,699개).
- 웨이퍼 수와 그룹 분포:

| split | 전체 | A-456 | B-456 | A-123 |
|---|---|---|---|---|
| train | 1,981 | 798 | 815 | 368 (이상치 제외 후 364) |
| val | 424 | 185 | 172 | 67 |
| test | 424 | 165 | 186 | 73 |

- **이상치 4장(MRR > 400)은 전부 A-123 그룹**이다. 제외 후 train 1,977장.
  MRR 4,130~4,326, 정상 A-123 범위의 약 27배 → 기록 또는 단위 오류로 추정.
- train MRR 범위(이상치 제외): A-456 53~89, B-456 54~102, A-123 139~163.
- train·val·test는 **시간상 뒤섞여 있다** (시작 시각 순으로 정렬하면 split이 1,300번 넘게 바뀜).

### 5.3 웨이퍼 내부 구조
- 그룹 판정: 웨이퍼가 거친 챔버가 {1,2,3}이면 `123`, {4,5,6}이면 `456`. 그룹명 = `STAGE-챔버셋`.
- 챔버 순서는 4→5→6 (또는 1→2→3). **train에서 3장은 챔버가 불완전**하다 (챔버 4만 있는 2장, 4·6만 있는 1장) → `flag_incomplete_chamber`.
- 챔버별 체류 시간 중앙값 (중복 제거 후): 4 = 186초, 5 = 84초, 6 = 71초 / 1 = 146초, 2 = 55초, 3 = 71초.
  → Di 방식 2구간 분할 = **(챔버 4 | 챔버 5·6)**, **(챔버 1 | 챔버 2·3)**.
- 한 웨이퍼 행 수 중앙값 약 340행(A-123은 약 275행). 단, train B 그룹에 2,000행 이상인 웨이퍼 3장, test에 14,678행인 웨이퍼 1장이 있다 → `flag_long_trace`.
- **완전히 같은 행이 중복**되어 있다: train 1,819 / val 434 / test 98 / 합계 2,351. split 사이에 걸친 중복은 없음 → 먼저 제거. 그 뒤에도 같은 (웨이퍼, TIMESTAMP)에 값이 다른 행이 train에서 약 400개(402쌍) 남는다 (val 64, test 90) → 처리 방식 TODO(결정 필요).

### 5.4 결측은 NaN이 아니라 0으로 나타난다
- 원본에 NaN은 없다. 대신 A-123 웨이퍼 일부가 압력·유량·회전 센서 중 하나 이상이 웨이퍼 전체에서 0이다 (Deivendran이 보고한 결측 221장과 같은 현상). A-456·B-456에는 없다.

| split | A-123 전체 | 전체-0 센서 웨이퍼 |
|---|---|---|
| train | 368 | 228 |
| val | 67 | 40 |
| test | 73 | 46 |

- 처리 규칙은 세 split에 동일하게 적용한다.
- 따라서 결측 확인은 `isna()`가 아니라 "웨이퍼 전체에서 해당 센서가 모두 0인가"로 한다 → `n_allzero_sensors` 컬럼.
- **전체-0 판정 대상 센서: 압력 6 + 슬러리 3 + 회전 3, 총 12개.** `DRESSING_WATER_STATUS`는 제외 (0/1 상태값). 이 기준에서 위 수치가 나온다.
- 회전 변수는 연마 중에도 0인 행이 많다. `WAFER_ROTATION`과 `STAGE_ROTATION`이 **동시에 0이 아닌 행은 전체의 1~3%** 뿐이다. V(상대속도)를 정의하기 전에 챔버별 회전 변수 분포부터 확인한다.

### 5.5 소모품 카운터 (train만, 스테이지별 시작 시각 순, 범위의 20% 이상 감소를 리셋으로 본 사전 확인치)
| 변수 | Stage A 리셋 | Stage B 리셋 |
|---|---|---|
| POLISHING_TABLE | 98 | 86 |
| DRESSER | 3 | 3 |
| BACKING_FILM, MEMBRANE, PRESSURIZED_SHEET | 각 5 | 각 5 |
| DRESSER_TABLE | 0 | 0 |

- 리셋 감지 임계값은 결과에 민감하다. 데이터 담당이 train·val·test 센서를 합쳐 다시 확정한다 (라벨 불필요).
- 3개(BACKING_FILM, MEMBRANE, PRESSURIZED_SHEET)가 같은 횟수 → 헤드 부품이 함께 교체되는 것으로 추정, 확인 필요.

**`src/aggregate.py`의 리셋 감지 (2026-09-22)**
- 리셋은 **스테이지 단위**로 감지한다. A-123과 A-456이 카운터를 공유한다(스테이지 공유)고 보았으나 (시작 시각 순으로 값이 한 줄로 이어짐),
  **10/28 카운터 진단(PR #14)으로 정정:** 카운터 6개는 장비 전체(A-123, A-456, B-456)가 값 하나를 공유한다.
  따라서 S1(A-456) 대 S2(A+B-456) 비교는 "같은 부품 공유" 근거로 쓰지 않는다.
- 비교 기준은 **웨이퍼 시작값**이다. 직전 웨이퍼(같은 스테이지, 시작 시각 순)의 시작값과 비교한다. 웨이퍼끼리 시간이 겹쳐서 직전 웨이퍼의 마지막 값과 비교하면 가짜 감소가 생긴다.
- 세 split 합산 결과: POLISH_TABLE A 100 / B 87, DRESSER 3 / 3, BACKING_FILM·MEMBRANE·PSHEET 각 5 / 5 (세 변수가 같은 웨이퍼에서 동시에 리셋), DRESSER_TABLE 0 / 0.
- 임계값 20%는 **임시값**이다. 10%·30%와 비교한 뒤 확정한다.
- `since_*`(마지막 교체 이후 누적 사용량)는 스테이지의 첫 리셋 이전 웨이퍼에서 NaN이다. DRESSER_TABLE은 리셋이 없어 `since_*` 컬럼을 만들지 않는다.

**POLISH_TABLE 리셋의 해석 (2026-09-27 확인)**
- **실제 테이블 교체로 본다.** 리셋 직전 최고값이 중앙값 307~310 / 최대 353~357이고 웨이퍼당 증가량이 11.1로,
  Deivendran 2025(교체 기준 약 350, 웨이퍼당 9~11)와 맞는다.
- 카운터가 스테이지 공유라는 위 항목을 원본 행에서 재확인했다: A-456과 A-123의 같은 시각(±60초) 값 차이가
  중앙값 0, 98.5%가 1 이내다. 챔버군으로 쪼개 세면 같은 교체를 두 번 센다 (A 100 → 140회).
  → 스테이지 공유로 보았으나, **10/28 카운터 진단(PR #14)으로 정정:** 카운터 6개는 장비 전체(A-123, A-456, B-456)가 값 하나를 공유한다.
  따라서 S1(A-456) 대 S2(A+B-456) 비교는 "같은 부품 공유" 근거로 쓰지 않는다.
- **데이터에는 장비가 처리한 웨이퍼의 약 절반만 있다.** 카운터 증가분 중 웨이퍼 기록 안에서 생긴 몫이
  A 48.2% / B 51.8%다. 그래서 교체 간격이 우리 데이터에서 약 15장, 장비 기준으로는 약 35장이 된다. 둘 다 맞다.
- **웨이퍼 시작값 기준은 리셋의 약 10%를 놓친다** (센서 행 단위로 세면 A 111 / B 100, 시작값 기준은 A 100 / B 87).
- A-123 웨이퍼의 55.3%는 웨이퍼 내 카운터 증가가 0이라 (456 그룹은 0장), 그 그룹의 `start_/delta_/since_POLISH_TABLE_w`는
  자기가 쓴 테이블이 아니라 옆 챔버군 테이블의 상태일 수 있다고 보았으나(B8), **10/28 카운터 진단(PR #14)으로 철회:**
  소모품 카운터 6개는 장비 전체(A-123, A-456, B-456)에서 값 하나를 공유하지만 A-123 처리 중에도 증가한다
  (정상 웨이퍼 기준 시간형 4개 100%, 드레서 계열은 A-456과 같은 이벤트형). 증가 0인 55%는 센서 기록 누락 웨이퍼에서
  카운터도 함께 멈춘 결과다. 공유 범위가 장비 전체이므로 S1(A-456) 대 S2(A+B-456) 비교를 "같은 부품 공유"의 근거로 쓰지 않는다.
  근거: `results/diagnostics/2026-10-28_김혜연_a123_counters_v2.md`.
- 근거: `results/diagnostics/2026-09-27_김혜연_polish_table_reset.csv`, `src/polish_table_reset.py`,
  `docs/reports/week4_data.md` 3.1절.

## 6. 피처셋 구조 (피처 설계서 v1)

| 세트 | 목적 | 규모 | 쓰는 곳 |
|---|---|---|---|
| **R** (재현용) | Di et al. 125개 원문대로 | 125개 → 그룹별 선택 | 공식 분할의 A안 (4~5주차, 게이트 ①) |
| **R-c** (인과 버전) | R의 라벨 피처(ID 1~21)를 공개 라벨만으로 재계산 | 125개 동일 구조 | 시간순 시뮬레이터의 A안 |
| **S** (종합용) | 물리·소모품·트리거 피처 | 핵심 약 40개 | A+, B, C, 샘플링 정책 (8주차~) |

층 이름 F1(공정 통계), F2(소모품 상태 c), F3(물리 파생), F4(라벨 의존)는 연구설계 문서와 같다.

### 6.1 R: Di 125개 피처 ID

| ID | 피처 | 개수 | 층 | 우리 해석 (원문 모호점) |
|---|---|---|---|---|
| 1–11 | MRR 시간 지연 r(t−1)…r(t−11) | 11 | F4 | 같은 그룹, 시작 시각 기준 |
| 12–21 | 사용량 최근접 이웃 10개의 MRR | 10 | F4 | 거리 공간 = z-score한 사용량 6개의 웨이퍼 평균. 자기 자신 제외 |
| 22–23 | 구간별 연마 시간 | 2 | F1 | 구간 마지막 − 첫 TIMESTAMP |
| 24–59 | 사용량 6개 × {평균, 표준편차, 감소율} × 2구간 | 36 | F2 | **감소율 = 구간 내 선형회귀 기울기** |
| 60–95 | 압력 6개 × {평균, 표준편차, AUC} × 2구간 | 36 | F1 | AUC = TIMESTAMP 기준 사다리꼴 적분 |
| 96–119 | 유량 4개 × {평균, 표준편차, AUC} × 2구간 | 24 | F1 | 슬러리 A·B·C + **DRESSING_WATER_STATUS**(개수로 역산한 추정) |
| 120–125 | 회전 3개 평균 × 2구간 | 6 | F1 | 그대로 |

- 압력 6개 = PRESSURIZED_CHAMBER, MAIN_OUTER_AIR_BAG, CENTER_AIR_BAG, RETAINER_RING, RIPPLE_AIR_BAG, EDGE_AIR_BAG.
- 사용량 6개 = BACKING_FILM, DRESSER, POLISHING_TABLE, DRESSER_TABLE, MEMBRANE, PRESSURIZED_SHEET.
- 피처 이름 규칙: `{통계}_{변수 약어}_{구간}` 예) `mean_RETAINER_s1`, `auc_SLURRY_C_s2` (s1 = 첫 챔버, s2 = 나머지 두 챔버).
- 선택 절차: Monte Carlo CV 20회마다 LR t-검정과 tree bagging OOB 순열 중요도로 투표, 많이 뽑힌 피처만 사용. 임계값은 원문에 없음 → TODO(결정 필요), 정한 값은 여기에 기록.

### 6.2 R-c: 공개 라벨만 쓰는 버전 (ID 1~21만 재정의)
- 시간 지연: 웨이퍼 t보다 **먼저 시작했고 그 시점에 라벨이 공개된** 웨이퍼 중 최근 11개.
- 사용량 이웃: 탐색 풀 = t 이전에 공개된 웨이퍼. 최근 L개 창으로 제한하는 옵션을 둔다.
- 학습 행의 피처도 그 웨이퍼 시점에 공개돼 있던 라벨만으로 만든다 (학습·예측 분포 일치).
- 공개 라벨이 11개보다 적으면 그룹 과거 평균으로 채우고 `lag_missing` 표시 피처를 추가한다.

### 6.3 S: 종합 피처 (핵심 = S2~S5, S10)

| 코드 | 층 | 내용 | 라벨 | 상태 |
|---|---|---|---|---|
| S1 | F1 확장 | median, min, max, range, PtP, skew, kurtosis | 무관 | 확장 (하면 A·A+ 모두) |
| **S2** | F1·F2 | 시작 시각, 구간별 연마 시간, 직전 웨이퍼와의 유휴 간격, 유휴 후 첫 웨이퍼 플래그 | 무관 | 핵심 |
| **S3** | F2 | 사용량 6개 시작값, 웨이퍼 내 증가분·감소율, 리셋 플래그, 마지막 교체 이후 누적 사용량, 교체 횟수 | 무관 | 핵심 (C의 k 입력) |
| **S4** | F3 | P, V, P·V, ln P, ln V, P^(7/6)·V, ∫P·V dt, P·V·연마시간 | 무관 | 핵심 |
| **S5** | F4 | 공개 라벨 웨이퍼의 k_obs = MRR/(P·V): 직전 k, c 공간 이웃 k, k의 EWMA | 의존 | 핵심 (C의 F4 허용 버전) |
| S6 | F4 확장 | JIT·MMD 참조 MRR (과거 웨이퍼로만 제한) | 의존 | 확장 |
| S7 | F1 확장 | 위상(phase) 분할 통계 | 무관 | 확장 |
| S9 | — | 슬러리 온도·연마입자 크기 역산값 | **파생** | 해석 전용, 입력 금지 |
| **S10** | — | 샘플링 트리거 (6.5) | 무관 | 핵심 |

- S8(주파수 피처)은 제외.
- F1을 확장(S1, S7)하면 **A와 A+에 동시에** 넣는다. 한쪽에만 넣으면 A+의 이득이 F3 덕분인지 구분할 수 없다.

### 6.4 안(案)별 입력

| 모델 | 입력 | 출력 구조 |
|---|---|---|
| A | F1 + F2 + F4(MRR 이웃, 라벨 나이) | MRR 직접 회귀 |
| A+ | A + F3 | MRR 직접 회귀 |
| B | F3의 P, V + 그룹 | MRR = k·P·V, k는 상수 또는 그룹별 |
| C | 구조 F3 고정, k 모델 입력 = F2 (허용 버전은 + S5) | MRR = k(c)·P·V |

### 6.5 샘플링 트리거 (S10, 실측 전에 계산 가능한 것만)

| 정책 | 트리거 | 역할 |
|---|---|---|
| 고정주기 | 마지막 실측 이후 웨이퍼 수 | 기준선 (ISD) |
| 불확실성 | 부트스트랩 앙상블 예측 표준편차 | 기준선 (Kang & Kang) |
| 입력 분포 거리 | 공개 라벨 웨이퍼 대비 Mahalanobis 거리 | 기준선 (ISD GSI, Kurz) |
| 이벤트 | 사용량 카운터 리셋 직후 실측 | 기준선 (ISD) |
| **k 상태** | 부트스트랩 C 모델들의 k 산포, 마지막 실측 이후 c 변화량. **리셋 플래그는 쓰지 않음** | 제안 방법 |

"연속 k장" 규칙이나 라벨 나이 상한을 쓰면 **모든 정책에 똑같이** 적용한다.

## 7. `neighbor_features` 함수 규칙

F4(R의 ID 1~21, S5)는 **이 함수에서만** 계산한다.

```python
def neighbor_features(wafers, revealed, n_lag=11, k_usage=10, window=None):
    """revealed: MRR이 공개된 웨이퍼 집합 (키 = (WAFER_ID, STAGE)).
    - R (재현): train 1,977장 전체를 넘긴다.
    - R-c (시뮬레이션): 정책이 고른 웨이퍼 중 해당 시점 이전 것만 넘긴다.
    반환: MRR 지연 11개, 사용량 이웃 MRR 10개, 라벨 나이, lag_missing,
          (P·V 확정 후) k_obs 지연과 이웃 k."""
```

- 항상 **같은 그룹** 안에서만 찾는다.
- 시간 지연은 대상 웨이퍼보다 **먼저 시작한** 웨이퍼만.
- 사용량 이웃은 **자기 자신 제외**.
- test·val 웨이퍼의 이웃도 train(공개) 웨이퍼에서만 찾는다 (Di 원문에 명시 없음 → 우리 가정).
- 테스트(`tests/test_neighbor.py`): 미래 웨이퍼·다른 그룹·자기 자신·미공개 라벨이 절대 쓰이지 않는지 확인.

## 8. P·V 후보 (5주차에 대표안 1개 확정, 나머지는 민감도 분석)

| 후보 | 정의 | 근거 |
|---|---|---|
| P1 | RETAINER_RING_PRESSURE 평균 | Yu 2019 |
| P2 | PRESSURIZED_CHAMBER_PRESSURE 중앙값 | Deivendran |
| P3 | 에어백 압력 4개(MAIN_OUTER, CENTER, RIPPLE, EDGE) 평균 | 우리 제안 |
| P4 | P1~P3를 연마 구간(압력 > 0)·챔버 구간별로 | Di 분할 |
| V1 | STAGE_ROTATION − WAFER_ROTATION | Yu 2019 |
| V2 | HEAD_ROTATION − (STAGE_ROTATION − WAFER_ROTATION), 웨이퍼별 중앙값 | Deivendran Eq. 8 |
| V3 | STAGE_ROTATION 단독 | 우리 제안 |
| 지수 | MRR = k·P^α·V^β, α·β ∈ [0,1] 적합 또는 α = 7/6 | Cai, Jia, Yu |

- 압력·회전 단위는 데이터 설명에 없다.
- **5.4 주의**: 두 회전이 동시에 0이 아닌 행이 1~3%뿐이고, 그 행들에서 V2의 약 68%가 음수다. 챔버별로 어떤 회전이 언제 켜지는지 먼저 확인한다.
- P2(PRESSURIZED_CHAMBER)는 두 회전이 모두 켜진 행 기준으로 A/B-456의 챔버 4에서만 0이 아닌 값이 나왔다 (사전 확인치, 전체 행으로 재확인 필요).

## 9. 재현 목표 (Di et al. 2017, MSE)

| 모델 | A-456 | B-456 | A-123 | 전체 | 출처 |
|---|---|---|---|---|---|
| 지속 모델, train CV | 7.55 | 11.4 | 5.62 | 8.78 | Di Table 4 |
| LR, train CV 20회 | 5.77 | 10.95 | 5.00 | 7.76 | Di Table 4 |
| Tree bagging, train CV 20회 | 5.65 | 8.32 | 5.32 | 6.48 | Di Table 4 |
| 통합 모델, train CV 20회 | 5.43 | 7.33 | 4.91 | 6.18 | Di Table 4 |
| LR / Tree bagging / 통합, test | — | — | — | 7.32 / 7.22 / 7.07 | Di Table 5 |

- **지속 모델(학습 없음)부터 맞춘다.** 다르면 시간 정렬·이웃 코드 문제다.
- 게이트 ①은 train CV 값으로 판정하고, 공식 test는 확인용으로 한 번만 본다.
- 피처 단계 점검: Jia 2018 Table 5 (LR test MSE) 통계 피처만 16.24 → + 사용량 이웃 7.77 → + 시간 이웃 7.32.
- Tree bagging = `RandomForestRegressor(max_features=1.0, random_state=SEED)`.

## 10. 전처리 순서 (`src/aggregate.py`)

1. split별로 185개 파일 로드 → `split` 컬럼 추가
2. 완전 중복 행 제거 (5.3)
3. `(WAFER_ID, STAGE)`로 웨이퍼 구분, TIMESTAMP 정렬
4. 그룹 부여, `t_start` = 첫 TIMESTAMP
5. 플래그: `is_outlier`(train MRR > 400), `flag_incomplete_chamber`, `flag_long_trace`, `n_allzero_sensors`
6. 구간 분할 (s1 = 챔버 4 또는 1, s2 = 나머지) → R의 ID 22~125 계산
7. 사용량 리셋 감지 → S3 컬럼
8. 저장: `data/processed/wafer_table.csv` (train·val 라벨 포함, test 라벨 없음)

행을 삭제하지 말고 플래그만 붙인다. 어떤 플래그를 제외할지는 실험 설정에서 정한다.

## 11. 논문별 그룹 번호 (수치 비교 시 반드시 확인)

| Stage-Chamber | Di | Jia 2018 | Zhang 2021 | Cai 2020·2021(Meas.) | Cai 2021(IJAMT) | SSDKAL |
|---|---|---|---|---|---|---|
| A-456 | Cond1 | Group1 | Group2 | Group1 | Group1 | Mode I |
| B-456 | Cond2 | Group2 | Group3 | Group3 | Group2 | Mode II |
| A-123 | Cond3 | Group3 | Group1 | Group2 | Group3 | Mode III |

## 12. 결과 저장

`results/{날짜}_{이름}_{R|Rc|S|FV1}_{실험명}.csv` + 같은 이름의 `.yaml`(사용 설정·피처 목록·시드·입력 파일 버전).
csv 컬럼: `group, model, featureset, fold, mse, mae, rmse, n_train, n_eval`. **`featureset` 컬럼은 필수다** (14절). FV1 태그 결과의 `featureset` 값은 `features_v1`이다.
진단·점검 결과는 `results/diagnostics/`에 두고 다른 규칙을 따른다 (14.4).

## 13. 미결정 (결정되면 이 절에서 지우고 해당 절에 반영)

- [ ] 같은 (웨이퍼, TIMESTAMP)에 값이 다른 행 약 400개 처리
- [ ] A-123 전체-0 센서 웨이퍼 (train 228 / val 40 / test 46) 처리 (제외 / 가능한 피처만 / 표시 피처)
- [ ] 불완전 챔버 3장, 긴 트레이스 웨이퍼 처리
- [ ] 리셋 감지 임계값 (train·val·test 센서 합쳐 확정)
- [x] 리셋 감지 단위: 스테이지별인가 기계 전체인가. 시간이 가장 가까운 A·B 웨이퍼끼리 카운터 시작값을 비교하면
      POLISH_TABLE을 뺀 5개 변수의 차이 중앙값이 변수 범위의 0.05~0.21%뿐이고, 리셋 시각도 거의 같다
      (BACKING_FILM: A 1.13·15.27·31.66·47.76·64.51일째 / B 1.14·15.32·32.39·47.86·64.24일째, 2026-09-26 확인).
      → 5.5절의 "A 5회 / B 5회"는 서로 다른 교체 10번이 아니라 같은 교체를 두 번 센 것일 수 있다.
      **해결: 기계 전체 (10/28 카운터 진단, PR #14).**
- [ ] POLISH_TABLE 리셋을 샘플링 이벤트 트리거(6.5절)에 넣을지 여부. 리셋이 우리 데이터 기준 웨이퍼 15장마다 한 번꼴
      (2,829장 중 187회)이라 다른 부품(60일에 3~5회)과 성격이 다르다.
      ~~부품 교체가 아니라 주기적 동작일 가능성~~ → **2026-09-27 확인으로 근거가 약해졌다.** 5.5절대로 실제 교체로 보이고,
      간격이 짧은 것은 데이터에 웨이퍼가 절반만 있기 때문이다. **교체 주기가 짧은 소모품으로 두고 트리거 후보로 다시 검토한다.**
- [ ] 리셋 감지를 웨이퍼 시작값 기준에서 **센서 행 단위**로 바꿀지. 시작값 기준은 POLISH_TABLE 리셋의 약 10%를 놓친다
      (행 단위 A 111 / B 100 대 시작값 A 100 / B 87). 바꾸면 `wafer_table.csv`를 다시 만들어야 하므로 14.2절의 `_v2` 규칙을 따른다.
- [ ] **A-123의 소모품 상태 입력**을 무엇으로 할지. A-123 웨이퍼의 55.3%는 웨이퍼 내 POLISH_TABLE 증가가 0이라,
      그 값이 자기 테이블의 마모가 아닐 수 있다고 보았으나(B8), **10/28 카운터 진단(PR #14)으로 철회:** 카운터는 장비 전체 공유값이지만
      A-123 처리 중에도 증가하고, 증가 0인 55%는 센서 기록 누락 웨이퍼에서 카운터도 함께 멈춘 결과다 (5.5절).
      "A-123 전용 카운터를 따로 만들지"는 필요 없다. 쓰는 방식은 preregistration.md를 따른다.
- [ ] R 피처 선택 임계값
- [ ] P·V 대표안, k_obs를 반환하는 `neighbor_features` 인터페이스
- [ ] 시뮬레이션에서 시간 지연 개수(11개 유지 여부)
- [ ] F1 확장(S1·S7) 여부

## 14. 실험 세트 분리 규칙

재현(Di 원문 그대로)과 우리 연구를 한 파일에서 섞으면, 재현 수치가 나중 변경 때문에 달라졌는지 알 수 없다. 그래서 코드·데이터·결과를 세트별로 분리한다.

### 14.1 피처 코드

| 파일 | 담는 것 | 규칙 |
|---|---|---|
| `src/features_repro.py` | **R 전용** (Di 125개) | 재현 확정 후 **동결**한다. 그 뒤로는 수정 금지 |
| `src/features.py` | **우리 연구용** (F1~F4, S) | 2막부터 새 피처는 **여기에만** 추가한다 |

- R-c(공개 라벨만 쓰는 버전)는 `src/features.py` 쪽이다. `features_repro.py`를 고쳐서 만들지 않는다.
- 모델 파일 안에서 피처를 만들지 않는 규칙(3절 5번)은 그대로다.

### 14.2 집계표와 피처 파일

- `data/processed/wafer_table.csv`는 **재현용으로 동결**한다.
- 새 피처는 `data/processed/features_S.csv`에 **웨이퍼 키 `(WAFER_ID, STAGE)`와 함께** 따로 저장하고, 쓸 때 키로 병합한다.
- 집계표를 다시 만들어야 하면 덮어쓰지 말고 파일명에 `_v2`를 붙인다 (`wafer_table_v2.csv`). 3절 3번과 같은 취지다.
- **어떤 결과가 어느 버전으로 나왔는지 `results/`의 `.yaml`에 기록한다.** 예: `input: data/processed/wafer_table.csv` 또는 `wafer_table_v2.csv + features_S.csv`.

### 14.3 결과 파일: `results/` 최상위 = 모델 실험 결과만

- 파일명: `results/{날짜}_{이름}_{R|Rc|S|FV1}_{실험명}.csv` + 같은 이름의 `.yaml` (12절).
- csv에 `featureset` 컬럼을 **반드시** 넣는다. 파일명만으로는 한 파일에 여러 세트가 섞였을 때 구분할 수 없다.
- 모델 성능 수치가 아닌 것은 여기 두지 않는다 (14.4).

### 14.4 진단·점검 결과: `results/diagnostics/`

리셋 임계값 비교, P·V 후보 표, 분포 확인처럼 **모델 성능이 아닌 점검 결과**를 둔다.

- 14.3의 이름 규칙(`{R|Rc|S|FV1}`)과 `featureset` 컬럼 요건을 **적용하지 않는다.** 피처셋과 무관한 결과이기 때문이다.
- 대신 파일명에 **날짜와 작성자**는 넣는다: `results/diagnostics/{날짜}_{이름}_{내용}.csv`.
- 설정 `.yaml`을 같이 두는 것과 기존 파일을 덮어쓰지 않는 규칙(3절 3번)은 최상위와 같다.

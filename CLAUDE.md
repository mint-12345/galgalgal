# CLAUDE.md — Drift.k CMP 가상계측 프로젝트

> 저장소 루트의 이 파일은 Claude Code가 작업을 시작할 때 자동으로 읽는다.
> 팀원 4명의 Claude가 같은 규칙으로 일하게 하는 **단일 기준 문서**다.
> 규칙을 바꾸려면 이 파일을 고치는 PR을 올리고 팀원 1명 이상의 승인을 받는다.
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
results/              실험 결과 csv + 설정 yaml
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
- train MRR 범위(이상치 제외): A-456 53~89, B-456 54~102, A-123 139~163.
- train·val·test는 **시간상 뒤섞여 있다** (시작 시각 순으로 정렬하면 split이 1,300번 넘게 바뀜).

### 5.3 웨이퍼 내부 구조
- 그룹 판정: 웨이퍼가 거친 챔버가 {1,2,3}이면 `123`, {4,5,6}이면 `456`. 그룹명 = `STAGE-챔버셋`.
- 챔버 순서는 4→5→6 (또는 1→2→3). **train에서 3장은 챔버가 불완전**하다 (챔버 4만 있는 2장, 4·6만 있는 1장) → `flag_incomplete_chamber`.
- 챔버별 체류 시간 중앙값: 4 = 184초, 5 = 84초, 6 = 71초 / 1 = 145초, 2 = 55초, 3 = 71초.
  → Di 방식 2구간 분할 = **(챔버 4 | 챔버 5·6)**, **(챔버 1 | 챔버 2·3)**.
- 한 웨이퍼 행 수 중앙값 약 340행(A-123은 약 275행). 단, train B 그룹에 2,000행 이상인 웨이퍼 3장, test에 14,678행인 웨이퍼 1장이 있다 → `flag_long_trace`.
- **완전히 같은 행이 train에서 1,819개 중복**되어 있다 (val 434, test 98, split 간 중복 없음) → 먼저 제거. 그 뒤에도 같은 (웨이퍼, TIMESTAMP)에 값이 다른 행이 train에서 약 400개(402쌍) 남는다 (val 64, test 90) → 처리 방식 TODO(결정 필요).

### 5.4 결측은 NaN이 아니라 0으로 나타난다
- 원본에 NaN은 없다. 대신 **A-123 train 368장 중 228장**이 압력·유량·회전 센서 중 하나 이상이 웨이퍼 전체에서 0이다 (Deivendran이 보고한 결측 221장과 같은 현상). A-456·B-456에는 없다.
- 따라서 결측 확인은 `isna()`가 아니라 "웨이퍼 전체에서 해당 센서가 모두 0인가"로 한다 → `n_allzero_sensors` 컬럼.
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

`results/{날짜}_{작성자}_{실험명}.csv` + 같은 이름의 `.yaml`(사용 설정·피처 목록·시드).
csv 컬럼: `group, model, featureset, fold, mse, mae, rmse, n_train, n_eval`

## 13. 미결정 (결정되면 이 절에서 지우고 해당 절에 반영)

- [ ] 같은 (웨이퍼, TIMESTAMP)에 값이 다른 행 약 400개 처리
- [ ] A-123 전체-0 센서 웨이퍼 228장 처리 (제외 / 가능한 피처만 / 표시 피처)
- [ ] 불완전 챔버 3장, 긴 트레이스 웨이퍼 처리
- [ ] 리셋 감지 임계값 (train·val·test 센서 합쳐 확정)
- [ ] R 피처 선택 임계값
- [ ] P·V 대표안, k_obs를 반환하는 `neighbor_features` 인터페이스
- [ ] 시뮬레이션에서 시간 지연 개수(11개 유지 여부)
- [ ] F1 확장(S1·S7) 여부

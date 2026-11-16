# PCA/PLS 차원 축소 + 그리드 서치 계획 (SR 입력용, A-456·B-456 공통 축)

- 작성·실행: 김혜연. 결과를 받는 사람: 박종화 (SR 재실행)
- 작성일: 2026-11-16. 작업 브랜치 `explore/dimred` (main `61d7178`에서 분기)
- 배경: 교수님 제안. 기존 SR은 사전 규칙(대표 자리 보장 7개 계열 + 상관 순 3개)으로 고른 10개 변수를 썼다.
  이 실험은 그 **규칙 기반 선택의 대안**으로, 피처 풀 전체를 PCA/PLS 성분으로 줄여 SR 입력을 만든다.
  공동식 탐색이 목적이므로 성분은 A-456과 B-456이 **같은 축을 공유**한다.
- **성격: 사후 탐색.** 정식 SR(PR #27)·공동 SR 계획(`sr/joint`) 이후에 세운 계획이다.
  `preregistration.md`의 사전확정 연구가 아니며, **정식 판정을 바꾸지 않는다.**
- 이 파일은 그리드·비교를 계산하기 **전에** 커밋·푸시한다. 커밋 시각이 이 실험 안에서의 사전 고정 근거다.
  계획을 고치면 맨 아래 "변경 기록"에 날짜·이유·결과를 본 뒤인지 여부를 적는다.
- 참고 코드(읽기만 함): `origin/sr/results`의 `src/sr_candidates.py`, `src/sr_run.py`, `src/sr_select.py`,
  `results/2026-11-01_박종화_FV1_sr_folds.yaml`. main의 `src/cv_common.py`, `data/processed/folds_v1.md`.

## 1. 데이터

### 1.1 대상
- `data/processed/features_v1.csv` + `wafer_table.csv`의 MRR. `split ∈ {train, val}`, `is_outlier = 0`.
  **A-456·B-456 행만** 쓴다 (A-123 제외). test 라벨과 `data/locked/`는 읽지 않는다.
- 키 `(WAFER_ID, STAGE)`. 전체 A-456 983 / B-456 987.

### 1.2 피처 풀 (106개, 라벨 미사용으로 확인)
`sr_candidates.py`의 규칙을 그대로 적용한다.

1. `feature_families_v1.csv`의 `sr_candidate = 1` (113개 중 112개, `allzero` 제외)
2. `EXCLUDED_FEATURES`: `since_POLISH_TABLE_w` (리셋 누락, 사전확정 4.8 11번)
3. 테이블 안 결측 비율 5% 초과: `since_DRESSER_w` (A 12.4% / B 11.5%)
4. 절댓값 중앙값 1e-9 미만: `std_DRESSER_s2`, `slope_DRESSER_s2`, `std_DRESSER_TABLE_s2`, `slope_DRESSER_TABLE_s2`

→ **남은 피처 106개. A-456과 B-456에서 목록이 정확히 같다** (`sr_candidates.py`의 `EXPECTED_POOL = 106`과 일치).
두 테이블 목록이 같으므로 공통 축을 만들 때 합집합·교집합 문제가 없다.

| 계열 | 개수 | 피처 |
|---|---|---|
| 시간 | 2 | dur_POLISH_s1, dur_POLISH_s2 |
| 소모품 카운터 | 32 | {mean, std, slope}\_{BACKING_FILM, POLISH_TABLE, MEMBRANE, PSHEET}\_{s1, s2} (24), mean_DRESSER_s1, mean_DRESSER_s2, std_DRESSER_s1, slope_DRESSER_s1, mean_DRESSER_TABLE_s1, mean_DRESSER_TABLE_s2, std_DRESSER_TABLE_s1, slope_DRESSER_TABLE_s1 |
| 헤드 압력 | 30 | {mean, std, auc}\_{PCHAMBER, MAIN_OUTER, CENTER, RIPPLE, EDGE}\_{s1, s2} |
| 리테이너 링 압력 | 8 | {mean, std, auc}\_RETAINER\_{s1, s2}, P1_on, frac_hi |
| 슬러리 유량 | 18 | {mean, std, auc}\_{SLURRY_A, SLURRY_B, SLURRY_C}\_{s1, s2} |
| 기타 | 7 | {mean, std, auc}\_DRESS_WATER\_{s1, s2}, ch56_step |
| 회전 | 6 | mean\_{WAFER_ROT, STAGE_ROT, HEAD_ROT}\_{s1, s2} |
| 리셋 후 경과량 | 3 | since_BACKING_FILM_w, since_MEMBRANE_w, since_PSHEET_w |

### 1.3 웨이퍼 집합 두 가지

| 집합 | 정의 | A-456 | B-456 | 결측 처리 |
|---|---|---|---|---|
| **주(主)** | `sr_run.load_frame`과 같은 방식: 그 테이블의 정식 SR 후보 10개(`results/diagnostics/2026-11-01_박종화_sr_candidates.csv`, `selected = 1`)에 결측이 없는 웨이퍼 | **940** | **951** | 학습 fold 중앙값 대치 |
| 보조 | 전체 | 983 | 987 | 학습 fold 중앙값 대치 |

- **주 집합 940 / 951은 재현됐다** (라벨 미사용 확인, 2026-11-16). 정식 SR·공동 SR 계획과 같은 웨이퍼다.
- 주 집합 안에서는 피처 풀 106개에 **남은 결측이 0개**였다. 따라서 주 집합에서 중앙값 대치는 실제로는 작동하지 않는다
  (코드는 그대로 두고, 대치한 칸 수를 결과에 기록한다).
- 보조 집합의 결측: `since_{BACKING_FILM, MEMBRANE, PSHEET}_w` 각 63장(A+B), `P1_on`·`frac_hi` 각 14장, `_s2` 피처 48개 각 2장.

## 2. 분할

- **바깥**: `folds_v1.csv`의 25개 fold 그대로 (repeat 0–4 × fold 0–4, r = repeat × 5 + fold).
  A-456·B-456 행만 쓴다. 학습 쪽 = 같은 repeat의 `fold != k`, 평가 쪽 = `fold == k`.
- **안쪽**: 각 바깥 학습 데이터(A+B 합친 것) 안에서
  `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + repeat)`
  (= `folds_v1`이 그 repeat에 쓴 시드), groups = `WAFER_ID`, 층화 = `group`.
- A·B가 공유하는 WAFER_ID는 바깥(`folds_v1` 검증 2번)·안쪽 모두 같은 조각에 들어간다.
- 코드에서 매 fold마다 학습·평가 쪽의 키 겹침 0, WAFER_ID 겹침 0을 assert한다.

## 3. 전처리 (모두 그 fold의 학습 쪽 데이터만으로 학습)

순서:

1. **결측 대치**: 테이블별 학습 쪽 중앙값.
2. **상수 열 제거**: 학습 쪽에서 어느 한 테이블이라도 표준편차(ddof=0) < 1e-12인 열은 뺀다 (제거한 열은 기록).
3. **X 표준화**: 테이블별 학습 쪽 평균·표준편차(ddof=0). 적용 쪽 웨이퍼는 **자기 테이블**의 학습 쪽 값으로 변환한다.
4. **y(MRR) 중심화**: 테이블별 학습 쪽 평균을 뺀다 (PLS 학습에만 쓰임. PCA는 y를 쓰지 않는다).
   → 테이블 수준 차이(MRR 평균 A 73.5 / B 80.3)와 피처 수준 차이가 성분을 지배하지 않게 하기 위함.
5. **성분 학습**: A+B를 합친 학습 쪽 데이터로 **하나만** 학습한다 (공통 축).
   - PCA: `sklearn.decomposition.PCA(n_components=k, svd_solver="full")`
   - PLS: `sklearn.cross_decomposition.PLSRegression(n_components=k, scale=False)` (표준화는 3단계에서 이미 함)
   - 부호: 성분마다 절댓값이 가장 큰 가중치가 양수가 되도록 뒤집는다 (fold 간 비교·loading 보고용. 예측에는 영향 없음).
6. 출력 Z = 성분 점수 k개.

**한계(명시)**: 성분은 **테이블 안의 상대 변화 축**이다 (테이블별 표준화·중심화 뒤에 학습). 이 성분으로 공동식이 나와도
원 단위 물리식이 아니며, 각 성분은 loading(가중치)으로만 해석한다.

## 4. 그리드

- method ∈ {PCA, PLS} × 성분 수 k ∈ {1, 2, 3, 5, 8, 10} → 12개 설정.

## 5. 그리드 평가 모형

- SR은 비용 때문에 그리드에 쓰지 않는다.
- 성분 k개에 대한 **테이블별 선형회귀** `MRR_g = a_g · Z + b_g` (`LinearRegression`, 절편·기울기 모두 테이블별, 원척도 MRR).
- **한계(명시)**: 이 선택은 선형 기준이다. SR(비선형)에 최적인 k는 이보다 클 수도 작을 수도 있다.

## 6. 선택 규칙 (중첩 교차검증)

1. 각 바깥 fold(25개)마다, 그 바깥 학습 쪽 안의 안쪽 5-fold로 12개 설정을 평가한다.
   - 안쪽 fold 지표 = 두 테이블 안쪽 평가 RMSE의 **평균** (공동 SR 계획 6·7절의 "두 테이블 RMSE 평균"과 같은 방식).
   - 설정마다 안쪽 5개 지표의 평균과 표준오차 = sd(ddof=1)/√5.
2. **1-SE 규칙**: 안쪽 평균이 최저인 설정의 (평균 + 1SE) 이내인 설정 중 **가장 작은 k**를 고른다.
   그 k에서 PCA·PLS가 둘 다 이내면 안쪽 평균 RMSE가 낮은 쪽.
3. 고른 설정으로 바깥 학습 쪽 전체에서 전처리·성분·선형회귀를 다시 맞추고, 바깥 평가 쪽에서 **테이블별 RMSE**를 잰다.
   → 25개 바깥 RMSE의 평균이 **"차원 축소 + 선택" 절차 전체의 정직한 성능**이다.
4. 바깥 fold별 선택 설정과 그 바깥 RMSE를 모두 기록한다.
5. **최종 추천 설정** = 25개 바깥 fold에서 가장 많이 선택된 (method, k).
   동률이면 작은 k → 25개 바깥 fold의 안쪽 평균 RMSE 평균이 낮은 쪽 순으로 정한다 (바깥 평가 쪽 오차는 쓰지 않는다).
6. **주의(명시)**: 박종화님이 최종 추천 설정 하나를 25-fold 전체에 쓰면, 그 설정을 고를 때 25개 fold의 학습 쪽,
   즉 결과적으로 전체 데이터가 쓰였으므로 **약하게 낙관적**이다. 그래서 8절에서 **바깥 fold별 선택 설정으로 돌린 SR 결과**도
   함께 내도록 제안한다.

## 7. 비교 (주 집합, 바깥 25-fold, 테이블별 RMSE)

| 기호 | 모형 | 비고 |
|---|---|---|
| (a) | 차원 축소 + 선택 절차의 선형회귀 (6절 3번의 바깥 RMSE) | 검정 대상 |
| (b) | 기존 SR 후보 10개 변수(테이블마다 자기 후보)의 테이블별 선형회귀 | 검정 대상. **A 4.451 / B 5.066 재현 확인** (`sr_folds.yaml`의 `rmse_lr`) |
| (c) | RF, 피처 풀 106개, 테이블별 (`RandomForestRegressor(max_features=1.0, random_state=42, n_jobs=-1)`), 결측은 학습 쪽 중앙값 | 참고 |
| (d) | 평균 예측 (테이블별 학습 쪽 MRR 평균) | 참고 |

- **검정**: 테이블별로 fold별 평가 MSE 차이 d = (a) − (b)에 `cv_common.corrected_ttest`
  (J = 25, df = 24, 양측, n_train·n_eval = 그 테이블의 fold별 학습·평가 장수). (c)·(d)는 검정하지 않는다.
- (c)는 정식 SR의 RF(입력 112개, 결측 그대로, A 3.277 / B 3.446)와 입력이 달라 수치가 다를 수 있다. 같은 값이라고 하지 않는다.
- **참고값(검정 제외)**: 기존 SR 최종 식 A 4.240 / B 4.177. **식 선택에 쓴 데이터로 잰 낙관적 값**이라고 표기한다.
- 보조 집합에서도 (a)·(b)·(c)·(d)와 (a) 대 (b) 검정을 같은 방식으로 계산한다. 보조 집합의 (b)는 후보 10개의 결측을
  학습 쪽 중앙값으로 채우므로 4.451 / 5.066과 직접 비교되지 않는다.
- 해석은 예측 구조에 대한 것이다. "유의하지 않음"을 "차이 없음"으로 쓰지 않는다.

## 8. SR 단계 (박종화님 작성 칸)

> 아래는 기본 제안만 적는다. 실행 설정·판정은 박종화님이 채운다.

- 입력: 최종 추천 (method, k)의 성분 점수 Z (각 fold에서 `fit_transform`으로 학습 쪽만 써서 만든 것).
- 모형(제안): 공동 SR 계획과 같이 `MRR_g = a_g · f(Z) + b_g`, f는 두 테이블 공통.
- **공동식 인정 기준(제안)**: 같은 성분의 테이블별 선형회귀(5절 모형)보다 보정 t-검정 p < 0.05로 낮고,
  **바깥 5-fold(반복 0의 fold 0–4) 중 3개 이상에서 같은 구조가 재현**될 때만 인정.
- 보고(제안): (1) 최종 추천 설정 하나로 돌린 결과, (2) 바깥 fold별 선택 설정으로 돌린 결과를 **모두** 보고한다.
  (1)은 6절 6번 이유로 약하게 낙관적이다.
- 실행 설정: _(박종화님 작성)_
- 판정 규칙 확정: _(박종화님 작성)_

## 9. 예상 결과 (실행 전 기록)

- **PLS 1번 성분**이 dur_POLISH_s1, auc_PCHAMBER_s1, auc_MAIN_OUTER_s1, auc_SLURRY_C_s1의 **합 축**과
  테이블별 스피어만 순위상관 |ρ| ≥ 0.9로 나올 것으로 예상한다.
- 계산 방법: 주 집합 전체(A+B)로 PLS(k = 추천 k, 추천이 PCA여도 PLS로)를 3절 전처리 그대로 한 번 맞춘다.
  합 축 = 네 피처의 테이블별 표준화 값의 합. 테이블마다 PLS 1번 성분 점수와 합 축의 스피어만 상관을 구한다.
  성분 부호는 임의이므로 |ρ|로 판정하고 부호도 함께 적는다.
- 실행 후 맞았는지 그대로 보고한다 (틀려도 계획을 고치지 않는다).

## 10. 코드와 확인

- 코드: `src/explore/dimred.py`
  - 핵심 함수 `fit_transform(train_df, apply_dfs, method, k, seed) -> (Z_train, [Z_apply...], model_info)`
    - 3절 전처리와 성분을 `train_df`로만 맞추고 `apply_dfs`에 같은 변환을 적용한다.
    - MRR은 `train_df`에서만 읽는다. `apply_dfs`는 함수 시작 시 MRR 열을 제거한 복사본으로 다룬다 (누수 방지 assert).
    - docstring에 박종화님이 25-fold 각각에서 그대로 호출하는 예시를 넣는다.
  - 모든 난수는 seed 고정 (`SEED = 42`, 안쪽 fold는 42 + repeat).
- 순서
  1. 주 집합 웨이퍼 수(940 / 951)와 (b) 선형회귀 RMSE(4.451 / 5.066) 재현 확인. **다르면 멈추고 보고.**
  2. 그리드 + 중첩 선택 + 비교 (주 집합 → 보조 집합).
- 확인
  - 학습 fold 밖 데이터가 대치·표준화·상수 열 판정·성분 학습에 쓰이지 않았는지 코드 리뷰.
  - 같은 seed로 두 번 실행해 모든 결과 csv가 일치하는지 확인.

## 11. 결과 파일

CLAUDE.md 12·14절에 맞춰, **모델 성능 수치**는 `results/` 최상위(FV1 태그, `featureset = features_v1` 열 포함),
**성능이 아닌 해석·점검 결과**는 `results/diagnostics/`에 둔다. 모든 csv에 같은 이름의 `.yaml`
(코드 커밋 해시, 이 계획 파일 경로, 실행 일시, 웨이퍼 집합, 피처 풀 개수, 패키지 버전, "사후 탐색" 표시)을 둔다.
같은 이름이 있으면 덮어쓰지 않고 `_v2`를 붙인다.

| 파일 | 내용 |
|---|---|
| `results/<실행일>_김혜연_FV1_dimred_grid.csv` | 집합 × 바깥 fold × 안쪽 fold × (method, k) × 테이블 RMSE (12절 공통 열 + `set, outer_r, inner_fold, method, k`) |
| `results/<실행일>_김혜연_FV1_dimred_selection.csv` | 집합 × 바깥 fold별 선택 설정, 안쪽 평균·SE, 테이블별 바깥 RMSE |
| `results/<실행일>_김혜연_FV1_dimred_compare.csv` | 7절 비교(fold별 행 + 요약)와 보정 t-검정 결과 |
| `results/diagnostics/<실행일>_김혜연_dimred_loadings.csv` | 추천 설정을 주 집합 전체로 한 번 맞춘 성분별 가중치 상위 15개 (가중치, 그 피처와 성분 점수의 상관). **해석 전용, 성능 평가에 쓰지 않음**을 yaml에 명시 |
| `results/diagnostics/<실행일>_김혜연_dimred_pls1_check.csv` | 9절 예상 확인 (테이블별 순위상관) |

## 12. 요청에서 해석으로 정한 것 (확인 필요)

1. 안쪽 fold 지표 = 두 테이블 RMSE의 평균 (요청에 하나로 합치는 방법이 없어 공동 SR 계획 방식을 따름).
2. 결측 대치 중앙값·상수 열 판정은 **테이블별** 학습 쪽 값 (표준화가 테이블별이므로 맞춤).
3. 최종 추천 동률 처리 (6절 5번).
4. 성분 부호 고정 규칙, PLS `scale=False`.
5. 9절의 합 축 정의(테이블별 표준화 값의 합)와 |ρ| 판정.
6. 결과 파일 이름·위치: 요청한 `<실행일>_김혜연_dimred_*.csv` 대신 CLAUDE.md 12·14절에 맞춰 성능 파일 3개에 `FV1` 태그를 붙이고,
   loadings·pls1_check는 `results/diagnostics/`에 둔다.
7. (c) RF 입력 = 피처 풀 106개 + 학습 쪽 중앙값 대치 (요청의 "전체 피처 풀").

## 변경 기록

| 날짜 | 내용 | 이유 | 결과를 본 뒤인가 |
|---|---|---|---|

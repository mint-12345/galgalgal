# reveal_masks_v1 — 전이학습 라벨 공개 마스크와 전진 검증 분할

> 기준: `preregistration.md` v0.2.1 5.2절(평가 구조), 5.3절(라벨 공개 조건)
> 생성 코드: `src/make_masks.py` / 입력: `data/processed/folds_v1.csv`, `data/processed/wafer_table.csv`의 `t_start`
> MRR은 읽지 않는다. 이상치는 folds_v1 대상 행에서 이미 빠져 있다.

## 구현 세부 (2026-10-29 확정, 실행 전)

1. r = repeat×5 + fold
2. 반올림은 사사오입 floor(x + 0.5): 공개 장수 n, 전진 검증 경계 모두
3. 무작위 공개: 시드 42 + r, 장수는 고정주기와 같은 n
4. 전진 검증 공개: 고정주기 r = 5×단계(단계 0~4), 무작위 시드 42 + 단계
5. 전진 검증 파일: walkforward_v1.csv(A-123 행별 step, role) + walkforward_cutoffs_v1.csv(step, train_end, eval_start, eval_end).
   456은 eval_start 이전 웨이퍼만 쓰는 규칙을 이 경계로 적용
6. A-123 안 WAFER_ID 고유성만 확인
7. 파일: data/processed/reveal_masks_v1.csv, walkforward_v1.csv, walkforward_cutoffs_v1.csv, 설명 reveal_masks_v1.md
8. δ 상한이 1개 남았는데 다음 후보가 P1_on·allzero 쌍이면 넣지 않고 멈춤
9. T3: ρ·b 최소제곱 → ρ를 0~2로 자르면 b만 재추정 → 잔차에 δ GP (순차)
10. 90% 예측구간은 WhiteKernel 포함 관측 예측구간. f_456·ρ·b 불확실성은 무시하고 한계로 기록

11) 456 소스 모델 예측값 전달 형식: 합의 중 (전준혁·김혜연)
12) 연마 시작 시각은 data/processed/wafer_table.csv의 t_start(웨이퍼 기록 시작 시각)를 (WAFER_ID, STAGE)로 붙여 씀. 새 피처 열은 만들지 않음.
13) 공개 8%에서 센서 정상 공개가 10장 미만인 fold는 규칙대로 P1_on을 빼고, 결과표에 P1_on 포함/제외 fold를 나눠 함께 보고.

## 시간 순서

`features_v1.csv`에는 연마 시작 시각 열이 없다. 그래서 시간 순서는 `wafer_table.csv`의 `t_start`(웨이퍼 첫 TIMESTAMP, CLAUDE.md 10절)로 정한다.
이 열만 키와 함께 읽는다. A-123 431장에서 결측과 동률이 없다.
`recipe_cluster_wafer.csv`의 `t_press`는 웨이퍼 안의 상대 시간이라 순서에 쓸 수 없다.

## 파일

| 파일 | 행 수 | 열 |
|---|---|---|
| `reveal_masks_v1.csv` | 26,366 | `design, repeat, fold, r, step, method, ratio, n_train, n_reveal, pos, WAFER_ID, STAGE` |
| `walkforward_v1.csv` | 2,155 (431장 × 5단계) | `WAFER_ID, STAGE, step, role, t_rank` |
| `walkforward_cutoffs_v1.csv` | 5 | `step, n_train, n_eval, train_end, eval_start, eval_end` |

**reveal_masks_v1.csv**: 공개된 행만 들어 있다.
- `design`: `folds`(주 평가, 25개 fold) 또는 `walkforward`(보조 평가, 5단계)
- `repeat`, `fold`: folds 행에만 있다. `step`: walkforward 행에만 있다.
- `r`: 고정주기 식의 r. folds는 repeat×5 + fold, walkforward는 5×step
- `method`: `fixed`(고정주기, 주 결과) 또는 `random`(무작위, 보조 결과)
- `ratio`: 2 / 4 / 8 / 16 / 100 (%)
- `n_train`: 그 fold·단계의 A-123 학습 쪽 장수 N. `n_reveal`: 공개 장수 n
- `pos`: 학습 쪽을 `t_start` 순으로 정렬했을 때의 위치(0…N−1)

공개 위치 계산:
- 고정주기: n = floor(비율 × N + 0.5). j번째 위치 = floor((j + r/25) × N/n), j = 0…n−1. 정수 연산으로 계산한다.
- 무작위: `numpy.random.default_rng(시드).choice(N, n, replace=False)`. 시드는 folds가 42 + r, walkforward가 42 + step이다.
- 학습 쪽 정의: folds는 같은 repeat에서 fold ≠ k인 A-123 행이다. walkforward는 그 단계의 `role = train` 행이다.

**walkforward_v1.csv**: `role`은 `train` / `eval` / `unused`(평가 구간 뒤). `t_rank`는 431장 안의 시간 순위(0부터)다.

**walkforward_cutoffs_v1.csv**: `train_end`는 마지막 학습 웨이퍼의 `t_start`, `eval_start`는 첫 평가 웨이퍼의 `t_start`, `eval_end`는 마지막 평가 웨이퍼의 `t_start`다.
456 소스는 각 단계에서 `t_start < eval_start`인 웨이퍼만 쓴다.

## 검증

- **학습 쪽만 공개된다.** folds 마스크 22,440행 중 평가 쪽 웨이퍼는 0행이다. walkforward 마스크 중 `role ≠ train`인 웨이퍼도 0행이다.
- **재생성:** 두 번 생성한 파일의 SHA-256이 같다.
  - `reveal_masks_v1.csv`: `00d9ed3ef9436450…`
  - `walkforward_v1.csv`: `041916b59fe38e78…`
  - `walkforward_cutoffs_v1.csv`: `b414e3ed870b8a24…`
- **고정주기 위치는 r에 따라 다르다.** 8% 공개 위치 묶음은 25개 fold에서 모두 다르다(25/25). 첫 공개 위치 ÷ 간격(N/n)의 범위는 2%에서 0~0.954, 8%에서 0~0.893, 16%에서 0~0.957이다. 시작 위치가 간격 전체에 퍼져 있다. 무작위 8%도 25/25가 다르다.
- 100% 공개는 학습 쪽 전부다.
- A-123 안 WAFER_ID는 고유하다(431장).
- 실행 환경: numpy 2.5.3, pandas 3.0.6

### 주 평가: fold당 공개 장수 (25개 fold의 최소~최대, 센서 정상 = `features_v1.csv`의 `allzero` = 0)

| 비율 | 학습 쪽 N | 공개 n | 센서 정상, 고정주기 | 센서 정상, 무작위 |
|---|---|---|---|---|
| 2% | 344~345 | 7 | 1~5 (평균 2.8) | 0~6 (평균 2.8) |
| 4% | 344~345 | 14 | 1~9 (평균 5.2) | 1~9 (평균 5.2) |
| 8% | 344~345 | 28 | 5~16 (평균 10.2) | 4~14 (평균 10.3) |
| 16% | 344~345 | 55 | 12~31 (평균 21.0) | 15~28 (평균 20.6) |
| 100% | 344~345 | 344~345 | 123~141 (평균 132.8) | 123~141 (평균 132.8) |

### 보조 평가: 전진 검증 단계별 장수

| 단계 | 학습 | 평가 | 이후(미사용) | 공개 n (2 / 4 / 8 / 16%) | 센서 정상, 고정주기 (2 / 4 / 8 / 16 / 100%) | 센서 정상, 무작위 (2 / 4 / 8 / 16 / 100%) |
|---|---|---|---|---|---|---|
| 0 | 216 | 43 | 172 | 4 / 9 / 17 / 35 | 2 / 6 / 10 / 19 / 98 | 1 / 4 / 9 / 10 / 98 |
| 1 | 259 | 43 | 129 | 5 / 10 / 21 / 41 | 0 / 5 / 10 / 18 / 106 | 3 / 5 / 7 / 14 / 106 |
| 2 | 302 | 43 | 86 | 6 / 12 / 24 / 48 | 2 / 4 / 5 / 22 / 120 | 2 / 5 / 9 / 20 / 120 |
| 3 | 345 | 43 | 43 | 7 / 14 / 28 / 55 | 6 / 7 / 11 / 18 / 134 | 2 / 7 / 9 / 20 / 134 |
| 4 | 388 | 43 | 0 | 8 / 16 / 31 / 62 | 3 / 8 / 10 / 20 / 151 | 3 / 7 / 12 / 27 / 151 |

- 경계: floor(431 × 0.5 + 0.5) = 216, 259, 302, 345, 388, 431.
- 센서 정상 공개 장수가 10장 미만이면 δ 입력에 `P1_on`을 넣지 않는다(5.4절 4번). 주 판정 비율 8%에서 10장 미만인 fold는 고정주기 9개, 무작위 8개다(25개 중).

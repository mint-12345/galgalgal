# source_preds_v1 — 전이학습 소스 모델 f_456의 A-123 예측값

> **상태 (2026-11-01): 검증·참고용. 전이학습 공식 소스 모델이 아니다.**
> 전이학습의 공식 f_456는 전이학습 코드(PR #20 `src/a123_transfer.py`의 `fit_sources()`)가 3.2절 고정 XGBoost로
> 직접 학습한 모델이다. 이 파일을 전이학습 T2·T3 입력으로 섞어 쓰지 않는다.
>
> - 두 모델은 학습 데이터(S1 983장 / S2 1,970장, 같은 행 순서), 피처(113 / 36개, 같은 순서), 3.2절 하이퍼파라미터가 같다.
> - **다른 것은 `n_jobs`뿐이다**(이 파일 4, PR #20은 2). xgboost 3.2.0은 `subsample < 1`이면 스레드 수에 따라 결과가 달라져
>   A-123 예측이 조합별로 최대 1.79~2.91, 평균 0.47~0.63 다르다.
> - `src/source_models.py`를 `n_jobs = 2`로 돌리면 PR #20 예측과 4종 모두 비트 단위로 같다(PR #20 커밋 `ece0357`로 확인).
> - 쓰임: PR #20 소스 모델의 독립 재현 확인, 소스 모델 품질 표(456 안 교차검증), 깊이 3·6 민감도 참고.
> - 근거: `results/diagnostics/2026-11-01_전준혁_xgb_thread_check.csv`. `n_jobs` 고정값은 팀 결정 필요(3.2절에 없음).

> 기준: `preregistration.md` v0.2.1 3.2절(고정 하이퍼파라미터), 5.1절(S1·S2), 5.2절 #21(소스 시간 범위), 5.4절 #25·#27(XGBoost 주 소스, 입력 두 버전)
> 생성 코드: `src/source_models.py` (실행: `python src/source_models.py`, 출력이 있으면 멈춘다) / 작성: 전준혁, 2026-10-31
> 설정·학습 기록·점검 결과: `source_preds_v1.yaml` (SHA-256 `78886c0d1d07b5f9…`)

## 무엇이 들어 있나

소스 2개 × 입력 2개 = **소스 모델 4종**의 A-123 train+val 431장 예측값. 모델은 XGBoost, 3.2절 고정값
(`n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=3`, `random_state=42`). 튜닝 없음.

| 기호 | 소스 (학습 데이터) | 입력 | 피처 수 |
|---|---|---|---|
| S1 · F-free | A-456 983장 | `features_v1.csv` 피처 전체 | 113 |
| S1 · F-free-usage | A-456 983장 | 소모품 카운터 계열 `{mean,std,slope}_{U}_{s1,s2}` | 36 |
| S2 · F-free | A-456 983 + B-456 987 = 1,970장 | 전체 | 113 |
| S2 · F-free-usage | 1,970장 | 소모품 카운터 계열 | 36 |

- 그룹·스테이지 범주형 열은 넣지 않았다. F-free(2절) 정의 밖이고, A-123은 456 소스에 없는 범주라 의미 있는 값이 없다.
- A-123 라벨은 **읽지 않았다.** 예측값만 만들었고 A-123에 대한 오차·상관은 계산하지 않았다(전이학습 실험 전 사전 열람 방지).

## 두 가지 설계 (5.2절 #21)

| `design` | `step` | 소스 학습 데이터 | `role` |
|---|---|---|---|
| `folds` (주 평가) | −1 | 456 train+val **전체** | `all` |
| `walkforward` (보조 평가) | 0~4 | 456 중 `t_start < eval_start(step)`인 웨이퍼만 (`walkforward_cutoffs_v1.csv`) | 그 단계의 A-123 역할 `train` / `eval` / `unused` (`walkforward_v1.csv`) |

- `folds` 설계는 456 전체로 학습한 모델 1개라서 **25개 fold 모두 같은 예측값**을 쓴다. A-123 fold 분할과 무관하므로 fold별 열을 두지 않았다.
- `walkforward` 단계별 소스 장수: A-456 435 / 592 / 637 / 794 / 903, B-456 414 / 571 / 639 / 806 / 877 (단계 0~4).
- `walkforward`에서는 A-123 431장 전부에 예측값이 있다(`unused` 포함). 그 단계에서 쓸 행은 `role`로 고른다. 예측에는 미래 456이 들어가지 않으므로 `unused` 행의 예측이 있어도 누수가 아니다.

## 열

| 열 | 값 |
|---|---|
| `design` | `folds` / `walkforward` |
| `step` | −1(folds) 또는 0~4 |
| `source` | `S1` / `S2` |
| `featureset` | `F-free` / `F-free-usage` |
| `model` | `xgb` |
| `max_depth` | 4 (민감도 분석용 깊이 3·6은 다른 파일, 같은 열 구조) |
| `role` | `all` / `train` / `eval` / `unused` |
| `WAFER_ID`, `STAGE` | 웨이퍼 키 |
| `n_source` | 그 모델의 학습 행 수 |
| `pred` | f_456(x) 예측 MRR (456 척도, 약 55~94) |

행 수: 6(folds 1 + walkforward 5) × 소스 2 × 입력 2 × 431장 = **10,344행**.

### 형식을 이렇게 정한 이유 (5.7절 11번 "전달 형식 합의 중"에 대한 제안)

- **긴 형식 + 키 열**: `reveal_masks_v1.csv`처럼 `(WAFER_ID, STAGE)`로 바로 병합할 수 있고, 소스·입력·설계·단계를 열 값으로 거를 수 있다.
- **모델 종류·깊이를 열로 명시**: 보조 소스(SR 식)나 깊이 민감도(3·6) 예측을 같은 열 구조의 다른 파일로 추가해 `concat`만 하면 된다.
- **fold별 열 없음**: 주 평가 소스는 456 전체 1개 모델이라 fold마다 다시 학습할 이유가 없다(5.2절 #21 "주 평가는 456 전체 사용").
- 합의 결과 형식이 바뀌면 `_v2`로 새로 만든다.

## 사용 예 (전이학습 T2·T3)

```python
import pandas as pd
sp = pd.read_csv("data/processed/source_preds_v1.csv")
KEY = ["WAFER_ID", "STAGE"]

# 주 평가: S1, 전체 피처
f = sp.query("design == 'folds' and source == 'S1' and featureset == 'F-free'")[KEY + ["pred"]]
a123 = a123.merge(f.rename(columns={"pred": "f456"}), on=KEY, how="left", validate="1:1")
# → 각 fold에서 공개 행(reveal_masks_v1.csv)으로 ρ·b 추정, 평가 쪽에 ρ·f456 + b (+ δ)

# 보조 평가: 단계 2
w = sp.query("design == 'walkforward' and step == 2 and source == 'S2' and featureset == 'F-free'")
tr = w[w.role == "train"]; ev = w[w.role == "eval"]
```

## 점검 (`source_preds_v1.yaml`의 `checks`, 모두 통과)

1. 조합 24개(설계·단계 6 × 소스 2 × 입력 2) × A-123 431장, 예측 NaN 0, 키 중복 0
2. 예측 키 = `folds_v1.csv`의 A-123 431장
3. 소스 학습 데이터에 A-123 0장. 주 평가 소스 = S1 983 / S2 1,970장
4. 전진 검증: 모든 단계에서 소스 456의 최대 `t_start` < `eval_start`
5. 피처 수 113 / 36
6. A-123과 456이 공유하는 WAFER_ID 0개
7. 같은 시드로 두 번 학습한 예측값이 비트 단위로 같음

## 소스 모델 품질 (456 안 교차검증, folds_v1 25 fold)

`results/2026-10-31_전준혁_source456_cv.csv`. 학습·평가 모두 소스 그룹 안. **전이 성능이 아니다.**

| 소스 · 입력 | RMSE (평가 묶음별 25 fold 평균) |
|---|---|
| S1 · F-free | A-456 3.004 |
| S1 · F-free-usage | A-456 3.115 |
| S2 · F-free | A-456 3.105 / B-456 3.192 / 전체 3.151 |
| S2 · F-free-usage | A-456 4.731 / B-456 5.383 / 전체 5.072 |

S2 · 사용량 버전만 크게 나쁘다. 소모품 카운터는 장비 전체가 공유하는 값(PR #14)이라 A·B 스테이지를 구분하지 못하고,
A-456과 B-456의 MRR 수준 차이(평균 72.9 대 79.9)를 설명할 입력이 없기 때문으로 보인다.

## 깊이 민감도용 파일 (6절 "트리 깊이 3, 6", #9)

같은 코드·같은 열 구조로 `max_depth`만 바꾼 파일. 전이 대 단독 결론의 깊이 민감도를 볼 때 `source_preds_v1.csv` 대신 쓴다.

| 파일 | max_depth | 생성 명령 |
|---|---|---|
| `source_preds_v1_d3.csv` (+ `.yaml`) | 3 | `python src/source_models.py --depth 3 --out data/processed/source_preds_v1_d3.csv` |
| `source_preds_v1.csv` | 4 (주) | `python src/source_models.py` |
| `source_preds_v1_d6.csv` (+ `.yaml`) | 6 | `python src/source_models.py --depth 6 --out data/processed/source_preds_v1_d6.csv` |

세 파일을 `concat`한 뒤 `max_depth` 열로 거르면 된다. 점검 7개는 세 파일 모두 통과(각 `.yaml`의 `checks`).
456 안 교차검증 RMSE(전체 평균, 깊이 3 / 4 / 6): S1·F-free 3.042 / 3.004 / 3.040, S1·사용량 3.218 / 3.115 / 3.053,
S2·F-free 3.251 / 3.151 / 3.136, S2·사용량 5.161 / 5.072 / 5.012 (`results/2026-10-31_전준혁_depth_sensitivity.csv`).

## 알려진 한계

- A-123 전체-0 센서 웨이퍼(265장)는 압력·유량·회전 피처가 NaN이다. 456 학습 데이터에는 이 피처들의 NaN이 없어서,
  XGBoost가 기본 방향으로 보낸다. `F-free` 예측은 이 웨이퍼들에서 의미가 약할 수 있다. `ch56_step`도 A-123 전부 NaN.
  사용량 버전은 이 문제가 없다(카운터는 A-123에서도 기록됨).
- `TODO(결정 필요)`: "소모품 사용량만"을 카운터 계열 36개로 해석했다. 리셋 후 경과량 `since_*_w` 5종을 넣을지 확인 필요.

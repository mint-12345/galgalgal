# cv_common — 공통 fold 평가와 보정 t-검정 (작성: 전준혁, 2026-10-31)

`src/cv_common.py`. preregistration.md 1절(공통 fold), 3.2절, 5.5절, 5.7절 16번을 코드로 옮긴 것이다.
SR 식 선택, 모델 정비(#6·#8·#9), 전이학습이 **같은 25개 fold와 같은 검정 함수**를 쓰게 하는 것이 목적이다.

## 함수

| 함수 | 입력 | 출력 |
|---|---|---|
| `load_dataset()` | — | `features_v1.csv` 113개 피처 + `MRR`, `t_start`, `split`, `group`. 2,401행. 키 `(WAFER_ID, STAGE)`로 1:1 병합, 행 수·그룹별 수·MRR 결측 0을 확인. test 행은 버리고 `data/locked/`는 읽지 않는다 |
| `feature_columns(kind)` | `"full"`(113개) / `"usage"`(소모품 카운터 36개) | 피처 이름 목록 |
| `load_folds(data)` | `load_dataset()` 결과 | (행 수, 5) 정수 배열. `[i, r]` = 반복 r에서 행 i가 평가되는 조각 |
| `iter_folds(data)` | 위 데이터 | `Fold(r_id, repeat, fold, train, eval)` 25개. `train`·`eval`은 `data` 행 위치. **매 fold마다 `check_no_leak` 실행** |
| `check_no_leak(data, train, eval)` | 행 위치 배열 2개 | 행·키·WAFER_ID가 겹치면 `ValueError` |
| `cv_evaluate(data, fit_predict, train_filter=, eval_filter=)` | `fit_predict(train_df, eval_df, fold) -> 예측 배열` | `(scores, preds)`. `scores` 열: `r_id, repeat, fold, subset, n_train, n_eval, mse, mae, rmse` (`subset` = `all` + 평가 쪽에 있는 그룹). `preds` 열: `r_id, WAFER_ID, STAGE, group, y, pred` |
| `metrics(y, pred)` | 배열 2개 | `{"mse", "mae", "rmse"}`. 예측에 NaN이 있으면 멈춤 |
| `corrected_ttest(diff, n_train, n_eval)` | fold별 차이(A − B), fold별 실제 학습·평가 수 | `mean_diff, sd_diff, se, t, df, p, n_folds, ratio` |
| `compare(scores, a, b, subset=, metric=)` | 모델 열(`model`)을 붙여 쌓은 `scores` | `corrected_ttest` 결과 + `mean_a, mean_b`. fold는 `r_id`로 짝짓는다 |
| `ttest_ntrain_sensitivity(diff, n_eval, n_train_full, n_revealed)` | fold별 차이, 평가 수, A-123 학습 쪽 전체 수, 공개 라벨 수 | 두 해석(`full` / `revealed`)별 `corrected_ttest` 결과 + `significant`. 전이학습 민감도용 (2026-11-01 추가) |

상수: `XGB_PARAMS`, `CAT_PARAMS`(3.2절 고정값 + `random_state=42`), `SEED = 42`, `GROUPS`, `KEY`.

> **주의 (2026-11-01):** xgboost 3.2.0은 `subsample < 1`이면 **`n_jobs`(스레드 수)에 따라 예측이 달라진다.**
> `XGB_PARAMS`는 `n_jobs=4`, PR #20 전이학습 코드는 `n_jobs=2`라 같은 설정인데도 소스 모델 예측이 최대 약 2~3 다르다.
> 결과를 서로 맞춰 보려면 `n_jobs`까지 같게 둔다. 3.2절에 `n_jobs`가 없어 팀 결정이 필요하다
> (`results/diagnostics/2026-11-01_전준혁_xgb_thread_check.csv`).

## 보정 t-검정 식

d_j = 지표_A(j) − 지표_B(j), j = 1…J (J = 25)

t = mean(d) / sqrt((1/J + n_eval/n_train) · var(d)), 자유도 J − 1, 양측 p = 2·P(T > |t|)

- `var`는 표본분산(ddof = 1).
- n_eval/n_train은 **fold마다 실제 크기로 나눈 비의 평균**(5.7절 16번). folds_v1에서는 어떤 부분집합으로 세어도 약 0.25다(5조각 중 1조각이 평가).
- 판정(5.5·5.7절 16번): **p < 0.05이고 mean_diff < 0**이면 A가 낫다.
- 반복 k-fold에 J = k × 반복을 쓰는 것은 Bouckaert & Frank(2004)의 반복 교차검증 판본을 따른다.

## 사용 예 1 — 두 모델을 25개 fold에서 비교

```python
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import cv_common as cv

data = cv.load_dataset()
cols = cv.feature_columns("full")

def xgb(tr, ev, fd):
    from xgboost import XGBRegressor
    return XGBRegressor(**cv.XGB_PARAMS).fit(tr[cols], tr["MRR"]).predict(ev[cols])

def mean_only(tr, ev, fd):
    return np.full(len(ev), tr["MRR"].mean())

a123 = lambda df: df["group"] == "A-123"
s_x, p_x = cv.cv_evaluate(data, xgb, train_filter=a123, eval_filter=a123)
s_m, p_m = cv.cv_evaluate(data, mean_only, train_filter=a123, eval_filter=a123)
scores = pd.concat([s_x.assign(model="xgb"), s_m.assign(model="mean")])
print(cv.compare(scores, "xgb", "mean"))   # mean_diff < 0 이고 p < 0.05 이면 xgb 우위
```

`scores`의 한 행 예:

| r_id | repeat | fold | subset | n_train | n_eval | mse | mae | rmse |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | all | 345 | 86 | … | … | … |

## 사용 예 2 — 전이학습처럼 fold 안에서 직접 다룰 때

```python
for fd in cv.iter_folds(data):              # 25회, 누수 점검 포함
    tr = data.iloc[fd.train]; ev = data.iloc[fd.eval]
    tr_t = tr[tr["group"] == "A-123"]        # 이 안에서 reveal_masks_v1.csv로 공개 행 선택 (r = fd.r_id)
    ev_t = ev[ev["group"] == "A-123"]
    ...                                       # 모델 학습·예측
    rows.append({"r_id": fd.r_id, "model": "T3", "mse": ..., "n_train": len(tr_t), "n_eval": len(ev_t)})

res = pd.DataFrame(rows)
t3 = res[res.model == "T3"].set_index("r_id").sort_index()
t1 = res[res.model == "T1"].set_index("r_id").sort_index()
cv.corrected_ttest(t3["mse"] - t1["mse"], t3["n_train"], t3["n_eval"])
# → {'mean_diff': ..., 'sd_diff': ..., 'se': ..., 't': ..., 'df': 24, 'p': ..., 'n_folds': 25, 'ratio': ...}
```

`n_train`에 무엇을 넣을지: 5.7절 16번은 "평가/학습 비율은 실제 fold 크기"다. 이 모듈과 PR #20 `nb_test`는
A-123 학습 쪽 전체(약 345장, 비율 0.25)를 넣는다. 이 기준은 바꾸지 않았다.

공개 라벨 수를 넣는 해석과의 차이 (`src/ttest_ntrain_sensitivity.py`, `results/diagnostics/2026-11-01_전준혁_ttest_ntrain_sensitivity.csv`):

| 공개 비율 | 공개 장수 | 비율 full / revealed | 표준오차 배율 | revealed로도 p < 0.05가 되려면 full 기준 t | full p = 0.05일 때 revealed p | full p = 0.001일 때 revealed p |
|---|---|---|---|---|---|---|
| 2% | 7 | 0.25 / 12.3 | 6.53 | 13.5 | 0.75 | 0.57 |
| 4% | 14 | 0.25 / 6.16 | 4.62 | 9.54 | 0.66 | 0.43 |
| **8% (주 판정)** | **28** | **0.25 / 3.08** | **3.28** | **6.77** | **0.54** | **0.26** |
| 16% | 55 | 0.25 / 1.57 | 2.35 | 4.86 | 0.39 | 0.12 |
| 100% | 345 | 0.25 / 0.25 | 1.00 | 2.06 | 0.05 | 0.001 |

- 8%에서 full 기준으로 유의해도(|t| ≥ 2.06) |t| < 6.77이면 revealed 기준으로는 유의하지 않다. **두 해석이 판정을 가를 수 있다.**
- 통계적 근거는 어느 쪽도 확실하지 않다. Nadeau & Bengio의 비율 n_eval/n_train은 fold 사이 학습·평가 집합이 겹쳐서 생기는
  상관을 근사하는 경험식이고, 일부 라벨만 공개하는 설계를 다루지 않는다. full은 사전확정 문구에 맞고, revealed는 보수적인 상한으로 볼 수 있다.
- 권장: 주 판정은 사전확정대로 full로 하고, revealed 결과를 **나란히 보고**한다. 전이학습 결과가 나오면
  `python src/ttest_ntrain_sensitivity.py --transfer results/<전이학습 fold별 csv>`로 비교마다 두 p와 판정 변화를 계산한다.
  → 보고 방식은 전이학습 담당·팀 확인 필요 (TODO(결정 필요)).

## 확인한 것 (`tests/test_cv_common_전준혁.py`, 10개 통과)

- 비율 ≈ 0일 때 `scipy.stats.ttest_1samp`와 t·p 일치, 비율이 있을 때 손 계산식과 일치, 보정 t가 일반 t보다 작음
- `compare`가 행 순서가 아니라 `r_id`로 짝지음
- 같은 행·같은 WAFER_ID가 양쪽에 있으면 `check_no_leak`이 멈춤
- 실제 데이터: 25 fold, 반복마다 2,401행이 정확히 한 번 평가, 조각 크기 479~481, 누수 0
- A-123 필터: 반복마다 평가 431장, n_train + n_eval = 431
- `ttest_ntrain_sensitivity`가 두 해석의 `corrected_ttest`와 같고, revealed 쪽 p가 더 크다
- 재검증(2026-11-01): 독립 계산식과 t 일치(차이 0). 비율을 fold별 비의 평균으로 구하든(이 모듈) 평균 크기의 비로 구하든(PR #20)
  A-123 fold에서 0.250002 대 0.250000, #8의 p 0.712101 대 0.712100으로 판정에 영향 없음

실행 예 `python src/cv_common.py` (그룹 평균 대 전체 평균 기준선): MSE 56.0 대 889.4, t = −113.9, df = 24, ratio = 0.250.

## 결과 csv 읽을 때 (2026-10-31 파일)

`results/2026-10-31_전준혁_*`의 `model_compare.csv`, `a123_split_vs_pooled*.csv`, `source456_cv*.csv`는 헤더에 `fold`가 두 번 있다.
pandas로 읽으면 `fold`(0~24, `r_id`)와 `fold.1`(반복 안 0~4)이 된다. 값은 맞고, 짝지을 때는 `fold`(= `r_id`)를 쓴다.
코드는 2026-11-01에 고쳐서 다시 실행하면 `fold`(0~24) 하나만 나온다. 기존 파일은 덮어쓰지 않았다.

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

상수: `XGB_PARAMS`, `CAT_PARAMS`(3.2절 고정값 + `random_state=42`), `SEED = 42`, `GROUPS`, `KEY`.

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

`n_train`에 무엇을 넣을지: 5.7절 16번은 "실제 fold 크기"다. A-123 학습 쪽 전체(약 345)를 넣으면 비율이 0.25가 된다.
공개 라벨 수(예: 8% = 28장)를 넣으면 비율이 약 3이 되어 검정이 크게 보수적으로 바뀐다.
보정 항은 fold 사이 학습 집합이 겹치는 정도를 반영하는 것이므로 **fold 분할 단위(A-123 학습 쪽 전체)** 를 권장한다.
→ 전이학습 담당과 확인 필요 (TODO(결정 필요)).

## 확인한 것 (`tests/test_cv_common_전준혁.py`, 9개 통과)

- 비율 ≈ 0일 때 `scipy.stats.ttest_1samp`와 t·p 일치, 비율이 있을 때 손 계산식과 일치, 보정 t가 일반 t보다 작음
- `compare`가 행 순서가 아니라 `r_id`로 짝지음
- 같은 행·같은 WAFER_ID가 양쪽에 있으면 `check_no_leak`이 멈춤
- 실제 데이터: 25 fold, 반복마다 2,401행이 정확히 한 번 평가, 조각 크기 479~481, 누수 0
- A-123 필터: 반복마다 평가 431장, n_train + n_eval = 431

실행 예 `python src/cv_common.py` (그룹 평균 대 전체 평균 기준선): MSE 56.0 대 889.4, t = −113.9, df = 24, ratio = 0.250.

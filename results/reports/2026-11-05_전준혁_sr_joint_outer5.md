# 공동 SR 바깥 5-fold 추가 검증 보고서

- 작성: 전준혁, 2026-11-05. 브랜치 `sr/joint` (로컬 전용, push·PR·merge 안 함).
- 계획: `docs/plans/2026-11-05_joint_sr_outer5_plan.md`. 탐색 시작 전 커밋 20030aa로 확정했고, 결과 후 변경 기록에는 표기·실행 방식만 적었다.
- **성격: 사후 추가 검증.**
  - 2026-11-04 공동 SR 결과("공통 형태 없음")와 바깥 fold 0 결과를 본 뒤 정했다.
  - 사전등록이 아니고 독립 test 평가도 아니다. test 라벨과 `data/locked/`는 읽지 않았다.
  - 기존 판정과 결과(정식 SR, 2026-11-04 공동 SR)는 바꾸지 않았다.
- **완전한 외부 검증이 아니다.** 후보 변수 14개는 과거 정식 SR(2026-11-01)이 **전체 Train+Val 라벨**로 고른 후보의 합집합이다. 각 fold의 평가 웨이퍼 라벨도 후보 선정에 들어갔다.
- **이번 검증 대상은 전체 데이터로 찾은 최종 f가 아니다.** 5-fold 성능은 "fold마다 새로 찾은 f들"의 성능이다. 그 특정 최종 식이 독립적으로 검증됐다고 말할 수 없다.

## 1. 요약

| 항목 | 결과 | 분류 (계획의 결과 전 규칙) |
|---|---|---|
| 구조 재현 | 사전 정의 구조 S\*가 5개 fold 중 **1개**(r = 0)에서만 나옴. 전체 데이터 최종 f와 문자열·골격이 같은 fold는 **0개** | **반복 낮음** |
| A-456 성능 | OOF RMSE 공동 4.548 / LR 4.527, 공동 SR이 나은 fold **2/5** | **차이 불확실** |
| B-456 성능 | OOF RMSE 공동 4.800 / LR 4.937, 공동 SR이 나은 fold **4/5** | **공동 SR 우세** (규칙상) |
| 결론 방향 (계획 5절) | 구조 분류가 "반복 낮음"이므로 성능과 무관하게 정해짐 | **"수식 구조의 재현성이 낮음"** |

- **B-456 우세는 작은 차이다.**
  - fold별 RMSE 차이 평균은 −0.126, SD는 0.131이다.
  - 가장 큰 차이(r = 2의 −0.309)는 극단 웨이퍼 1장이 있는 fold에서 나왔다.
  - fold가 5개뿐이라 유의성은 주장하지 않는다. 참고용 Nadeau–Bengio 보정 t는 −1.44(p = 0.22)다.
- **반복된 것은 변수다.** 슬러리 유량 C 적분과 메인 에어백 압력 적분은 5/5 fold, 챔버 압력 적분은 4/5 fold에 들어갔다. 식의 형태는 반복되지 않았다.
- 각 fold f의 값은 전체 데이터 f와 테이블별 |r| = 0.95~0.99로 상관이 높다. 이것은 "비슷한 순서로 웨이퍼를 줄 세운다"는 뜻일 뿐이다. 같은 식이라는 근거로 쓰지 않는다 (계획 4절 4).

## 2. 실행 내역

| 항목 | 내용 |
|---|---|
| 완료 fold | r = 0~4 전체 (그림 표기 Fold 1~5) |
| fold 0 | 기존 탐색(`holdout_joint_p0~p4.jsonl`, c3542df) 재사용. 새 코드로 다시 평가해 기존 RMSE와 1e-5 이내로 일치 확인 (공동 5.069 / 4.412, LR 5.009 / 4.428) |
| fold 1~4 | 새로 탐색. 시드 42~51, niterations 500, populations 31, population_size 27, deterministic, serial, 지수 제한 손실. **설정 축소 없음** |
| 탐색 데이터 | 바깥 학습 쪽만 (n_train 1,510~1,519). 평가 쪽 웨이퍼 키 해시를 시드 캐시마다 저장하고 평가 때 대조 |
| 실행 시간 | fold 1~4: 시드 40개, 합계 17.1 CPU시간. 프로세스 3개로 11:09~18:10, 약 7.0시간. fold 0: 시드 10개, 2.9 CPU시간(이전 세션). 집계 13초 |
| 동시 실행 | RAM 확인 후 3개. 프로세스당 실사용 약 1.2 GB, 3개일 때 가용 약 1.8 GB. 시드별 결정적 실행이라 결과와 무관하다 |
| 느린 시드 | 초반 노트북 배터리 구동 때 약 1,600초였고 전원 연결 뒤 약 1,050초. fold 4 시드 51은 마지막에 혼자 돌며 7,041초가 걸렸다. 결과에는 영향 없음 |
| 환경 | `C:\Users\junju\venvs\galgalgal_sr272` (Python 3.12.10, PySR 2.7.2, juliacall 0.9.36). 설치·변경 없음. YesPatent 환경과 공유하지 않음. 버전과 입력 해시는 결과 `.yaml`에 있음. yaml의 `workers: 8`은 집계 쪽 설정이고, 탐색 프로세스 수는 3이다 |

## 3. 성능: 공동 SR 대 같은 변수 테이블별 LR

같은 변수 LR은 각 fold의 선택 f에 들어간 변수만 쓴다. 그래서 fold마다 변수가 다르다. 둘 다 바깥 학습 쪽에서 적합했다.

### 3.1 fold별 RMSE (차이 = 공동 − LR, 음수면 공동 SR이 나음)

| r (그림) | A 공동 | A LR | A 차이 | B 공동 | B LR | B 차이 | n_eval A / B |
|---|---|---|---|---|---|---|---|
| 0 (1) | 5.069 | 5.009 | +0.060 | 4.412 | 4.428 | −0.015 | 188 / 192 |
| 1 (2) | 4.426 | 4.466 | −0.040 | 4.464 | 4.596 | −0.132 | 186 / 194 |
| 2 (3) | 4.310 | 4.166 | +0.143 | 5.885 | 6.194 | −0.309 | 190 / 191 |
| 3 (4) | 4.533 | 4.523 | +0.010 | 4.540 | 4.728 | −0.188 | 188 / 184 |
| 4 (5) | 4.364 | 4.432 | −0.068 | 4.528 | 4.516 | +0.013 | 188 / 190 |

### 3.2 5-fold 집계

fold 평균과 OOF(5개 평가 fold 예측을 합쳐 웨이퍼마다 한 번 계산)는 서로 다른 값이므로 나눠서 적는다.

| 테이블 | 모형 | RMSE fold 평균 ± SD | MAE fold 평균 ± SD | **OOF RMSE** | **OOF MAE** | n |
|---|---|---|---|---|---|---|
| A-456 | 공동 SR | 4.540 ± 0.307 | 3.592 ± 0.198 | **4.548** | **3.591** | 940 |
| A-456 | LR(같은 변수) | 4.519 ± 0.306 | 3.591 ± 0.208 | **4.527** | **3.590** | 940 |
| A-456 | 차이 | +0.021 ± 0.084 | +0.001 | +0.021 | +0.001 | |
| B-456 | 공동 SR | 4.766 ± 0.627 | 3.529 ± 0.120 | **4.800** | **3.528** | 951 |
| B-456 | LR(같은 변수) | 4.892 ± 0.736 | 3.600 ± 0.164 | **4.937** | **3.599** | 951 |
| B-456 | 차이 | −0.126 ± 0.131 | −0.071 | −0.137 | −0.071 | |

- 공동 SR이 나은 fold 수: A 2/5, B 4/5.
- 참고 Nadeau–Bengio 보정 t (판정에 쓰지 않음): A t = 0.37, p = 0.73 / B t = −1.44, p = 0.22.
- 참고 행 (결론에 쓰지 않음):
  - 학습 쪽 평균 예측의 RMSE fold 평균은 A 약 6.30, B 약 9.04다.
  - 후보 14개 전부 쓴 테이블별 LR은 A 4.452, B 5.014다.
  - A에서는 14개 LR이 공동 SR보다 낮고, B에서는 높다.

## 4. 수식 구조의 재현성

### 4.1 fold별 선택 식 (바깥 학습 쪽 탐색, 안쪽 4-fold 1-표준오차)

| r | 선택 f (탐색 문자열) | 복잡도 | 변수 수 | S\* | 전체 f와 Jaccard | corr(f, 전체 f) A / B | a_A / a_B |
|---|---|---|---|---|---|---|---|
| 전체 | `(log(P1_on) − auc_RETAINER_s1) · log(auc_SLURRY_C_s1 · (auc_PCHAMBER_s1 + 0.090) · auc_MAIN_OUTER_s1)` | 13 | 5 | 예 | 1 | 1 / 1 | 34.5 / 42.3 (0.816) |
| 0 | `(auc_RETAINER_s1 + log(std_POLISH_TABLE_s1)) · log(auc_SLURRY_C_s1 · auc_PCHAMBER_s1 · auc_MAIN_OUTER_s1)` | 11 | 5 | **예** | 0.67 | −0.984 / −0.986 | −39.7 / −47.6 (0.833) |
| 1 | `(std_MEMBRANE_s1 + log(std_POLISH_TABLE_s1)) · log(auc_MAIN_OUTER_s1 · auc_PCHAMBER_s1 · auc_SLURRY_C_s1)` | 11 | 5 | 아니오 (리테이너 없음) | 0.43 | −0.984 / −0.986 | −38.2 / −47.7 (0.802) |
| 2 | `sqrt(auc_PCHAMBER_s1 · dur_POLISH_s1) + auc_SLURRY_C_s1 + auc_MAIN_OUTER_s1 / dur_POLISH_s1` | 10 | 4 | 아니오 (곱 아님) | 0.50 | −0.962 / −0.963 | −54.4 / −71.0 (0.767) |
| 3 | `auc_SLURRY_C_s1 + ((auc_RETAINER_s1 − P1_on) + auc_RETAINER_s1) · log(auc_MAIN_OUTER_s1)` | 10 | 4 | 아니오 (맨 바깥이 합) | 0.80 | −0.962 / −0.985 | −53.0 / −70.7 (0.749) |
| 4 | `auc_MAIN_OUTER_s1 · (auc_PCHAMBER_s1 + auc_SLURRY_C_s1) − log(dur_POLISH_s1) / dur_POLISH_s1` | 10 | 4 | 아니오 (곱 아님) | 0.50 | −0.954 / −0.971 | −34.9 / −42.7 (0.817) |

- 위의 `·`는 곱셈(`*`)이다. 원문 문자열은 `_selection.csv`와 `_structure.csv`에 있다.
- 괄호 안 값은 a_A/a_B 비다.
- fold별 f는 서로 다른 식이므로 a의 크기는 비교하지 않는다 (계획 4절 5).

### 4.2 문자열 일치와 구조적 유사성

- **문자열 일치·골격 일치: 0/5.** 전체 데이터 최종 f와 같은 식, 또는 상수만 다른 같은 골격을 다시 찾은 fold는 없다.
- **사전 정의 구조 S\***("리테이너 압력 항 × log(압력·슬러리 적분의 곱)"): **1/5** (r = 0). 계획 4절 분류로 **반복 낮음**이다.
- 사후 관찰(분류에는 쓰지 않음):
  - 각 fold의 후보 식 묶음(91~108개) 안에서도 S\* 식은 r = 0에 3개, r = 4에 1개만 있었다. r = 4의 그 식은 안쪽 점수가 선택 식보다 나빴다.
  - 따라서 1-표준오차 규칙이 S\*를 걸러낸 것이 아니다. 대부분의 fold에서 탐색이 그 구조를 아예 찾지 않았다.
- **비슷한 구조끼리 묶어 보기** (관찰):
  - r = 0과 r = 1은 `(x + log(std_POLISH_TABLE_s1)) · log(세 적분의 곱)`으로 골격이 같다. x만 리테이너 적분과 멤브레인 사용량 SD로 다르다.
  - r = 2와 r = 4는 세 적분에 연마 시간을 더하는 형태다.
  - r = 3은 리테이너 항 × log(에어백 적분)에 슬러리 적분을 더한 형태다.

### 4.3 변수 반복 (5회 중)

| 변수 | 등장 fold | 전체 f에 있음 |
|---|---|---|
| auc_MAIN_OUTER_s1 (메인 에어백 압력 적분) | 5/5 | 예 |
| auc_SLURRY_C_s1 (슬러리 유량 C 적분) | 5/5 | 예 |
| auc_PCHAMBER_s1 (챔버 압력 적분) | 4/5 | 예 |
| auc_RETAINER_s1 (리테이너 압력 적분) | 2/5 | 예 |
| std_POLISH_TABLE_s1 | 2/5 | 아니오 |
| dur_POLISH_s1 (연마 시간) | 2/5 | 아니오 |
| P1_on (리테이너 압력 연마 평균) | 1/5 | 예 |
| std_MEMBRANE_s1 | 1/5 | 아니오 |

나머지 후보 6개(DRESSER_TABLE, WAFER_ROT, since_MEMBRANE, PSHEET, frac_hi, since_BACKING_FILM)는 한 번도 쓰이지 않았다.

### 4.4 그룹별 계수

- 다섯 fold 모두 a_A, a_B의 부호가 같다 (fold 안에서 둘 다 음수).
- a_A/a_B 비는 0.75~0.83이고, 전체 데이터 0.816과 비슷하다.
- 계수 부호가 전체 데이터와 반대인 것은 f의 부호가 반대이기 때문이다 (corr이 음수). 관계 방향이 다르다는 뜻은 아니다.
- 이 비가 비슷한 것은 "B-456의 MRR 변동 폭이 A-456보다 크다"는 데이터 성질과 맞는다. 식 구조가 재현됐다는 근거는 아니다.

## 5. 전체 데이터 탐색 결과와의 비교

| 항목 | 2026-11-04 전체 데이터 탐색 | 이번 5-fold 재탐색 |
|---|---|---|
| 탐색 데이터 | A·B Train+Val 전체 | fold마다 그 fold를 뺀 80% |
| 선택 식 | 위 표의 "전체" 행, 복잡도 13 | 5개 모두 다른 식, 복잡도 10~11 |
| 성능 근거 | 안쪽 CV 점수와 fold 0 바깥 검증 1회 | 5개 fold의 바깥 평가와 OOF |
| 이번 검증에서 확인된 것 | (검증 대상 아님) | "공동 탐색 절차"로 얻은 f들의 성능과 구조 반복성 |

기존 판정 "공통 형태 없음"(2026-11-04 계획 9절)은 그대로 둔다. 이번 결과는 그 판정을 바꾸지 않는 추가 기술 통계다.

## 6. 극단 B 웨이퍼 사후 민감도 (주 결론을 바꾸지 않음)

- 대상: B-456 웨이퍼 −4230160594, 연마 98초, r = 2 평가 쪽.
- OOF 예측은 실측 70에 대해 공동 SR 131, LR 137이었다.
- 탐색은 다시 하지 않았다.

| 변형 | B 공동 RMSE 평균 | B LR RMSE 평균 | 차이 평균 | 공동이 나은 fold | r = 2 차이 |
|---|---|---|---|---|---|
| 전체 웨이퍼 (주 결과) | 4.766 | 4.892 | −0.126 | 4/5 | −0.309 |
| score_only (채점에서만 제외) | 4.362 | 4.434 | −0.071 | 4/5 | −0.035 |
| refit_without (계수 재적합에서도 제외) | 4.362 | 4.466 | −0.105 | 3/5 | −0.035 |

- 이 웨이퍼를 빼면 B-456의 차이는 절반 정도(−0.13 → −0.07)로 줄어든다. 방향은 그대로다.
- refit_without에서는 공동이 나은 fold가 3/5로 줄어 "차이 불확실" 범위에 들어간다.
- 같은 변형에서 r = 1의 LR RMSE가 4.60에서 5.31로 커졌다. LR 계수가 이 웨이퍼 하나에 민감하다는 관찰이다. 이번에는 원인을 더 보지 않았다.
- A-456은 이 웨이퍼와 무관해 모든 변형에서 같다.

## 7. 결론

1. **수식 구조의 재현성은 낮다** (계획 5절 결론 방향).
   - fold마다 새로 탐색하면 다른 식이 나왔다.
   - 사전 정의 구조는 1/5 fold에서만 다시 나왔고, 전체 데이터 최종 식을 다시 찾은 fold는 없다.
   - 2026-11-04 전체 데이터에서 찾은 특정 식은 이번 검증으로 재현되지 않았다. 독립적으로 검증됐다고 말할 수 없다.
2. **반복된 것은 변수 묶음이다.** 메인 에어백 압력·슬러리 유량 C·챔버 압력의 챔버 4 적분이 대부분의 fold에 들어갔다. 다만 이 변수들은 전체 Train+Val 라벨로 골라진 후보 안에 있었다.
3. **같은 변수 LR 대비 개선은 크지 않다.**
   - A-456은 차이 불확실이다 (2/5, OOF RMSE +0.02).
   - B-456은 규칙상 공동 SR 우세다 (4/5, OOF RMSE −0.14). 다만 차이가 작고, 그 일부는 극단 웨이퍼 1장이 있는 fold에서 나왔다.
   - 유의성은 주장하지 않는다.

## 8. 한계

- 사후 추가 검증이고, 후보 변수 14개가 전체 Train+Val 라벨로 골라졌다. 외부 검증이 아니다.
- 반복 0 하나(5개 fold)만 봤다. 다른 분할 반복에서는 선택 식이 또 달라질 수 있다.
- fold 0은 이번 계획 전에 이미 결과를 본 탐색이다.
- 같은 변수 LR은 공동 SR이 고른 변수를 그대로 받는다. 그래서 변수 선택의 이득은 LR 쪽에도 들어 있다.
- fold 사이 학습 쪽이 75% 겹친다. fold별 차이는 독립 표본이 아니다.
- test 세트는 쓰지 않았다.

## 9. 포스터 권장 그림과 영어 캡션

**추천: Fig 2 + Fig 3** (성능과 구조 재현성을 함께 보여줌). 한 칸뿐이면 요약 패널을 쓴다. 파일과 전체 캡션은 `results/reports/poster_joint_sr/README.md`에 있다.

> **Fig. 2. Joint SR versus linear regression with the same variables.** Left: out-of-fold RMSE pooled over five folds (each wafer predicted once; A-456 n = 940, B-456 n = 951); white dots are per-fold RMSEs. Right: per-fold RMSE difference (joint SR − linear regression); negative values favour joint SR, the bar is the mean. Joint SR was better in 2/5 folds for A-456 (OOF RMSE 4.55 vs 4.53) and in 4/5 folds for B-456 (4.80 vs 4.94). With five folds, no significance is claimed.

> **Fig. 3. Which variables and structure the re-discovered equations used.** Rows: the equation found on all Train+Val data (Nov 4, dark) and the equation re-discovered in each fold without its held-out 20% (blue). Filled cells mark variables used in f. Slurry-flow and main air-bag pressure integrals appeared in all five folds and the chamber-pressure integral in four, but the pre-defined form "retainer pressure × log(product of integrals)" reappeared in only one fold, and no fold reproduced the full-data equation.

## 10. 파일과 커밋

브랜치 `sr/joint`(로컬, 워크트리 `C:/GitHub/galgalgal_joint`)에서 작업했다. 원격 push·PR·merge는 하지 않았다.

| 커밋 | 내용 |
|---|---|
| 20030aa | 계획 문서 + 실행 코드 `src/sr_joint_outer5.py` (fold 1~4 탐색 전) |
| e3b95f9 | 결과 csv·yaml 7쌍, 그림 코드 `src/sr_joint_figs_outer5.py`, 테스트 `tests/test_sr_joint_outer5_전준혁.py`(13개), 포스터 그림·README·conclusion.json, 계획 변경 기록 |
| (이 보고서) | `results/reports/2026-11-05_전준혁_sr_joint_outer5.md` |

새 결과 파일 (기존 파일 덮어쓰기 없음):
- `results/2026-11-05_전준혁_FV1_sr_joint_outer5.csv` — fold별 지표 (주 비교 + 참고 행)
- `…_oof.csv` — 웨이퍼별 OOF 예측
- `…_selection.csv` — fold별 선택 식·계수·LR 계수
- `…_inner.csv` — 안쪽 CV 후보 점수
- `…_summary.csv` — 5-fold 집계
- `…_structure.csv` — 구조 비교
- `…_posthoc_onewafer.csv` — 극단 웨이퍼 민감도
- 각 csv에 같은 이름의 `.yaml` (환경·입력 해시·시드별 탐색 시간)
- `results/reports/poster_joint_sr/` — Fig 1~4, 요약 패널 (PNG 300 dpi + PDF), README, conclusion.json
- 탐색 캐시(저장소 밖): `C:/GitHub/galgalgal_joint_cache/outer5/fold{1..4}_seed{42..51}.json`

테스트: SR venv에서 65개, xgboost가 필요한 2개 파일은 Python 3.11에서 12개, 모두 통과했다.

건드리지 않은 것: preregistration.md, CLAUDE.md, 정식 SR 코드·결과, 2026-11-04 공동 SR 결과, features_v1.csv, 팀원 파일, `.venv`, 메인 폴더의 docs/ 삭제분과 압축.zip.

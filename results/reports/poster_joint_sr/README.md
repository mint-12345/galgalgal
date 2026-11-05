# 공동 SR 바깥 5-fold 추가 검증 — 포스터 그림

- 작성: 전준혁, 2026-11-05. 브랜치 `sr/joint` (로컬 전용).
- 생성 코드: `src/sr_joint_figs_outer5.py`. 입력은 `results/2026-11-05_전준혁_FV1_sr_joint_outer5*.csv`와 `results/2026-11-04_전준혁_FV1_sr_joint_coefs.csv`이고, 요약 패널 문구는 `conclusion.json`에서 읽는다.
- 형식: PNG 300 dpi + PDF(벡터, 글꼴 포함). 그림 글자는 영어다.
- **그림의 Fold 1~5는 `folds_v1` 반복 0의 fold r = 0~4다.** 극단 B 웨이퍼는 Fold 3(r = 2)에 있다.
- 성격: **사후 추가 검증**이다. 사전등록도, 독립 test 평가도 아니다. 후보 변수 14개는 전체 Train+Val 라벨로 골랐다. 그래서 완전한 외부 검증이 아니다.

## 한 줄 결론 (계획 5절 규칙)

구조 분류는 **반복 낮음**이다. 사전 정의 구조 S\*가 5개 fold 중 1개(Fold 1)에서만 나왔다. 그래서 결론 방향은 계획 5절의 **"수식 구조의 재현성이 낮음"**이다.

성능은 테이블마다 다르다.
- A-456: 차이 불확실. 공동 SR이 2/5 fold에서만 나았다.
- B-456: 규칙상 공동 SR 우세. 4/5 fold에서 나았고, OOF RMSE는 4.80 대 4.94다.
  - 다만 fold별 차이 평균은 −0.13 ± 0.13이다.
  - fold 수가 5개뿐이라 통계적 유의성은 주장하지 않는다.

**전체 데이터로 찾은 최종 f 자체는 이번 검증의 대상이 아니다.** 5-fold 성능은 fold마다 새로 찾은 f들의 성능이다.

## 포스터 추천

- **1순위: Fig 2 + Fig 3.**
  - Fig 2는 성능(공동 SR 대 같은 변수 LR)을 보여준다.
  - Fig 3은 구조 재현성(fold마다 다른 식, 공통 변수)을 보여준다.
  - 둘을 함께 놓아야 결론의 두 부분이 다 보인다.
- 지면이 한 칸뿐이면 **요약 패널**을 쓴다. 전체 데이터 식과 5-fold 결과를 나눠 놓은 그림이다.
- Fig 1(흐름)은 방법 칸에, Fig 4(실측 대 예측)는 보충으로 쓴다.

## 그림별 설명과 영어 캡션

### Fig 1 — `fig1_validation_flow`
검증 절차를 그린 그림이다.
- Train+Val 웨이퍼를 5등분한다.
- 각 fold에서 80%만으로 공동 SR 탐색, 안쪽 4-fold 식 선택, 재적합을 한다.
- 남은 20%를 예측하고 LR과 비교한다.

> **Fig. 1. Validation workflow.** Train+Val wafers of tables A-456 and B-456 were split into five folds. In each fold, the joint equation MRR_g = a_g·f(x) + b_g was searched from scratch on the remaining 80% (PySR template, 14 candidate variables, 10 seeds × 500 iterations), f was chosen by inner 4-fold cross-validation with the one-standard-error rule, and its constants and a_g, b_g were refit on the same 80%. The held-out 20% was predicted once and compared with per-table linear regression on the same variables. Held-out labels were used only for scoring.

### Fig 2 — `fig2_joint_vs_lr`
- 왼쪽 패널: 5개 fold 예측을 합친 OOF RMSE 막대이고, y축은 0부터다. 흰 점은 fold별 RMSE다.
- 오른쪽 패널: fold별 RMSE 차이(공동 SR − LR)다. 0보다 아래면 공동 SR이 나은 것이다.
- B-456의 Fold 3 점이 위쪽에 따로 떨어져 있다. 극단 웨이퍼 1장(연마 98초) 때문이다.

> **Fig. 2. Joint SR versus linear regression with the same variables.** Left: out-of-fold RMSE pooled over five folds (each wafer predicted once; A-456 n = 940, B-456 n = 951); white dots are per-fold RMSEs. Right: per-fold RMSE difference (joint SR − linear regression); negative values favour joint SR, the bar is the mean. Joint SR was better in 2/5 folds for A-456 (OOF RMSE 4.55 vs 4.53) and in 4/5 folds for B-456 (4.80 vs 4.94). With five folds, no significance is claimed.

### Fig 3 — `fig3_equation_reproducibility`
- 행: 전체 데이터 최종 f(검은색)와 fold별로 새로 찾은 f(파란색).
- 열: 사용 변수.
- 오른쪽 두 열: 복잡도, 사전 정의 구조 S\* 해당 여부.

슬러리 유량 C 적분과 메인 에어백 압력 적분은 5/5 fold에서 쓰였고, 챔버 압력 적분은 4/5에서 쓰였다. 식의 형태는 fold마다 다르다.

> **Fig. 3. Which variables and structure the re-discovered equations used.** Rows: the equation found on all Train+Val data (Nov 4, dark) and the equation re-discovered in each fold without its held-out 20% (blue). Filled cells mark variables used in f. Slurry-flow and main air-bag pressure integrals appeared in all five folds and the chamber-pressure integral in four, but the pre-defined form "retainer pressure × log(product of integrals)" reappeared in only one fold, and no fold reproduced the full-data equation.

### Fig 4 — `fig4_measured_vs_predicted`
실측 MRR 대 OOF 예측 MRR이다.
- 행: 공동 SR(위), LR(아래).
- 열: A-456, B-456.
- 점선: y = x.

축 밖 점 1개는 극단 B 웨이퍼다(실측 70, 예측 131/137). 축 가장자리에 삼각형으로 표시했다.

> **Fig. 4. Measured versus out-of-fold predicted MRR.** Top: joint SR; bottom: per-table linear regression with the same variables. Dashed line: y = x. One B-456 wafer with an unusually short polishing time (98 s; measured 70) was predicted far above its value by both models and is drawn as a triangle at the axis edge.

### 요약 패널 — `summary_panel`
- 왼쪽 상자: 2026-11-04에 전체 Train+Val로 찾은 최종 f와 테이블별 a·b. 이 f는 5-fold 검증 대상이 아니라고 그림에 적었다.
- 오른쪽 상자: fold마다 새로 찾은 f의 5-fold 결과.
- 아래: 계획 5절 규칙으로 고른 결론.

> **Summary.** Left: the joint equation found on all Train+Val wafers (Nov 4) with its per-table slope a_g and offset b_g; this exact equation was not what the 5-fold check tested. Right: five-fold re-discovery check, in which the equation was searched again in each fold without its held-out 20%. The equation form was not reproducible across folds, and the gain over linear regression with the same variables was unclear for A-456 and small for B-456. Post-hoc analysis on Train+Val only; candidate variables were pre-selected on all Train+Val data, so this is not an external validation.

## 파일

| 파일 | 내용 |
|---|---|
| `fig1_validation_flow.{png,pdf}` | 검증 흐름 |
| `fig2_joint_vs_lr.{png,pdf}` | OOF RMSE와 fold별 차이 |
| `fig3_equation_reproducibility.{png,pdf}` | 변수 × fold, 복잡도, S\* |
| `fig4_measured_vs_predicted.{png,pdf}` | 실측 대 OOF 예측 |
| `summary_panel.{png,pdf}` | 전체 데이터 식과 5-fold 결과 요약 |
| `conclusion.json` | 요약 패널 문구(결론 방향의 근거 포함) |

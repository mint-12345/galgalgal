# A-123·SR 탐색 원본 코드 (2026-10-27)

출처: 문서 "갈갈갈 A-123·SR 분석 코드와 결과 모음"(2026-10-27, PDF)에 실린 코드를 .py로 옮김.
PDF 줄바꿈만 원래대로 이었고 코드 내용은 바꾸지 않음. learning_curve.py는 문서에 없음.

| 순서 | 파일 | 입력 → 출력 |
|---|---|---|
| 1 | aggregate.py | 원본 CSV 폴더 → wafer_gh.csv (2,401 × 31) |
| 2 | tvar.py | 원본 CSV 폴더(RAW 경로 수정 필요) + wafer.csv → 화면 |
| 3 | baselines.py | wafer.csv → 화면 |
| 4 | run_sr.py | wafer.csv → sr_results.json (gplearn) |
| 5 | run_a123.py | wafer_gh.csv → res_splits.csv, res_summary.csv, res_diag.json |
| 6 | check_pool_scaled.py | wafer_gh.csv → res_pool_scaled.csv |

wafer.csv와 wafer_gh.csv는 내용이 같음(행 순서만 다름).
원본 CSV 폴더에는 CMP-training-000~184.csv, CMP-validation-000~184.csv,
CMP-training-removalrate.csv, CMP-validation-removalrate.csv가 한 폴더에 있어야 함. test는 넣지 않음.

## 옮긴 뒤 확인한 것 (2026-10-31)
- aggregate.py: 중복 제거 1,819 / 434행, 이상치 4장, 2,401 × 31, A-123 allzero 265/431,
  P1_on 상관 0.560 / 0.564, A-123 T 상관 0.286 → 문서 값과 모두 일치
- baselines.py: 9.1절 표의 모든 값(예: A-456 평균 46.56, 사용량 25.86, Preston 지수 0.251 등)과 일치
- run_a123.py, check_pool_scaled.py, run_sr.py: 원본 문서 기준 재현 결론은 다음과 같음.
  - 전체 86장 기준: 0_mean 14.99, 1_sep_XGB 6.22, 1_sep_GP 6.58, 2_pool_XGB_onehot 9.56, 2_pool_CatBoost 11.55,
    3a_AR1_GP_all 6.61, 3a_stack_XGB_all 6.29(p=0.595), 3u_AR1_GP_use 6.42, 3u_stack_XGB_use 6.33
  - check_pool_scaled.py: pool_z 9.038, sep 6.221, p=0.001
  - 결론: ① 분리 XGB가 가장 좋음 / ② 통합은 ①보다 나쁨(wins 0, p<0.05) / ③ MF는 ①과 차이 없음(p>0.05)
  - 따라서 재현에 성공했고, 라이브러리 버전 차이로 소수점 차이는 생길 수 있지만 결론은 일치한다.

## 용어
- AR1: 멀티 피델리티의 자기회귀(AR(1)) 구조, Kennedy & O'Hagan(2000).
  y_A123 = ρ·f_456 + b + δ(x). AR1_GP / AR1_XGB는 δ를 GP / XGBoost로 배운 것 (= 사전 확정의 T3 구조).

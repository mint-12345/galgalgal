# 최종 주장과 한계

- 이 문서는 Step 1~27 의 저장된 결과만으로 작성했습니다. 새로운 학습이나 모델 선택은 하지 않았습니다.
- 모든 수치는 `reports/final_summary/` 의 CSV 에서 읽었습니다.

## 1. 근거가 되는 핵심 수치

| 항목 | 값 |
|---|---|
| test 에서 Pure ML RMSE | 3.1856 |
| test 에서 Hybrid RMSE | 3.4650 |
| test 에서 Fixed-k Preston RMSE | 9.1367 |
| test 에서 Fixed-k Preston R² | -0.0700 |
| test 에서 Hybrid 가 Fixed-k 대비 줄인 RMSE | 5.6717 |
| test 에서 Hybrid 의 상대 감소율 (%) | 62.0762 |
| test 에서 Hybrid − Pure ML RMSE | +0.2793 |
| test 에서 Hybrid 의 상대 차이 (%) | 8.7685 |
| Step 25 에서 17개 조건 그룹 중 Hybrid 가 더 좋은 그룹 수 | 0 |
| Step 25 에서 7개 shift test 중 Hybrid 가 더 좋은 test 수 | 1 |
| Step 25 의 shift RMSE 평균 차이 (Hybrid − Pure ML) | +0.0061 |
| Hybrid k(x) 에서 PV 를 구성하는 4개 feature 의 비중 (%) | 10.7 |
| training 과 test 사이 공유 WAFER_ID 수 | 113 |

## 2. 결과로 지지되는 주장

**1) 고정 k 를 쓰는 Preston baseline 은 이 데이터의 MRR 을 거의 설명하지 못합니다.**

세 데이터 모두에서 R² 가 0 이하입니다. training -0.0082, validation -0.0312, test -0.0700 입니다. test RMSE 는 9.1367 로, 평가 구간의 평균으로 예측하는 것보다도 나쁩니다. 센서 기반 PV proxy 에 상수 하나를 곱하는 형태로는 wafer 마다 달라지는 MRR 을 따라가지 못합니다.

**2) k 를 k(x) 로 바꾸면 성능이 크게 회복됩니다.**

구조와 PV 정의를 그대로 둔 채 k 만 공정상태의 함수로 바꿨을 때 RMSE 가 세 데이터에서 각각 5.1873, 5.6988, 5.6717 줄어듭니다. 상대적으로는 59.22%, 62.54%, 62.08% 입니다. R² 는 음수에서 0.83~0.86 으로 올라갑니다. **이 연구에서 가장 크고 가장 재현성 있는 효과입니다.**

**3) 모델 순위가 세 데이터에서 반복됩니다.**

training, validation, test 모두에서 Pure ML, Hybrid, Fixed-k Preston 순입니다. 순위뿐 아니라 격차의 크기도 비슷합니다. 외부 데이터에서 결론이 뒤집히지 않았습니다.

**4) Hybrid 는 구조적으로 해석 가능합니다.**

예측이 PV_proxy 와 k(x) 로 분리되어 있어, 어떤 공정정보가 k 쪽에 기여하는지 따로 볼 수 있습니다. Step 21 에서 Stage 구분과 dresser 계열 사용량이 k(x) 기여의 대부분을 차지하고, pressure 비중이 Pure ML 보다 약 1.8배 크며, 기여의 89.3% 가 PV 밖 정보에서 나온다는 것을 확인했습니다. **이는 모델이 예측에 사용한 연관이며 인과가 아닙니다.**

**5) Hybrid 의 정확도는 Pure ML 과 크게 떨어지지 않습니다.**

test 에서 RMSE 차이가 0.2793, 상대적으로 8.8% 입니다. 해석 가능한 구조를 얻는 대가로 지불하는 정확도가 이 정도라는 점은 보고할 수 있습니다.

## 3. 결과로 지지되지 않는 주장

**1) "Hybrid 가 Pure ML 보다 정확하다" — 지지되지 않습니다.**

training +0.2530, validation +0.1582, test +0.2793 으로 세 데이터 모두에서 Hybrid 의 RMSE 가 더 높습니다. MAE 도 세 데이터 모두에서 더 높습니다. 어떤 데이터에서도 Hybrid 가 앞선 적이 없습니다.

**2) "Hybrid 가 일반적으로 더 robust 하다" — 지지되지 않습니다.**

Step 22 와 Step 23 에서는 Hybrid 가 유리해 보였지만, 그 비교 상대는 tuning 을 받지 않은 Pure ML 이었습니다. 같은 크기의 tuning 을 준 뒤 Step 25 에서 다시 비교하면 **17개 조건 그룹 전부에서 Pure ML 이 더 낮고, shift 7개 중 6개에서도 Pure ML 이 더 낮습니다.** shift RMSE 평균 차이는 +0.0061 로 사실상 동률입니다. 가장 극단적인 T1 에서만 Hybrid 가 낫습니다. **하나의 극단 조건을 일반적 우위로 확대해 말할 수 없습니다.**

**3) "P9 와 V3 가 실제 물리적 P 와 V 이다" — 지지되지 않습니다.**

PHM 데이터의 변수는 스케일이 가려져 있어 물리 단위가 없습니다. P9 은 6개 pressure 채널의 raw row 별 평균이고 V3 은 abs(HEAD_ROTATION) 입니다. 둘 다 **센서 기반 proxy** 이며, 실제 polishing pressure 나 actual relative linear velocity 로 부를 수 없습니다. 따라서 k 도 물리 상수가 아니라 **effective k** 입니다.

**4) "인과관계를 발견했다" — 지지되지 않습니다.**

Step 21 의 결과는 모델 기여도(contribution)와 연관(association)입니다. 예를 들어 dresser 사용량이 커질수록 k(x) 기여가 낮아지는 방향으로 나타나지만, 이 데이터만으로 마모가 MRR 을 낮춘다는 인과를 확인할 수 없습니다. 관측 데이터이고 개입 실험이 아닙니다.

**5) "Hybrid 가 외부 일반화에서 더 낫다" — 지지되지 않습니다.**

test 와 training 의 RMSE 차이가 Pure ML -0.1336, Hybrid -0.1073 으로 비슷합니다. 두 모델 모두 외부에서 크게 무너지지 않았고, 어느 쪽이 더 잘 일반화된다고 말할 근거가 없습니다.

## 4. 연구의 한계

**1) PHM 변수는 스케일이 가려져 있어 실제 물리 단위가 없습니다.**

압력이 몇 kPa 인지, 회전이 어떤 선속도에 해당하는지 알 수 없습니다. 따라서 이 연구는 Preston 식의 **구조**를 검증한 것이지 물리 계수를 추정한 것이 아닙니다.

**2) true Preston pressure 와 relative linear velocity 를 직접 복원할 수 없습니다.**

실제 Preston 식의 V 는 패드와 wafer 의 상대 선속도입니다. 데이터에는 회전 속도 채널만 있고 반경 정보나 기하 정보가 없어 상대 선속도를 계산할 수 없습니다. 결과 해석은 proxy 수준으로 제한됩니다.

**3) P9 × V3 는 sensor 기반 proxy 입니다.**

Step 14, 14b, 15 에서 P 후보 12개와 V 후보 6개, 요약 방식 3가지를 비교해 고른 조합입니다. 다른 조합을 쓰면 fixed-k 의 절대값이 달라집니다. 실제로 training fixed-k RMSE 는 PV 정의에 따라 8.6511 과 8.7597 로 다릅니다. **다만 어떤 조합을 써도 fixed-k 의 R² 가 0 근처이거나 음수라는 결론은 바뀌지 않습니다.**

**4) 세 모델의 엄밀한 비교는 CHAMBER_PATH = 4-5-6 에만 적용됩니다.**

1-2-3 경로는 압력과 회전 기록이 대부분 0 이어서 PV_proxy 를 계산할 수 없는 sample 이 절반을 넘습니다. 따라서 세 모델을 같은 조건에서 비교할 수 없어 제외했습니다. 삭제한 것이 아니라 비교 대상에서 뺀 것이고, 개수와 구조는 모두 보고했습니다.

**5) 1-2-3 은 별도의 process regime 입니다.**

test 에서 1-2-3 의 MRR 평균은 152.22 로 4-5-6 의 76.98 과 두 배 가까이 다릅니다. 전부 Stage A 입니다. **이 연구의 결론을 1-2-3 에 그대로 적용할 수 없습니다.**

**6) training, validation, test 가 wafer 수준에서 완전히 독립적이지 않습니다.**

같은 wafer+Stage 기록이 두 데이터에 동시에 들어간 경우는 0건이지만, 같은 WAFER_ID 가 training 과 test 사이 113개, training 과 validation 사이 115개, validation 과 test 사이 34개 공유됩니다. 같은 wafer 가 한쪽에서는 Stage A, 다른 쪽에서는 Stage B 로 나타납니다. **같은 wafer 의 소모품 상태가 부분적으로 공유되므로 외부 성능이 약간 낙관적일 수 있습니다.**

**7) 하나의 공개 CMP 데이터셋에 대한 결과입니다.**

장비, 레시피, 기간이 모두 이 데이터에 한정됩니다. 다른 장비나 다른 공정에서 같은 결론이 나온다고 말할 수 없습니다. 특히 k(x) 가 dresser 사용량과 강하게 연관된다는 관찰은 이 데이터의 소모품 교체 주기에 의존할 수 있습니다.

## 5. 한 문장 결론

**센서 기반 PV proxy 에 상수 k 를 쓰는 Preston baseline 은 이 데이터에서 실패하지만, k 를 공정상태의 함수 k(x) 로 바꾸면 RMSE 가 약 60% 회복되며, 그 결과는 validation 과 test 에서 그대로 재현됩니다. 다만 순수 데이터 기반 모델이 세 데이터 모두에서 더 정확하므로, Hybrid 의 가치는 정확도가 아니라 구조적 해석 가능성에 있습니다.**

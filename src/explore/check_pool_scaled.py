# 원본: "갈갈갈 A-123·SR 분석 코드와 결과 모음"(2026-10-27) 5절 코드 3을 PDF에서 옮겨 적음.
#       PDF 줄바꿈만 원래대로 이었고, 코드 내용은 바꾸지 않음.
"""② 통합 학습이 나쁜 이유 가설 확인: 그룹별로 MRR을 표준화(z-score)하면 회복되는가?
가설: 456의 MRR 분산(44~84)이 A-123(15)보다 커서, 제곱오차 학습이 456에 집중한다.
분할·SEED·피처·XGB 설정은 run_a123.py와 동일. 표준화 통계는 학습 데이터에서만 계산.
"""
import sys, numpy as np, pandas as pd
from scipy import stats
from xgboost import XGBRegressor
W = pd.read_csv(sys.argv[1])
PROC = ['P1_on', 'frac_hi', 'P3_on', 'Pch_on', 'T', 't_on', 'QA_on', 'QB_on', 'QC_on', 'Vh_on', 'Vw']
USE = ['U_BACKING_FILM', 'U_DRESSER', 'U_POLISHING_TABLE', 'U_DRESSER_TABLE', 'U_MEMBRANE', 'U_PRESSURIZED_SHEET', 'dU_table']
FEATS = PROC + USE + ['allzero']
A = W[W.group == 'A-123'].reset_index(drop=True); L = W[W.group != 'A-123'].reset_index(drop=True)
P = pd.concat([L, A], ignore_index=True); OH = pd.get_dummies(P.group).astype(int); P = pd.concat([P, OH], axis=1)
X = P[FEATS + list(OH.columns)].values.astype(float); nL = len(L)
def xgb(s): return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=s, n_jobs=2)
rng = np.random.RandomState(42); out = []
for s in range(20):
    idx = rng.permutation(len(A)); nt = int(round(0.2 * len(A))); te, tr = idx[:nt], idx[nt:]
    ptr = np.r_[np.arange(nL), nL + tr]; pte = nL + te
    g = P.group.values; y = P.MRR.values.copy()
    mu = {k: y[ptr][g[ptr] == k].mean() for k in np.unique(g)}; sd = {k: y[ptr][g[ptr] == k].std() for k in np.unique(g)}
    z = np.array([(y[i] - mu[g[i]]) / sd[g[i]] for i in range(len(y))])
    pz = xgb(42 + s).fit(X[ptr], z[ptr]).predict(X[pte]) * sd['A-123'] + mu['A-123']
    sep = xgb(42 + s).fit(A.loc[tr, FEATS].values.astype(float), A.MRR.values[tr]).predict(A.loc[te, FEATS].values.astype(float))
    out.append(dict(split=s, pool_z=np.mean((y[pte] - pz) ** 2), sep=np.mean((y[pte] - sep) ** 2)))
O = pd.DataFrame(out); d = O.pool_z - O.sep
t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 20 + 86 / 345)); p = 2 * stats.t.sf(abs(t), 19)
print(O.mean().round(3).to_dict(), 'diff', round(d.mean(), 3), 'wins', int((d < 0).sum()), 'p', round(p, 3))
print('MRR var by group (train+val):', W.groupby('group').MRR.var().round(1).to_dict())
O.to_csv('res_pool_scaled.csv', index=False)

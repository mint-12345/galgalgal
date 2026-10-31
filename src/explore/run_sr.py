# 원본: "갈갈갈 A-123·SR 분석 코드와 결과 모음"(2026-10-27) 8.1절을 PDF에서 옮겨 적음.
#       PDF 줄바꿈만 원래대로 이었고, 코드 내용은 바꾸지 않음. (입력: 같은 폴더의 wafer.csv)
"""그룹별 symbolic regression (gplearn).
- 탐색/선택: train만 사용 (gplearn max_samples=0.8 → OOB 적합도로 모델 선택)
- val: 최종 점수만 계산 (선택에 쓰지 않음)
- 입력은 train 중앙값으로 나눠 무차원화 (x/median) → 식의 상수 해석이 쉬움
- 출력: 선택된 식을 MRR = a + b*f(x) 로 train에서 선형 보정 (SR 표준 관행)
"""
import json, sys, warnings
import numpy as np, pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

SEED0 = 42
W = pd.read_csv('wafer.csv')

PHYS = ['P1_on', 'frac_hi', 'P3_on', 'Pch_on', 'T', 't_on', 'QA_on', 'QB_on', 'QC_on', 'Vh_on', 'Vw']
USE = ['U_DRESSER', 'U_DRESSER_TABLE', 'U_POLISHING_TABLE', 'U_MEMBRANE']
SETS = {'P': PHYS, 'P+U': PHYS + USE}


def _pow2(x):
    return np.clip(x, -1e3, 1e3) ** 2
sq = make_function(function=_pow2, name='sq', arity=1)
FUNCS = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', sq)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


results = []
for grp in ["A-123", "A-456", "B-456"]:
    tr = W[(W.split == 'train') & (W.group == grp) & (W.allzero == 0)]
    va = W[(W.split == 'val') & (W.group == grp) & (W.allzero == 0)]
    for sname, feats in SETS.items():
        a = tr.dropna(subset=feats + ['MRR']); b = va.dropna(subset=feats + ['MRR'])
        med = a[feats].median().replace(0, 1.0)
        Xa = (a[feats] / med).values; Xb = (b[feats] / med).values
        ya = a['MRR'].values; yb = b['MRR'].values
        rec = dict(group=grp, set=sname, n_train=len(a), n_val=len(b),
                   n_train_dropped=len(tr) - len(a), n_val_dropped=len(va) - len(b),
                   val_mse_mean=mse(yb, ya.mean()), val_var=float(np.var(yb)))
        lr = LinearRegression().fit(Xa, ya)
        rec['val_mse_linear'] = mse(yb, lr.predict(Xb))
        cands = []
        for pc in [0.0005, 0.002, 0.01]:
            for s in range(4):
                sr = SymbolicRegressor(population_size=3000, generations=30, tournament_size=20,
                                       function_set=FUNCS, metric='mse', const_range=(-2., 2.),
                                       parsimony_coefficient=pc, max_samples=0.8,
                                       p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                                       p_point_mutation=0.1, init_depth=(2, 5),
                                       random_state=SEED0 + s, n_jobs=1)
                sr.fit(Xa, ya)
                prog = sr._program
                cands.append((prog.oob_fitness_, prog.length_, pc, s, sr))
        # 선택 규칙 (계산 전에 정함): OOB MSE가 최소값의 1.05배 이내인 후보 중 가장 짧은 식
        omin = min(c[0] for c in cands)
        ok = [c for c in cands if c[0] <= 1.05 * omin]
        oob, _, pc, s, sr = min(ok, key=lambda c: (c[1], c[0]))
        rec['candidates'] = [dict(oob=float(c[0]), length=int(c[1]), parsimony=c[2], seed=SEED0 + c[3]) for c in cands]
        rec['oob_min'] = float(omin)
        fa = sr.predict(Xa); fb = sr.predict(Xb)
        cal = LinearRegression().fit(fa.reshape(-1, 1), ya)
        import re
        raw = str(sr._program)
        expr = re.sub(r'X(\d+)', lambda m: feats[int(m.group(1))], raw)
        used = sorted({feats[int(i)] for i in re.findall(r'X(\d+)', raw)})
        rec.update(expr=expr, used=used, length=int(sr._program.length_), parsimony=pc, seed=SEED0 + s,
                   oob_mse=float(oob), cal_a=float(cal.intercept_), cal_b=float(cal.coef_[0]),
                   train_mse_sr=mse(ya, cal.predict(fa.reshape(-1, 1))),
                   val_mse_sr=mse(yb, cal.predict(fb.reshape(-1, 1))),
                   medians={f: float(med[f]) for f in feats})
        print(json.dumps({k: v for k, v in rec.items() if k != 'medians'}, ensure_ascii=False), flush=True)
        results.append(rec)
        json.dump(results, open('sr_results.json', 'w'), ensure_ascii=False, indent=1)

json.dump(results, open('sr_results.json', 'w'), ensure_ascii=False, indent=1)

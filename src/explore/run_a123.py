# 원본: "갈갈갈 A-123·SR 분석 코드와 결과 모음"(2026-10-27) 4절 코드 2를 PDF에서 옮겨 적음.
#       PDF 줄바꿈만 원래대로 이었고, 코드 내용은 바꾸지 않음. (gp()는 원문에도 쓰이지 않는 남은 코드)
"""A-123 예측: ① 분리 학습 / ② 통합 학습 / ③ 멀티 피델리티 비교.
팀 규칙 (진행보고 2026-10-12, README, 자문보고서 2026-10-14)
- 웨이퍼 키 = (WAFER_ID, STAGE), 완전 중복 행 제거, train 이상치 4장(A-123, MRR>1000) 제외
- test 센서/라벨은 읽지 않음 (라벨 파일은 잠긴 폴더로 이동)
- 자문: test를 뺀 나머지(train+val) 전체를 교차검증에 사용
- Monte Carlo CV 20회, A-123 무작위 80/20, SEED 42
- 범주형(그룹)은 0/1 단일 변수 대신 one-hot 또는 CatBoost 범주형 처리 (자문)

평가 대상은 항상 A-123의 평가 조각(20%)뿐이다. 456 데이터는 평가에 쓰이지 않으므로
②, ③에서 매 분할 전부 학습에 써도 누수가 아니다.
"""
import json, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

SEED, N_SPLIT, TEST_FRAC = 42, 20, 0.2
W = pd.read_csv(sys.argv[1])
OUT = sys.argv[2]

PROC = ['P1_on', 'frac_hi', 'P3_on', 'Pch_on', 'T', 't_on', 'QA_on', 'QB_on', 'QC_on', 'Vh_on', 'Vw']
USE = ['U_BACKING_FILM', 'U_DRESSER', 'U_POLISHING_TABLE', 'U_DRESSER_TABLE', 'U_MEMBRANE',
       'U_PRESSURIZED_SHEET', 'dU_table']
FEATS = PROC + USE + ['allzero']          # 자문보고 6.4 방식 4(결측 표시 피처) — 전체-0 웨이퍼 유지
A = W[W.group == 'A-123'].reset_index(drop=True)          # HF: train+val 431장
L = W[W.group != 'A-123'].reset_index(drop=True)          # LF 원천: A-456 + B-456 1,970장
L_stageA = (L.STAGE == 'A').astype(int).values


def xgb(seed):          # 하이퍼파라미터 고정, 탐색 안 함 (진행보고 3절과 같은 방침)
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, random_state=seed, n_jobs=2)


def gp(seed):
    k = C(1.0) * RBF(length_scale=np.ones(1)) + WhiteKernel(1.0)
    return k


def fit_gp(X, y, seed):
    """GP: 결측은 학습 조각 중앙값으로 채우고 표준화, ARD-RBF + 백색잡음."""
    med = np.nanmedian(X, axis=0); med = np.where(np.isnan(med), 0, med)
    Xf = np.where(np.isnan(X), med, X)
    mu, sd = Xf.mean(0), Xf.std(0); sd[sd == 0] = 1
    k = C(1.0, (1e-2, 1e3)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(0.5, (1e-3, 10))
    m = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=seed)
    m.fit((Xf - mu) / sd, y)
    return lambda Z: m.predict((np.where(np.isnan(Z), med, Z) - mu) / sd)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ---- LF 모델: 456 전체로 한 번만 학습 (A-123 라벨을 쓰지 않으므로 분할과 무관) ----
def make_lf(cols, seed=SEED):
    X = np.column_stack([L[cols].values.astype(float), L_stageA])
    m = xgb(seed).fit(X, L.MRR.values)
    return lambda D: m.predict(np.column_stack([D[cols].values.astype(float), np.ones(len(D))]))  # A-123은 stage A


lf_all = make_lf(PROC + USE)          # ③a: 전체 피처 LF
lf_use = make_lf(USE)                 # ③b: 456과 범위가 겹치는 사용량 피처만 쓴 LF
A['f_L_all'] = lf_all(A); A['f_L_use'] = lf_use(A)
diag = {
    'n_A123': len(A), 'n_A123_allzero': int(A.allzero.sum()), 'n_LF': len(L),
    'corr_fL_all_vs_y_A123': float(np.corrcoef(A.f_L_all, A.MRR)[0, 1]),
    'corr_fL_use_vs_y_A123': float(np.corrcoef(A.f_L_use, A.MRR)[0, 1]),
    'corr_fL_use_vs_y_A123_clean': float(np.corrcoef(A.f_L_use[A.allzero == 0], A.MRR[A.allzero == 0])[0, 1]),
    'corr_fL_use_vs_y_A123_zero': float(np.corrcoef(A.f_L_use[A.allzero == 1], A.MRR[A.allzero == 1])[0, 1]),
    'fL_all_range_on_A123': [float(A.f_L_all.min()), float(A.f_L_all.max())],
    'y_A123_mean': float(A.MRR.mean()), 'y_A123_var': float(A.MRR.var()),
}
print(json.dumps(diag, indent=1), flush=True)

# ---- 통합 학습용 데이터 (one-hot 3열) ----
P = pd.concat([L.assign(_g=L.group), A.assign(_g='A-123')], ignore_index=True)
OH = pd.get_dummies(P._g).astype(int)
OHC = list(OH.columns)
P = pd.concat([P, OH], axis=1)
nL = len(L)

rng = np.random.RandomState(SEED)
splits = []
for _ in range(N_SPLIT):
    idx = rng.permutation(len(A)); nt = int(round(TEST_FRAC * len(A)))
    splits.append((idx[nt:], idx[:nt]))

rec = []
for s, (tr, te) in enumerate(splits):
    a_tr, a_te = A.iloc[tr], A.iloc[te]
    y_tr, y_te = a_tr.MRR.values, a_te.MRR.values
    X_tr, X_te = a_tr[FEATS].values.astype(float), a_te[FEATS].values.astype(float)
    pred = {}
    pred['0_mean'] = np.full(len(te), y_tr.mean())
    # ① 분리 학습
    pred['1_sep_XGB'] = xgb(SEED + s).fit(X_tr, y_tr).predict(X_te)
    pred['1_sep_GP'] = fit_gp(X_tr, y_tr, SEED + s)(X_te)
    # ② 통합 학습: 456 전체 + A-123 학습 조각
    ptr = np.r_[np.arange(nL), nL + tr]; pte = nL + te
    Xp = P[FEATS + OHC].values.astype(float)
    pred['2_pool_XGB_onehot'] = xgb(SEED + s).fit(Xp[ptr], P.MRR.values[ptr]).predict(Xp[pte])
    cb = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05, random_seed=SEED + s, verbose=0,
                           cat_features=[len(FEATS)])
    Xc = P[FEATS].copy(); Xc['grp'] = P._g.values
    cb.fit(Xc.iloc[ptr], P.MRR.values[ptr]); pred['2_pool_CatBoost'] = cb.predict(Xc.iloc[pte])
    # ③ 멀티 피델리티(AR1, Kennedy & O'Hagan 2000; 재귀형 추정)
    #    y_H(x) = rho * f_L(x) + b + delta(x)
    for tag in ['all', 'use']:
        fl_tr, fl_te = a_tr[f'f_L_{tag}'].values, a_te[f'f_L_{tag}'].values
        lr = LinearRegression().fit(fl_tr.reshape(-1, 1), y_tr)
        base_tr, base_te = lr.predict(fl_tr.reshape(-1, 1)), lr.predict(fl_te.reshape(-1, 1))
        pred[f'3{tag[0]}_LFonly_{tag}'] = base_te                                   # delta 없이 LF만 보정
        pred[f'3{tag[0]}_AR1_GP_{tag}'] = base_te + fit_gp(X_tr, y_tr - base_tr, SEED + s)(X_te)
        dx = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=3, random_state=SEED + s, n_jobs=2)
        pred[f'3{tag[0]}_AR1_XGB_{tag}'] = base_te + dx.fit(X_tr, y_tr - base_tr).predict(X_te)
        pred[f'3{tag[0]}_stack_XGB_{tag}'] = xgb(SEED + s).fit(
            np.column_stack([X_tr, fl_tr]), y_tr).predict(np.column_stack([X_te, fl_te]))
        rec.append(dict(split=s, model=f'3{tag[0]}_rho_{tag}', mse=float(lr.coef_[0]), subset='rho'))
    z = a_te.allzero.values == 1
    for k, p in pred.items():
        rec.append(dict(split=s, model=k, subset='all', mse=mse(y_te, p)))
        rec.append(dict(split=s, model=k, subset='clean', mse=mse(y_te[~z], p[~z])))
        rec.append(dict(split=s, model=k, subset='zero', mse=mse(y_te[z], p[z])))
    print(f'split {s} done: ' + ' '.join(f'{k}={mse(y_te, p):.2f}' for k, p in pred.items()), flush=True)

R = pd.DataFrame(rec)
R.to_csv(OUT + '_splits.csv', index=False)

# ---- 요약: 평균±표준편차, ①-XGB 대비 쌍대 비교 (Nadeau & Bengio 2003 보정 t) ----
n_tr, n_te = len(splits[0][0]), len(splits[0][1])
summ = []
for sub in ['all', 'clean', 'zero']:
    S = R[R.subset == sub].pivot(index='split', columns='model', values='mse')
    for ref in ['1_sep_XGB', '1_sep_GP']:
        for m in S.columns:
            d = S[m] - S[ref]; J = len(d)
            var = d.var(ddof=1) * (1 / J + n_te / n_tr)
            t = d.mean() / np.sqrt(var) if var > 0 else np.nan
            p = 2 * stats.t.sf(abs(t), J - 1) if var > 0 else np.nan
            summ.append(dict(subset=sub, model=m, ref=ref, mse_mean=S[m].mean(), mse_sd=S[m].std(),
                             diff_vs_ref=d.mean(), wins=int((d < 0).sum()), p_corrected=p))
rho = R[R.subset == 'rho'].groupby('model').mse.agg(['mean', 'std'])
Sm = pd.DataFrame(summ); Sm.to_csv(OUT + '_summary.csv', index=False)
json.dump(dict(diag=diag, rho=rho.to_dict(), n_train_split=n_tr, n_test_split=n_te),
          open(OUT + '_diag.json', 'w'), indent=1)
pd.set_option('display.width', 250)
print(Sm[Sm.ref == '1_sep_XGB'].round(3).to_string(index=False))
print(rho)

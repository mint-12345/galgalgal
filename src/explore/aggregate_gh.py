# 원본: "갈갈갈 A-123·SR 분석 코드와 결과 모음"(2026-10-27) 3절 코드 1을 PDF에서 옮겨 적음.
#       PDF 줄바꿈만 원래대로 이었고, 코드 내용은 바꾸지 않음.
"""웨이퍼 단위 물리 피처 집계 (train + val).
정의는 김혜연 진행보고(2026-10-12) 5~6절을 따름.
키 = (WAFER_ID, STAGE), 그룹 = STAGE-챔버셋 (A-456, B-456, A-123).
"""
import glob, sys
import numpy as np, pandas as pd

RAW = sys.argv[1]
OUT = sys.argv[2]

USAGE = ['USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER', 'USAGE_OF_POLISHING_TABLE',
         'USAGE_OF_DRESSER_TABLE', 'USAGE_OF_MEMBRANE', 'USAGE_OF_PRESSURIZED_SHEET']
AIRBAG = ['MAIN_OUTER_AIR_BAG_PRESSURE', 'CENTER_AIR_BAG_PRESSURE',
          'RIPPLE_AIR_BAG_PRESSURE', 'EDGE_AIR_BAG_PRESSURE']
SENS = AIRBAG + ['PRESSURIZED_CHAMBER_PRESSURE', 'RETAINER_RING_PRESSURE',
                 'SLURRY_FLOW_LINE_A', 'SLURRY_FLOW_LINE_B', 'SLURRY_FLOW_LINE_C',
                 'WAFER_ROTATION', 'STAGE_ROTATION', 'HEAD_ROTATION']


def load(split):
    fs = sorted(glob.glob(f'{RAW}/CMP-{split}-[0-9][0-9][0-9].csv'))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d.dropna(subset=['WAFER_ID'])
    n0 = len(d); d = d.drop_duplicates(); print(split, 'files', len(fs), 'dup rows removed', n0 - len(d))
    d['split'] = split
    return d


def agg_wafer(g):
    g = g.sort_values('TIMESTAMP')
    on = g['STAGE_ROTATION'] > 0
    rr = g['RETAINER_RING_PRESSURE']
    r = {}
    r['t_start'] = g['TIMESTAMP'].min()
    r['n_rows'] = len(g)
    r['T_all'] = on.mean()
    r['T'] = on[rr > 0].mean() if (rr > 0).any() else np.nan   # 연마 구간(리테이너>0) 중 stage on 비율 (보고서 5.4 r와 일치 확인)
    r['t_on'] = on.sum()                                        # 연마 행 수(≈초)
    m = (rr > 0) & on
    r['P1_on'] = rr[m].mean() if m.any() else np.nan            # 리테이너 링, 압력>0 & stage on
    r['P1'] = rr[rr > 0].mean() if (rr > 0).any() else np.nan
    r['frac_hi'] = (rr[m] > 4000).mean() if m.any() else np.nan  # 7,135 단계 체류 비율
    ab = g[AIRBAG].where(g[AIRBAG] > 0)
    r['P3'] = ab.mean(axis=1).mean()                            # 에어백 4개 평균(압력 구간)
    r['P3_on'] = ab[on].mean(axis=1).mean() if on.any() else np.nan
    pc = g['PRESSURIZED_CHAMBER_PRESSURE']
    r['Pch_on'] = pc[on & (pc > 0)].mean() if (on & (pc > 0)).any() else np.nan
    for c, k in [('SLURRY_FLOW_LINE_A', 'QA'), ('SLURRY_FLOW_LINE_B', 'QB'), ('SLURRY_FLOW_LINE_C', 'QC')]:
        r[k + '_on'] = g.loc[on, c].mean() if on.any() else np.nan
    r['Vh_on'] = g.loc[on, 'HEAD_ROTATION'].mean() if on.any() else np.nan
    r['Vs_on'] = g.loc[on, 'STAGE_ROTATION'].mean() if on.any() else np.nan
    r['Vw'] = g['WAFER_ROTATION'][g['WAFER_ROTATION'] > 0].mean()
    for u in USAGE:
        r[u.replace('USAGE_OF_', 'U_')] = g[u].mean()
    r['dU_table'] = g['USAGE_OF_POLISHING_TABLE'].max() - g['USAGE_OF_POLISHING_TABLE'].min()
    r['allzero'] = int(any((g[c] == 0).all() for c in SENS))
    r['chamber_set'] = '123' if g['CHAMBER'].max() <= 3 else '456'
    return pd.Series(r)


rows = []
for split in ['training', 'validation']:
    d = load(split)
    w = d.groupby(['WAFER_ID', 'STAGE']).apply(agg_wafer).reset_index()
    lab = pd.read_csv(f'{RAW}/CMP-{split}-removalrate.csv')
    w = w.merge(lab, on=['WAFER_ID', 'STAGE'], how='left')
    w['split'] = 'train' if split == 'training' else 'val'
    rows.append(w)
W = pd.concat(rows, ignore_index=True)
W['group'] = W['STAGE'] + '-' + W['chamber_set']
W = W.rename(columns={'AVG_REMOVAL_RATE': 'MRR'})
# 팀 규칙: train 이상치(MRR>1000, A-123 4장) 제외
out = (W['split'] == 'train') & (W['MRR'] > 1000)
print('outliers dropped', out.sum(), W.loc[out, 'group'].tolist())
W = W[~out]
print(W.groupby(['split', 'group']).size())
W.to_csv(OUT, index=False)

# 원본: "갈갈갈 A-123·SR 분석 코드와 결과 모음"(2026-10-27) 8.3절을 PDF에서 옮겨 적음.
#       PDF 줄바꿈만 원래대로 이었고, 코드 내용은 바꾸지 않음.
#       RAW 경로는 원문에 고정돼 있음 → 실행 환경에 맞게 바꿔야 함. (입력: 같은 폴더의 wafer.csv)
import glob, numpy as np, pandas as pd
RAW='/home/claude/data/raw'
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RAW}/CMP-training-[0-9]*.csv'))]).dropna(subset=['WAFER_ID']).drop_duplicates()
d=d[d.CHAMBER<=3]
lab=pd.read_csv(f'{RAW}/CMP-training-removalrate.csv')
def f(g):
    on=g.STAGE_ROTATION>0; rr=g.RETAINER_RING_PRESSURE>0
    r={}
    r['T_all']=on.mean()
    r['T_rr']=on[rr].mean() if rr.any() else np.nan
    for c in [1,2,3]:
        m=g.CHAMBER==c; r[f'T_c{c}']=on[m].mean() if m.any() else np.nan; r[f'n_c{c}']=m.sum()
    r['T_c1rr']=on[(g.CHAMBER==1)&rr].mean() if ((g.CHAMBER==1)&rr).any() else np.nan
    pres=(g[['MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE']]>0).any(axis=1)
    r['T_press']=on[pres].mean() if pres.any() else np.nan
    sl=g.SLURRY_FLOW_LINE_C>0
    r['T_slurry']=on[sl].mean() if sl.any() else np.nan
    return pd.Series(r)
w=d.groupby(['WAFER_ID','STAGE']).apply(f).reset_index().merge(lab,on=['WAFER_ID','STAGE'])
W=pd.read_csv('wafer.csv'); W=W[(W.split=='train')&(W.group=='A-123')][['WAFER_ID','STAGE','allzero']]
w=w.merge(W,on=['WAFER_ID','STAGE']); w=w[w.allzero==0]
print(len(w)); print(w.drop(columns=['WAFER_ID','STAGE']).corr()['AVG_REMOVAL_RATE'].round(3).to_string())
print(w[['T_all','T_rr','T_c1','T_press']].describe().round(3))

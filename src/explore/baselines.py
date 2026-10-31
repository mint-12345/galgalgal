# 원본: "갈갈갈 A-123·SR 분석 코드와 결과 모음"(2026-10-27) 8.2절을 PDF에서 옮겨 적음.
#       PDF 줄바꿈만 원래대로 이었고, 코드 내용은 바꾸지 않음. (입력: 같은 폴더의 wafer.csv)
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
W=pd.read_csv('wafer.csv')
def mse(a,b): return float(np.mean((a-b)**2))
def fit(tr,va,cols,log=False):
    a=tr.dropna(subset=cols); b=va.dropna(subset=cols)
    if log:
        a=a[(a[cols]>0).all(1)]; b=b[(b[cols]>0).all(1)]
        m=LinearRegression().fit(np.log(a[cols]),np.log(a.MRR)); p=np.exp(m.predict(np.log(b[cols])))
        return mse(b.MRR.values,p), dict(zip(cols,m.coef_.round(3))), len(b)
    m=LinearRegression().fit(a[cols],a.MRR); return mse(b.MRR.values,m.predict(b[cols])), None, len(b)
rows=[]
for g in ['A-456','B-456','A-123']:
    tr=W[(W.split=='train')&(W.group==g)&(W.allzero==0)]; va=W[(W.split=='val')&(W.group==g)&(W.allzero==0)]
    r={'group':g,'val_mse_mean':mse(va.MRR.values,tr.MRR.mean())}
    for name,cols,lg in [('lin_P1on',['P1_on'],0),('lin_T',['T'],0),('lin_frachi',['frac_hi'],0),
                         ('preston_P1on_Vh',['P1_on','Vh_on'],1),('preston_P1on_T',['P1_on','T'],1),
                         ('time_only',['t_start'],0),('usage_only',['U_DRESSER','U_DRESSER_TABLE','U_POLISHING_TABLE','U_MEMBRANE'],0)]:
        m,c,n=fit(tr,va,cols,lg); r[name]=round(m,2)
        if c: r[name+'_exp']=c
    rows.append(r)
pd.set_option('display.width',250)
print(pd.DataFrame(rows).T.to_string())

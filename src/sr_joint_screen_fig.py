"""공동 SR 1단계 포스터 그림: 그룹별 RMSE (상위 1개 공동 모형 대 기준선), 오차 막대 = fold 표준오차, 2.3인치 너비.

입력: results/2026-11-06_김혜연_FV1_sr_joint_screen_folds.csv. 실행: python src/sr_joint_screen_fig.py
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Malgun Gothic","font.size":7,"axes.unicode_minus":False})
f=pd.read_csv("results/2026-11-06_김혜연_FV1_sr_joint_screen_folds.csv",encoding="utf-8-sig")
def get(g,key,model,sc):
    x=f[(f.group==g)&(f.rank_key==key)&(f.model==model)&(f.scale==sc)].rmse; return x.mean(), x.std(ddof=1)/np.sqrt(len(x))
rows=[("공동 상위 1*","top1","joint","pooled"),("(가) 그룹별 LR","top1","lr_group","pooled"),("(나) 공통기울기 LR","top1","lr_shared","pooled"),
      ("(다) 정식 SR","official","sr_final","group"),("(라) RF","official","rf","group")]
fig,axes=plt.subplots(2,1,figsize=(2.3,2.6),sharex=True)
for ax,g in zip(axes,["A-456","B-456"]):
    for i,(lab,k,m,sc) in enumerate(rows):
        mu,se=get(g,k,m,sc); c="#2a78d6" if m=="joint" else "#52514e"
        ax.errorbar(mu,i,xerr=se,fmt="o",ms=4,color=c,ecolor=c,elinewidth=1.2,capsize=2)
        ax.text(mu+se+0.06,i,f"{mu:.2f}",va="center",fontsize=6.5,color="#0b0b0b")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_title(g,fontsize=7.5,loc="left",fontweight="bold",pad=2)
    ax.grid(axis="x",color="#e4e3df",lw=0.6); ax.set_axisbelow(True)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y",length=0); ax.set_ylim(len(rows)-0.5,-0.6)
axes[1].set_xlabel("평가 RMSE (평균 ± fold SE)"); axes[1].set_xlim(3.0,5.4)
fig.text(0.01,0.005,"* 246식 중 선택: 선택 낙관성 있음",fontsize=5.5,color="#52514e")
fig.tight_layout(pad=0.3,h_pad=0.6); fig.subplots_adjust(bottom=0.17)
fig.savefig("results/reports/2026-11-06_김혜연_sr_joint_screen_rmse.png",dpi=300)

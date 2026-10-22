import sys; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
import figstyle as F
ROOT="/home/claude/proj"
sut=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_unit_time.csv")
aud=pd.read_csv(f"{ROOT}/outputs/r4pde/audpc_per_unit.csv")
sut["group3"]=np.where(sut.treatment=="control","Control",sut.accession)
aud["group3"]=np.where(aud.treatment=="control","Control",aud.accession)
TPn={"T1":1,"T2":2,"T3":3}; sut["t"]=sut.timepoint.map(TPn)

fig=plt.figure(figsize=(182*F.MM,150*F.MM))
gs=fig.add_gridspec(3,3,height_ratios=[2.15,1,1],hspace=0.40,wspace=0.42)

# A severity-calculation panel (embed)
axA=fig.add_subplot(gs[0,:]); axA.imshow(np.asarray(Image.open(f"{ROOT}/outputs/supp_figures/sev_calc_panel.png")))
F.noticks(axA)
fig.text(0.012,0.975,"A",fontsize=9,fontweight="bold")

# B disease progress curves (mean ± SE) by group
axB=fig.add_subplot(gs[1,0])
for g,col in F.GRP3.items():
    s=sut[sut.group3==g].groupby("t").severity.agg(["mean","sem"]).reindex([1,2,3])
    axB.plot([1,2,3],s["mean"],"-o",color=col,ms=3.5,label=g)
    axB.fill_between([1,2,3],s["mean"]-s["sem"],s["mean"]+s["sem"],color=col,alpha=.18)
axB.set_xticks([1,2,3]);axB.set_xticklabels(["T1","T2","T3"]);axB.set_xlabel("evaluation")
axB.set_ylabel("severity (%)"); axB.legend(fontsize=5.6); F.panel_label(axB,"B")
axB.set_title("disease progress curves",fontsize=6.6,pad=3)

# C per-unit spaghetti
axC=fig.add_subplot(gs[1,1])
for uid,gdf in sut.groupby("biological_unit_id"):
    gdf=gdf.sort_values("t"); col=F.GRP3[gdf.group3.iloc[0]]
    axC.plot(gdf.t,gdf.severity,"-",color=col,alpha=.13,lw=0.5)
for g,col in F.GRP3.items():
    s=sut[sut.group3==g].groupby("t").severity.mean().reindex([1,2,3])
    axC.plot([1,2,3],s,"-",color=col,lw=1.8)
axC.set_xticks([1,2,3]);axC.set_xticklabels(["T1","T2","T3"]);axC.set_xlabel("evaluation");axC.set_ylabel("severity (%)")
axC.set_title("per-unit progress (n=257)",fontsize=6.6,pad=3); F.panel_label(axC,"C")

# D AUDPC raincloud (half-violin + strip + mean)
axD=fig.add_subplot(gs[1,2])
rng=np.random.default_rng(0)
for i,(g,col) in enumerate(F.GRP3.items()):
    v=aud[aud.group3==g].AUDPC.dropna().values
    if len(v)<2: continue
    vp=axD.violinplot([v],positions=[i],widths=0.8,showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor(col); b.set_alpha(.35); b.set_edgecolor(col)
        m=np.mean(b.get_paths()[0].vertices[:,0]); b.get_paths()[0].vertices[:,0]=np.clip(b.get_paths()[0].vertices[:,0],m,np.inf)
    axD.scatter(np.full(len(v),i)-0.12+rng.normal(0,.03,len(v)),v,s=3,color=col,alpha=.5,linewidths=0)
    axD.plot([i-0.2,i+0.02],[v.mean()]*2,color="k",lw=1.2)
axD.set_xticks(range(3));axD.set_xticklabels(list(F.GRP3),fontsize=5.6,rotation=15)
axD.set_ylabel("AUDPC (relative)"); axD.set_title("area under progress curve",fontsize=6.4,pad=3); F.panel_label(axD,"D")

# E functional comparison ASC vs CAL (mean curves + shaded difference) + bottom sq diff
axE=fig.add_subplot(gs[2,:2])
tt=np.linspace(1,3,100)
curves={}
for g in ["ASC 160","CAL 96"]:
    s=sut[sut.group3==g].groupby("t").severity.mean().reindex([1,2,3]).values
    curves[g]=np.interp(tt,[1,2,3],s)
    axE.plot(tt,curves[g],color=F.GRP3[g],lw=1.8,label=g)
    a=aud[aud.group3==g].AUDPC.mean()
    axE.fill_between(tt,0,curves[g],color=F.GRP3[g],alpha=.10)
axE.fill_between(tt,curves["ASC 160"],curves["CAL 96"],color="#bbbbbb",alpha=.45)
d=np.trapezoid((curves["ASC 160"]-curves["CAL 96"])**2,tt)**0.5
axE.text(0.98,0.05,f"functional distance = {d:.2f}",transform=axE.transAxes,ha="right",fontsize=6,
         bbox=dict(fc="white",ec="#999",lw=0.5,boxstyle="round,pad=0.3"))
axE.set_xticks([1,2,3]);axE.set_xticklabels(["T1","T2","T3"]);axE.set_xlabel("evaluation");axE.set_ylabel("severity (%)")
axE.legend(fontsize=5.8,loc="upper left"); axE.set_title("functional comparison of progress curves",fontsize=6.4,pad=3)
F.panel_label(axE,"E",x=-0.09)
# F squared difference
axF=fig.add_subplot(gs[2,2])
axF.fill_between(tt,0,(curves["ASC 160"]-curves["CAL 96"])**2,color="#3A9D6E",alpha=.6)
axF.plot(tt,(curves["ASC 160"]-curves["CAL 96"])**2,color="#2f7d58",lw=1)
axF.set_xticks([1,2,3]);axF.set_xticklabels(["T1","T2","T3"]);axF.set_xlabel("evaluation");axF.set_ylabel("squared difference")
F.panel_label(axF,"F",x=-0.3)
fig.savefig(f"{ROOT}/outputs/figures/Figure_7.png"); print("saved Figure_7.png","func dist",round(d,2))

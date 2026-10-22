import sys, json; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figstyle as F
ROOT="/home/claude/proj"
co=np.load(f"{ROOT}/outputs/discrim/coords.npz"); U=co["g_umap"]; y=co["y"]
meta=pd.read_csv(f"{ROOT}/outputs/discrim/coords_meta.csv")
cls=["ASC 160","CAL 96","Control"]
U0=pd.read_csv(f"{ROOT}/outputs/trajectories/unit_level.csv"); UV=np.load(f"{ROOT}/outputs/trajectories/unit_vectors.npy")
U0["group3"]=np.where(U0.treatment=="control","Control",U0.accession)
tm=json.load(open(f"{ROOT}/outputs/trajectories/trajectory_metrics.json"))
meta["group3"]=meta.group3

fig=plt.figure(figsize=(180*F.MM,120*F.MM)); gs=fig.add_gridspec(2,2,hspace=0.32,wspace=0.28,height_ratios=[1.15,1])

# A trajectories in UMAP (3 groups, arrows) + faint cloud
axA=fig.add_subplot(gs[0,:])
axA.scatter(U[:,0],U[:,1],c="#e6e6e6",s=1.2,linewidths=0)
center=U.mean(0)
for k,g in enumerate(cls):
    C=[]
    for t in ["T1","T2","T3"]:
        m=((y==k)&(meta.timepoint==t).values); C.append([U[m,0].mean(),U[m,1].mean()])
    C=np.array(C); col=F.GRP3[g]
    axA.plot(C[:,0],C[:,1],"-",color=col,lw=1.6,zorder=3)
    for i in range(2): axA.annotate("",xy=C[i+1],xytext=C[i],arrowprops=dict(arrowstyle="-|>",color=col,lw=1.4),zorder=4)
    for i,t in enumerate(["T1","T2","T3"]): axA.scatter(*C[i],color=col,s=[20,34,52][i],ec="w",lw=0.6,zorder=5)
    gc=U[y==k].mean(0); d=gc-center; d=d/(np.linalg.norm(d)+1e-9)
    lp=gc+d*np.array([np.ptp(U[:,0]),np.ptp(U[:,1])])*0.16
    axA.text(lp[0],lp[1],g,color=col,fontsize=6.6,ha="center",va="center",fontweight="bold",zorder=6,
             bbox=dict(fc="white",ec=col,lw=0.6,alpha=0.85,boxstyle="round,pad=0.15"))
F.noticks(axA); axA.set_xlabel("UMAP-1");axA.set_ylabel("UMAP-2")
axA.text(0.01,0.02,"T1 → T2 → T3",transform=axA.transAxes,fontsize=6,color="#555"); F.panel_label(axA,"A")

# B dispersion evolution (violin of distance-to-group-centroid per timepoint)
axB=fig.add_subplot(gs[1,0])
def dists(g,tp):
    m=(U0.group3==g)&(U0.timepoint==tp)
    if m.sum()<2: return np.array([])
    V=UV[m.values]; return np.linalg.norm(V-V.mean(0),axis=1)
pos=0; xt=[]; xl=[]
for g in ["ASC 160","CAL 96"]:
    for j,tp in enumerate(["T1","T2","T3"]):
        v=dists(g,tp)
        if len(v)<2: pos+=1; continue
        vp=axB.violinplot([v],positions=[pos],widths=0.8,showextrema=False)
        for b in vp['bodies']: b.set_facecolor(F.GRP3[g]); b.set_alpha(.35+0.12*j); b.set_edgecolor(F.GRP3[g])
        axB.plot([pos-0.3,pos+0.3],[v.mean()]*2,color="k",lw=1)
        xt.append(pos); xl.append(f"{tp}"); pos+=1
    pos+=0.6
axB.set_xticks(xt);axB.set_xticklabels(xl,fontsize=5.6)
axB.set_ylabel("distance to centroid")
axB.text(0.02,0.96,"ASC 160",transform=axB.transAxes,fontsize=5.6,color=F.GRP3["ASC 160"],va="top")
axB.text(0.55,0.96,"CAL 96",transform=axB.transAxes,fontsize=5.6,color=F.GRP3["CAL 96"],va="top")
axB.set_title("within-group dispersion (PERMDISP P=0.001)",fontsize=5.8,pad=3); F.panel_label(axB,"B")

# C per-unit displacement raincloud (3 groups)
axC=fig.add_subplot(gs[1,1]); rng=np.random.default_rng(0)
def paired(g):
    out=[]
    for uid,gg in U0[U0.group3==g].groupby("biological_unit_id"):
        d={r.timepoint:UV[r.Index] for r in gg.itertuples()}
        if "T1" in d and "T3" in d: out.append(np.linalg.norm(d["T3"]-d["T1"]))
    return np.array(out)
for i,g in enumerate(cls):
    v=paired(g)
    if len(v)<2: continue
    vp=axC.violinplot([v],positions=[i],widths=0.8,showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor(F.GRP3[g]); b.set_alpha(.35); b.set_edgecolor(F.GRP3[g])
        m=np.mean(b.get_paths()[0].vertices[:,0]); b.get_paths()[0].vertices[:,0]=np.clip(b.get_paths()[0].vertices[:,0],m,np.inf)
    axC.scatter(np.full(len(v),i)-0.12+rng.normal(0,.03,len(v)),v,s=3,color=F.GRP3[g],alpha=.5,linewidths=0)
    axC.plot([i-0.2,i+0.02],[v.mean()]*2,color="k",lw=1.1)
axC.set_xticks(range(3));axC.set_xticklabels(cls,fontsize=5.6,rotation=12)
axC.set_ylabel("latent displ. T1→T3")
p=tm["per_unit_displacement_T1T3"]["ASC_vs_CAL_perm_p"]
axC.text(0.5,0.95,f"ASC vs CAL perm P={p:.0e}",transform=axC.transAxes,ha="center",fontsize=5.4,va="top")
axC.set_title("per-unit displacement",fontsize=6,pad=3); F.panel_label(axC,"C")
fig.savefig(f"{ROOT}/outputs/figures/Figure_4.png"); print("saved Figure_4.png")

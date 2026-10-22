import sys, json; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import seaborn as sns
from sklearn.manifold import MDS
import figstyle as F
ROOT="/home/claude/proj"
cz=np.load(f"{ROOT}/outputs/integration2/corr.npz",allow_pickle=True); R=cz["R"]; labels=list(cz["labels"])
co=np.load(f"{ROOT}/outputs/discrim/coords.npz"); U=co["g_umap"]; y=co["y"]; cls=["ASC 160","CAL 96","Control"]
ig=json.load(open(f"{ROOT}/outputs/pls/integration_metrics.json")); tm=json.load(open(f"{ROOT}/outputs/trajectories/trajectory_metrics.json"))
rv=ig["RV"]
fr=json.load(open(f"{ROOT}/frozen/frozen_results.json")); dc=json.load(open(f"{ROOT}/outputs/discrim/discrim_compare.json"))
sg=pd.read_csv(f"{ROOT}/outputs/spatial/spatial_by_group.csv"); audg=pd.DataFrame(fr["audpc_by_group"])

fig=plt.figure(figsize=(180*F.MM,140*F.MM))
gs=fig.add_gridspec(2,2,hspace=0.30,wspace=0.24,height_ratios=[1,0.92])

# ---------- A: correlation network (collinear classical metrics collapsed to one node) ----------
axA=fig.add_subplot(gs[0,0])
sel=[0,3,4,5,6,7]  # classical severity, latent displ, Moran, heterogeneity, lesion frac, lesion score
Ls=["classical severity","latent displ.","Moran's I","heterogeneity","lesion frac.","lesion score"]
bl=[0,1,2,2,2,2]; bcol={0:"#B2182B",1:"#3B6FB6",2:"#3A9D6E"}
Rs=R[np.ix_(sel,sel)]; Dd=1-np.abs(Rs)
pos=MDS(n_components=2,dissimilarity="precomputed",random_state=3,normalized_stress='auto').fit_transform(Dd)
for i in range(len(sel)):
    for j in range(i+1,len(sel)):
        if abs(Rs[i,j])>0.3:
            axA.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]],"-",
                     color=("#c25b6b" if Rs[i,j]>0 else "#5b83c2"),lw=0.5+3.2*abs(Rs[i,j])**2,alpha=0.55,zorder=1)
deg=(np.abs(Rs)>0.3).sum(1)
for i,l in enumerate(Ls):
    axA.scatter(pos[i,0],pos[i,1],s=90+45*deg[i],color=bcol[bl[i]],alpha=.92,ec="white",lw=1.2,zorder=3)
    dy=0.11 if i not in(0,) else -0.13
    axA.annotate(l,(pos[i,0],pos[i,1]),xytext=(0,-13 if bl[i]!=1 else 12),textcoords="offset points",ha="center",fontsize=5.4,zorder=4)
axA.axis("off"); axA.margins(0.18)
from matplotlib.lines import Line2D
axA.legend(handles=[Line2D([0],[0],marker="o",ls="",mfc=bcol[k],mec="w",ms=6,
    label=["classical severity","latent (global)","spatial (patch)"][k]) for k in range(3)],
    fontsize=5,loc="upper left",ncol=1,handletextpad=0.1,borderpad=0.2)
axA.set_title("metric similarity network  (edge = |ρ|)",fontsize=6.6,pad=2); F.panel_label(axA,"A",x=-0.02,y=1.05)

# ---------- B: phenotype landscape (density contours per group on UMAP) ----------
axB=fig.add_subplot(gs[0,1])
for k,c in enumerate(cls):
    m=y==k
    sns.kdeplot(x=U[m,0],y=U[m,1],ax=axB,fill=True,alpha=0.28,levels=4,color=F.GRP3[c],thresh=0.15,bw_adjust=1.1)
    axB.scatter(U[m,0],U[m,1],s=1.2,color=F.GRP3[c],alpha=.35,linewidths=0)
    cx,cy=U[m,0].mean(),U[m,1].mean(); axB.text(cx,cy,c,fontsize=6,fontweight="bold",color=F.GRP3[c],ha="center",
        path_effects=[])
axB.set_xlabel("UMAP-1");axB.set_ylabel("UMAP-2");axB.set_title("phenotype landscape",fontsize=6.6,pad=2)
F.noticks(axB); F.panel_label(axB,"B",x=-0.04)

# ---------- C: parallel-coordinate signatures per group (4 conceptual axes) ----------
axC=fig.add_subplot(gs[1,0])
def audpc(g): return float(audg[(audg.accession==g)&(audg.treatment=="inoculated")].AUDPC_mean.iloc[0]) if g!="Control" else float(audg[audg.treatment=="control"].AUDPC_mean.mean())
def net(g): return tm["centroid_trajectory"].get(f"{g}|inoculated",{}).get("net_T1T3",0) if g!="Control" else np.mean([tm["centroid_trajectory"][f"{a}|control"]["net_T1T3"] for a in ["ASC 160","CAL 96"]])
def moran(g):
    tr="control" if g=="Control" else "inoculated"; d=sg[(sg.treatment==tr)&(sg.timepoint=="T3")]; d=d[d.accession==g] if g!="Control" else d
    return float(d.morans_I.mean())
def disp(g):
    key=f"{g}|inoculated" if g!="Control" else "ASC 160|control"
    return tm["dispersion"][key]["T3"]-tm["dispersion"][key]["T1"]
axes=["magnitude\n(AUDPC)","state\n(latent displ.)","organization\n(Moran I)","progression\n(Δdispersion)"]
M=np.array([[audpc(g),net(g),moran(g),disp(g)] for g in cls]); Mn=(M-M.min(0))/(M.max(0)-M.min(0)+1e-9)
xs=np.arange(4)
for row,g in zip(Mn,cls):
    axC.plot(xs,row,"-o",color=F.GRP3[g],lw=1.6,ms=4,label=g,alpha=.9)
    axC.fill_between(xs,row,alpha=.06,color=F.GRP3[g])
for x in xs: axC.axvline(x,color="#eee",lw=6,zorder=0)
axC.set_xticks(xs);axC.set_xticklabels(axes,fontsize=5.4);axC.set_yticks([0,0.5,1]);axC.set_yticklabels(["low","","high"],fontsize=5.5)
axC.set_ylim(-0.05,1.1);axC.legend(fontsize=5.6,loc="upper center",ncol=3,columnspacing=0.8,handletextpad=0.2)
axC.set_title("phenotypic signatures",fontsize=6.6,pad=2); F.panel_label(axC,"C",x=-0.08)

# ---------- D: four-lens synthesis ----------
axD=fig.add_subplot(gs[1,1])
bars=[("classical severity\nrecovered (R²cv)",ig["pls_cv_r2"],"#B2182B"),
      ("latent complementary\nto severity",ig["latent_frac_unexplained_by_severity"],"#6a51a3"),
      ("global ∩ severity (RV)",rv["global_vs_severity"],"#3B6FB6"),
      ("global ∩ patch (RV)",rv["global_vs_spatial"],"#756bb1"),
      ("patch ∩ severity (RV)",rv["spatial_vs_severity"],"#3A9D6E")]
labs=[b[0] for b in bars]; vals=[b[1] for b in bars]; colr=[b[2] for b in bars]
yb=np.arange(len(bars))[::-1]
axD.barh(yb,vals,color=colr,alpha=.9,height=0.62,edgecolor="white",lw=0.6)
for yv,v in zip(yb,vals): axD.text(v+0.015,yv,f"{v:.2f}",va="center",fontsize=6,fontweight="bold",color="#222")
axD.set_yticks(yb);axD.set_yticklabels(labs,fontsize=5.4);axD.set_xlim(0,1.05)
axD.set_xlabel("coefficient (0–1)",fontsize=6)
axD.axvline(0.5,color="#ddd",lw=0.6,zorder=0)
axD.set_title("shared vs complementary information",fontsize=6.4,pad=3); F.panel_label(axD,"D",x=-0.42)
for s in ("top","right"): axD.spines[s].set_visible(False)
fig.savefig(f"{ROOT}/outputs/figures/Figure_8.png"); print("saved Figure_8.png")

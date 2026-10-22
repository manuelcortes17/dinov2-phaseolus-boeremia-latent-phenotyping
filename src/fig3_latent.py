import sys; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figstyle as F
from mpl_toolkits.mplot3d import Axes3D  # noqa
ROOT="/home/claude/proj"
co=np.load(f"{ROOT}/outputs/discrim/coords.npz"); y=co["y"]
meta=pd.read_csv(f"{ROOT}/outputs/discrim/coords_meta.csv")
sev=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")[["image_id","severity"]]
meta=meta.merge(sev,on="image_id",how="left")
pex=pd.read_csv(f"{ROOT}/outputs/latent/pca_explained.csv")
cls=["ASC 160","CAL 96","Control"]; gcol=[F.GRP3[c] for c in cls]
U=co["g_umap"]; P2=co["g_pca2"]; P3=co["g_pca3"]

fig=plt.figure(figsize=(184*F.MM,112*F.MM)); gs=fig.add_gridspec(2,3,hspace=0.34,wspace=0.52)

# A PCA by group (hulls)
axA=fig.add_subplot(gs[0,0])
for k,c in enumerate(cls):
    m=y==k; axA.scatter(P2[m,0],P2[m,1],s=2.2,color=gcol[k],alpha=.32,linewidths=0,label=c); F.hull(P2[m,0],P2[m,1],axA,gcol[k],alpha=.025,lw=1.0)
axA.set_xlabel("PC1 (%.1f%%)"%(pex.explained_var_ratio[0]*100));axA.set_ylabel("PC2 (%.1f%%)"%(pex.explained_var_ratio[1]*100))
axA.legend(markerscale=1.5,handletextpad=0.1,fontsize=4.4,loc="lower right",labelspacing=0.25,borderpad=0.25,framealpha=0.85); F.panel_label(axA,"A")

# B UMAP by group
axB=fig.add_subplot(gs[0,1])
for k,c in enumerate(cls):
    m=y==k; axB.scatter(U[m,0],U[m,1],s=2.5,color=gcol[k],alpha=.5,linewidths=0,label=c)
axB.set_xlabel("UMAP-1");axB.set_ylabel("UMAP-2");axB.legend(markerscale=2.2,fontsize=5.6,loc="best"); F.panel_label(axB,"B")

# C UMAP by timepoint
axC=fig.add_subplot(gs[0,2])
for t in ["T1","T2","T3"]:
    m=(meta.timepoint==t).values; axC.scatter(U[m,0],U[m,1],s=2.5,color=F.TP[t],alpha=.55,linewidths=0,label=t)
axC.set_xlabel("UMAP-1");axC.set_ylabel("UMAP-2");axC.legend(markerscale=2.2,fontsize=5.6,loc="best"); F.panel_label(axC,"C")

# D UMAP by severity (continuous)
axD=fig.add_subplot(gs[1,0])
o=np.argsort(np.nan_to_num(meta.severity.values))
p=axD.scatter(U[o,0],U[o,1],c=np.log1p(np.nan_to_num(meta.severity.values)[o]),cmap=F.SEVMAP,s=2.6,linewidths=0)
axD.set_xlabel("UMAP-1");axD.set_ylabel("UMAP-2")
cb=fig.colorbar(p,ax=axD,fraction=0.045,pad=0.03);cb.set_label("log(1+sev %)",fontsize=5.5,labelpad=1);cb.ax.tick_params(labelsize=5.5)
F.panel_label(axD,"D")

# E 3D cube point cloud (PCA-3) by group
axE=fig.add_subplot(gs[1,1],projection="3d")
for k,c in enumerate(cls):
    m=y==k; axE.scatter(P3[m,0],P3[m,1],P3[m,2],s=2,color=gcol[k],alpha=.35,linewidths=0)
axE.set_xlabel("PC1",fontsize=5.5,labelpad=-8);axE.set_ylabel("PC2",fontsize=5.5,labelpad=-8);axE.set_zlabel("PC3",fontsize=5.5,labelpad=-8)
axE.tick_params(labelsize=4,pad=-2)
axE.xaxis.pane.set_alpha(0.04);axE.yaxis.pane.set_alpha(0.04);axE.zaxis.pane.set_alpha(0.04)
axE.view_init(elev=18,azim=-60); axE.set_title("latent space (PCA-3)",fontsize=6.2,pad=-2)
fig.text(0.40,0.47,"E",fontsize=10,fontweight="bold")

# F PCA variance
axF=fig.add_subplot(gs[1,2])
axF.bar(pex.pc[:12],pex.explained_var_ratio[:12]*100,color="#c2c6cc",width=0.72,label="per PC")
axF.plot(pex.pc[:12],pex["cum"][:12]*100,"-o",color="#B2182B",ms=2.6,lw=1.2,label="cumulative")
axF.set_xlabel("principal component");axF.set_ylabel("variance explained (%)",labelpad=1)
axF.legend(fontsize=5.2,loc="center right",handlelength=1.2,borderpad=0.2); F.panel_label(axF,"F",x=-0.24)
fig.savefig(f"{ROOT}/outputs/figures/Figure_3.png"); print("saved Figure_3.png")

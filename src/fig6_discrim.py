import sys, json; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figstyle as F
ROOT="/home/claude/proj"; D=f"{ROOT}/outputs/discrim"
cg=pd.read_csv(f"{D}/curves_global.csv"); cp=pd.read_csv(f"{D}/curves_patch.csv")
cmg=np.load(f"{D}/confusion_global.npy"); cmp=np.load(f"{D}/confusion_patch.npy")
dc=json.load(open(f"{D}/discrim_compare.json")); cls=dc["classes"]
co=np.load(f"{D}/coords.npz"); y=co["y"]; gcol=[F.GRP3[c] for c in cls]

fig=plt.figure(figsize=(192*F.MM,110*F.MM)); gs=fig.add_gridspec(2,4,hspace=0.55,wspace=0.52)

# A loss vs epoch
axA=fig.add_subplot(gs[0,0])
axA.plot(cg.epoch,cg.train_loss,color="#3B6FB6",lw=1,label="global train")
axA.plot(cg.epoch,cg.val_loss,color="#3B6FB6",lw=1,ls="--",label="global val")
axA.plot(cp.epoch,cp.train_loss,color="#E08214",lw=1,label="patch train")
axA.plot(cp.epoch,cp.val_loss,color="#E08214",lw=1,ls="--",label="patch val")
axA.set_xlabel("epoch"); axA.set_ylabel("cross-entropy loss"); axA.legend(fontsize=4.8,ncol=1,handlelength=1.3)
F.panel_label(axA,"A")
# B accuracy & recall vs epoch
axB=fig.add_subplot(gs[0,1])
axB.plot(cg.epoch,cg.val_acc,color="#3B6FB6",lw=1.1,label="global acc")
axB.plot(cg.epoch,cg.val_recall,color="#3B6FB6",lw=1.1,ls=":",label="global recall")
axB.plot(cp.epoch,cp.val_acc,color="#E08214",lw=1.1,label="patch acc")
axB.plot(cp.epoch,cp.val_recall,color="#E08214",lw=1.1,ls=":",label="patch recall")
axB.set_xlabel("epoch"); axB.set_ylabel("validation score"); axB.set_ylim(0,1); axB.legend(fontsize=4.8,handlelength=1.3)
F.panel_label(axB,"B")
# C,D confusion matrices
def confmat(ax,cm,title,pl):
    n=cm/cm.sum(1,keepdims=True)
    im=ax.imshow(n,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(3));ax.set_yticks(range(3))
    ax.set_xticklabels(cls,fontsize=5,rotation=30,ha="right");ax.set_yticklabels(cls,fontsize=5)
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{cm[i,j]}",ha="center",va="center",fontsize=6,color="white" if n[i,j]>0.5 else "#333")
    ax.set_ylabel("true");ax.set_xlabel("predicted");ax.set_title(title,fontsize=6.6,pad=3); F.panel_label(ax,pl,x=-0.20,y=1.14)
confmat(fig.add_subplot(gs[0,2]),cmg,f"global (acc {dc['results'][0]['test_accuracy']:.2f})" if 'results' in dc else "global","C")
confmat(fig.add_subplot(gs[0,3]),cmp,"patch","D")

# E discrimination heatmap: representation x embedding CV accuracy
axE=fig.add_subplot(gs[1,0])
reps=["PCA","UMAP","PLS"]; embs=["global","patch"]
Mx=np.array([[dc["discrimination"][f"{e}|{r}"]["cv_accuracy"] for r in reps] for e in embs])
im=axE.imshow(Mx,cmap="YlGnBu",vmin=0.7,vmax=1)
axE.set_xticks(range(3));axE.set_xticklabels(reps,fontsize=6);axE.set_yticks(range(2));axE.set_yticklabels(embs,fontsize=6)
for i in range(2):
    for j in range(3): axE.text(j,i,f"{Mx[i,j]:.2f}",ha="center",va="center",fontsize=6.5,color="white" if Mx[i,j]>0.9 else "#333")
axE.set_title("grouped-CV accuracy",fontsize=6.6,pad=3); F.panel_label(axE,"E",x=-0.22,y=1.14)
# F global clustering hulls (PCA2)
axF=fig.add_subplot(gs[1,1])
XY=co["g_pca2"]
for k,c in enumerate(cls):
    m=y==k; axF.scatter(XY[m,0],XY[m,1],s=3,color=gcol[k],alpha=.32,linewidths=0,label=c)
    F.hull(XY[m,0],XY[m,1],axF,gcol[k],alpha=.04,lw=1.0)
axF.set_xlabel("PC1");axF.set_ylabel("PC2");axF.set_title(f"global  (GMM ARI={dc['clustering']['global']['gmm_ARI']:.2f})",fontsize=6.2,pad=3)
axF.legend(fontsize=5,markerscale=1.6,handletextpad=0.1,loc="best"); F.panel_label(axF,"F")
# G patch clustering hulls
axG=fig.add_subplot(gs[1,2])
XY=co["p_pca2"]
for k,c in enumerate(cls):
    m=y==k; axG.scatter(XY[m,0],XY[m,1],s=3,color=gcol[k],alpha=.32,linewidths=0)
    F.hull(XY[m,0],XY[m,1],axG,gcol[k],alpha=.04,lw=1.0)
axG.set_xlabel("PC1");axG.set_ylabel("PC2");axG.set_title(f"patch  (GMM ARI={dc['clustering']['patch']['gmm_ARI']:.2f})",fontsize=6.2,pad=3)
F.panel_label(axG,"G")
# H silhouette comparison (elegant lollipop)
axH=fig.add_subplot(gs[1,3])
labs=[f"{e[0]}·{r}" for e in embs for r in reps]
sil=[dc["discrimination"][f"{e}|{r}"]["silhouette"] for e in embs for r in reps]
cols=["#3B6FB6"]*3+["#E08214"]*3; yy=np.arange(len(labs))[::-1]
axH.hlines(yy,0,sil,color=cols,lw=1.2); axH.plot(sil,yy,"o",color="none",mec="k",ms=4)
for v,c in zip(sil,cols): pass
axH.scatter(sil,yy,c=cols,s=22,zorder=3)
axH.set_yticks(yy);axH.set_yticklabels(labs,fontsize=5);axH.set_xlabel("silhouette")
axH.set_title("separation",fontsize=6.6,pad=3); F.panel_label(axH,"H",x=-0.30,y=1.14)
fig.savefig(f"{ROOT}/outputs/figures/Figure_6.png"); print("saved Figure_6.png")

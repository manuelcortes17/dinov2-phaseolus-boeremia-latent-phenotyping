import sys; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import figstyle as F
ROOT="/home/claude/proj"; GS=16; SIZE=224
order=pd.read_csv(f"{ROOT}/outputs/embeddings/order.csv")
sev=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")[["image_id","severity"]]
spat=pd.read_csv(f"{ROOT}/outputs/spatial/spatial_per_image.csv")
order=order.merge(sev,on="image_id",how="left").merge(spat[["image_id","morans_I"]],on="image_id",how="left")
P=np.load(f"{ROOT}/outputs/embeddings/patch_f16.npy",mmap_mode="r")
lesmaps=np.load(f"{ROOT}/outputs/spatial/lesion_score_maps_f16.npy")
id2row={i:k for k,i in enumerate(order.image_id)}
def nr(A): return A/(np.linalg.norm(A,axis=-1,keepdims=True)+1e-8)
def leaf_cells(path):
    im=np.asarray(Image.open(path).convert("RGB").resize((SIZE,SIZE),Image.BICUBIC)).astype(float)/255
    mx=im.max(-1);mn=im.min(-1);sat=(mx-mn)/(mx+1e-6)
    ch=(sat>0.18)&(mx<0.95)&(im[...,1]>=im[...,2]*0.8)
    return im,(ch.reshape(GS,SIZE//GS,GS,SIZE//GS).mean(axis=(1,3))>0.5)
def LM(iid): return lesmaps[id2row[iid]].astype(float).reshape(GS,GS)

samp=order[order.use_unsup==True].sample(180,random_state=0); pool=[]
for _,r in samp.iterrows():
    _,m=leaf_cells(r.img_path); pool.append(nr(np.asarray(P[id2row[r.image_id]],float))[m.reshape(-1)])
pcp=PCA(3,random_state=0).fit(np.vstack(pool)); scv=pcp.transform(np.vstack(pool))
lo,hi=np.percentile(scv,2,0),np.percentile(scv,98,0)
def rgbmap(iid,mask):
    s=np.clip((pcp.transform(nr(np.asarray(P[id2row[iid]],float)))-lo)/(hi-lo+1e-8),0,1).reshape(GS,GS,3)
    out=np.ones((GS,GS,3))*0.93; out[mask]=s[mask]; return out
def overlay_map(ax,im,mask,grid,cmap,vmin,vmax):
    ax.imshow(im*0.55); ax.imshow(np.ma.masked_invalid(zoom(np.where(mask,grid,np.nan),SIZE/GS,order=1)),cmap=cmap,alpha=.85,vmin=vmin,vmax=vmax); F.noticks(ax)

exs=order[(order.accession=="ASC 160")&(order.treatment=="inoculated")&(order.timepoint=="T3")].dropna(subset=["severity"]).sort_values("severity")
ex=exs.iloc[len(exs)//2]; imE,mE=leaf_cells(ex.img_path)
cand=order[(order.treatment=="inoculated")&order.severity.notna()&order.morans_I.notna()&(order.severity.between(8,25))].sort_values("morans_I")
low=cand.iloc[3]; high=cand.iloc[-4]

fig=plt.figure(figsize=(190*F.MM,116*F.MM))
gs=fig.add_gridspec(2,4,hspace=0.30,wspace=0.32,height_ratios=[1,1.05])
# row1
axA=fig.add_subplot(gs[0,0]); axA.imshow(imE)
for k in range(0,SIZE+1,SIZE//GS): axA.axhline(k,color="w",lw=0.25);axA.axvline(k,color="w",lw=0.25)
F.noticks(axA);axA.set_title("leaf + 16×16 patches",fontsize=6.4,pad=2);F.panel_label(axA,"A")
axB=fig.add_subplot(gs[0,1]); axB.imshow(zoom(rgbmap(ex.image_id,mE),(SIZE/GS,SIZE/GS,1),order=0));F.noticks(axB);axB.set_title("local phenotype (patch PCA)",fontsize=5.6,pad=2);F.panel_label(axB,"B")
axC=fig.add_subplot(gs[0,2]); overlay_map(axC,imE,mE,LM(ex.image_id),F.SEVMAP,.2,.7);axC.set_title("lesion-score map",fontsize=6.4,pad=2);F.panel_label(axC,"C")
axD=fig.add_subplot(gs[0,3]); pv=nr(np.asarray(P[id2row[ex.image_id]],float));c=nr(pv[mE.reshape(-1)].mean(0,keepdims=True))[0]
overlay_map(axD,imE,mE,(1-pv@c).reshape(GS,GS),"magma",None,None);axD.set_title("local heterogeneity",fontsize=6.4,pad=2);F.panel_label(axD,"D")
# row2: two same-severity different-organization + hexbin
imL,mL=leaf_cells(low.img_path); axE=fig.add_subplot(gs[1,0]); overlay_map(axE,imL,mL,LM(low.image_id),F.SEVMAP,.2,.7)
axE.set_title(f"fragmented\nsev={low.severity:.0f}%  I={low.morans_I:.2f}",fontsize=5.8,pad=2);F.panel_label(axE,"E")
imH,mH=leaf_cells(high.img_path); axF=fig.add_subplot(gs[1,1]); overlay_map(axF,imH,mH,LM(high.image_id),F.SEVMAP,.2,.7)
axF.set_title(f"aggregated\nsev={high.severity:.0f}%  I={high.morans_I:.2f}",fontsize=5.8,pad=2);F.panel_label(axF,"F")
axG=fig.add_subplot(gs[1,2:]); d=order[(order.treatment=="inoculated")&order.severity.notna()&order.morans_I.notna()]
hb=axG.hexbin(d.severity,d.morans_I,gridsize=22,cmap="BuPu",mincnt=1,linewidths=0)
axG.scatter([low.severity,high.severity],[low.morans_I,high.morans_I],s=30,facecolor="none",edgecolor="k",lw=1.2,zorder=3)
axG.annotate("E",(low.severity,low.morans_I),fontsize=6,fontweight="bold",xytext=(3,3),textcoords="offset points")
axG.annotate("F",(high.severity,high.morans_I),fontsize=6,fontweight="bold",xytext=(3,3),textcoords="offset points")
axG.axhline(0,color="#bbb",lw=0.5);axG.set_xlabel("classical severity (%)");axG.set_ylabel("Moran's I",labelpad=1)
cb=fig.colorbar(hb,ax=axG,fraction=0.045,pad=0.02);cb.set_label("leaves",fontsize=5.5);cb.ax.tick_params(labelsize=5)
axG.set_title("severity and spatial organization are decoupled",fontsize=6.0,pad=6);F.panel_label(axG,"G",x=-0.09,y=1.14)
fig.savefig(f"{ROOT}/outputs/figures/Figure_5.png"); print("saved Figure_5.png")

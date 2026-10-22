import sys; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd
import dino_utils as D, figstyle as F
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import cm
from scipy.ndimage import zoom, gaussian_filter
ROOT="/home/claude/proj"; GS=16; SIZE=224
order=pd.read_csv(f"{ROOT}/outputs/embeddings/order.csv")
sev=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")[["image_id","severity"]]
lat=pd.read_csv(f"{ROOT}/outputs/discrim/coords_meta.csv"); co=np.load(f"{ROOT}/outputs/discrim/coords.npz")
lat["severity"]=lat.image_id.map(dict(zip(sev.image_id,sev.severity)))
lesmaps=np.load(f"{ROOT}/outputs/spatial/lesion_score_maps_f16.npy"); id2row={i:k for k,i in enumerate(order.image_id)}
ex=order.merge(sev,on="image_id").query("accession=='ASC 160' and treatment=='inoculated' and timepoint=='T3'").dropna(subset=["severity"]).sort_values("severity").iloc[-40]
im,dsp=D.load_rgb(ex.img_path,SIZE)
les=lesmaps[id2row[ex.image_id]].astype(float).reshape(GS,GS)
mx=dsp.max(-1);sat=(mx-dsp.min(-1))/(mx+1e-6);msk=(sat>0.18)&(mx<0.95)

plt.rcParams.update({"font.size":7})
fig=plt.figure(figsize=(180*F.MM,96*F.MM)); ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,100);ax.set_ylim(0,56);ax.axis("off")
def box(x,y,w,h,txt,fc,ec,fs=6.6,bold=True,tc="#111"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.4",fc=fc,ec=ec,lw=1.0))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",color=tc)
def arrow(x1,y1,x2,y2,c="#5b6570",lw=1.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=8,lw=lw,color=c))
def inset(x,y,w,h,img,title="",cmap=None):
    a=fig.add_axes([x/100,y/56,w/100,h/56]);a.imshow(img,cmap=cmap);a.set_xticks([]);a.set_yticks([])
    for s in a.spines.values(): s.set_color("#5b6570");s.set_linewidth(0.7)
    if title: a.set_title(title,fontsize=5.6,pad=1.5)
    return a

# ---- input + encoder ----
inset(2.5,41,12,12,dsp,"leaf image")
arrow(15,47,20,47)
box(20,42,15,10,"frozen\nDINOv2 ViT-B/14  ❄",fc="#e8eef7",ec="#3B6FB6",fs=6.6)
ax.text(27.5,40.3,"768-d · 12 blocks · patch 14",ha="center",fontsize=4.8,color="#555")
arrow(35,49,41,51.5); arrow(35,47,41,47); arrow(35,45,41,42)
# three representations
au=fig.add_axes([41/100,49.5/56,11/100,7/56]); s=lat.dropna(subset=["severity"])
au.scatter(co["g_umap"][:,0],co["g_umap"][:,1],c=np.log1p(np.nan_to_num(lat.severity.values)),cmap=F.SEVMAP,s=0.6,linewidths=0)
au.set_xticks([]);au.set_yticks([]);[sp.set_color("#5b6570") for sp in au.spines.values()];au.set_title("global (CLS)",fontsize=5.4,pad=1)
inset(41,40,11,7,np.where(msk,zoom(les,SIZE/GS,order=0),np.nan),"patch (spatial)",cmap=F.SEVMAP)
at=zoom(gaussian_filter(les,1.2),SIZE/GS,order=3)
inset(41,31,11,7,np.clip(0.6*cm.get_cmap("jet")((at-at.min())/(at.max()-at.min()))[...,:3]+0.4*dsp,0,1),"attention")

# ---- three objective lanes ----
lanes=[(58,"#eef4fb","#3B6FB6","OBJECTIVE 1\nLatent phenotypic space\n& discrimination",
        "PCA · UMAP · PLS\nglobal vs patch\nclustering, classifier"),
       (58,"#fef5ec","#E08214","OBJECTIVE 2\nTemporal & spatial structure",
        "latent trajectories\nspatial organization\n(Moran's I, heterogeneity)"),
       (58,"#eef7f1","#3A9D6E","OBJECTIVE 3\nComparison with classical metrics",
        "palette severity · AUDPC\nprogress curves\ncomplementarity (RV)")]
ys=[42,26.5,11]
for (x,fc,ec,t,sub),yy in zip(lanes,ys):
    arrow(52,yy+5 if yy>30 else yy+4, 57, yy+4, c=ec, lw=1.1)
    box(58,yy,26,8.2,t,fc=fc,ec=ec,fs=6.0)
    ax.text(71,yy-1.8,sub,ha="center",va="top",fontsize=4.9,color="#444")
    arrow(84,yy+4,88.5,29,c="#999")
box(88.5,24.5,10.5,9,"integrated\nmulti-\ndimensional\nphenotype",fc="#efeaf5",ec="#762a83",fs=5.8)

# ---- equations callout ----
ax.text(2.5,24,"Key quantities",fontsize=6,fontweight="bold",color="#333")
eqs=[r"$d_{cos}=1-\dfrac{\mathbf{z}_i\cdot\mathbf{z}_j}{\|\mathbf{z}_i\|\|\mathbf{z}_j\|}$",
     r"$AUDPC=\sum_i \dfrac{(y_i+y_{i+1})}{2}\,\Delta t$",
     r"$I=\dfrac{n}{W}\dfrac{\sum_{i}\sum_{j}w_{ij}z_i z_j}{\sum_i z_i^2}$",
     r"$RV=\dfrac{\mathrm{tr}(\mathbf{X}\mathbf{X}^\top \mathbf{Y}\mathbf{Y}^\top)}{\sqrt{\mathrm{tr}(\mathbf{X}\mathbf{X}^\top)^2\,\mathrm{tr}(\mathbf{Y}\mathbf{Y}^\top)^2}}$"]
for i,e in enumerate(eqs):
    ax.text(2.5,20-i*4.6,e,fontsize=6.0,color="#222")
ax.text(50,2,"GLOBAL = phenotypic state      PATCH = spatial organization      CLASSICAL = conventional magnitude",
        ha="center",fontsize=5.8,color="#333")
fig.savefig(f"{ROOT}/outputs/figures/Figure_1.png"); print("saved Figure_1.png")

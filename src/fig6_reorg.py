import sys; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import figstyle as F
ROOT="/home/claude/proj"; GS=16; SIZE=224
order=pd.read_csv(f"{ROOT}/outputs/embeddings/order.csv")
sev=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")[["image_id","severity"]]
order=order.merge(sev,on="image_id",how="left")
lesmaps=np.load(f"{ROOT}/outputs/spatial/lesion_score_maps_f16.npy")
sg=pd.read_csv(f"{ROOT}/outputs/spatial/spatial_by_group.csv")
id2row={i:k for k,i in enumerate(order.image_id)}
def leaf(path):
    im=np.asarray(Image.open(path).convert("RGB").resize((SIZE,SIZE),Image.BICUBIC)).astype(float)/255
    mx=im.max(-1);mn=im.min(-1);sat=(mx-mn)/(mx+1e-6)
    ch=(sat>0.18)&(mx<0.95)&(im[...,1]>=im[...,2]*0.8)
    return im,(ch.reshape(GS,SIZE//GS,GS,SIZE//GS).mean(axis=(1,3))>0.5)
def rep(acc,tp):
    s=order[(order.accession==acc)&(order.treatment=="inoculated")&(order.timepoint==tp)].dropna(subset=["severity"]).sort_values("severity")
    return s.iloc[int(len(s)*0.7)]

fig=plt.figure(figsize=(180*F.MM,120*F.MM))
gtop=fig.add_gridspec(2,3,left=.10,right=.995,top=.955,bottom=.46,hspace=.10,wspace=.05)
gbot=fig.add_gridspec(1,3,left=.09,right=.985,top=.36,bottom=.09,wspace=.36)
for i,acc in enumerate(["ASC 160","CAL 96"]):
    for j,tp in enumerate(["T1","T2","T3"]):
        r=rep(acc,tp); im,m=leaf(r.img_path); lg=lesmaps[id2row[r.image_id]].astype(float).reshape(GS,GS)
        ax=fig.add_subplot(gtop[i,j]); ax.imshow(im*0.55)
        ax.imshow(np.ma.masked_invalid(zoom(np.where(m,lg,np.nan),SIZE/GS,order=1)),cmap=F.SEVMAP,alpha=.85,vmin=.2,vmax=.7)
        F.noticks(ax)
        if i==0: ax.set_title(tp,fontsize=7,pad=2)
        if j==0: ax.set_ylabel(acc,fontsize=7,color=F.ACC[acc])
fig.text(0.015,0.95,"A",fontsize=10,fontweight="bold")
fig.text(0.015,0.70,"B",fontsize=10,fontweight="bold")
# bottom quantitative
metrics=[("morans_I","Moran's I (aggregation)","C"),("lesion_fraction","lesion fraction","D"),("mean_lesion_score","mean lesion score","E")]
for k,(mt,lab,pl) in enumerate(metrics):
    ax=fig.add_subplot(gbot[0,k])
    for a,col in F.ACC.items():
        s=sg[(sg.accession==a)&(sg.treatment=="inoculated")].set_index("timepoint").reindex(["T1","T2","T3"])
        ax.plot([1,2,3],s[mt],"-o",color=col,ms=3,label=a)
        sc=sg[(sg.accession==a)&(sg.treatment=="control")].set_index("timepoint").reindex(["T1","T2","T3"])
        ax.plot([1,2,3],sc[mt],"--",color=col,alpha=.45,lw=0.8)
    if mt=="morans_I": ax.axhline(0,color="#999",lw=0.5)
    ax.set_xticks([1,2,3]);ax.set_xticklabels(["T1","T2","T3"]);ax.set_ylabel(lab)
    if k==0: ax.legend(fontsize=5.6,loc="upper left")
    F.panel_label(ax,pl,x=-0.22)
fig.text(0.5,0.40,"solid = inoculated, dashed = control",ha="center",fontsize=5.6,color="#666")
fig.savefig(f"{ROOT}/outputs/figures/Figure_6.png"); print("saved Figure_6.png")

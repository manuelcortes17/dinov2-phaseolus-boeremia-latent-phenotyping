import sys; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, torch
import dino_utils as D, figstyle as F
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image
ROOT="/home/claude/proj"; HS=448; GH=HS//14
order=pd.read_csv(f"{ROOT}/outputs/embeddings/order.csv")
sev=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")[["image_id","severity"]]
order=order.merge(sev,on="image_id",how="left")
healthy=np.load(f"{ROOT}/outputs/spatial/healthy_patch_centroid.npy")
model,_=D.build_model(); tfH=D.make_transform(HS); tf2=D.make_transform(224); torch.set_num_threads(2)
def nr(A): return A/(np.linalg.norm(A,axis=-1,keepdims=True)+1e-8)
def disp_of(p,S): return np.asarray(Image.open(p).convert("RGB").resize((S,S),Image.BICUBIC)).astype(np.float32)/255
def leafpx(d):
    mx=d.max(-1);mn=d.min(-1);sat=(mx-mn)/(mx+1e-6); return (sat>0.18)&(mx<0.95)&(d[...,1]>=d[...,2]*0.8)
def cells(px,G): return px.reshape(G,px.shape[0]//G,G,px.shape[1]//G).mean(axis=(1,3))>0.5
@torch.no_grad()
def symptom_hr(p):
    im=Image.open(p).convert("RGB"); res=D.extract(model,tfH(im).unsqueeze(0))
    ls=(1-nr(res["patch"])@healthy).reshape(GH,GH); disp=disp_of(p,HS); return disp,cells(leafpx(disp),GH),ls
@torch.no_grad()
def attn224(p): return D.extract(model,tf2(Image.open(p).convert("RGB").resize((224,224))).unsqueeze(0))["rollout"].reshape(16,16)
def overlay(disp,grid,mask,cmap,alpha,gamma=1.0):
    S=disp.shape[0]; g=zoom(grid,S/grid.shape[0],order=3); mk=zoom(mask.astype(float),S/mask.shape[0],order=1)
    g=g*mk; ins=mk>0.3
    if ins.sum(): g=g-g[ins].min(); g=np.clip(g/(g[ins].max()+1e-8),0,1)**gamma
    heat=cm.get_cmap(cmap)(g)[...,:3]; return np.where(ins[...,None],alpha*heat+(1-alpha)*disp,disp*0.55)

pk=lambda a,t,tp,q: order[(order.accession==a)&(order.treatment==t)&(order.timepoint==tp)].dropna(subset=["severity"]).sort_values("severity").iloc[q]
ROWS=[("control · T1",order[(order.accession=="ASC 160")&(order.treatment=="control")&(order.timepoint=="T1")].iloc[0]),
      ("inoculated · T1",pk("ASC 160","inoculated","T1",5)),
      ("inoculated · T2",pk("ASC 160","inoculated","T2",-30)),
      ("inoculated · T3",pk("ASC 160","inoculated","T3",-20))]
dd=[(lab,symptom_hr(r.img_path),attn224(r.img_path)) for lab,r in ROWS]

fig=plt.figure(figsize=(150*F.MM,178*F.MM))
gs=fig.add_gridspec(4,3,left=.115,right=.99,top=.945,bottom=.015,hspace=0.16,wspace=0.10)
cols=["leaf","symptom localization (DINOv2 patch)","attention (rollout)"]
for j,(lab,(disp,cm_,ls),att) in enumerate(dd):
    a0=fig.add_subplot(gs[j,0]); a0.imshow(disp); F.noticks(a0)
    a0.text(-0.10,0.5,lab,transform=a0.transAxes,fontsize=7.5,rotation=90,va="center",ha="right",fontweight="bold")
    a1=fig.add_subplot(gs[j,1]); a1.imshow(overlay(disp,ls,cm_,F.SEVMAP,.85,gamma=1.3)); F.noticks(a1)
    a2=fig.add_subplot(gs[j,2]); a2.imshow(overlay(disp,att,cm_,"jet",.58)); F.noticks(a2)
    if j==0:
        for a,t in zip([a0,a1,a2],cols): a.set_title(t,fontsize=7.5,pad=4)
fig.text(0.02,0.94,"A",fontsize=11,fontweight="bold")
fig.text(0.02,0.71,"B",fontsize=11,fontweight="bold")
fig.text(0.02,0.48,"C",fontsize=11,fontweight="bold")
fig.savefig(f"{ROOT}/outputs/figures/Figure_2.png"); print("saved Figure_2.png")

# ---- supplementary quant figure S3 (from saved csv) ----
st=pd.read_csv(f"{ROOT}/outputs/discrim/attn_symptom.csv"); rng=np.random.default_rng(1)
figs=plt.figure(figsize=(150*F.MM,62*F.MM)); g2=figs.add_gridspec(1,2,wspace=.36,left=.10,right=.97,top=.86,bottom=.20)
axq=figs.add_subplot(g2[0,0])
for i,c in enumerate(["rho_patch","rho_attn"]):
    v=st[c].dropna().values; axq.scatter(np.full(len(v),i)+rng.normal(0,.05,len(v)),v,s=9,color=["#B2182B","#888"][i],alpha=.6,linewidths=0)
    axq.plot([i-.2,i+.2],[np.nanmean(v)]*2,color="k",lw=1.5)
axq.axhline(0,color="#bbb",lw=0.5);axq.set_xticks([0,1]);axq.set_xticklabels(["patch\nembedding","raw\nattention"],fontsize=7)
axq.set_ylabel("ρ with visual symptom");axq.set_title("what localizes the symptom?",fontsize=7.5);F.panel_label(axq,"A",x=-0.18)
axr=figs.add_subplot(g2[0,1])
for t,cc in zip(["T1","T2","T3"],[F.TP[k] for k in ["T1","T2","T3"]]):
    v=st[st.timepoint==t].rho_patch.dropna();axr.scatter(np.full(len(v),{"T1":1,"T2":2,"T3":3}[t])+rng.normal(0,.05,len(v)),v,s=9,color=cc,alpha=.6,linewidths=0)
axr.plot([1,2,3],st.groupby("timepoint").rho_patch.mean().reindex(["T1","T2","T3"]),"k-o",ms=3,lw=1)
axr.set_xticks([1,2,3]);axr.set_xticklabels(["T1","T2","T3"]);axr.set_ylabel("ρ(patch, symptom)")
axr.set_title("symptom localization by stage",fontsize=7.5);F.panel_label(axr,"B",x=-0.18)
figs.savefig(f"{ROOT}/outputs/supp_figures/Figure_S3.png"); print("saved Figure_S3.png")

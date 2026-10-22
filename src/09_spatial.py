"""09_spatial.py — Patch-level spatial phenotyping from DINOv2 patch embeddings.
Per image (16x16 grid): leaf-patch mask, lesion-score map (cosine distance to the
healthy control-patch centroid), spatial heterogeneity, Moran's I (spatial
autocorrelation), lesion fraction, and lesion aggregation (connected components).
Preserves 2D coordinates. Outputs per-image metrics + lesion-score maps."""
import os, json, numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import label
ROOT="/home/claude/proj"; EMB=f"{ROOT}/outputs/embeddings"; OUT=f"{ROOT}/outputs/spatial"
os.makedirs(OUT, exist_ok=True)
GS=16; SIZE=224; CELL=SIZE//GS

order=pd.read_csv(f"{EMB}/order.csv")
P=np.load(f"{EMB}/patch_f16.npy", mmap_mode="r")     # (N,256,768) f16
N=len(order)

def leaf_patch_mask(path):
    im=np.asarray(Image.open(path).convert("RGB").resize((SIZE,SIZE),Image.BICUBIC)).astype(np.float32)/255.
    r,g,b=im[...,0],im[...,1],im[...,2]; mx=im.max(-1); mn=im.min(-1)
    sat=(mx-mn)/(mx+1e-6); chroma=(sat>0.18)&(mx<0.95)&(g>=b*0.8)     # leaf tissue (green/chlorotic/necrotic)
    frac=chroma.reshape(GS,CELL,GS,CELL).mean(axis=(1,3))             # 16x16 leaf fraction
    return (frac>0.5)

# rook adjacency weight indices for 16x16
def morans_I(vals, mask):
    v=vals.copy().astype(float); m=mask.copy()
    if m.sum()<8: return np.nan
    mu=v[m].mean(); z=np.where(m, v-mu, 0.0); W=0.0; num=0.0
    for i in range(GS):
        for j in range(GS):
            if not m[i,j]: continue
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii,jj=i+di,j+dj
                if 0<=ii<GS and 0<=jj<GS and m[ii,jj]:
                    W+=1; num+=z[i,j]*z[ii,jj]
    den=(z[m]**2).sum()
    return float((m.sum()/W)*(num/den)) if W>0 and den>0 else np.nan

# ---- pass 1: healthy centroid from control leaf-patches ----
def norm_rows(A): return A/ (np.linalg.norm(A,axis=-1,keepdims=True)+1e-8)
ctrl_idx=order.index[(order.treatment=="control")&(order.is_augmented==False)&(order.use_unsup==True)].tolist()
acc=[]
for i in ctrl_idx:
    m=leaf_patch_mask(order.img_path.iloc[i]).reshape(-1)
    pv=norm_rows(np.asarray(P[i],dtype=np.float32))
    acc.append(pv[m])
healthy=np.vstack(acc); healthy_c=norm_rows(healthy.mean(0,keepdims=True))[0]
np.save(f"{OUT}/healthy_patch_centroid.npy", healthy_c.astype("float32"))
# control lesion-score distribution -> threshold
cs=1-healthy@healthy_c   # not used; compute per-control patch below
ctrl_scores=[]
for i in ctrl_idx:
    m=leaf_patch_mask(order.img_path.iloc[i]).reshape(-1)
    pv=norm_rows(np.asarray(P[i],dtype=np.float32))
    ctrl_scores.append(1-pv[m]@healthy_c)
ctrl_scores=np.concatenate(ctrl_scores)
THR=float(np.percentile(ctrl_scores,95))
np.save(f"{OUT}/control_lesionscore_dist.npy", ctrl_scores.astype("float32"))

# ---- pass 2: per-image spatial metrics ----
maps=np.full((N,GS*GS), np.nan, dtype="float32")
rows=[]
use=order.use_unsup==True
for i in range(N):
    if not use.iloc[i]:
        rows.append(dict(image_id=order.image_id.iloc[i], leaf_patches=0)); continue
    m2=leaf_patch_mask(order.img_path.iloc[i]); m=m2.reshape(-1)
    pv=norm_rows(np.asarray(P[i],dtype=np.float32))
    ls=1-pv@healthy_c                       # lesion score per patch (256,)
    maps[i]=np.where(m, ls, np.nan)
    lp=pv[m]
    # heterogeneity: mean pairwise cosine distance among leaf patches
    if lp.shape[0]>=3:
        S=lp@lp.T; iu=np.triu_indices(lp.shape[0],1); het=float(1-S[iu].mean())
    else: het=np.nan
    grid=np.where(m2, ls.reshape(GS,GS), np.nan)
    mI=morans_I(np.nan_to_num(grid), m2)
    lesion_mask=m2 & (ls.reshape(GS,GS)>THR)
    lf=float(lesion_mask.sum()/max(m2.sum(),1))
    # aggregation: (largest CC size)/(n lesion patches); and n components
    if lesion_mask.sum()>0:
        lab,nc=label(lesion_mask)
        sizes=np.bincount(lab.ravel())[1:]
        agg=float(sizes.max()/lesion_mask.sum()); ncomp=int(nc)
    else: agg=np.nan; ncomp=0
    rows.append(dict(image_id=order.image_id.iloc[i], accession=order.accession.iloc[i],
        treatment=order.treatment.iloc[i], timepoint=order.timepoint.iloc[i],
        biological_unit_id=order.biological_unit_id.iloc[i], leaf_patches=int(m2.sum()),
        mean_lesion_score=float(np.nanmean(np.where(m,ls,np.nan))), heterogeneity=het,
        morans_I=mI, lesion_fraction=lf, aggregation=agg, n_components=ncomp))
    if i%400==0: open(f"{ROOT}/logs/spatial.log","a").write(f"{i}/{N}\n")
np.save(f"{OUT}/lesion_score_maps_f16.npy", maps.astype("float16"))
sp=pd.DataFrame(rows); sp.to_csv(f"{OUT}/spatial_per_image.csv", index=False)
# aggregate
a=sp[sp.leaf_patches>0].groupby(["accession","treatment","timepoint"]).agg(
    heterogeneity=("heterogeneity","mean"), morans_I=("morans_I","mean"),
    lesion_fraction=("lesion_fraction","mean"), aggregation=("aggregation","mean"),
    mean_lesion_score=("mean_lesion_score","mean")).reset_index()
a.to_csv(f"{OUT}/spatial_by_group.csv", index=False)
json.dump({"lesion_threshold_p95_control":THR,"n_control_patches":int(len(ctrl_scores))},
          open(f"{OUT}/spatial_info.json","w"), indent=2)
print("DONE spatial"); print(a.to_string(index=False))

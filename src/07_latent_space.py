"""07_latent_space.py — Latent phenotypic space from global DINOv2 embeddings.
Raw (L2-normalized) space -> PCA -> UMAP (visualization only). Saves coordinates,
explained variance, and the PCA feature matrix for downstream stats/PLS."""
import os, json, numpy as np, pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import umap

ROOT="/home/claude/proj"; EMB=f"{ROOT}/outputs/embeddings"; OUT=f"{ROOT}/outputs/latent"
os.makedirs(OUT, exist_ok=True)
SEED=42; np.random.seed(SEED)

order=pd.read_csv(f"{EMB}/order.csv")
G=np.load(f"{EMB}/global.npy")                      # (N,768) all images
assert len(order)==len(G)

# main analysis set: originals, no copies, known timepoint (leakage-safe, no dup)
mask=(order.use_unsup==True).values
meta=order[mask].reset_index(drop=True)
X=normalize(G[mask], norm="l2")                     # unit vectors (cosine geometry)

pca=PCA(n_components=50, random_state=SEED).fit(X)
Z=pca.transform(X)                                  # (n,50)
np.save(f"{OUT}/pca_scores.npy", Z.astype("float32"))
pd.DataFrame({"pc":np.arange(1,51),
              "explained_var_ratio":pca.explained_variance_ratio_,
              "cum":np.cumsum(pca.explained_variance_ratio_)}).to_csv(f"{OUT}/pca_explained.csv",index=False)

reducer=umap.UMAP(n_neighbors=30,min_dist=0.10,n_components=2,metric="euclidean",random_state=SEED)
U=reducer.fit_transform(Z)                           # UMAP on PCA-50

meta["PC1"]=Z[:,0]; meta["PC2"]=Z[:,1]; meta["PC3"]=Z[:,2]
meta["UMAP1"]=U[:,0]; meta["UMAP2"]=U[:,1]
meta.to_csv(f"{OUT}/latent_coords.csv", index=False)
# feature matrix + metadata for R (PERMANOVA etc.)
np.save(f"{OUT}/feat_l2.npy", X.astype("float32"))
meta.to_csv(f"{OUT}/meta_unsup.csv", index=False)

info=dict(n=int(mask.sum()), pc_var_top3=[round(float(v),4) for v in pca.explained_variance_ratio_[:3]],
          cum_var_pc10=float(np.cumsum(pca.explained_variance_ratio_)[9]),
          cum_var_pc50=float(np.cumsum(pca.explained_variance_ratio_)[49]))
json.dump(info, open(f"{OUT}/latent_info.json","w"), indent=2)
print(json.dumps(info,indent=2))

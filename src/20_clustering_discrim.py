"""20_clustering_discrim.py — Which representation discriminates the 3 groups best
(PCA vs UMAP vs PLS-DA), and how frozen embeddings cluster (global vs patch).
Grouped-CV accuracy + silhouette; KMeans/GMM with ARI; 2D/3D coordinates + hulls."""
import numpy as np, pandas as pd, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import umap
ROOT="/home/claude/proj"; OUT=f"{ROOT}/outputs/discrim"; os.makedirs(OUT,exist_ok=True)
np.random.seed(0)
order=pd.read_csv(f"{ROOT}/outputs/embeddings/order.csv"); id2row={i:k for k,i in enumerate(order.image_id)}
ci=pd.read_csv(f"{ROOT}/outputs/embeddings/class_index.csv"); rows=ci.image_id.map(id2row).values
G=np.load(f"{ROOT}/outputs/embeddings/global.npy")[rows]; Pp=np.load(f"{ROOT}/outputs/embeddings/patch_pooled.npy")[rows]
classes=["ASC 160","CAL 96","Control"]; y=ci.group3.map({c:i for i,c in enumerate(classes)}).values
groups=ci.biological_unit_id.values

def reps(X):
    Xs=StandardScaler().fit_transform(X)
    pca=PCA(50,random_state=0).fit(Xs); Z=pca.transform(Xs)
    U=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=42).fit_transform(Z)
    Yoh=np.eye(3)[y]; pls=PLSRegression(3).fit(Xs,Yoh); T=pls.transform(Xs)
    return dict(PCA=Z[:,:10], UMAP=U, PLS=T, PCA2=Z[:,:2], PCA3=Z[:,:3])

def disc(space):
    gkf=GroupKFold(5)
    pred=cross_val_predict(LogisticRegression(max_iter=2000,C=1.0),space,y,groups=groups,cv=gkf)
    acc=accuracy_score(y,pred)
    sil=silhouette_score(space,y) if space.shape[1]<=10 else np.nan
    return float(acc),float(sil)

out={"classes":classes,"discrimination":{}, "clustering":{}}
coords={}
for name,X in [("global",G),("patch",Pp)]:
    R=reps(X); coords[name]=R
    for rep in ["PCA","UMAP","PLS"]:
        acc,sil=disc(R[rep]); out["discrimination"][f"{name}|{rep}"]={"cv_accuracy":acc,"silhouette":sil}
    # clustering on PCA-10
    km=KMeans(3,n_init=10,random_state=0).fit_predict(R["PCA"])
    gm=GaussianMixture(3,random_state=0).fit_predict(R["PCA"])
    out["clustering"][name]={"kmeans_ARI":float(adjusted_rand_score(y,km)),
                             "gmm_ARI":float(adjusted_rand_score(y,gm))}
# save coords for figures
np.savez(f"{OUT}/coords.npz",
         g_pca2=coords["global"]["PCA2"], g_pca3=coords["global"]["PCA3"], g_umap=coords["global"]["UMAP"], g_pls=coords["global"]["PLS"],
         p_pca2=coords["patch"]["PCA2"], p_pca3=coords["patch"]["PCA3"], p_umap=coords["patch"]["UMAP"], p_pls=coords["patch"]["PLS"],
         y=y)
ci[["image_id","group3","timepoint","accession","split"]].to_csv(f"{OUT}/coords_meta.csv",index=False)
json.dump(out,open(f"{OUT}/discrim_compare.json","w"),indent=2)
print(json.dumps(out,indent=2))

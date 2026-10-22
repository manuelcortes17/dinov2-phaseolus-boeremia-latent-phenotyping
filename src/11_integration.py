"""11_integration.py — Integration of DINOv2 latent phenotype with R4PDE severity.
Central question: does DINOv2 merely recover conventional severity, or add
complementary phenotypic information? Grouped-CV PLS (leakage-safe), exploratory
vs predictive PLS, complementarity (similar severity / divergent latent), and
RV coefficients between blocks (global latent | spatial | severity)."""
import os, json, numpy as np, pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
ROOT="/home/claude/proj"; OUT=f"{ROOT}/outputs/pls"; os.makedirs(OUT, exist_ok=True)
rng=np.random.default_rng(42)

meta=pd.read_csv(f"{ROOT}/outputs/latent/meta_unsup.csv")
X=np.load(f"{ROOT}/outputs/latent/feat_l2.npy")     # (n,768) aligned to meta
Z=np.load(f"{ROOT}/outputs/latent/pca_scores.npy")  # (n,50)
sev=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")[["image_id","severity"]]
spat=pd.read_csv(f"{ROOT}/outputs/spatial/spatial_per_image.csv")
meta=meta.reset_index().rename(columns={"index":"row"})
m=meta.merge(sev,on="image_id",how="left").merge(
    spat[["image_id","heterogeneity","morans_I","lesion_fraction","aggregation","mean_lesion_score"]],
    on="image_id",how="left")
ok=m.severity.notna().values
m=m[ok].reset_index(drop=True); X=X[m.row.values]; Z=Z[m.row.values]
y=m.severity.values.astype(float); ylog=np.log1p(y)
groups=m.biological_unit_id.values
print("n images with severity:", len(m))

# ---------- 1. correlations latent PCs vs severity ----------
cors={f"PC{i+1}": [float(pearsonr(Z[:,i],ylog)[0]), float(spearmanr(Z[:,i],y)[0])] for i in range(10)}

# ---------- 2. predictive PLS (grouped CV) severity ~ global embedding ----------
def grouped_pls_cv(Xf,yf,groups,ncomp,folds=5):
    gkf=GroupKFold(n_splits=folds); pred=np.zeros_like(yf);
    for tr,te in gkf.split(Xf,yf,groups):
        sc=StandardScaler().fit(Xf[tr])
        pls=PLSRegression(n_components=ncomp)
        pls.fit(sc.transform(Xf[tr]), yf[tr])
        pred[te]=pls.predict(sc.transform(Xf[te])).ravel()
    ss_res=((yf-pred)**2).sum(); ss_tot=((yf-yf.mean())**2).sum()
    return 1-ss_res/ss_tot, float(np.sqrt(((yf-pred)**2).mean())), pred
# choose ncomp by CV
scan={}
for nc in [2,4,6,8,10,15]:
    r2,rmse,_=grouped_pls_cv(X,ylog,groups,nc); scan[nc]=[float(r2),float(rmse)]
best_nc=max(scan,key=lambda k:scan[k][0])
r2_cv,rmse_cv,pred=grouped_pls_cv(X,ylog,groups,best_nc)
# exploratory (fit all -> apparent R2) to contrast with CV (overfit demo)
sc=StandardScaler().fit(X); pls=PLSRegression(best_nc).fit(sc.transform(X),ylog)
app=pls.predict(sc.transform(X)).ravel()
r2_app=1-((ylog-app)**2).sum()/((ylog-ylog.mean())**2).sum()
# bootstrap CI on CV R2 (resample groups)
uq=np.array(sorted(set(groups)))
def boot_r2(n=300):
    out=[]
    for _ in range(n):
        gs=rng.choice(uq,len(uq),replace=True)
        idx=np.concatenate([np.where(groups==g)[0] for g in gs])
        try:
            r2,_,_=grouped_pls_cv(X[idx],ylog[idx],groups[idx],best_nc); out.append(r2)
        except Exception: pass
    return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]
r2_ci=boot_r2(200)

# ---------- 3. complementarity: similar severity, divergent latent ----------
# residual latent variation not explained by severity: R2 of latent~severity (multivariate)
from numpy.linalg import lstsq
Ys=np.column_stack([np.ones_like(y),ylog,ylog**2])
B,_,_,_=lstsq(Ys,Z,rcond=None); resid=Z-Ys@B
frac_unexpl=float((resid**2).sum()/((Z-Z.mean(0))**2).sum())
# among near-equal severity pairs (|dsev|<2%), latent distances still large?
def near_pairs_stats(band=2.0,nmax=60000):
    idx=np.argsort(y); yy=y[idx]; ZZ=Z[idx]; gg=groups[idx]
    lo=np.searchsorted(yy,yy-band); hi=np.searchsorted(yy,yy+band)
    ds=[]; rng2=np.random.default_rng(0)
    for i in range(len(yy)):
        js=list(range(lo[i],hi[i]))
        for j in js:
            if j<=i or gg[j]==gg[i]: continue
            ds.append(np.linalg.norm(ZZ[i]-ZZ[j]))
            if len(ds)>=nmax: break
        if len(ds)>=nmax: break
    ds=np.array(ds)
    alld=[]
    for _ in range(20000):
        a,b=rng2.integers(0,len(y),2)
        if groups[a]!=groups[b]: alld.append(np.linalg.norm(Z[a]-Z[b]))
    return float(ds.mean()), float(np.median(ds)), float(np.mean(alld))
near_mean,near_med,all_mean=near_pairs_stats()

# ---------- 4. RV coefficients between blocks (image level, standardized) ----------
def RV(A,B):
    A=A-A.mean(0); B=B-B.mean(0)
    AA=A@A.T; BB=B@B.T
    return float(np.trace(AA@BB)/np.sqrt(np.trace(AA@AA)*np.trace(BB@BB)))
Sblk=StandardScaler().fit_transform(m[["heterogeneity","morans_I","lesion_fraction","aggregation","mean_lesion_score"]].fillna(0).values)
sevblk=StandardScaler().fit_transform(np.column_stack([ylog]))
rv={"global_vs_severity":RV(Z,sevblk),
    "spatial_vs_severity":RV(Sblk,sevblk),
    "global_vs_spatial":RV(Z,Sblk)}

res=dict(n_images=int(len(m)), pc_severity_corr=cors,
         pls_cv_scan=scan, pls_best_ncomp=int(best_nc),
         pls_cv_r2=float(r2_cv), pls_cv_r2_ci=r2_ci, pls_cv_rmse=float(rmse_cv),
         pls_apparent_r2=float(r2_app),
         latent_frac_unexplained_by_severity=frac_unexpl,
         near_severity_latent_dist_mean=near_mean, near_severity_latent_dist_median=near_med,
         overall_latent_dist_mean=all_mean, RV=rv)
json.dump(res, open(f"{OUT}/integration_metrics.json","w"), indent=2)
# save arrays for figures
np.save(f"{OUT}/pls_pred_logsev.npy", pred); m.to_csv(f"{OUT}/merged_image_level.csv",index=False)
np.save(f"{OUT}/pls_scores.npy", pls.x_scores_.astype("float32"))
print(json.dumps({k:v for k,v in res.items() if k not in ("pc_severity_corr","pls_cv_scan")}, indent=2))

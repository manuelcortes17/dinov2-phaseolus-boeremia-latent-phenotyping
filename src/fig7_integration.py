import sys, json; sys.path.insert(0,"/home/claude/proj/scripts")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import figstyle as F
ROOT="/home/claude/proj"
ig=json.load(open(f"{ROOT}/outputs/pls/integration_metrics.json"))
tm=json.load(open(f"{ROOT}/outputs/trajectories/trajectory_metrics.json"))
merged=pd.read_csv(f"{ROOT}/outputs/pls/merged_image_level.csv")
pred=np.load(f"{ROOT}/outputs/pls/pls_pred_logsev.npy")
Z=np.load(f"{ROOT}/outputs/latent/pca_scores.npy")
meta=pd.read_csv(f"{ROOT}/outputs/latent/meta_unsup.csv")
sev=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")[["image_id","severity"]]
meta=meta.merge(sev,on="image_id",how="left"); ok=meta.severity.notna().values
Zc=Z[ok]; sv=meta.severity.values[ok]
healthy=Zc[(meta.treatment.values[ok]=="control")].mean(0)
dist=np.linalg.norm(Zc-healthy,axis=1)

fig=plt.figure(figsize=(180*F.MM,115*F.MM))
gs=fig.add_gridspec(2,3,hspace=0.36,wspace=0.40)

# A: PLS CV predicted vs observed
axA=fig.add_subplot(gs[0,0])
obs=np.log1p(merged.severity.values)
axA.scatter(obs,pred,s=2,c=merged.severity.values,cmap=F.SEVMAP,linewidths=0,alpha=.6)
lim=[min(obs.min(),pred.min()),max(obs.max(),pred.max())]
axA.plot(lim,lim,"k--",lw=0.6)
axA.set_xlabel("observed log(1+sev)"); axA.set_ylabel("PLS predicted (grouped CV)")
axA.text(0.04,0.92,f"$R^2_{{CV}}$={ig['pls_cv_r2']:.2f}\nRMSE={ig['pls_cv_rmse']:.2f}",
         transform=axA.transAxes,fontsize=6,va="top"); F.panel_label(axA,"A")

# B: apparent vs CV R2 (overfitting) + ncomp scan
axB=fig.add_subplot(gs[0,1])
axB.bar([0,1],[ig["pls_apparent_r2"],ig["pls_cv_r2"]],0.5,color=["#bdbdbd","#B2182B"])
axB.errorbar([1],[ig["pls_cv_r2"]],yerr=[[ig["pls_cv_r2"]-ig["pls_cv_r2_ci"][0]],
    [ig["pls_cv_r2_ci"][1]-ig["pls_cv_r2"]]],fmt="none",ecolor="k",capsize=2,lw=0.8)
axB.set_xticks([0,1]); axB.set_xticklabels(["apparent","grouped CV"]); axB.set_ylabel("$R^2$ (log severity)")
axB.set_ylim(0,1); F.panel_label(axB,"B")

# C: RV coefficients
axC=fig.add_subplot(gs[0,2])
rv=ig["RV"]; keys=["global_vs_severity","spatial_vs_severity","global_vs_spatial"]
labs=["global\n vs sev","spatial\n vs sev","global\n vs spatial"]
axC.bar(range(3),[rv[k] for k in keys],0.6,color=["#3182bd","#31a354","#756bb1"])
axC.set_xticks(range(3)); axC.set_xticklabels(labs,fontsize=5.6); axC.set_ylabel("RV coefficient")
axC.set_ylim(0,1); F.panel_label(axC,"C")

# D: severity vs latent distance-to-healthy (spread = complementary info)
axD=fig.add_subplot(gs[1,0])
axD.scatter(sv,dist,s=2,c=sv,cmap=F.SEVMAP,linewidths=0,alpha=.5)
rho=spearmanr(sv,dist)[0]
axD.set_xlabel("R4PDE severity (%)"); axD.set_ylabel("latent dist. to healthy")
axD.text(0.5,0.06,f"Spearman ρ={rho:.2f}\nspread at fixed severity = complementary",
         transform=axD.transAxes,fontsize=5.4,ha="center"); F.panel_label(axD,"D")

# E: PC vs severity correlations
axE=fig.add_subplot(gs[1,1])
pcs=[f"PC{i}" for i in range(1,11)]; r=[ig["pc_severity_corr"][p][1] for p in pcs]
axE.bar(range(1,11),r,0.7,color=["#B2182B" if abs(v)>0.2 else "#9aa0a6" for v in r])
axE.axhline(0,color="k",lw=0.5); axE.set_xlabel("Principal component")
axE.set_ylabel("Spearman ρ with severity"); F.panel_label(axE,"E")

# F: R4PDE severity curve + latent net displacement
axF=fig.add_subplot(gs[1,2])
sg=pd.read_csv(f"{ROOT}/outputs/r4pde/severity_per_image.csv")
g=sg[sg.treatment=="inoculated"].groupby(["accession","timepoint"]).severity.mean().reset_index()
for a,col in F.ACC.items():
    s=g[g.accession==a].set_index("timepoint").reindex(["T1","T2","T3"])
    axF.plot([1,2,3],s.severity,"-o",color=col,ms=4,label=a)
axF.set_xticks([1,2,3]); axF.set_xticklabels(["T1","T2","T3"])
axF.set_xlabel("evaluation"); axF.set_ylabel("R4PDE severity (%)")
axF.legend(fontsize=6,loc="upper left"); F.panel_label(axF,"F")
fig.savefig(f"{ROOT}/outputs/figures/Figure_7.png"); print("saved Figure_7.png")

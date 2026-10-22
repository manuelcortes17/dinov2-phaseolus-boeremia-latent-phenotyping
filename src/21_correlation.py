"""21_correlation.py — unit-level integrated metrics + Spearman correlation matrix
linking classic R4PDE phytopathometry with DINOv2 latent/spatial descriptors."""
import numpy as np, pandas as pd, json, os
from scipy.stats import spearmanr
ROOT="/home/claude/proj"; OUT=f"{ROOT}/outputs/integration2"; os.makedirs(OUT,exist_ok=True)
aud=pd.read_csv(f"{ROOT}/outputs/r4pde/audpc_per_unit.csv")
U=pd.read_csv(f"{ROOT}/outputs/trajectories/unit_level.csv")
UV=np.load(f"{ROOT}/outputs/trajectories/unit_vectors.npy")
sp=pd.read_csv(f"{ROOT}/outputs/spatial/spatial_per_image.csv")

# latent net displacement T1->T3 per unit
disp={}
for uid,g in U.groupby("biological_unit_id"):
    d={r.timepoint:UV[r.Index] for r in g.itertuples()}
    if "T1" in d and "T3" in d: disp[uid]=np.linalg.norm(d["T3"]-d["T1"])
# spatial at T3 per unit
spT3=sp[sp.timepoint=="T3"].groupby("biological_unit_id").agg(
    moransI_T3=("morans_I","mean"),het_T3=("heterogeneity","mean"),
    lesfrac_T3=("lesion_fraction","mean"),lscore_T3=("mean_lesion_score","mean")).reset_index()

m=aud.rename(columns={"severity.T1":"sev_T1","severity.T2":"sev_T2","severity.T3":"sev_T3"})
m["latent_displ"]=m.biological_unit_id.map(disp)
m=m.merge(spT3,on="biological_unit_id",how="left")
m=m[m.treatment=="inoculated"].copy()

cols=["sev_T3","AUDPC","rAUDPC","latent_displ","moransI_T3","het_T3","lesfrac_T3","lscore_T3"]
lab={"sev_T3":"severity T3","AUDPC":"AUDPC","rAUDPC":"rAUDPC","latent_displ":"latent displ.",
     "moransI_T3":"Moran's I","het_T3":"heterogeneity","lesfrac_T3":"lesion frac.","lscore_T3":"lesion score"}
M=m[cols].dropna()
R=np.zeros((len(cols),len(cols))); Pv=np.ones_like(R)
for i in range(len(cols)):
    for j in range(len(cols)):
        r,p=spearmanr(M.iloc[:,i],M.iloc[:,j]); R[i,j]=r; Pv[i,j]=p
np.savez(f"{OUT}/corr.npz",R=R,P=Pv,labels=[lab[c] for c in cols])
m.to_csv(f"{OUT}/unit_metrics.csv",index=False)
json.dump({"n_units":int(len(M)),"labels":[lab[c] for c in cols],
           "R":R.round(3).tolist()},open(f"{OUT}/corr.json","w"),indent=2)
print("n units",len(M)); print(pd.DataFrame(R.round(2),index=[lab[c] for c in cols],columns=[lab[c] for c in cols]).to_string())

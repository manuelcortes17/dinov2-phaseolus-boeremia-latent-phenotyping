"""08_trajectories.py — Temporal phenotypic trajectories in latent space.
Unit-level aggregation (mean over technical images) -> centroids per
accession x treatment x timepoint -> displacement / length / velocity /
direction / accession divergence / within-group dispersion, with
unit-level bootstrap CIs and a permutation test on per-unit displacement."""
import os, json, numpy as np, pandas as pd
ROOT="/home/claude/proj"; LAT=f"{ROOT}/outputs/latent"; OUT=f"{ROOT}/outputs/trajectories"
os.makedirs(OUT, exist_ok=True)
rng=np.random.default_rng(42)

meta=pd.read_csv(f"{LAT}/meta_unsup.csv")
Z=np.load(f"{LAT}/pca_scores.npy")            # (n,50) aligned to meta
meta=meta.reset_index(drop=True)
TP=["T1","T2","T3"]

# ---- unit-level vectors: mean over technical images per unit x timepoint ----
rows=[]; vecs=[]
for (uid,tp),g in meta.groupby(["biological_unit_id","timepoint"]):
    idx=g.index.values
    rows.append(dict(biological_unit_id=uid, timepoint=tp,
                     accession=g.accession.iloc[0], treatment=g.treatment.iloc[0],
                     n_img=len(idx)))
    vecs.append(Z[idx].mean(0))
U=pd.DataFrame(rows); UV=np.vstack(vecs)          # unit-level
np.save(f"{OUT}/unit_vectors.npy", UV.astype("float32")); U.to_csv(f"{OUT}/unit_level.csv",index=False)

def centroid(acc,tr,tp,uv=UV,u=U):
    m=(u.accession==acc)&(u.treatment==tr)&(u.timepoint==tp)
    return uv[m.values].mean(0) if m.sum()>0 else None

def traj_metrics(acc,tr,uv=UV,u=U):
    C={tp:centroid(acc,tr,tp,uv,u) for tp in TP}
    if any(C[t] is None for t in TP): return None
    d12=np.linalg.norm(C["T2"]-C["T1"]); d23=np.linalg.norm(C["T3"]-C["T2"])
    net=np.linalg.norm(C["T3"]-C["T1"]); length=d12+d23
    v1=C["T2"]-C["T1"]; v2=C["T3"]-C["T2"]
    straight=float(net/length) if length>0 else np.nan
    cos_steps=float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12))
    return dict(step_T1T2=float(d12),step_T2T3=float(d23),net_T1T3=float(net),
                length=float(length),straightness=straight,cos_consecutive=cos_steps)

def dispersion(acc,tr,tp):
    m=(U.accession==acc)&(U.treatment==tr)&(U.timepoint==tp)
    if m.sum()<2: return np.nan
    V=UV[m.values]; c=V.mean(0); return float(np.mean(np.linalg.norm(V-c,axis=1)))

res={"centroid_trajectory":{}, "dispersion":{}, "accession_divergence":{}}
for acc in ["ASC 160","CAL 96"]:
    for tr in ["inoculated","control"]:
        tm=traj_metrics(acc,tr)
        if tm: res["centroid_trajectory"][f"{acc}|{tr}"]=tm
        res["dispersion"][f"{acc}|{tr}"]={tp:dispersion(acc,tr,tp) for tp in TP}

# accession divergence (inoculated): distance between accession centroids per tp,
# and angle between disease-progression vectors (T3-T1)
for tr in ["inoculated"]:
    perc={tp: float(np.linalg.norm(centroid("ASC 160",tr,tp)-centroid("CAL 96",tr,tp))) for tp in TP}
    a=centroid("ASC 160",tr,"T3")-centroid("ASC 160",tr,"T1")
    c=centroid("CAL 96",tr,"T3")-centroid("CAL 96",tr,"T1")
    ang=float(np.degrees(np.arccos(np.clip(np.dot(a,c)/(np.linalg.norm(a)*np.linalg.norm(c)+1e-12),-1,1))))
    res["accession_divergence"]={"centroid_dist_per_tp":perc,
        "angle_progression_vectors_deg":ang,
        "progress_mag_ASC":float(np.linalg.norm(a)),"progress_mag_CAL":float(np.linalg.norm(c))}

# ---- per-unit paired displacement T1->T3 (units present at both) + permutation test ----
def paired_disp(acc,tr):
    u=U[(U.accession==acc)&(U.treatment==tr)]
    piv={};
    for uid,g in u.groupby("biological_unit_id"):
        d={r.timepoint: UV[r.Index] for r in g.itertuples()}
        if "T1" in d and "T3" in d: piv[uid]=np.linalg.norm(d["T3"]-d["T1"])
    return np.array(list(piv.values()))
disp_asc=paired_disp("ASC 160","inoculated"); disp_cal=paired_disp("CAL 96","inoculated")
disp_ctrl=np.concatenate([paired_disp("ASC 160","control"),paired_disp("CAL 96","control")])
def boot_ci(x,n=5000):
    if len(x)==0: return [np.nan,np.nan,np.nan]
    bs=[rng.choice(x,len(x),replace=True).mean() for _ in range(n)]
    return [float(np.mean(x)),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
res["per_unit_displacement_T1T3"]={
    "ASC160_inoc_mean_ci":boot_ci(disp_asc),"n_ASC":int(len(disp_asc)),
    "CAL96_inoc_mean_ci":boot_ci(disp_cal),"n_CAL":int(len(disp_cal)),
    "control_mean_ci":boot_ci(disp_ctrl),"n_ctrl":int(len(disp_ctrl))}
# permutation test ASC vs CAL inoculated displacement
def perm_test(a,b,n=10000):
    obs=abs(a.mean()-b.mean()); pool=np.concatenate([a,b]); na=len(a); cnt=0
    for _ in range(n):
        rng.shuffle(pool);
        if abs(pool[:na].mean()-pool[na:].mean())>=obs: cnt+=1
    return float(obs), (cnt+1)/(n+1)
obs,p=perm_test(disp_asc,disp_cal)
res["per_unit_displacement_T1T3"]["ASC_vs_CAL_diff"]=obs
res["per_unit_displacement_T1T3"]["ASC_vs_CAL_perm_p"]=p
# inoculated vs control (disease-driven movement)
obs2,p2=perm_test(np.concatenate([disp_asc,disp_cal]),disp_ctrl)
res["per_unit_displacement_T1T3"]["inoc_vs_ctrl_diff"]=obs2
res["per_unit_displacement_T1T3"]["inoc_vs_ctrl_perm_p"]=p2

json.dump(res, open(f"{OUT}/trajectory_metrics.json","w"), indent=2)
print(json.dumps(res, indent=2))

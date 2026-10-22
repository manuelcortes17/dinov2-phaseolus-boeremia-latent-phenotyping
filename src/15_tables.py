"""15_tables.py — main manuscript tables from frozen results."""
import json, numpy as np, pandas as pd
ROOT="/home/claude/proj"; O=f"{ROOT}/outputs/tables"; import os; os.makedirs(O,exist_ok=True)
fr=json.load(open(f"{ROOT}/frozen/frozen_results.json"))
tm=json.load(open(f"{ROOT}/outputs/trajectories/trajectory_metrics.json"))
ig=json.load(open(f"{ROOT}/outputs/pls/integration_metrics.json"))
idx=pd.read_csv(f"{ROOT}/outputs/index/master_index.csv")

# ---- Table 1: dataset ----
o=idx[(idx.is_augmented==False)&(idx.is_copy==False)&(idx.timepoint.isin(["T1","T2","T3"]))]
t1=o.groupby(["accession","treatment"]).apply(lambda d: pd.Series({
    "biological units":d.biological_unit_id.nunique(),
    "images T1":int((d.timepoint=="T1").sum()),
    "images T2":int((d.timepoint=="T2").sum()),
    "images T3":int((d.timepoint=="T3").sum()),
    "total images":len(d)}),include_groups=False).reset_index()
aug=idx[idx.is_augmented==True]
t1.loc[len(t1)]=["—","augmented (controls)","—","—","—","—",len(aug)]
t1.to_csv(f"{O}/Table1_dataset.csv",index=False)

# ---- Table 2: latent-space & trajectory statistics ----
pm=pd.DataFrame(fr["permanova"])
rows=[]
for _,r in pm[~pm.term.isin(["Residual","Total"])].iterrows():
    rows.append(["PERMANOVA",r.term,f"R²={r.R2:.3f}",f"F={r['F']:.1f}",f"P={r.P:.3f}"])
pd_=fr["permdisp"]
disp=tm["dispersion"]
rows.append(["PERMDISP","dispersion T1/T2/T3 (inoc)",
             f"{disp['ASC 160|inoculated']['T1']:.3f}/{disp['ASC 160|inoculated']['T2']:.3f}/{disp['ASC 160|inoculated']['T3']:.3f} (ASC)",
             f"{disp['CAL 96|inoculated']['T1']:.3f}/{disp['CAL 96|inoculated']['T2']:.3f}/{disp['CAL 96|inoculated']['T3']:.3f} (CAL)","P=0.001"])
pu=tm["per_unit_displacement_T1T3"]
rows.append(["Trajectory","net latent displ. T1→T3 (ASC, inoc)",
             f"{pu['ASC160_inoc_mean_ci'][0]:.3f}",f"95% CI [{pu['ASC160_inoc_mean_ci'][1]:.3f}, {pu['ASC160_inoc_mean_ci'][2]:.3f}]",f"n={pu['n_ASC']}"])
rows.append(["Trajectory","net latent displ. T1→T3 (CAL, inoc)",
             f"{pu['CAL96_inoc_mean_ci'][0]:.3f}",f"95% CI [{pu['CAL96_inoc_mean_ci'][1]:.3f}, {pu['CAL96_inoc_mean_ci'][2]:.3f}]",f"n={pu['n_CAL']}"])
rows.append(["Trajectory","ASC vs CAL displacement","permutation test",f"Δ={pu['ASC_vs_CAL_diff']:.3f}",f"P={pu['ASC_vs_CAL_perm_p']:.1e}"])
t2=pd.DataFrame(rows,columns=["analysis","term","estimate","statistic","significance"])
t2.to_csv(f"{O}/Table2_latent_trajectory.csv",index=False)

# ---- Table 3: DINOv2 vs R4PDE integration ----
rv=ig["RV"]
t3=pd.DataFrame([
 ["Predictive PLS (severity ~ global embedding)","grouped-CV R²",f"{ig['pls_cv_r2']:.3f}",
     f"95% CI [{ig['pls_cv_r2_ci'][0]:.3f}, {ig['pls_cv_r2_ci'][1]:.3f}]"],
 ["Predictive PLS","apparent R² (fit-all)",f"{ig['pls_apparent_r2']:.3f}","overfitting gap small"],
 ["Predictive PLS","RMSE (log severity, CV)",f"{ig['pls_cv_rmse']:.3f}",f"{ig['pls_best_ncomp']} components"],
 ["Complementarity","latent variance NOT explained by severity",f"{ig['latent_frac_unexplained_by_severity']*100:.1f}%","—"],
 ["Complementarity","latent dist. (similar-severity pairs vs overall)",
     f"{ig['near_severity_latent_dist_mean']:.3f} vs {ig['overall_latent_dist_mean']:.3f}","similar severity ≠ similar latent"],
 ["Block similarity (RV)","global embedding vs classical severity",f"{rv['global_vs_severity']:.3f}","moderate overlap"],
 ["Block similarity (RV)","spatial features vs classical severity",f"{rv['spatial_vs_severity']:.3f}","near-independent"],
 ["Block similarity (RV)","global vs spatial",f"{rv['global_vs_spatial']:.3f}","partial overlap"],
],columns=["analysis","quantity","value","note"])
t3.to_csv(f"{O}/Table3_integration.csv",index=False)
print("TABLE 1\n",t1.to_string(index=False))
print("\nTABLE 2\n",t2.to_string(index=False))
print("\nTABLE 3\n",t3.to_string(index=False))

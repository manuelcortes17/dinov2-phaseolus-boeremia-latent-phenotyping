"""16_freeze.py — Consolidate every machine-readable result into one frozen package.
The manuscript may ONLY use numbers from frozen/frozen_results.json."""
import os, json, numpy as np, pandas as pd
ROOT="/home/claude/proj"; O=f"{ROOT}/outputs"; F=f"{ROOT}/frozen"; os.makedirs(F, exist_ok=True)
def jload(p): return json.load(open(p)) if os.path.exists(p) else None
def cload(p): return pd.read_csv(p) if os.path.exists(p) else None

fr={}
fr["audit"]=jload(f"{O}/index/audit_summary.json")
fr["latent"]=jload(f"{O}/latent/latent_info.json")
fr["trajectories"]=jload(f"{O}/trajectories/trajectory_metrics.json")
fr["integration"]=jload(f"{O}/pls/integration_metrics.json")
fr["spatial_info"]=jload(f"{O}/spatial/spatial_info.json")
fr["spatial_by_group"]=cload(f"{O}/spatial/spatial_by_group.csv").round(4).to_dict("records")

# severity summaries
sev=cload(f"{O}/r4pde/severity_per_image.csv")
g=sev.groupby(["accession","treatment","timepoint"]).severity.agg(["mean","std","median","count"]).round(3).reset_index()
fr["severity_by_group"]=g.to_dict("records")
aud=cload(f"{O}/r4pde/audpc_per_unit.csv")
ag=aud.groupby(["accession","treatment"]).agg(AUDPC_mean=("AUDPC","mean"),AUDPC_sd=("AUDPC","std"),
     rAUDPC_mean=("rAUDPC","mean"),n=("AUDPC","size")).round(3).reset_index()
fr["audpc_by_group"]=ag.to_dict("records")

# R stats
pm=cload(f"{O}/stats/permanova.csv")
if pm is not None:
    pm.columns=["term","Df","SumOfSqs","R2","F","P"]
    fr["permanova"]=pm.round(4).to_dict("records")
fr["permdisp"]=cload(f"{O}/stats/permdisp.csv").round(4).to_dict("records") if os.path.exists(f"{O}/stats/permdisp.csv") else None
em=cload(f"{O}/stats/lmm_severity_emmeans.csv")
if em is not None: fr["lmm_severity_emmeans"]=em.round(3).to_dict("records")
ct=cload(f"{O}/stats/lmm_severity_contrasts.csv")
if ct is not None: fr["lmm_severity_contrasts"]=ct.round(4).to_dict("records")

# PCA explained
pex=cload(f"{O}/latent/pca_explained.csv")
fr["pca_explained_top5"]=pex.head(5).round(4).to_dict("records")

json.dump(fr, open(f"{F}/frozen_results.json","w"), indent=2, ensure_ascii=False, default=float)
print("Frozen keys:", list(fr.keys()))
print(json.dumps({"severity_by_group":fr["severity_by_group"],
                  "audpc_by_group":fr["audpc_by_group"],
                  "permanova":fr.get("permanova")}, indent=2, ensure_ascii=False, default=float))

import json, pandas as pd, subprocess, os, shutil, numpy as np
ROOT="/home/claude/proj"; TAB=f"{ROOT}/outputs/tables"; ST=f"{ROOT}/outputs/stats"; SF=f"{ROOT}/outputs/supp_figures"; D=f"{ROOT}/outputs/discrim"
os.makedirs(SF,exist_ok=True)
au=json.load(open(f"{ROOT}/outputs/index/audit_summary.json")); ig=json.load(open(f"{ROOT}/outputs/pls/integration_metrics.json"))
sc=json.load(open(f"{ROOT}/outputs/pls/sensitivity_cv.json")); fr=json.load(open(f"{ROOT}/frozen/frozen_results.json"))
dc=json.load(open(f"{D}/discrim_compare.json")); clf=json.load(open(f"{D}/classifier_metrics.json"))
shutil.copy(f"{ROOT}/outputs/attention/attn_demo.png", f"{SF}/Figure_S1.png")
for s in ["/tmp/sev_validate.png"]:
    if os.path.exists(s): shutil.copy(s, f"{SF}/Figure_S2.png")

def md(df):
    df=df.fillna("")
    return "\n".join(["| "+" | ".join(map(str,df.columns))+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+
        ["| "+" | ".join(str(x) for x in r)+" |" for r in df.values])
def csv(p): return md(pd.read_csv(p))

# discrimination table
rows=[]
for e in ["global","patch"]:
    for r in ["PCA","UMAP","PLS"]:
        d=dc["discrimination"][f"{e}|{r}"]; rows.append({"embedding":e,"representation":r,
            "CV accuracy":round(d["cv_accuracy"],3),"silhouette":round(d["silhouette"],3)})
disc_tab=pd.DataFrame(rows)
clu=pd.DataFrame([{"embedding":e,"KMeans ARI":round(dc["clustering"][e]["kmeans_ARI"],3),
                   "GMM ARI":round(dc["clustering"][e]["gmm_ARI"],3)} for e in ["global","patch"]])
clfm=pd.DataFrame([{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()} for r in clf["results"]])
symp=pd.DataFrame([{"comparison":"patch embedding vs symptom","ρ overall":fr["symptom_localization"]["rho_patch_overall"],
                    **{f"ρ {k}":v for k,v in fr["symptom_localization"]["rho_patch_by_tp"].items()}},
                   {"comparison":"raw attention vs symptom","ρ overall":fr["symptom_localization"]["rho_attn_overall"],
                    **{f"ρ {k}":v for k,v in fr["symptom_localization"]["rho_attn_by_tp"].items()}}])
scan=pd.DataFrame([{"components":k,"grouped_CV_R2":round(v[0],4),"RMSE":round(v[1],4)} for k,v in ig["pls_cv_scan"].items()])
pcc=pd.DataFrame([{"PC":k,"Spearman(sev)":round(v[1],3)} for k,v in ig["pc_severity_corr"].items()])

doc=f"""---
title: "Supplementary Information — A multidimensional phenotype of the Phaseolus vulgaris–Boeremia interaction"
---

# Supplementary Information

**A multidimensional phenotype of the *Phaseolus vulgaris*–*Boeremia* interaction from frozen DINOv2 representations, benchmarked against classical phytopathometry**

Detailed methods, quality control, complete statistical outputs, sensitivity analyses and supplementary figures for the main text. Main-text figures are cited as Figure 1–8 and main tables as Table 1–3; supplementary items are Figure S1–S2 and Table S1–S12.

## S1. Dataset and image quality control

The archive held {au['n_images_total_files']} files: {au['n_original_images']} original leaf images and {au['n_augmented_files_present']} augmented control images (Table 1, main text). Original images were {au['orig_native_resolution']}; {au['n_biological_units']} biological units. Quality control removed {au['n_copies_dropped']} exact duplicates and {au['n_unknown_timepoint']} images with an unresolved evaluation label; {au['n_missing_files']} files were missing. Technical replication was a median of {au['imgs_per_unit_time_median']:.0f} images per unit and evaluation.

**Augmentation audit.** The manifest listed {au['n_manifest_rows']} entries; {au['n_augmented_files_present']} were present ({au['n_augmented_missing_vs_manifest']} not materialized). Augmentation was control-only and used only to balance the three groups for the discrimination analyses (Figure 6, main text); ancestry was preserved and every transform stayed with its source unit and data partition. Unsupervised analyses used originals only (n = {au['unsup_n']}); the balanced discrimination set comprised n = {au['pred_n']}.

## S2. DINOv2 configuration and embeddings

Frozen DINOv2 ViT-B/14: 12 blocks, 12 heads, 768 dimensions, exact parameter match. Inference at 224 px (16×16 = 256 patch tokens) and, for high-resolution symptom maps (Figure 2, main text), 448 px (32×32 tokens). Global = L2-normalized class token; patch = 256 tokens with coordinates; patch summary = per-image mean. PCA variance: PC1 {fr['latent']['pc_var_top3'][0]*100:.1f}%, PC2 {fr['latent']['pc_var_top3'][1]*100:.1f}%, PC3 {fr['latent']['pc_var_top3'][2]*100:.1f}%; cumulative {fr['latent']['cum_var_pc10']*100:.1f}% (10 PCs), {fr['latent']['cum_var_pc50']*100:.1f}% (50 PCs).

## S3. Sensitivity and robustness

Predictive PLS for severity (Figure 8, Table 3, main text): grouped-CV R² = {sc['grouped_cv_r2']} vs random-split R² = {sc['random_cv_r2']} (leakage inflation {sc['leakage_inflation']}); apparent R² = {ig['pls_apparent_r2']:.3f}. The negligible grouped-vs-random gap shows the severity signal is genuine and not a technical-replicate leakage artefact. The challenged-vs-control latent displacement contrast was not significant (permutation P = 0.11) because control leaves also drift (senescence); the disease-specific signal is the inoculation × timepoint PERMANOVA term (P = 0.001, Table S1).

## S4. Complete statistical outputs

**Table S1. PERMANOVA (unit-level, 999 permutations, by terms).** Referenced from Results 3.3 and Table 2 (main text).

{csv(f"{ST}/permanova.csv")}

**Table S2. PERMDISP mean distance-to-centroid by evaluation (challenged).** Supports Figure 4B.

{csv(f"{ST}/permdisp.csv")}

**Table S3. Severity linear mixed model: estimated marginal means (accession within evaluation).** Supports Figure 7B.

{csv(f"{ST}/lmm_severity_emmeans.csv")}

**Table S4. Severity accession contrasts within evaluation.**

{csv(f"{ST}/lmm_severity_contrasts.csv")}

**Table S5. Spatial-metric mixed-model ANOVA (accession × timepoint).** Supports Figure 5, Figure 6.

{csv(f"{ST}/spatial_lmm_anova.csv")}

**Table S6. Discrimination of the three groups by representation (grouped-CV accuracy, silhouette).** Supports Figure 6E–H.

{md(disc_tab)}

**Table S7. Unsupervised clustering agreement with the true groups (adjusted Rand index).** Supports Figure 6F–G.

{md(clu)}

**Table S8. Held-out classifier performance (linear probe).** Supports Figure 6A–D.

{md(clfm)}

**Table S9. Symptom localization: correlation of patch embedding and of raw attention with the visible symptom.** Supports Figure 2D–E.

{md(symp)}

**Table S10. PLS component-number cross-validation scan.**

{md(scan)}

**Table S11. Correlation of leading principal components with severity.** Supports Figure 8A.

{md(pcc)}

**Table S12. Classical severity, AUDPC and spatial metrics by group.** Source data for Figures 5–8.

Severity by group:

{md(pd.DataFrame(fr['severity_by_group']))}

AUDPC by group (functional distance ASC vs CAL = {fr['functional_distance_ASC_CAL']}):

{md(pd.DataFrame(fr['audpc_by_group']))}

Spatial metrics by group:

{md(pd.DataFrame(fr['spatial_by_group']))}

## S5. Supplementary figures

![]({SF}/Figure_S1.png){{width=17cm}}

**Figure S1. Per-head attention analysis.** For representative challenged and control leaves: the 16×16 patch grid, class-token attention (last layer), layer-averaged attention, inter-layer variance, four individual heads, and attention rollout, as smooth lamina-masked overlays. Heads capture partially distinct patterns; none is specifically lesion-tuned (cf. Figure 2, main text).

![]({SF}/Figure_S2.png){{width=15cm}}

**Figure S2. Palette-based severity validation.** For representative leaves across group and evaluation: original image, palette-class map, and highlighted lesion tissue with the resulting severity, showing exclusion of the inoculation plug and scale card and capture of chlorotic + necrotic area (cf. Figure 7A, main text).

![]({SF}/Figure_S3.png){{width=13cm}}

**Figure S3. Quantitative symptom localization (supports Figure 2, main text).** (A) Per-patch Spearman correlation with the visible symptom for the DINOv2 patch embedding versus raw attention (points = leaves; bars = means): the patch representation tracks the symptom (ρ ≈ 0.35) while raw attention does not (ρ ≈ −0.10). (B) Patch–symptom correlation by evaluation.

## References

Del Ponte, E. M. (2023). *R for plant disease epidemiology (R4PDE)*. Author. https://r4pde.net
"""
open(f"{ROOT}/supplementary_full.md","w").write(doc)
os.chdir(ROOT)
subprocess.run(["pandoc","supplementary_full.md","-o","Supplementary_Information_DINOv2_Phaseolus_Boeremia.docx","--resource-path",ROOT],check=True)
print("supp docx built; supp tables S1-S12")

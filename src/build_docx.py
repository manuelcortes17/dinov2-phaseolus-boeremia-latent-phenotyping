import pandas as pd, re, subprocess, os
ROOT="/home/claude/proj"; FIG=f"{ROOT}/outputs/figures"; TAB=f"{ROOT}/outputs/tables"
md=open(f"{ROOT}/manuscript.md").read()

caps={
1:"**Figure 1. The latent phenotyping workflow and its mapping to the three objectives.** A standardized leaf image passes once through a frozen DINOv2 ViT-B/14 encoder, producing a global class-token embedding (phenotypic state), a grid of patch embeddings (spatial organization) and attention (interpretability). Objective 1 builds and discriminates the latent space (PCA/UMAP/PLS, global vs patch, clustering, classifier); Objective 2 quantifies temporal trajectories and spatial organization; Objective 3 compares the representation with classical phytopathometry (palette severity, AUDPC, progress curves). Key quantities (cosine distance, AUDPC, Moran's I, RV) are defined at lower left.",
2:"**Figure 2. The representation localizes the symptom in its patches, not in its attention.** (A–C) Representative leaves (control and challenged ASC 160 at T1–T3): original image; symptom-localization map from high-resolution (32×32) DINOv2 patch embeddings (cosine distance to the healthy control-patch centroid); and attention rollout, all masked to the lamina. The patch representation concentrates on chlorotic and necrotic tissue as disease advances, whereas attention remains on the lamina and venation. Quantitative correlations with the visible symptom are in Figure S3.",
3:"**Figure 3. A continuous latent phenotypic space with three groups.** (A) PCA of global embeddings with convex hulls per group. (B) UMAP by group (two accessions and control). (C) UMAP by evaluation. (D) UMAP coloured continuously by classical severity. (E) Three-dimensional point cloud (PCA-1–3). (F) PCA variance explained (bars) and cumulative (line).",
4:"**Figure 4. Disease progression as a latent trajectory.** (A) Group centroid trajectories T1→T2→T3 in UMAP (grey = all leaves; arrows = temporal order); the control barely moves. (B) Within-group dispersion (distance to centroid) per evaluation as violins with group means (PERMDISP P = 0.001). (C) Per-unit net T1→T3 displacement as half-violin rainclouds for the three groups (ASC vs CAL permutation P = 1×10⁻⁴).",
5:"**Figure 5. The spatial phenotype and its decoupling from severity.** (A) Leaf with 16×16 patch grid. (B) Patch-PCA local-phenotype map. (C) Lesion-score map. (D) Local heterogeneity. (E–F) Two challenged leaves of near-identical severity but opposite spatial organization (fragmented vs aggregated). (G) Density of all challenged leaves in the severity–Moran's I plane; the two example leaves (E, F) are marked. Severity and spatial organization vary almost independently.",
6:"**Figure 6. Discrimination and clustering: global versus patch embeddings.** (A) Training cross-entropy loss and (B) validation accuracy/recall per epoch for a linear probe on global and patch descriptors (70/15/15 unit-grouped split). (C–D) Held-out confusion matrices (global, patch). (E) Grouped-CV accuracy for a probe in PCA, UMAP and PLS spaces (global vs patch). (F–G) PCA scatter with convex hulls per group and Gaussian-mixture ARI (global vs patch). (H) Silhouette of the true groups by representation. Global embeddings carry group identity (ARI = 0.77); patch embeddings do not (ARI = 0.10).",
7:"**Figure 7. Classical phytopathometry (Objective 3).** (A) Palette-based severity on one leaf: original → palette classification (healthy/lesion) → segmented output with severity. (B) Disease-progress curves by group (mean ± SE). (C) Per-unit progress curves (n = 257). (D) AUDPC by group as half-violin rainclouds with means. (E) Functional comparison of the accession progress curves (shaded difference; L² functional distance). (F) Squared difference over time.",
8:"**Figure 8. Integrated comparison of classical and latent descriptors.** (A) Metric similarity network: nodes are metrics coloured by block (classical severity, latent-global, spatial-patch) and positioned by multidimensional scaling of 1−|ρ|, with edge width scaling with |Spearman ρ|; the classical severity metric sits apart from the tight spatial cluster, bridged by latent displacement. (B) Phenotype landscape: kernel-density territories of the three groups on the UMAP layout. (C) Phenotypic signatures across four conceptual axes — magnitude (AUDPC), state (latent displacement), organization (Moran's I), progression (Δdispersion) — normalized per axis. (D) Shared versus complementary information as horizontal bars: classical severity recovered by the global embedding (R²_CV), latent variance complementary to severity, and pairwise RV coefficients (global∩severity, global∩patch, patch∩severity).",
}
for n in range(1,9):
    img=f"![]({FIG}/Figure_{n}.png){{width=16.8cm}}\n\n{caps[n]}"
    md=md.replace(f"*[Figure {n} near here]*", img)

def csv_md(path):
    df=pd.read_csv(path).fillna("")
    h="| "+" | ".join(df.columns)+" |"
    s="| "+" | ".join(["---"]*len(df.columns))+" |"
    rows=["| "+" | ".join(str(x) for x in r)+" |" for r in df.values]
    return "\n".join([h,s]+rows)

tables=f"""

## Tables

**Table 1. Image dataset and experimental representation.**

{csv_md(f'{TAB}/Table1_dataset.csv')}

**Table 2. Multivariate latent-space and trajectory statistics.**

{csv_md(f'{TAB}/Table2_latent_trajectory.csv')}

**Table 3. Integration between DINOv2 representations and classical severity.**

{csv_md(f'{TAB}/Table3_integration.csv')}
"""
# insert Tables section just before References
md=md.replace("\n## References", tables+"\n## References")
open(f"{ROOT}/manuscript_full.md","w").write(md)
print("words:", len(re.findall(r"\\w+", md)), "approx chars", len(md))

os.chdir(ROOT)
subprocess.run(["pandoc","manuscript_full.md","-o","Manuscript_DINOv2_Phaseolus_Boeremia.docx",
    "--resource-path",ROOT], check=True)
print("docx built")

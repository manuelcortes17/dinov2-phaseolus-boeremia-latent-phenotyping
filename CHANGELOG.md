# 📝 Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/).

## [1.0.0] – 2026-10-21

### Added
- Full reproducible pipeline (Python + R + Quarto).
- Eight main figures and three supplementary figures at 650 dpi.
- Frozen results JSON that pins every number cited in the manuscript.
- Quarto walk-through notebook (`analysis_notebook.qmd`).
- Manuscript and supplementary information (DOCX) under `docs/manuscript/`.
- Repository scaffolding: README with figure grid, CITATION.cff, LICENSE,
  CONTRIBUTING, data-structure doc, publish script and `.gitignore`.

### Terminology
- Unified terminology across figures and documents to **"conventional"**
  (replacing *traditional / traditionality / classical*).
- `challenged` → `inoculated` in figures and documents per co-author review.

### Changed
- Figure 3C → PLS-DA by severity with group confidence ellipses.
- Figure 3D → latent-distance boxplot across T1→T3.
- Figure 3E → UMAP-3D density view, distinct from the PCA shown in 3A.
- Figure 4A → 2-D axonometric projection of the latent space with group
  centroid trajectories.
- Figure 5G → severity × Moran's I quadrant chart with density layer and
  leaf thumbnails for the two highlighted examples.
- Figure 6B → validation score axis rescaled so curves rise from baseline.
- Figure 8A → clustered correlogram replacing the earlier similarity network.
- Figure 8C → single-panel quadrant chart (magnitude / state / organization /
  progression) with high–low outside the box and italic unit labels.


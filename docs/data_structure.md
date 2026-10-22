# 📂 Data Structure

The raw image dataset is not stored in this repository (archive quota).
The pipeline expects the following layout under `data/raw/` (built from the field folder convention):

```
data/raw/
├── ASC_160/
│   ├── control/
│   │   └── Asc_<id>/           first → third evaluation subfolders
│   │       ├── Primera_evaluación/
│   │       ├── Segunda_evaluación/
│   │       └── Tercera_evaluación/
│   └── inoculated/
│       └── Asc_<id>/...
└── CAL_96/
    ├── control/
    └── inoculated/
```

- Each leaf image: **3,648 × 2,736 px**, Canon G12 at 5× zoom, uniform light-grey background.
- **T1 / T2 / T3** correspond to the three post-inoculation evaluations (4 / 6 / 8 days after inoculation).
- One **biological unit** = one trifoliate leaf tracked across the three evaluations.
- `01_build_index.py` parses this tree, filters duplicates, tags augmented images (control-only) and writes a tidy `master_index.csv`.

## Expected auxiliary files

| Path | Produced by | Purpose |
|---|---|---|
| `outputs/index/master_index.csv` | `01_build_index.py` | row = image; columns = accession, treatment, biological_unit_id, timepoint, img_path, is_augmented |
| `outputs/embeddings/global.npy` | `04_extract_embeddings.py` | (N, 768) global CLS |
| `outputs/embeddings/patch_f16.npy` | `04_extract_embeddings.py` | (N, 256, 768) patch grid |
| `outputs/r4pde/severity_per_image.csv` | `10_r4pde_severity.R` | per-image % severity |
| `outputs/spatial/spatial_per_image.csv` | `09_spatial.py` | Moran's I, heterogeneity, lesion metrics |
| `frozen/frozen_results.json` | `16_freeze.py` | every number cited in the manuscript |

## DINOv2 weights

Download the ViT-B/14 release once and point the loader to a local path:

```bash
mkdir -p models && cd models
curl -L -o dinov2_vitb14.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
```

Then set `DINOV2_WEIGHTS=models/dinov2_vitb14.pth` in your shell before running `04_extract_embeddings.py`.

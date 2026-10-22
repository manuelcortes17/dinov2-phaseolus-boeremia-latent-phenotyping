"""01_build_index.py — Master index for all images.
Backbone table linking every image to accession / biological unit / treatment /
timepoint / augmentation ancestry, with grouping keys for leakage-safe CV and
mixed models. Also emits the data-audit summary (Agent 1)."""
import os, re, json, numpy as np, pandas as pd

ROOT = "/home/claude/proj"
ORIG_DIR = f"{ROOT}/data/resized/orig"
AUG_DIR  = f"{ROOT}/data/resized/aug"
IDX_ORIG = "/mnt/user-data/uploads/Prueba/_pipeline_tmp/index_orig.csv"
MANIFEST = f"{ROOT}/data/manifest_fullres.csv"
OUT = f"{ROOT}/outputs/index"; os.makedirs(OUT, exist_ok=True)

def norm(s): return re.sub(r"[\\/ ]", "_", str(s))
def tp_int(e):
    return {1:"T1",2:"T2",3:"T3"}.get(int(e), "?")

# ---------- originals ----------
o = pd.read_csv(IDX_ORIG)
# fix ambiguous timepoints from full relpath
mask = o.timepoint == "?"
o.loc[mask, "timepoint"] = o.loc[mask, "relpath"].str.lower().map(
    lambda s: "T3" if "tercer" in s else ("T2" if "segunda" in s else ("T1" if "primera" in s else "?")))
o["is_augmented"] = False
o["transform"] = "orig"
o["origen"] = o["relpath"]
o["img_path"] = o["key"].map(lambda k: f"{ORIG_DIR}/{k}")
o["exists"] = o["img_path"].map(os.path.exists)

# ---------- augmented ----------
man = pd.read_csv(MANIFEST)
man["nkey"] = man["archivo"].map(norm)
aug_files = sorted(os.listdir(AUG_DIR))
arows = []
man_by = {r.nkey: r for r in man.itertuples(index=False)}
for f in aug_files:
    mrow = man_by.get(f)
    if mrow is None:
        # fallback: parse transform + unit from filename
        acc = None; unit = None; ev = None; tr = f.split("__")[-1].replace(".jpg", "")
    else:
        acc = mrow.accesion; unit = str(mrow.id_planta); ev = mrow.evaluacion; tr = mrow.transform
    arows.append(dict(key=f, relpath=f, accession=acc, unit=unit,
                      treatment="control",
                      timepoint=tp_int(ev) if ev is not None and not pd.isna(ev) else "?",
                      filename=f, is_copy=False, is_augmented=True, transform=tr,
                      origen=(mrow.origen if mrow is not None else ""),
                      img_path=f"{AUG_DIR}/{f}", exists=True))
a = pd.DataFrame(arows)

cols = ["key","relpath","accession","unit","treatment","timepoint","filename",
        "is_copy","is_augmented","transform","origen","img_path","exists"]
df = pd.concat([o[cols], a[cols]], ignore_index=True)

# biological unit + grouping
df["biological_unit_id"] = df["accession"].astype(str) + "|" + df["unit"].astype(str)
df["group_id"] = df["biological_unit_id"]
df["image_id"] = ["img%05d" % i for i in range(len(df))]

# analysis flags
df["is_orig_transform"] = (df["transform"] == "orig")
# unsupervised set: originals only, drop copies and unknown-timepoint
df["use_unsup"] = (~df["is_augmented"]) & (~df["is_copy"]) & (df["timepoint"] != "?")
# predictive/balanced set: originals + true augmentations (exclude aug 'orig' dupes and copies)
df["use_pred"] = (~df["is_copy"]) & (df["timepoint"] != "?") & (~((df["is_augmented"]) & (df["is_orig_transform"])))

df.to_csv(f"{OUT}/master_index.csv", index=False)

# ---------- audit summary ----------
audit = {}
audit["n_images_total_files"] = int(len(df))
audit["n_original_images"] = int((~df.is_augmented).sum())
audit["n_augmented_files_present"] = int((df.is_augmented).sum())
audit["n_manifest_rows"] = int(len(man))
audit["n_augmented_missing_vs_manifest"] = int(len(man) - (df.is_augmented).sum())
audit["n_copies_dropped"] = int(df.is_copy.sum())
audit["n_unknown_timepoint"] = int((df.timepoint == "?").sum())
audit["n_missing_files"] = int((~df.exists).sum())
audit["n_biological_units"] = int(df.biological_unit_id.nunique())
audit["n_units_original"] = int(df.loc[~df.is_augmented, "biological_unit_id"].nunique())
audit["orig_native_resolution"] = "3648x2736 (resized to <=1024 long side for compute)"
tab = (df[~df.is_augmented & ~df.is_copy & (df.timepoint!='?')]
       .groupby(["accession","treatment","timepoint"]).size().rename("n").reset_index())
tab.to_csv(f"{OUT}/audit_counts_originals.csv", index=False)
# per unit images (technical replication)
tech = (df[~df.is_augmented].groupby(["accession","treatment","biological_unit_id","timepoint"])
        .size().rename("imgs_per_unit_time").reset_index())
tech.to_csv(f"{OUT}/audit_tech_replication.csv", index=False)
audit["imgs_per_unit_time_median"] = float(tech.imgs_per_unit_time.median())
audit["imgs_per_unit_time_mean"] = round(float(tech.imgs_per_unit_time.mean()),3)

# transforms present
audit["aug_transforms"] = df[df.is_augmented]["transform"].value_counts().to_dict()
audit["unsup_n"] = int(df.use_unsup.sum())
audit["pred_n"] = int(df.use_pred.sum())

json.dump(audit, open(f"{OUT}/audit_summary.json","w"), indent=2, ensure_ascii=False)
print(json.dumps(audit, indent=2, ensure_ascii=False))
print("\nCounts (originals, analysis set):")
print(tab.to_string(index=False))

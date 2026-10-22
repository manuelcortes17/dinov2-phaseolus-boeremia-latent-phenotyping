"""04_extract_embeddings.py — Frozen DINOv2 ViT-B/14 global (CLS) + patch embeddings
for every image in the master index. Writes memmaps + order table. Resumable-safe
(single pass, incremental flush, progress log)."""
import sys, os, time, json, numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/claude/proj/scripts")
import dino_utils as D
from PIL import Image

ROOT = "/home/claude/proj"; OUT = f"{ROOT}/outputs/embeddings"; os.makedirs(OUT, exist_ok=True)
LOG = f"{ROOT}/logs/extract.log"; os.makedirs(f"{ROOT}/logs", exist_ok=True)
SIZE, GS, D_EMB, BATCH = 224, 16, 768, 24
def log(m):
    open(LOG, "a").write(f"[{time.strftime('%H:%M:%S')}] {m}\n")

torch.set_num_threads(2); torch.manual_seed(0)
df = pd.read_csv(f"{ROOT}/outputs/index/master_index.csv")
df = df[df.exists].reset_index(drop=True)
N = len(df); log(f"START N={N}")
df[["image_id","key","img_path","accession","treatment","timepoint",
    "biological_unit_id","is_augmented","transform","use_unsup","use_pred"]].to_csv(f"{OUT}/order.csv", index=False)

g_mm = np.lib.format.open_memmap(f"{OUT}/global.npy", mode="w+", dtype="float32", shape=(N, D_EMB))
p_mm = np.lib.format.open_memmap(f"{OUT}/patch_f16.npy", mode="w+", dtype="float16", shape=(N, GS*GS, D_EMB))

model, info = D.build_model(); tf = D.make_transform(SIZE)
log(f"model {info}")
t0 = time.time()
for s in range(0, N, BATCH):
    paths = df.img_path.iloc[s:s+BATCH].tolist()
    xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    with torch.no_grad():
        out = model.forward_features(xs)
    g_mm[s:s+len(paths)] = out["x_norm_clstoken"].numpy().astype("float32")
    p_mm[s:s+len(paths)] = out["x_norm_patchtokens"].numpy().astype("float16")
    if (s // BATCH) % 10 == 0:
        el = time.time() - t0; rate = (s+len(paths)) / max(el,1e-6)
        log(f"{s+len(paths)}/{N} {el:.0f}s {rate:.2f} img/s eta {(N-s-len(paths))/max(rate,1e-6):.0f}s")
g_mm.flush(); p_mm.flush()
json.dump(dict(N=int(N), grid=GS, dim=D_EMB, size=SIZE, model=info,
               patch_dtype="float16"), open(f"{OUT}/meta.json","w"), indent=2)
log("DONE")
print("DONE", N)

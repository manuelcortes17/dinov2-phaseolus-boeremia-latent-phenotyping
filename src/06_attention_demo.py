"""06_attention_demo.py — Composite DINOv2 attention panel (reference aesthetic).
Columns: patch grid | attn CLS (last) | attn mean(layers) | attn var(layers) |
head 0..3 | attention rollout. Rows: representative leaves."""
import sys, os, numpy as np
sys.path.insert(0, "/home/claude/proj/scripts")
import dino_utils as D
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import zoom, gaussian_filter

SIZE = 224; GS = 16
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/proj/outputs/attention/attn_demo.png"
DPI = int(sys.argv[2]) if len(sys.argv) > 2 else 170

ROWS = [
 ("ASC 160 · inoculated · T3", "/mnt/user-data/uploads/Prueba/ASC 160-20260626T233951Z-3-001/ASC 160/Asc 01/Tercera evaluación/IMG_8581.JPG"),
 ("CAL 96 · inoculated · T3",  "/mnt/user-data/uploads/Prueba/CAL 96-20260626T234035Z-3-001/CAL 96/Asc 01/Tercera evaluación/IMG_8611.JPG"),
 ("ASC 160 · control · T1",    "/mnt/user-data/uploads/Prueba/ASC 160-20260626T233951Z-3-001/ASC 160/Testigos 2/Testigo 1º evaluación.JPG"),
 ("ASC 160 · inoculated · T1", "/mnt/user-data/uploads/Prueba/ASC 160-20260626T233951Z-3-001/ASC 160/Asc 01/Primera evaluación/IMG_8335.JPG"),
]
COLS = ["patch grid", "attn CLS (last layer)", "attn mean (layers)",
        "attn variance (layers)", "head 0", "head 1", "head 2", "head 3",
        "attention rollout"]

def leaf_mask(disp):
    r, g, b = disp[..., 0], disp[..., 1], disp[..., 2]
    bright = disp.mean(-1)
    green = (g >= r * 0.92) & (g >= b * 0.85)     # green / chlorotic / necrotic leaf
    notbg = bright < 0.72                           # exclude light-gray background
    m = (green & notbg).astype(np.float32)
    m = gaussian_filter(m, sigma=2.0)
    return np.clip(m * 1.3, 0, 1)

def up(a16, mask, sigma=6.0, gamma=1.15):
    a = zoom(a16.reshape(GS, GS).astype(np.float32), SIZE / GS, order=3)
    a = gaussian_filter(a, sigma=sigma) * mask
    inside = mask > 0.15
    if inside.sum() > 0:
        a = a - a[inside].min()
        a = np.clip(a / (a[inside].max() + 1e-8), 0, 1)
    return a ** gamma

def overlay(disp, heat, cmap="jet", alpha=0.6):
    rgba = cm.get_cmap(cmap)(heat)[..., :3]
    return np.clip(alpha * rgba + (1 - alpha) * disp, 0, 1)

def grid_panel(disp):
    img = disp.copy()
    for k in range(0, SIZE + 1, SIZE // GS):
        kk = min(k, SIZE - 1)
        img[kk, :, :] = [1, 1, 0.0]; img[:, kk, :] = [1, 1, 0.0]
    return img

model, info = D.build_model()
tf = D.make_transform(SIZE)

n_r, n_c = len(ROWS), len(COLS)
fig, ax = plt.subplots(n_r, n_c, figsize=(n_c * 1.55, n_r * 1.62))
plt.subplots_adjust(left=0.055, right=0.998, top=0.93, bottom=0.005, wspace=0.03, hspace=0.03)

for r, (lab, path) in enumerate(ROWS):
    im, disp = D.load_rgb(path, SIZE)
    x = tf(im).unsqueeze(0)
    res = D.extract(model, x)
    ca = res["cls_attn"]                       # (L,H,256)
    last_mean = ca[-1].mean(0)                 # mean heads, last layer
    layer_mean = ca.mean(1)                    # (L,256) mean heads per layer
    mean_layers = layer_mean.mean(0)           # mean across layers
    var_layers = layer_mean.var(0)             # variance across layers
    heads = ca[-1]                             # (H,256) last layer per head
    roll = res["rollout"]
    msk = leaf_mask(disp)
    panels = [grid_panel(disp),
              overlay(disp, up(last_mean, msk)),
              overlay(disp, up(mean_layers, msk)),
              overlay(disp, up(var_layers, msk, sigma=5), cmap="magma", alpha=0.62),
              overlay(disp, up(heads[0], msk)), overlay(disp, up(heads[1], msk)),
              overlay(disp, up(heads[2], msk)), overlay(disp, up(heads[3], msk)),
              overlay(disp, up(roll, msk))]
    for c in range(n_c):
        ax[r, c].imshow(panels[c]); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        for s in ax[r, c].spines.values(): s.set_visible(False)
        if r == 0:
            ax[r, c].set_title(COLS[c], fontsize=6.6, pad=3)
    ax[r, 0].set_ylabel(lab, fontsize=6.6, rotation=90, labelpad=2)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
print("saved", OUT, "info", info)

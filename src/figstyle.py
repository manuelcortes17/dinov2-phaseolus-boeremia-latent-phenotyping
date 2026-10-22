"""figstyle.py — shared publication style (Nature-like) for all figures."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

DPI = 650
MM = 1/25.4
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":7,"axes.labelsize":7.5,"axes.titlesize":7.5,
    "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.5,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
    "xtick.major.size":2.5,"ytick.major.size":2.5,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":DPI,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "legend.frameon":False,"axes.grid":False,"pdf.fonttype":42,"ps.fonttype":42,
})
# palettes
ACC   = {"ASC 160":"#3B6FB6","CAL 96":"#E08214"}
TREAT = {"control":"#4DAF4A","inoculated":"#B2182B"}
TP    = {"T1":"#66C2A5","T2":"#3690C0","T3":"#08306B"}
# biological severity colormap: healthy green -> chlorotic yellow -> necrotic brown
SEVMAP = LinearSegmentedColormap.from_list("sev",
    ["#2E7D32","#9CCC65","#F9D423","#E08214","#8B3A1B"])

def panel_label(ax, s, x=-0.16, y=1.12, fs=8.5):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="top", ha="right")

def clean(ax):
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    return ax

def noticks(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    return ax

# three-group scheme (two accessions + control)
GRP3 = {"ASC 160":"#3B6FB6", "CAL 96":"#E08214", "Control":"#3A9D6E"}

def confidence_ellipse(x, y, ax, n_std=2.0, fc="none", **kw):
    import numpy as np
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    if len(x) < 3: return
    cov = np.cov(x, y); pear = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]+1e-12)
    rx = np.sqrt(1+pear); ry = np.sqrt(1-pear)
    ell = Ellipse((0,0), width=rx*2, height=ry*2, facecolor=fc, **kw)
    sx = np.sqrt(cov[0,0])*n_std; sy = np.sqrt(cov[1,1])*n_std
    tr = transforms.Affine2D().rotate_deg(45).scale(sx, sy).translate(np.mean(x), np.mean(y))
    ell.set_transform(tr + ax.transData); ax.add_patch(ell)

def hull(x, y, ax, color, alpha=0.10, lw=1.2):
    import numpy as np
    from scipy.spatial import ConvexHull
    pts = np.column_stack([x, y])
    if len(pts) < 3: return
    h = ConvexHull(pts); v = np.append(h.vertices, h.vertices[0])
    ax.fill(pts[v,0], pts[v,1], color=color, alpha=alpha, zorder=0)
    ax.plot(pts[v,0], pts[v,1], color=color, lw=lw, alpha=0.8, zorder=1)

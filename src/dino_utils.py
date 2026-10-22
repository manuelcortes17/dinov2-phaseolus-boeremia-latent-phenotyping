"""
dino_utils.py — Frozen DINOv2 ViT-B/14 loader + feature/attention extraction.
Loads architecture from torch.hub (facebookresearch/dinov2) with pretrained=False
and injects the LOCAL pretrained weights (dinov2_vitb14_pretrain.pth). CPU only.
"""
import os, torch, numpy as np
from PIL import Image
import torchvision.transforms as T

WEIGHTS = os.environ.get("DINO_WEIGHTS",
    "/mnt/user-data/uploads/Prueba/dinov2_vitb14_pretrain.pth")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
PATCH = 14

torch.manual_seed(0); np.random.seed(0)

DINO_PKG = os.environ.get("DINO_PKG", os.path.expanduser("~/dinov2_pkg"))

def build_model():
    if DINO_PKG not in os.sys.path:
        os.sys.path.insert(0, DINO_PKG)
    from dinov2.models.vision_transformer import vit_base
    model = vit_base(patch_size=14, img_size=518, init_values=1.0,
                     block_chunks=0, num_register_tokens=0,
                     interpolate_antialias=False, interpolate_offset=0.1,
                     ffn_layer="mlp")
    sd = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and "blocks.0.attn.qkv.weight" not in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    torch.set_grad_enabled(False)
    info = dict(n_blocks=len(model.blocks),
                n_heads=model.blocks[0].attn.num_heads,
                embed_dim=model.embed_dim,
                missing=len(missing), unexpected=len(unexpected))
    return model, info

def make_transform(size=224):
    # square resize (whole leaf preserved), matching reference grid over full leaf
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_rgb(path, size=224):
    im = Image.open(path).convert("RGB")
    disp = np.asarray(im.resize((size, size), Image.BICUBIC)).astype(np.float32) / 255.0
    return im, disp

class AttnGrabber:
    """Monkeypatch every block.attn to expose explicit softmax attention weights."""
    def __init__(self, model):
        self.model = model; self.store = {}; self._orig = {}
    def __enter__(self):
        for i, blk in enumerate(self.model.blocks):
            a = blk.attn; self._orig[i] = a.forward
            a.forward = self._mk(a, i)
        return self
    def _mk(self, a, idx):
        def fwd(x):
            B, N, C = x.shape
            qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * a.scale
            attn = attn.softmax(dim=-1)
            self.store[idx] = attn.detach()  # (B, heads, N, N)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = a.proj(x); x = a.proj_drop(x)
            return x
        return fwd
    def __exit__(self, *exc):
        for i, blk in enumerate(self.model.blocks):
            blk.attn.forward = self._orig[i]

@torch.no_grad()
def extract(model, x, n_reg=0):
    """x: (1,3,H,W). Returns dict with cls, patch tokens (grid), and attention tensors."""
    with AttnGrabber(model) as g:
        out = model.forward_features(x)  # dict for dinov2 hub model
    cls = out["x_norm_clstoken"][0].cpu().numpy()          # (768,)
    patch = out["x_norm_patchtokens"][0].cpu().numpy()     # (256,768)
    gs = int(round(patch.shape[0] ** 0.5))
    L = len(g.store); H = model.blocks[0].attn.num_heads
    n_prefix = 1 + n_reg
    # CLS->patch attention per layer per head: (L,H,Npatch)
    cls_attn = np.stack([g.store[i][0, :, 0, n_prefix:].cpu().numpy() for i in range(L)])  # (L,H,P)
    # attention rollout across layers (avg heads, add residual)
    roll = None
    N = g.store[0].shape[-1]
    R = np.eye(N)
    for i in range(L):
        A = g.store[i][0].mean(0).cpu().numpy()  # (N,N) avg heads
        A = A + np.eye(N)
        A = A / A.sum(-1, keepdims=True)
        R = A @ R
    roll = R[0, n_prefix:]  # CLS row -> patches
    return dict(cls=cls, patch=patch, grid=gs, n_heads=H, n_layers=L,
                cls_attn=cls_attn, rollout=roll)

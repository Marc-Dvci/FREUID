"""
Forensic streams: template-agnostic detectors for the unseen-document-type objective.

A semantic RGB backbone learns "what a Mauritius ID looks like" and fails on the 2 unseen
private types. Forensic branches instead key on *device / manipulation / recapture statistics*
that are largely document-independent, so they transfer across templates:

  * noise branch -- SRM high-pass residuals (fixed) + a learnable constrained BayarConv. Captures
                    splicing/editing/recapture noise inconsistencies.
  * dct  branch -- fixed 8x8 block-DCT (CAT-Net-style): double-compression / recapture leave
                    characteristic DCT-coefficient statistics.

Both consume the SAME normalized 3-channel RGB tensor the dataset already produces, so they are
drop-in ensemble members for train.py / inference.py via the "forensic_*:" backbone prefix.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ---- three classic SRM high-pass kernels (5x5) ----
_SRM = [
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4.0,
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32) / 2.0,
]


class SRMConv2d(nn.Module):
    """Fixed SRM residual extractor: 3 filters -> 3 residual channels (averaged over RGB)."""
    def __init__(self):
        super().__init__()
        w = np.zeros((3, 3, 5, 5), np.float32)
        for o in range(3):
            for c in range(3):
                w[o, c] = _SRM[o] / 3.0
        self.register_buffer("weight", torch.from_numpy(w))

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2)


class BayarConv2d(nn.Module):
    """Learnable constrained conv (Bayar & Stamm): center = -1, remaining weights sum to 1."""
    def __init__(self, in_ch=3, out_ch=3, k=5):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * 0.01)

    def _constrain(self):
        w = self.weight.clone()
        c = self.k // 2
        w[:, :, c, c] = 0.0
        w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-8)   # remaining weights sum to 1
        w[:, :, c, c] = -1.0
        return w

    def forward(self, x):
        return F.conv2d(x, self._constrain(), padding=self.k // 2)


def _dct_basis(n=8):
    """8x8 separable DCT-II basis -> conv weight (n*n, 1, n, n)."""
    k = np.arange(n)
    b = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    b[0] *= 1 / np.sqrt(2)
    b *= np.sqrt(2.0 / n)                       # (n, n) 1-D basis
    filt = np.einsum("ux,vy->uvxy", b, b).reshape(n * n, 1, n, n).astype(np.float32)
    return torch.from_numpy(filt)


class DCTStem(nn.Module):
    """Fixed 8x8 block DCT on luminance -> 64 coefficient maps at H/8."""
    def __init__(self, n=8):
        super().__init__()
        self.n = n
        self.register_buffer("basis", _dct_basis(n))
        self.register_buffer("rgb2y", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, x):
        y = (x * self.rgb2y).sum(1, keepdim=True)
        return F.conv2d(y, self.basis, stride=self.n)        # (B, 64, H/8, W/8)


class ForensicNet(nn.Module):
    """variant in {'noise','dct'}: forensic stem -> timm backbone -> single logit."""
    def __init__(self, variant="noise", backbone="convnextv2_nano.fcmae_ft_in22k_in1k",
                 pretrained=True, drop_rate=0.2, grad_checkpointing=False, **_):
        super().__init__()
        self.variant = variant
        if variant == "noise":
            self.srm = SRMConv2d()
            self.bayar = BayarConv2d(3, 3, 5)
            in_chans = 6
        elif variant == "dct":
            self.dct = DCTStem(8)
            in_chans = 64
        else:
            raise ValueError(variant)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          in_chans=in_chans, drop_rate=drop_rate)
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(self.backbone.num_features, 1))

    def _stem(self, x):
        if self.variant == "noise":
            return torch.cat([self.srm(x), self.bayar(x)], dim=1)
        return self.dct(x)

    def forward(self, x):
        return self.head(self.backbone(self._stem(x))).squeeze(-1)


def create_forensic_model(model_name, pretrained=True, **kw):
    """model_name like 'forensic_noise' or 'forensic_dct:convnextv2_nano.fcmae_ft_in22k_in1k'."""
    spec = model_name[len("forensic_"):]
    variant, _, backbone = spec.partition(":")
    if not backbone:
        backbone = "convnextv2_nano.fcmae_ft_in22k_in1k"
    return ForensicNet(variant=variant, backbone=backbone, pretrained=pretrained, **kw)


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    for name in ("forensic_noise:convnextv2_nano.fcmae_ft_in22k_in1k",
                 "forensic_dct:convnextv2_nano.fcmae_ft_in22k_in1k"):
        m = create_forensic_model(name, pretrained=False)
        out = m(x)
        print(f"{name:55s} -> logits {tuple(out.shape)}")
    # verify Bayar constraint holds after _constrain()
    b = BayarConv2d(3, 3, 5)
    w = b._constrain()
    c = 5 // 2
    print("Bayar center == -1:", torch.allclose(w[:, :, c, c], torch.tensor(-1.0)))
    print("Bayar non-center sum == 1:", torch.allclose(
        w.sum(dim=(2, 3)) + 1.0, torch.tensor(1.0), atol=1e-5))  # sum(all)= -1+1 =0 => +1 ==1

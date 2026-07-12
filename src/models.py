"""
Models for FREUID.

* FREUIDModel        -- single-stream timm backbone + small head (the workhorse for global-image
                        fine-tuning of foundation backbones: DINOv2 / EVA-02 / ConvNeXt-V2 / etc.).
* AttentionMILModel  -- high-res patch branch: score each patch with a shared backbone, aggregate
                        with gated attention pooling. Patches are document-agnostic (helps
                        cross-document generalization) and memory-friendly on 12 GB.

All models output a single logit per image (BCEWithLogitsLoss); apply sigmoid for scores.

Recommended backbones (timm names):
  vit_large_patch14_reg4_dinov2.lvd142m   (DINOv2, strong cross-domain)
  vit_base_patch14_reg4_dinov2.lvd142m    (lighter, fits 12 GB at higher res)
  eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
  convnextv2_large.fcmae_ft_in22k_in1k_384 / convnextv2_base...
"""
import torch
import torch.nn as nn
import timm


def _make_head(in_features, num_classes=1, kind="linear", drop_rate=0.2):
    if kind == "linear":
        return nn.Sequential(nn.Dropout(drop_rate), nn.Linear(in_features, num_classes))
    # small mlp head
    return nn.Sequential(
        nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.GELU(),
        nn.Dropout(drop_rate), nn.Linear(512, num_classes),
    )


class FREUIDModel(nn.Module):
    def __init__(self, model_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
                 pretrained=True, num_classes=1, drop_rate=0.2, head="linear",
                 grad_checkpointing=False, img_size=None):
        super().__init__()
        model_kwargs = dict(pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        if img_size is not None and _uses_fixed_image_size(model_name):
            model_kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **model_kwargs)
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        self.num_features = self.backbone.num_features
        self.head = _make_head(self.num_features, num_classes, head, drop_rate)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)  # logits (B,)

    @torch.no_grad()
    def features(self, x):
        return self.backbone(x)


def _uses_fixed_image_size(model_name):
    name = model_name.lower()
    return any(token in name for token in ("vit", "deit", "beit", "eva"))


class AttentionMILModel(nn.Module):
    """Gated-attention MIL over patches. Input x: (B, N, C, H, W)."""
    def __init__(self, model_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
                 pretrained=True, attn_dim=256, drop_rate=0.2, grad_checkpointing=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        d = self.backbone.num_features
        self.attn_V = nn.Linear(d, attn_dim)
        self.attn_U = nn.Linear(d, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.head = _make_head(d, 1, "linear", drop_rate)

    def forward(self, x):
        B, N, C, H, W = x.shape
        feats = self.backbone(x.view(B * N, C, H, W)).view(B, N, -1)   # (B, N, d)
        a = self.attn_w(torch.tanh(self.attn_V(feats)) * torch.sigmoid(self.attn_U(feats)))  # (B,N,1)
        a = torch.softmax(a, dim=1)
        pooled = (a * feats).sum(dim=1)                                # (B, d)
        return self.head(pooled).squeeze(-1)                           # logits (B,)


def create_model(model_name="convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=True, **kw):
    if model_name.startswith("forensic_"):
        from forensic_streams import create_forensic_model
        kw.pop("head", None)  # forensic head is fixed
        return create_forensic_model(model_name, pretrained=pretrained, **kw)
    return FREUIDModel(model_name=model_name, pretrained=pretrained, **kw)


if __name__ == "__main__":
    m = create_model("convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    print("single-stream logits:", out.shape)
    mil = AttentionMILModel("convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=False)
    out2 = mil(torch.randn(2, 4, 3, 224, 224))
    print("MIL logits:", out2.shape)

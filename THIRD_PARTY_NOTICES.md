# Third-party data, models, and software notices

The repository source written for this entry is licensed under the MIT License. That license does
not relicense competition data, external datasets, pretrained parameters, or their upstream works.
The following resources were used by at least one frozen selected checkpoint.

## Data

- **FREUID Challenge 2026 data** — supplied by Microblink through Kaggle under the competition's
  non-commercial research terms. It is never redistributed in this repository or Docker image.
- **MIDV-Holo** — Sheshkus et al., *MIDV-Holo: A New Dataset for Identity Document Analysis and
  Hologram Detection*, CC BY-SA 2.5. Source: <https://github.com/SmartEngines/midv-holo>.
  The selected DINOv2 checkpoint used 15,179 derived frames for training; external images are not
  redistributed. The source dataset incorporates faces from Generated Photos; attribution and
  source details are provided by the MIDV-Holo authors.

## Pretrained parameters

- **DINOv2 ViT-B/14 with registers**, timm identifier
  `vit_base_patch14_reg4_dinov2.lvd142m` — upstream DINOv2 project and pretrained parameters are
  published under Apache-2.0. Source: <https://github.com/facebookresearch/dinov2>.
- **ConvNeXt V2 Base FCMAE**, timm identifier
  `convnextv2_base.fcmae_ft_in22k_in1k_384` — Meta pretrained parameters are published under
  CC BY-NC 4.0; the upstream implementation is MIT. Source:
  <https://github.com/facebookresearch/ConvNeXt-V2>.

Redistributed `.pth` files contain fine-tuned parameters and are provided solely for competition
reproduction and non-commercial research. Users remain responsible for the upstream terms. No
checkpoint is claimed to be covered by this repository's MIT source-code license.

## Runtime libraries

The container uses PyTorch/torchvision, timm, Albumentations, OpenCV, NumPy, pandas, SciPy, Pillow,
and their pinned dependencies. Their respective upstream licenses remain in force. Package versions
are fixed in `docker-requirements.txt`; PyTorch and torchvision are supplied by the immutable base
image recorded in `Dockerfile`.

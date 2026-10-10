# SG-DWIF: RGB–DSM Semantic Segmentation for Remote Sensing

[Chinese README](README.md)

SG-DWIF is a semantic segmentation model that takes an RGB image and a single-channel digital surface model (DSM) as input and produces logits for six classes. The Python module is `SG_DWIF.py`, and the model class is `SGDWIF`. The current DSM encoder uses **shared modules** through `WTConvNeXt()`.

## Architecture

1. **RGB branch:** A SAM2 Hiera-L backbone with its original parameters frozen. Each Hiera block is wrapped with an adapter controlled by a learnable scalar gate.
2. **DSM branch:** A single-channel WTConvNeXt encoder that produces geometry features at four scales.
3. **Four-scale fusion:** Semantic-Guided Directional Wavelet Interaction Fusion (SG-DWIF), combining directional wavelet enhancement, semantic-guided subband selection, and reliability gating.
4. **Decoder:** The `Decoder` imported from `WTSAM.py` progressively merges the fused features and restores the input resolution. See the UNetFormer-based decoder used by the original project.

Call the model as `logits = model(rgb, dsm)`. The expected shapes are `[B, 3, H, W]` for `rgb`, `[B, 1, H, W]` for `dsm`, and `[B, 6, H, W]` for the default output. Both inputs must have the same batch size and spatial dimensions. The bundled `sam2_hiera_l.yaml` configuration requires the original **SAM2 Hiera-L** checkpoint, not a SAM2.1 checkpoint.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `SG_DWIF.py` | Main model, gated adapters, and SG-DWIF fusion |
| `wtresnet.py`, `wtconv2d.py`, `wavelet.py` | DSM encoder and wavelet operations |
| `sam2/`, `sam2_configs/` | Local SAM2 implementation and configuration used by this model |
| `utils.py` | Dataset and metric utilities from the original project |
| `third_party/` | Third-party license text |

**Required external source:** `SG_DWIF.py` imports `Decoder` from `WTSAM.py`, but `WTSAM.py` is not included in this directory. Add the decoder and its dependencies from the original project before importing the model. The original training script, `train.py`, is also not included here. Datasets, pretrained checkpoints, and training outputs are not bundled.

## Installation

Python 3.10 or 3.11 and an NVIDIA CUDA GPU are recommended. The model has been checked with Python 3.11, PyTorch 2.1.1 + CUDA 12.1, torchvision 0.16.1, and timm 1.0.28. Install a matching `torch` and `torchvision` build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
cd SG-DWIF-Git
python -m pip install -r requirements.txt
```

The bundled SAM2 builder currently initializes the model on CUDA, so a CUDA GPU is required. The optional compiled extension for SAM2 video post-processing is not bundled; SG-DWIF uses the Hiera image backbone.

## Checkpoints

Download the original `sam2_hiera_large.pt` checkpoint from the [Meta SAM2 project](https://github.com/facebookresearch/sam2) and place it at `Pweight/sam2_hiera_large.pt`. The original checkpoint URL is:

```text
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

The SAM2 checkpoint initializes the RGB backbone. A separately trained SG-DWIF segmentation checkpoint is needed for meaningful predictions; no such checkpoint is included here.

## Loading a Trained Segmentation Model

After adding the required `WTSAM.py` decoder, load a trained checkpoint as follows:

```python
import torch
from SG_DWIF import SGDWIF

model = SGDWIF(checkpoint_path="Pweight/sam2_hiera_large.pt")
weights = torch.load("result_TGRS/your_model_checkpoint", map_location="cpu")
if isinstance(weights, dict) and "model" in weights:
    weights = weights["model"]
weights = {key.removeprefix("module."): value for key, value in weights.items()}
model.load_state_dict(weights, strict=True)
model = model.cuda().eval()

rgb = torch.rand(1, 3, 512, 512, device="cuda")
dsm = torch.rand(1, 1, 512, 512, device="cuda")
with torch.inference_mode():
    logits = model(rgb, dsm)
    labels = logits.argmax(dim=1)
```

The random inputs above demonstrate the interface only. For real inference, use the same RGB and DSM preprocessing as during training.

## Potsdam Training

The original project uses a configuration-based `train.py` script, which is **not included in this directory**. Its default dataset root is `ISPRS_semantic_labeling_Potsdam/`, and its default SAM2 checkpoint path is `Pweight/sam2_hiera_large.pt`. The expected dataset subdirectories are:

```text
ISPRS_semantic_labeling_Potsdam/
├── 4_Ortho_RGBIR/
├── 1_DSM_normalisation/
└── 5_Labels_for_participants/
```

If you use the original training script, adjust `CUDA_VISIBLE_DEVICES`, `BATCH_SIZE`, and `WINDOW_SIZE` for your hardware before running `python train.py`. Its original settings use 512 × 512 crops and a batch size of 16, which will usually need to be reduced on a single consumer GPU. `INITIAL_BEST_OA = 90.0` is the initial checkpoint-saving threshold; set it to `0.0` to allow saving from the first evaluation. The six Potsdam classes and their order are defined in `utils.py`.

## Results on the Potsdam Dataset

| Model | OA | Roads | Buildings | Low veg. | Trees | Cars | Clutter | mF1 | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ABCNet | 89.57 ± 0.29 | 92.12 ± 1.32 | 96.79 ± 0.76 | 87.42 ± 0.49 | 83.96 ± 3.15 | 93.36 ± 1.83 | 53.75 ± 1.91 | 90.48 ± 0.58 | 82.93 ± 0.98 |
| ASMFNet | 88.75 ± 0.02 | 91.44 ± 0.23 | 96.98 ± 0.01 | 84.78 ± 0.49 | 82.84 ± 0.57 | 93.80 ± 0.08 | 54.80 ± 1.53 | 89.41 ± 0.01 | 81.35 ± 0.04 |
| UNetFormer | 90.16 ± 0.18 | 91.66 ± 0.69 | 97.09 ± 0.16 | 87.92 ± 0.99 | 85.66 ± 0.45 | 96.41 ± 0.06 | 59.52 ± 1.49 | 91.33 ± 0.23 | 84.42 ± 0.37 |
| FTransUNet | 89.94 ± 0.21 | 92.53 ± 0.18 | 97.09 ± 0.32 | 86.21 ± 0.02 | 85.43 ± 0.88 | 95.62 ± 0.21 | 57.72 ± 1.36 | 90.96 ± 0.19 | 83.83 ± 0.30 |
| CMFNet | 90.16 ± 0.06 | 92.76 ± 0.18 | 97.27 ± 0.33 | 87.20 ± 0.23 | 85.74 ± 1.14 | 95.69 ± 0.16 | 54.12 ± 2.77 | 91.29 ± 0.09 | 84.35 ± 0.14 |
| SAM2MSNet | 84.82 ± 0.09 | 87.49 ± 0.43 | 94.83 ± 0.17 | 82.00 ± 1.39 | 75.44 ± 1.26 | 94.15 ± 0.35 | 44.77 ± 2.18 | 85.97 ± 0.12 | 76.15 ± 0.20 |
| Ours (shared) | 91.18 ± 0.28 | 91.72 ± 0.41 | 97.93 ± 0.26 | 88.44 ± 0.35 | 88.95 ± 0.51 | 96.13 ± 0.32 | 63.28 ± 1.96 | 92.30 ± 0.14 | 86.01 ± 0.22 |
| Ours | 91.37 ± 0.21 | 92.82 ± 0.04 | 98.01 ± 0.27 | 88.93 ± 1.87 | 88.22 ± 1.10 | 95.94 ± 0.91 | 59.74 ± 2.81 | 92.32 ± 0.25 | 86.06 ± 0.45 |

## Third-Party Code and Licenses

- `sam2/` and `sam2_configs/` are based on [Meta SAM2](https://github.com/facebookresearch/sam2). Its Apache 2.0 license text is included at [`sam2/LICENSE`](sam2/LICENSE).
- The wavelet convolution implementation refers to [WTConv](https://github.com/BGU-CS-VIL/WTConv). Its MIT license text is included at [`third_party/WTConv_LICENSE`](third_party/WTConv_LICENSE).
- No repository-level license has been assigned to the original SG-DWIF code. The rights holder should select one and add a `LICENSE` file before public release.
## Important Statement
Except for the original code and experimental data, the overall organization and structure of the project files were arranged with the assistance of AI tools.
"""SG-DWIF model for RGB and DSM semantic segmentation."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import wavelet
from sam2.build_sam import build_sam2
from wtresnet import WTConvNeXt
from WTSAM import Decoder


class GatedAdapter(nn.Module):

    def __init__(self, block: nn.Module, gate_init: float = 3.0) -> None:
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU(),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt_learn(x)
        prompted = x + self.gate_value() * prompt
        return self.block(prompted)


def _group_count(channels: int, max_groups: int = 16) -> int:
    return math.gcd(channels, min(channels, max_groups))


class ConvNormAct(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class DirectionalBandEnhancer(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SemanticGuidedDirectionalWaveletFusion(nn.Module):

    def __init__(
        self,
        channels: int,
        fusion_channels: Optional[int] = None,
        strip_kernel: int = 7,
        layer_scale_init: float = 0.1,
        wavelet_type: str = "db1",
    ) -> None:
        super().__init__()
        if strip_kernel < 3 or strip_kernel % 2 == 0:
            raise ValueError("strip_kernel must be an odd integer >= 3")

        self.channels = channels
        self.fusion_channels = fusion_channels or min(max(channels // 4, 32), 128)
        inner = self.fusion_channels

        self.semantic_proj = ConvNormAct(channels, inner, kernel_size=1)
        self.geometry_proj = ConvNormAct(channels, inner, kernel_size=1)

        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(
            wavelet_type,
            inner,
            inner,
            torch.float32,
        )
        self.register_buffer("wt_filter", wt_filter, persistent=True)
        self.register_buffer("iwt_filter", iwt_filter, persistent=True)

        self.low_enhancer = DirectionalBandEnhancer(inner, (3, 3))
        self.lh_enhancer = DirectionalBandEnhancer(inner, (1, strip_kernel))
        self.hl_enhancer = DirectionalBandEnhancer(inner, (strip_kernel, 1))
        self.hh_enhancer = DirectionalBandEnhancer(inner, (3, 3))

        relation_channels = 4 * inner

        self.band_score = nn.Sequential(
            nn.Conv2d(relation_channels, inner, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(inner), inner),
            nn.GELU(),
            nn.Conv2d(inner, inner, kernel_size=1, bias=True),
        )

        self.low_gate = nn.Sequential(
            nn.Conv2d(relation_channels, inner, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(inner), inner),
            nn.GELU(),
            nn.Conv2d(inner, inner, kernel_size=1, bias=True),
        )

        self.reliability_gate = nn.Sequential(
            nn.Conv2d(relation_channels, inner, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(inner), inner),
            nn.GELU(),
            nn.Conv2d(
                inner,
                inner,
                kernel_size=3,
                padding=1,
                groups=inner,
                bias=False,
            ),
            nn.Conv2d(inner, inner, kernel_size=1, bias=True),
        )

        self.geometry_out = nn.Sequential(
            nn.Conv2d(inner, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )

        self._init_gates()

    def _init_gates(self) -> None:
        nn.init.zeros_(self.band_score[-1].weight)
        nn.init.zeros_(self.band_score[-1].bias)
        nn.init.zeros_(self.low_gate[-1].weight)
        nn.init.zeros_(self.low_gate[-1].bias)

        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(self.reliability_gate[-1].bias, -1.0)

    @staticmethod
    def _relation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat((a, b, a * b, torch.abs(a - b)), dim=1)

    @staticmethod
    def _pad_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (height, width)

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        filters = self.wt_filter.to(device=x.device, dtype=x.dtype)
        bands = F.conv2d(x, filters, stride=2, groups=channels)
        return bands.reshape(batch, channels, 4, height // 2, width // 2)

    def _iwt(
        self,
        bands: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        batch, channels, _, height, width = bands.shape
        filters = self.iwt_filter.to(device=bands.device, dtype=bands.dtype)
        x = bands.reshape(batch, channels * 4, height, width)
        x = F.conv_transpose2d(x, filters, stride=2, groups=channels)
        return x[..., : output_size[0], : output_size[1]]

    def forward(
        self,
        semantic: torch.Tensor,
        geometry: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if semantic.ndim != 4 or geometry.ndim != 4:
            raise ValueError("semantic and geometry features must both be BCHW tensors")
        if semantic.shape != geometry.shape:
            raise ValueError(
                "semantic and geometry features must have identical shapes, "
                f"got {tuple(semantic.shape)} and {tuple(geometry.shape)}"
            )
        if semantic.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {semantic.shape[1]}"
            )

        semantic_inner = self.semantic_proj(semantic)
        geometry_inner = self.geometry_proj(geometry)

        geometry_even, original_size = self._pad_even(geometry_inner)
        semantic_even, _ = self._pad_even(semantic_inner)
        semantic_half = F.avg_pool2d(semantic_even, kernel_size=2, stride=2)

        bands = self._dwt(geometry_even)
        ll, lh, hl, hh = bands.unbind(dim=2)

        low_delta = self.low_enhancer(ll)
        low_weight = torch.sigmoid(self.low_gate(self._relation(semantic_half, ll)))
        ll_refined = ll + low_weight * low_delta

        high_bands = (lh, hl, hh)
        high_deltas = (
            self.lh_enhancer(lh),
            self.hl_enhancer(hl),
            self.hh_enhancer(hh),
        )
        score_list = [
            self.band_score(self._relation(semantic_half, band_delta))
            for band_delta in high_deltas
        ]
        band_weights = torch.softmax(torch.stack(score_list, dim=2), dim=2)
        high_bands_tensor = torch.stack(high_bands, dim=2).contiguous()
        high_deltas_tensor = torch.stack(high_deltas, dim=2).contiguous()
        high_refined_tensor = (
            high_bands_tensor
            + band_weights.contiguous() * high_deltas_tensor
        )
        high_refined = high_refined_tensor.unbind(dim=2)

        refined_bands = torch.stack(
            (ll_refined, high_refined[0], high_refined[1], high_refined[2]),
            dim=2,
        )
        geometry_reconstructed = self._iwt(refined_bands, original_size)

        reliability = torch.sigmoid(
            self.reliability_gate(
                self._relation(semantic_inner, geometry_reconstructed)
            )
        )
        geometry_prompt = self.geometry_out(reliability * geometry_reconstructed)
        fused = semantic + self.layer_scale * geometry_prompt

        if not return_aux:
            return fused

        zero_low = torch.zeros_like(ll_refined)
        boundary_bands = torch.stack(
            (zero_low, high_refined[0], high_refined[1], high_refined[2]),
            dim=2,
        )
        boundary = self._iwt(boundary_bands, original_size)
        aux = {
            "boundary": boundary,
            "reliability": reliability,
            "band_weights": band_weights,
        }
        return fused, aux


class SGDWIF(nn.Module):
    """SAM2 and WTConvNeXt segmentation network with SG-DWIF fusion."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        decode_channels: int = 64,
        dropout: float = 0.0,
        window_size: int = 8,
        num_classes: int = 6,
        strip_kernel: int = 7,
        layer_scale_init: float = 0.1,
        adapter_gate_init: float = 3.0,
    ) -> None:
        super().__init__()

        self.resnet = WTConvNeXt()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)

        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.adapter_stage_ends = tuple(
            int(index) for index in self.encoder.stage_ends
        )
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                GatedAdapter(block, gate_init=adapter_gate_init)
            )
        self.encoder.blocks = nn.Sequential(*blocks)

        encoder_channels = (144, 288, 576, 1152)
        fusion_kwargs = {
            "strip_kernel": strip_kernel,
            "layer_scale_init": layer_scale_init,
        }
        self.fusion1 = SemanticGuidedDirectionalWaveletFusion(
            encoder_channels[0], **fusion_kwargs
        )
        self.fusion2 = SemanticGuidedDirectionalWaveletFusion(
            encoder_channels[1], **fusion_kwargs
        )
        self.fusion3 = SemanticGuidedDirectionalWaveletFusion(
            encoder_channels[2], **fusion_kwargs
        )
        self.fusion4 = SemanticGuidedDirectionalWaveletFusion(
            encoder_channels[3], **fusion_kwargs
        )

        self.decoder = Decoder(
            encoder_channels,
            decode_channels,
            dropout,
            window_size,
            num_classes,
        )

    def adapter_gate_values(self) -> torch.Tensor:

        values = [
            block.gate_value()
            for block in self.encoder.blocks
            if isinstance(block, GatedAdapter)
        ]
        if not values:
            raise RuntimeError("No GatedAdapter modules were found")
        return torch.stack(values)

    def adapter_gate_stage_values(self) -> Dict[str, torch.Tensor]:

        gates = self.adapter_gate_values()
        stages = {}
        start = 0
        for stage_index, end in enumerate(self.adapter_stage_ends, start=1):
            stages["x{}".format(stage_index)] = gates[start : end + 1]
            start = end + 1
        return stages

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "optical input must have shape [B, 3, H, W], "
                f"got shape {tuple(x.shape)}"
            )
        if y.ndim != 4 or y.shape[1] != 1:
            raise ValueError(
                "DSM input must have shape [B, 1, H, W], "
                f"got shape {tuple(y.shape)}"
            )
        if x.shape[0] != y.shape[0] or x.shape[-2:] != y.shape[-2:]:
            raise ValueError(
                "optical and DSM inputs must have identical batch and spatial sizes, "
                f"got {tuple(x.shape)} and {tuple(y.shape)}"
            )

        h, w = x.shape[-2:]

        x1, x2, x3, x4 = self.encoder(x)
        y0 = self.resnet.layer0(y)
        y1 = self.resnet.layer1(y0)
        y2 = self.resnet.layer2(y1)
        y3 = self.resnet.layer3(y2)
        y4 = self.resnet.layer4(y3)

        res1 = self.fusion1(x1, y1)
        res2 = self.fusion2(x2, y2)
        res3 = self.fusion3(x3, y3)
        res4 = self.fusion4(x4, y4)

        out = self.decoder(res1, res2, res3, res4, h, w)
        return out


__all__ = [
    "GatedAdapter",
    "SemanticGuidedDirectionalWaveletFusion",
    "SGDWIF",
]


if __name__ == "__main__":
    with torch.no_grad():
        model = SGDWIF().cuda().eval()
        x = torch.randn(1, 3, 256, 256).cuda()
        y = torch.randn(1, 1, 256, 256).cuda()
        out = model(x, y)
        print(out.shape)
 

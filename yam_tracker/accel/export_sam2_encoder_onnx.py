"""export_sam2_encoder_onnx.py — export SAM2's image encoder to ONNX.

The image encoder is the Hiera trunk + FPN neck (SAM2Base.image_encoder). It is
the per-frame hotspot: for the shell tiles the memory-attention/decoder are
cheap next to it. We export ONLY the encoder at a FIXED square resolution and
FP32; build_sam2_trt_engine.py then compiles it to an fp16 TRT .plan, and
tracker_accel wires it into forward_image (downstream conv_s0/conv_s1 + memory
attention stay torch). This mirrors the ENPIRE SAM3 split-export template.

Outputs (flat tuple, ONNX can't nest): backbone_fpn levels then vision_pos_enc
levels. The predictor's forward_image keeps only the last num_feature_levels of
each, so we emit all levels and slice in the runner.

Usage:
  CUDA_VISIBLE_DEVICES=4 python export_sam2_encoder_onnx.py \
      --image-size 512 --backbone large --out _trt/sam2_enc_l_512.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append("/shared/samurai/sam2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

import tracker_accel as tra  # noqa: E402


class _EncoderWrapper(nn.Module):
    """SAM2Base.image_encoder → flat tuple (fpn levels, then pos-enc levels)."""

    def __init__(self, model):
        super().__init__()
        self.image_encoder = model.image_encoder

    def forward(self, sample):
        out = self.image_encoder(sample)
        return (*out["backbone_fpn"], *out["vision_pos_enc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--backbone", default="large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true",
                    help="export the encoder in fp16 so TRT 11 builds a "
                         "strongly-typed fp16 engine (matches the eager "
                         "autocast(fp16) numerics the accurate baseline uses)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg, ckpt = tra._BACKBONES[args.backbone]
    model = build_sam2_video_predictor(
        cfg, ckpt, device="cuda:0",
        hydra_overrides_extra=[f"++model.image_size={args.image_size}"])
    wrapper = _EncoderWrapper(model).eval().to("cuda:0")
    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        wrapper = wrapper.half()

    dummy = torch.zeros(1, 3, args.image_size, args.image_size,
                        device="cuda:0", dtype=dtype)
    with torch.inference_mode():
        sanity = wrapper(dummy)
    n_fpn = len(sanity) // 2
    for i, t in enumerate(sanity):
        kind = "fpn" if i < n_fpn else "pos"
        print(f"  out {kind}_{i % n_fpn}: {tuple(t.shape)} {t.dtype}")

    names = [f"fpn_{i}" for i in range(n_fpn)] + [f"pos_{i}" for i in range(n_fpn)]
    print(f"exporting -> {out}")
    torch.onnx.export(
        wrapper, (dummy,), str(out),
        input_names=["image"], output_names=names,
        opset_version=args.opset, dynamo=False, do_constant_folding=True,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    # write the level count so the runner can bind unambiguously
    (out.with_suffix(".meta")).write_text(f"n_fpn={n_fpn}\nimage_size={args.image_size}\n")


if __name__ == "__main__":
    main()

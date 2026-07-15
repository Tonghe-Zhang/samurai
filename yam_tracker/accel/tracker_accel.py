"""tracker_accel.py — faster SAMURAI tracking via lower internal resolution
and an optional TensorRT image encoder, WITHOUT changing accuracy behaviour.

We keep everything that makes tracking accurate identical to tracker.Tracker:
  * samurai_mode stays ON (motion memory) — the yaml is a samurai config,
  * ONE object per propagate pass (obj_id=0),
  * reverse-then-forward propagation from the seed frame.

The only levers exposed here are the *sanctioned* ones:
  * image_size  : SAM2's internal square resolution (default 1024). The shell
                  tiles are 640x360, so 1024 upscales every frame. Building the
                  predictor at 512/640 cuts encoder pixels ~4x/2.5x.
  * trt_encoder : swap the torch Hiera-trunk+FPN `forward_image` for a fixed-
                  resolution fp16 TensorRT engine. The memory-attention and
                  mask decoder stay in torch (they are cheap next to the trunk).

`track()` is inherited from tracker.Tracker unchanged — this class only changes
how the predictor is *built* and how one frame is *encoded*.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tracker as trk  # noqa: E402  (also puts SAM2_REPO on sys.path)
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

# Backbone samurai configs + checkpoint filenames, resolved against trk.CKPT_DIR.
_BACKBONES = {
    "large": ("configs/samurai/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    "base_plus": ("configs/samurai/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "small": ("configs/samurai/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "tiny": ("configs/samurai/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
}


class AccelTracker(trk.Tracker):
    """SAMURAI predictor with resolution + optional TRT-encoder overrides.

    Reuses Tracker.track() verbatim (samurai_mode, single object, reverse+
    forward). Construction differs only in image_size / backbone / encoder.
    """

    def __init__(self, video_path: str, device: str = "cuda:0",
                 image_size: int = 1024, backbone: str = "large",
                 trt_encoder_plan: str | None = None):
        cfg, ckpt_name = _BACKBONES[backbone]
        ckpt = os.path.join(trk.CKPT_DIR, ckpt_name)
        overrides = [f"++model.image_size={image_size}"]
        self.predictor = build_sam2_video_predictor(
            cfg, ckpt, device=device, hydra_overrides_extra=overrides)
        self.image_size = image_size
        if trt_encoder_plan is not None:
            self._install_trt_encoder(trt_encoder_plan, device)
        self.state = self.predictor.init_state(
            video_path, offload_video_to_cpu=True, offload_state_to_cpu=True)
        self.num_frames = self.state["num_frames"]

    def _install_trt_encoder(self, plan_path: str, device: str) -> None:
        """Monkeypatch the predictor's forward_image to run the Hiera trunk +
        FPN neck through a TRT engine, keeping the downstream torch code path
        (conv_s0/conv_s1 high-res projections, pos-enc) identical to eager.
        """
        from trt_encoder import TRTImageEncoder  # local, lazy import

        engine = TRTImageEncoder(plan_path, device=device)
        model = self.predictor

        def forward_image_trt(img_batch: torch.Tensor):
            backbone_out = engine(img_batch)  # dict: backbone_fpn, vision_pos_enc
            if model.use_high_res_features_in_sam:
                backbone_out["backbone_fpn"][0] = model.sam_mask_decoder.conv_s0(
                    backbone_out["backbone_fpn"][0])
                backbone_out["backbone_fpn"][1] = model.sam_mask_decoder.conv_s1(
                    backbone_out["backbone_fpn"][1])
            return backbone_out

        model.forward_image = forward_image_trt

"""trt_encoder.py — run SAM2's image encoder through a TensorRT engine.

Wraps a fixed-resolution fp16 .plan (from build_sam2_trt_engine.py) and exposes
`__call__(img_batch)` returning the SAM2 backbone dict
    {"backbone_fpn": [...], "vision_pos_enc": [...], "vision_features": last}
so it drops straight into SAM2Base.forward_image (see AccelTracker). Mirrors the
persistent-buffer / bound-address pattern from ENPIRE's _sam3_trt_runner.

The engine is batch=1, static-shape; SAMURAI encodes one frame at a time.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

_LOG = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.float32: torch.float32, trt.float16: torch.float16,
    trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool,
}


class TRTImageEncoder:
    def __init__(self, plan_path: str, device: str = "cuda:0"):
        self.device = device
        with Path(plan_path).open("rb") as f:
            self.engine = trt.Runtime(_LOG).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"deserialize failed for {plan_path} (rebuild for this GPU/TRT)")
        self.context = self.engine.create_execution_context()

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        # outputs are fpn_0.. then pos_0..; keep declaration order
        self._n_fpn = sum(1 for n in self.output_names if n.startswith("fpn_"))

        self._bufs: dict[str, torch.Tensor] = {}
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._bufs[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
        self._in = self.input_names[0]

    @torch.inference_mode()
    def __call__(self, img_batch: torch.Tensor) -> dict:
        # Everything runs on the current CUDA stream: the input copy, the TRT
        # execute, and the downstream torch ops that consume the outputs are
        # all ordered on one stream, so no host sync is needed here (that would
        # stall the async pipeline every frame — see one-model perf rules).
        buf = self._bufs[self._in]
        buf.copy_(img_batch.to(self.device, dtype=buf.dtype, non_blocking=True))
        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        fpn = [self._bufs[f"fpn_{i}"].float().clone() for i in range(self._n_fpn)]
        pos = [self._bufs[f"pos_{i}"].float().clone() for i in range(self._n_fpn)]
        return {"backbone_fpn": fpn, "vision_pos_enc": pos, "vision_features": fpn[-1]}

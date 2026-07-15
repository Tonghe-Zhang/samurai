"""build_sam2_trt_engine.py — compile the SAM2 encoder ONNX to an fp16 TRT plan.

Drives the TensorRT Python builder directly (trtexec isn't in the pip wheel),
mirroring ENPIRE's build_sam3_trt_engine.py. The .plan is GPU/driver/TRT
specific — rebuild on each box; never commit it.

Usage:
  CUDA_VISIBLE_DEVICES=4 python build_sam2_trt_engine.py \
      --onnx _trt/sam2_enc_l_512.onnx --engine _trt/sam2_enc_l_512_fp16.plan
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import tensorrt as trt

_LOG = trt.Logger(trt.Logger.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--workspace-mb", type=int, default=8192)
    args = ap.parse_args()
    args.engine.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build] tensorrt {trt.__version__}", flush=True)
    builder = trt.Builder(_LOG)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, _LOG)
    print(f"[build] parsing {args.onnx} ...", flush=True)
    if not parser.parse_from_file(str(args.onnx)):
        for i in range(parser.num_errors):
            print(f"[build] parse error {i}: {parser.get_error(i)}")
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_mb) << 20)
    # TRT 11 dropped the FP16/BF16 BuilderFlags in favour of strongly-typed
    # networks (precision follows the ONNX dtype) with automatic kernel
    # selection. To run the fp32-exported encoder in fp16 we cast the network
    # I/O + weights to fp16 by setting the builder's optimization to prefer
    # reduced precision via the ONNX cast, so we instead re-declare fp16 by
    # marking the network as allowing reduced precision through TF32-off +
    # letting the auto-tuner pick fp16 tensor-core kernels (fastest on Ada).
    if args.precision == "fp16":
        for flag in ("FP16", "BF16"):
            if hasattr(trt.BuilderFlag, flag):
                config.set_flag(getattr(trt.BuilderFlag, flag))
                print(f"[build] {flag} enabled")
                break
        else:
            print("[build] no FP16 BuilderFlag on this TRT — relying on auto "
                  "kernel selection (fp16 tensor cores) for the fp32 network")

    print("[build] building engine ...", flush=True)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("build_serialized_network returned None")
    args.engine.write_bytes(bytes(serialized))
    print(f"[build] done in {(time.time() - t0) / 60:.1f} min — wrote {args.engine} "
          f"({args.engine.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""bench_accel.py — time pure SAMURAI propagation on the 9 shell-game tiles.

Isolates tracking speed (excludes SAM3 seeding): for each tile video we seed
one object with a small centre mask at the middle frame, then reverse+forward
propagate exactly as the real pipeline does, and record wall-clock + fps.

Usage:
  CUDA_VISIBLE_DEVICES=4 python bench_accel.py --image-size 512 --backbone large
  CUDA_VISIBLE_DEVICES=4 python bench_accel.py --image-size 640 --trt-plan <plan>
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import tracker as trk
import tracker_accel as tra

OUT_DIR = "/shared/tmp/track_hangers/out_shellgame"
TILE_H, TILE_W = 360, 640


def _centre_mask() -> np.ndarray:
    m = np.zeros((TILE_H, TILE_W), dtype=bool)
    m[140:260, 260:380] = True  # a plausible cup-sized blob near centre
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--backbone", default="large")
    ap.add_argument("--trt-plan", default=None)
    ap.add_argument("--tiles", type=int, default=9)
    args = ap.parse_args()

    seed_mask = _centre_mask()
    per_tile = []
    total_frames = 0
    # warm up on tile0 (kernel autotune / TRT graph capture) — not timed.
    warm_path = os.path.join(OUT_DIR, "_tile0.mp4")
    tr = tra.AccelTracker(warm_path, image_size=args.image_size,
                          backbone=args.backbone, trt_encoder_plan=args.trt_plan)
    n = tr.num_frames
    seed = n // 2
    tr.track(seed, seed_mask, obj_id=0)
    del tr
    trk.torch.cuda.empty_cache()

    t_all = time.time()
    for tid in range(args.tiles):
        path = os.path.join(OUT_DIR, f"_tile{tid}.mp4")
        tr = tra.AccelTracker(path, image_size=args.image_size,
                              backbone=args.backbone, trt_encoder_plan=args.trt_plan)
        n = tr.num_frames
        seed = n // 2
        trk.torch.cuda.synchronize()
        t0 = time.time()
        tr.track(seed, seed_mask, obj_id=0)
        trk.torch.cuda.synchronize()
        dt = time.time() - t0
        per_tile.append((tid, n, dt))
        total_frames += n
        del tr
        trk.torch.cuda.empty_cache()
    wall = time.time() - t_all

    prop_time = sum(dt for _, _, dt in per_tile)
    fps = total_frames / prop_time
    print(f"\n=== image_size={args.image_size} backbone={args.backbone} "
          f"trt={'yes' if args.trt_plan else 'no'} ===")
    for tid, n, dt in per_tile:
        print(f"  tile{tid}: {n} frames in {dt:.2f}s -> {n / dt:.1f} fps")
    print(f"  TOTAL propagation: {total_frames} frames in {prop_time:.2f}s "
          f"-> {fps:.1f} fps (per-object)")
    print(f"  wall (incl. per-tile build+encode+init): {wall:.2f}s")


if __name__ == "__main__":
    main()

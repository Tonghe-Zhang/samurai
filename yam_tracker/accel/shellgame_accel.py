"""shellgame_accel.py — accelerated shell-game cup tracker.

Identical pipeline to shellgame.py (SAM3 finds the red ball, picks the covering
cup, SAMURAI forward-tracks it, composites the 3x3 grid with green box + "track"
label, no fill) — the ONLY change is the tracker construction, which uses
AccelTracker with a lower internal resolution / smaller backbone / TRT encoder.

samurai_mode stays ON and one object per SAMURAI pass (track() is inherited
unchanged), so tracking accuracy is expected to match the baseline; that is
verified frame-by-frame against the ground-truth grid by verify_accuracy.py.

Reuses shellgame.detect_ball_x / pick_covering_cup / render_tile_overlay /
_tile_crop verbatim, so seeding is byte-identical to the baseline.

Usage:
  CUDA_VISIBLE_DEVICES=4 python shellgame_accel.py \
      --image-size 512 --backbone large --out shellgame_trt_grid.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shellgame as sg  # noqa: E402  (reuse ball/cup detection + rendering)
import tracker as trk  # noqa: E402
import tracker_accel as tra  # noqa: E402
import video_io  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--backbone", default="large")
    ap.add_argument("--trt-plan", default=None)
    ap.add_argument("--out", default="shellgame_trt_grid.mp4")
    args = ap.parse_args()

    os.makedirs(sg.OUT_DIR, exist_ok=True)
    serve_dir = os.path.join(sg.OUT_DIR, "_serve")
    os.makedirs(serve_dir, exist_ok=True)
    trk._static_server(serve_dir, sg.SERVE_PORT)

    frames, fps = video_io.read_frames(sg.VIDEO)
    n = len(frames)
    print(f"loaded {n} frames @ {fps}fps | image_size={args.image_size} "
          f"backbone={args.backbone} trt={'yes' if args.trt_plan else 'no'}")

    # Reuse the existing per-tile encode cache written by shellgame.py.
    tile_videos = [os.path.join(sg.OUT_DIR, f"_tile{tid}.mp4") for tid in range(9)]
    for tid in range(9):
        if not os.path.exists(tile_videos[tid]):
            crops = [sg._tile_crop(f, tid) for f in frames]
            rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
            video_io._encode_h264(rgb, tile_videos[tid], fps)

    tile_overlays: dict[int, list[np.ndarray]] = {}
    report = {}
    for tid in range(9):
        ball_x, ball_f, seed_f = sg.detect_ball_x(tid, frames, serve_dir)
        seed_crop = sg._tile_crop(frames[seed_f], tid)
        cup_mask = sg.pick_covering_cup(tid, seed_crop, ball_x, serve_dir)
        cup_box = video_io._mask_bbox(cup_mask)
        cup_cx = cup_box[0] + cup_box[2] / 2.0
        tr = tra.AccelTracker(tile_videos[tid], device="cuda:0",
                              image_size=args.image_size, backbone=args.backbone,
                              trt_encoder_plan=args.trt_plan)
        masks = tr.track(seed_f, cup_mask, obj_id=0)
        del tr
        trk.torch.cuda.empty_cache()
        tile_overlays[tid] = sg.render_tile_overlay(frames, tid, masks)
        report[tid] = (ball_x, ball_f, cup_cx, seed_f)
        print(f"tile{tid}: ball_x={ball_x:.0f}@f{ball_f} seed@f{seed_f} "
              f"cup_cx={cup_cx:.0f} tracked {len(masks)} frames")

    gw, gh = sg.TILE_W + sg.GUTTER, sg.TILE_H + sg.GUTTER
    grid_h = sg.TILE_H * 3 + sg.GUTTER * 4
    grid_w = sg.TILE_W * 3 + sg.GUTTER * 4
    rendered = []
    for idx in range(n):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for tid in range(9):
            r, c = tid // 3, tid % 3
            y0, x0 = sg.GUTTER + r * gh, sg.GUTTER + c * gw
            canvas[y0:y0 + sg.TILE_H, x0:x0 + sg.TILE_W] = tile_overlays[tid][idx]
        rendered.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    out_path = os.path.join(sg.OUT_DIR, args.out)
    video_io._encode_h264(rendered, out_path, fps)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

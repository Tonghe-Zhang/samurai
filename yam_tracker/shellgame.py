"""shellgame.py — fully automatic 9-tile shell-game cup tracker.

The input video is a 3x3 grid of 9 independent shell-game instances (each tile
640x360). For every tile we:
  1. detect the RED BALL via SAM3 on the frame where it is most visible,
  2. detect the CUPS ("cone") on a seed frame right after the covering cup has
     descended, and pick the cup whose base is directly over the ball,
  3. seed SAMURAI with that cup's mask and forward-track it through all frames,
  4. render a green mask + green box + "track" label above the cup.
Finally we composite the 9 overlaid tiles back into the 3x3 grid and write one
H.264 mp4.

Everything is automatic: no per-tile hand tuning, ball and cup positions come
from SAM3. Run:  CUDA_VISIBLE_DEVICES=1 python shellgame.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracker as trk  # noqa: E402
import video_io  # noqa: E402

# Input video: first CLI arg, else SHELLGAME_VIDEO env, else the cluster default.
VIDEO = (sys.argv[1] if len(sys.argv) > 1 else
         os.environ.get("SHELLGAME_VIDEO",
                        "/shared/shell-game-data/yam_shellgame/tiled_9.mp4"))
OUT_DIR = os.environ.get("SHELLGAME_OUT", "out_shellgame")
GRID_OUT = os.path.join(OUT_DIR, "shellgame_tracked_grid.mp4")
TILE_W, TILE_H = 640, 360
GUTTER = 8  # black border drawn between tiles in the output grid
SERVE_PORT = 8795
# Vision-driven timing: scan a broad early window for the ball; the seed frame
# (cup settled over the ball) is found per tile as the first frame after the
# ball is last seen — no hand-picked per-tile constants.
BALL_SCAN = list(range(6, 40, 2))
GREEN = (0, 255, 0)


def _tile_crop(frame: np.ndarray, tid: int) -> np.ndarray:
    r, c = tid // 3, tid % 3
    return frame[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W]


def _center_x(bbox_xywh: list[int]) -> float:
    return bbox_xywh[0] + bbox_xywh[2] / 2.0


def detect_ball_x(tid: int, frames: list[np.ndarray], serve_dir: str) -> tuple[float, int, int]:
    """Return (ball_center_x, best_frame, seed_frame), all from vision.

    ball_center_x/best_frame come from the highest-scoring "red ball" detection.
    seed_frame is the first frame after the ball is last detected (i.e. once a
    cup has covered it) — this is where we seed the cup tracker per tile.
    """
    best = None  # (score, cx, frame)
    last_seen = BALL_SCAN[0]
    for f in BALL_SCAN:
        crop = _tile_crop(frames[f], tid)
        name = f"_ball_t{tid}_f{f}.png"
        cv2.imwrite(os.path.join(serve_dir, name), crop)
        url = f"http://localhost:{SERVE_PORT}/{name}"
        r = trk.requests.post(
            trk.SAM3_URL, json={"image_url": url, "text": "red ball", "score_threshold": 0.2},
            timeout=60)
        r.raise_for_status()
        dets = r.json()["detections"]
        if dets:
            last_seen = f
        for d in dets:
            cx = _center_x(d["bbox_xywh"])
            if best is None or d["score"] > best[0]:
                best = (d["score"], cx, f)
    if best is None:
        raise RuntimeError(f"tile {tid}: no red ball detected")
    seed_frame = min(last_seen + 6, len(frames) - 1)  # just after the cover
    return best[1], best[2], seed_frame


def pick_covering_cup(tid: int, seed_crop: np.ndarray, ball_x: float,
                      serve_dir: str) -> np.ndarray:
    """Detect cones on the seed frame; return the mask of the cup over the ball."""
    name = f"_seed_t{tid}.png"
    cv2.imwrite(os.path.join(serve_dir, name), seed_crop)
    url = f"http://localhost:{SERVE_PORT}/{name}"
    r = trk.requests.post(
        trk.SAM3_URL, json={"image_url": url, "text": "cone", "score_threshold": 0.15},
        timeout=60)
    r.raise_for_status()
    dets = r.json()["detections"]
    if not dets:
        raise RuntimeError(f"tile {tid}: no cups detected at seed frame")
    best = min(dets, key=lambda d: abs(_center_x(d["bbox_xywh"]) - ball_x))
    return trk._det_to_mask(best, TILE_H, TILE_W)


def render_tile_overlay(frames: list[np.ndarray], tid: int,
                        masks: dict[int, np.ndarray]) -> list[np.ndarray]:
    """Draw a green box + 'track' label above the tracked cup on each tile crop.

    No mask fill: painting the mask tints the cup and makes it look a different
    colour from the identical cups next to it, so we mark it with the box only.
    """
    out = []
    for idx, frame in enumerate(frames):
        img = _tile_crop(frame, tid).copy()
        mask = masks.get(idx)
        if mask is not None and mask.any():
            box = video_io._mask_bbox(mask)
            if box is not None:
                x, y, bw, bh = box
                cv2.rectangle(img, (x, y), (x + bw, y + bh), GREEN, 2)
                ty = y - 8 if y - 8 > 12 else y + bh + 18
                cv2.putText(img, "track", (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, GREEN, 2, cv2.LINE_AA)
        out.append(img)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    serve_dir = os.path.join(OUT_DIR, "_serve")
    os.makedirs(serve_dir, exist_ok=True)
    trk._static_server(serve_dir, SERVE_PORT)

    frames, fps = video_io.read_frames(VIDEO)
    n = len(frames)
    print(f"loaded {n} frames @ {fps}fps")

    # Write 9 tile sub-videos once so SAMURAI can encode each cheaply.
    tile_videos = []
    for tid in range(9):
        crops = [_tile_crop(f, tid) for f in frames]
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        p = os.path.join(OUT_DIR, f"_tile{tid}.mp4")
        video_io._encode_h264(rgb, p, fps)
        tile_videos.append(p)

    tile_overlays: dict[int, list[np.ndarray]] = {}
    report = {}
    for tid in range(9):
        ball_x, ball_f, seed_f = detect_ball_x(tid, frames, serve_dir)
        seed_crop = _tile_crop(frames[seed_f], tid)
        cup_mask = pick_covering_cup(tid, seed_crop, ball_x, serve_dir)
        cup_cx = video_io._mask_bbox(cup_mask)
        cup_cx = cup_cx[0] + cup_cx[2] / 2.0
        tr = trk.Tracker(tile_videos[tid], device="cuda:0")
        masks = tr.track(seed_f, cup_mask, obj_id=0)
        del tr
        trk.torch.cuda.empty_cache()
        tile_overlays[tid] = render_tile_overlay(frames, tid, masks)
        report[tid] = (ball_x, ball_f, cup_cx, seed_f)
        print(f"tile{tid}: ball_x={ball_x:.0f}@f{ball_f} seed@f{seed_f} "
              f"cup_cx={cup_cx:.0f} tracked {len(masks)} frames")

    # Composite overlaid tiles into a 3x3 grid separated by black gutters.
    gw, gh = TILE_W + GUTTER, TILE_H + GUTTER
    grid_h = TILE_H * 3 + GUTTER * 4
    grid_w = TILE_W * 3 + GUTTER * 4
    rendered = []
    for idx in range(n):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)  # black frame
        for tid in range(9):
            r, c = tid // 3, tid % 3
            y0, x0 = GUTTER + r * gh, GUTTER + c * gw
            canvas[y0:y0 + TILE_H, x0:x0 + TILE_W] = tile_overlays[tid][idx]
        rendered.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    video_io._encode_h264(rendered, GRID_OUT, fps)
    print(f"\nwrote {GRID_OUT}")
    for tid in range(9):
        bx, bf, cx, sf = report[tid]
        print(f"  tile{tid}: ball@x{bx:.0f}(f{bf}) seed@f{sf} -> cup@x{cx:.0f} "
              f"(dx={abs(cx-bx):.0f})")


if __name__ == "__main__":
    main()

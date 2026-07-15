"""spell_blocks.py — track each lettered block of a spelling episode, per its hand.

Each block is routed by how it was manipulated:
  * single hand  -> reverse-track in that one wrist from the block's landing.
  * handover / left-then-right (with a pause) -> the handover protocol: seed in
    EACH wrist at a frame where that arm holds the block and reverse-track; the
    block stays visible in the right wrist near the landing and reappears in the
    left wrist further back.
Both wrist views are overlaid (one colour + letter per block) and tiled left|right.

Reuses tracker (SAM3 seed + SAMURAI), relabel_api (path window + jitter gate),
dataset_meta (camera blobs), video_io (H.264 + overlay). No hardcoded paths.

    python spell_blocks.py            # runs the BEAR and NOTE demos
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_meta as meta  # noqa: E402
import relabel_api as rl  # noqa: E402
import tracker as trk  # noqa: E402
import video_io  # noqa: E402
from video_io import mask_centroid  # noqa: E402

OUT_DIR = os.environ.get("SPELL_OUT", "out_spell")
PROMPT = "wooden block"
LCAM, RCAM = "left_wrist_0_rgb", "right_wrist_0_rgb"
# Each block: letter + phases [(camera, seed_frame)]. A phase reverse-tracks in
# that wrist from a frame where the arm holds the block. Handover / left-then-
# right blocks list both a left and a right phase.
EPISODES = {
    "xdof/019d2add-1850-7955-8e16-34724042c5fd": [  # BEAR — all right hand
        ("B", [(RCAM, 419)]), ("E", [(RCAM, 741)]),
        ("A", [(RCAM, 1162)]), ("R", [(RCAM, 1409)]),  # R clearly held just before landing
    ],
    "xdof/019d2c4e-6b30-790b-b841-1e8445c46ce9": [  # NOTE — N left, O handover, T/E left->right
        ("N", [(LCAM, 338)]),
        ("O", [(LCAM, 737), (RCAM, 915)]),
        ("T", [(LCAM, 1020), (RCAM, 1201)]),
        ("E", [(LCAM, 1360), (RCAM, 1563)]),
    ],
}


def _gripper_seed(frames, frame_idx, port) -> np.ndarray | None:
    """SAM3-detect wooden blocks; return the mask nearest the gripper region."""
    h, w = frames[frame_idx].shape[:2]
    cv2.imwrite(os.path.join(OUT_DIR, "_sp_seed.png"), frames[frame_idx])
    dets = trk.seed_from_sam3(f"http://localhost:{port}/_sp_seed.png", h, w, PROMPT, 0.2)
    dets = [s for s in dets if mask_centroid(s["mask"]) is not None]  # drop empty masks
    if not dets:
        return None
    gx, gy = w / 2.0, h * 0.9
    return min(dets, key=lambda s: (mask_centroid(s["mask"])[0] - gx) ** 2
               + (mask_centroid(s["mask"])[1] - gy) ** 2)["mask"]


def _track_one(video, seed, tr, port):
    """Reverse-track a block from `seed`, path-window + jitter-gate the result."""
    frames = tr._frames
    mask0 = _gripper_seed(frames, seed, port)
    if mask0 is None:
        return {}
    masks = tr.track(seed, mask0, obj_id=0, directions=(True,))
    return rl.reliable_run(rl.path_continuous_window(masks, seed), seed)


def run(episode, blocks, port=trk._SERVE_PORT) -> str:
    """Track every block per its routing and tile the two wrist views."""
    name = episode.split("/")[-1][:8]
    print(f"episode {name}: {''.join(b for b, _ in blocks)}")
    per_cam = {LCAM: {}, RCAM: {}}  # cam -> {block_idx: masks}
    frames_by_cam, fps = {}, 30.0
    for cam in (LCAM, RCAM):
        video = meta.camera_video(episode, cam)
        tr = trk.Tracker(video)
        tr._frames, fps = video_io.read_frames(video)
        for bi, (letter, phases) in enumerate(blocks):
            for pcam, seed in phases:
                if pcam != cam:
                    continue
                kept = _track_one(video, seed, tr, port)
                if kept:
                    per_cam[cam][bi] = kept
                    print(f"  [{cam[:5]}] {letter} @f{seed} -> [{min(kept)}-{max(kept)}] ({len(kept)}f)")
                else:
                    print(f"  [{cam[:5]}] {letter} @f{seed} -> no seed")
        frames_by_cam[cam] = tr._frames
        del tr
        trk.torch.cuda.empty_cache()

    n = min(len(frames_by_cam[LCAM]), len(frames_by_cam[RCAM]))
    gap = np.zeros((frames_by_cam[LCAM][0].shape[0], 8, 3), np.uint8)
    tiled = []
    for i in range(n):
        panels = []
        for cam in (LCAM, RCAM):
            img = frames_by_cam[cam][i].copy()
            for bi, kept in per_cam[cam].items():
                m = kept.get(i)
                if m is None or not m.any():
                    continue
                color = video_io._PALETTE[bi % len(video_io._PALETTE)]
                box = video_io._mask_bbox(m)
                if box:
                    x, y, w, h = box
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, blocks[bi][0], (x, max(11, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            panels.append(img)
        tiled.append(cv2.cvtColor(np.hstack([panels[0], gap, panels[1]]), cv2.COLOR_BGR2RGB))
    out = os.path.join(OUT_DIR, f"{name}_spell_wrists.mp4")
    video_io._encode_h264(tiled, out, fps)
    print(f"  wrote {out}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trk._static_server(OUT_DIR)
    outs = [run(ep, blocks) for ep, blocks in EPISODES.items()]
    print("\nDELIVERABLES:")
    for o in outs:
        print("  " + os.path.abspath(o))


if __name__ == "__main__":
    main()

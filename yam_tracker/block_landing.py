"""block_landing.py — back-track lettered blocks from their landing frames.

For an episode with known per-block LANDING frames, seed the block the gripper
just placed (the block nearest the gripper at that frame) and reverse-track it.
Overlay every block's track (distinct colour + label) on BOTH wrist cameras,
then tile the two wrist views side-by-side into one full-length H.264 video.

Reuses dataset_meta (camera blobs), tracker (SAM3 seeding + reverse tracking),
relabel_api (path-continuity window), and video_io (H.264 + mask helpers).

    python block_landing.py            # runs the two configured episodes
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

OUT_DIR = os.environ.get("BLOCK_OUT", "out_blocks")
PROMPT, THR = "block", 0.2
# Episode -> per-block landing frames (when each block is placed on the table).
EPISODES = {
    "xdof/019d2add-1850-7955-8e16-34724042c5fd": [419, 741, 1162, 1431],
    "xdof/019d2c4e-6b30-790b-b841-1e8445c46ce9": [338, 915, 1201, 1563],
}
WRISTS = [("lwrist", "left_wrist_0_rgb"), ("rwrist", "right_wrist_0_rgb")]


def seed_gripper_block(frames, land_frame, port) -> np.ndarray | None:
    """SAM3-detect blocks at the landing frame; pick the one nearest the gripper.

    The just-placed block sits by the gripper (bottom-centre of the wrist view),
    so we seed the detection whose centre is closest to that point.
    """
    h, w = frames[land_frame].shape[:2]
    cv2.imwrite(os.path.join(OUT_DIR, "_bl_seed.png"), frames[land_frame])
    dets = trk.seed_from_sam3(f"http://localhost:{port}/_bl_seed.png", h, w, PROMPT, THR)
    if not dets:
        return None
    gx, gy = w / 2.0, h * 0.9
    best = min(dets, key=lambda s: (mask_centroid(s["mask"])[0] - gx) ** 2
               + (mask_centroid(s["mask"])[1] - gy) ** 2)
    return best["mask"]


def track_blocks(video, land_frames, port) -> list[dict[int, np.ndarray]]:
    """Reverse-track each landed block; return one path-continuous mask dict per block."""
    frames, fps = video_io.read_frames(video)
    tr = trk.Tracker(video)
    tracks = []
    for i, lf in enumerate(land_frames):
        mask0 = seed_gripper_block(frames, lf, port)
        if mask0 is None:
            tracks.append({})
            print(f"    block{i+1} @f{lf}: no seed")
            continue
        masks = tr.track(lf, mask0, obj_id=i, directions=(True,))  # reverse only
        kept = rl.path_continuous_window(masks, lf)
        tracks.append(kept)
        print(f"    block{i+1} @f{lf}: window [{min(kept)}-{max(kept)}] ({len(kept)}f)")
    del tr
    trk.torch.cuda.empty_cache()
    return frames, fps, tracks


def render_camera(frames, tracks) -> list[np.ndarray]:
    """Overlay all blocks (palette colour + #k) on each frame; return BGR frames."""
    out = []
    for idx, frame in enumerate(frames):
        img = frame.copy()
        for k, masks in enumerate(tracks):
            m = masks.get(idx)
            if m is None or not m.any():
                continue
            color = video_io._PALETTE[k % len(video_io._PALETTE)]
            box = video_io._mask_bbox(m)
            if box is None:
                continue
            x, y, bw, bh = box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(img, f"#{k+1}", (x, max(11, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        out.append(img)
    return out


def run_episode(episode: str, land_frames: list[int], port: int = trk._SERVE_PORT):
    """Track blocks in both wrist cams and write one side-by-side tiled H.264 video."""
    name = episode.split("/")[-1][:8]
    print(f"episode {name}: landings {land_frames}")
    rendered = {}
    fps = 30.0
    for tag, cam in WRISTS:
        print(f"  [{tag}]")
        frames, fps, tracks = track_blocks(meta.camera_video(episode, cam), land_frames, port)
        rendered[tag] = render_camera(frames, tracks)
    # tile left | right (pad to equal length just in case), black divider column
    n = min(len(rendered["lwrist"]), len(rendered["rwrist"]))
    h, w = rendered["lwrist"][0].shape[:2]
    gap = np.zeros((h, 8, 3), np.uint8)
    tiled = [cv2.cvtColor(np.hstack([rendered["lwrist"][i], gap, rendered["rwrist"][i]]),
                          cv2.COLOR_BGR2RGB) for i in range(n)]
    out_mp4 = os.path.join(OUT_DIR, f"{name}_blocks_wrists.mp4")
    video_io._encode_h264(tiled, out_mp4, fps)
    print(f"  wrote {out_mp4}")
    return out_mp4


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trk._static_server(OUT_DIR)
    outs = [run_episode(ep, frs) for ep, frs in EPISODES.items()]
    print("\nDELIVERABLES:")
    for o in outs:
        print("  " + os.path.abspath(o))


if __name__ == "__main__":
    main()

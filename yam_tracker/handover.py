"""handover.py — track one object across a two-arm handover, both wrist views.

An object is passed from a giver arm to a receiver arm. We track the SAME object
in each wrist camera over that arm's possession phase, then tile the two views
side-by-side. Seeds are auto-detected (SAM3 "nearest the gripper"), so the
caller only supplies a per-arm seed frame — no hand-drawn boxes.

  giver wrist:    seed at the handover frame, reverse-track to the pickup.
  receiver wrist: seed while it clearly holds the object, track both directions
                  (reverse to the handover, forward to placement).

Reuses tracker (SAM3 seed + SAMURAI), relabel_api (path window + jitter gate),
dataset_meta (camera blobs), video_io (H.264 + overlay). No hardcoded paths.

    python handover.py            # runs the configured demo episodes
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

OUT_DIR = os.environ.get("HANDOVER_OUT", "out_handover")
GREEN = (0, 255, 0)
# Demo episodes: object prompt + giver/receiver arm and each arm's seed frame.
# giver seeds at the handover; receiver seeds where it clearly holds the object.
DEMOS = [
    {"episode": "xdof/019d2c4e-6b30-790b-b841-1e8445c46ce9", "prompt": "wooden block",
     "giver": ("left_wrist_0_rgb", 737), "receiver": ("right_wrist_0_rgb", 915)},
    {"episode": "xdof/019c0548-d850-7623-b17e-c86133937c64", "prompt": "headphones",
     "giver": ("left_wrist_0_rgb", 478), "receiver": ("right_wrist_0_rgb", 853)},
]


def _gripper_seed(frames, frame_idx, prompt, port) -> np.ndarray | None:
    """SAM3-detect `prompt`; return the mask nearest the gripper (bottom-centre)."""
    h, w = frames[frame_idx].shape[:2]
    cv2.imwrite(os.path.join(OUT_DIR, "_ho_seed.png"), frames[frame_idx])
    dets = trk.seed_from_sam3(f"http://localhost:{port}/_ho_seed.png", h, w, prompt, 0.2)
    if not dets:
        return None
    gx, gy = w / 2.0, h * 0.9
    return min(dets, key=lambda s: (mask_centroid(s["mask"])[0] - gx) ** 2
               + (mask_centroid(s["mask"])[1] - gy) ** 2)["mask"]


def _track_phase(episode, cam, seed, prompt, directions, port):
    """Seed at `seed`, track, path-window + jitter-gate; return (frames, fps, masks)."""
    video = meta.camera_video(episode, cam)
    frames, fps = video_io.read_frames(video)
    mask0 = _gripper_seed(frames, seed, prompt, port)
    if mask0 is None:
        raise RuntimeError(f"{cam}: no '{prompt}' detected at frame {seed}")
    tr = trk.Tracker(video)
    masks = tr.track(seed, mask0, directions=directions)
    del tr
    trk.torch.cuda.empty_cache()
    kept = rl.reliable_run(rl.path_continuous_window(masks, seed), seed)
    print(f"  {cam}: seed {seed} -> [{min(kept)}-{max(kept)}] ({len(kept)}f)")
    return frames, fps, kept


def _overlay(frames, masks, label):
    out = []
    for i, fr in enumerate(frames):
        img = fr.copy()
        m = masks.get(i)
        if m is not None and m.any():
            b = video_io._mask_bbox(m)
            if b:
                x, y, w, h = b
                cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
                cv2.putText(img, label, (x, max(11, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
        out.append(img)
    return out


def run(demo: dict, port: int = trk._SERVE_PORT) -> str:
    """Track the handover object in both wrists and tile into one H.264 video."""
    name = demo["episode"].split("/")[-1][:8]
    print(f"episode {name}: {demo['prompt']} handover")
    gcam, gseed = demo["giver"]
    rcam, rseed = demo["receiver"]
    gf, fps, gmask = _track_phase(demo["episode"], gcam, gseed, demo["prompt"], (True,), port)
    rf, _, rmask = _track_phase(demo["episode"], rcam, rseed, demo["prompt"], (True, False), port)
    go = _overlay(gf, gmask, "giver")
    ro = _overlay(rf, rmask, "receiver")
    n = min(len(go), len(ro))
    gap = np.zeros((go[0].shape[0], 8, 3), np.uint8)
    tiled = [cv2.cvtColor(np.hstack([go[i], gap, ro[i]]), cv2.COLOR_BGR2RGB) for i in range(n)]
    out = os.path.join(OUT_DIR, f"{name}_{demo['prompt'].split()[0]}_handover.mp4")
    video_io._encode_h264(tiled, out, fps)
    print(f"  wrote {out}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trk._static_server(OUT_DIR)
    outs = [run(d) for d in DEMOS]
    print("\nDELIVERABLES:")
    for o in outs:
        print("  " + os.path.abspath(o))


if __name__ == "__main__":
    main()

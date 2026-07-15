"""relabel_api.py — thin hindsight-relabeling API for goal-reaching datasets.

Given an object (named + located on one frame) it back-tracks the object to the
earliest frame where it is BOTH visible and NOT being contacted by the arm, and
returns that frame index + a bounding-box overlay screenshot. Intended use:
hindsight-relabel where a manipulated object "started" for goal-reaching labels.

    from relabel_api import find_origin
    res = find_origin(video, seed_frame=963, seed_box=[240,181,284,239],
                      arm="right", episode="xdof/019d...", label="O")
    # -> {"frame": 831, "bbox_xywh": [...], "overlay_png": "..."}

SAM3 supplies the seed mask (open-vocab, from a box/point prompt); SAMURAI
back-propagates; dataset contact_annotations define the "no contact" windows.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_meta as meta  # noqa: E402
import tracker as trk  # noqa: E402
import video_io  # noqa: E402
from video_io import mask_centroid  # noqa: E402

GREEN = (0, 255, 0)
OUT_DIR = os.environ.get("RELABEL_OUT", "out_relabel")  # served root + mp4 output


def _seed_mask(frame_bgr: np.ndarray, box_xyxy, prompt: str, port: int) -> np.ndarray:
    """Get a tight SAM3 mask for the object in `box_xyxy` on this frame.

    Query SAM3 with `prompt`, then keep the detection whose box best overlaps the
    requested region (so we seed the specific instance the user pointed at).
    """
    h, w = frame_bgr.shape[:2]
    cv2.imwrite(os.path.join(OUT_DIR, "_relabel_seed.png"), frame_bgr)
    dets = trk.seed_from_sam3(f"http://localhost:{port}/_relabel_seed.png",
                              h, w, prompt, thr=0.2)
    tx0, ty0, tx1, ty1 = box_xyxy
    tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    best, best_d = None, 1e9
    for s in dets:  # nearest detection centre to the requested box centre
        x0, y0, x1, y1 = s["box_xyxy"]
        d = ((x0 + x1) / 2 - tcx) ** 2 + ((y0 + y1) / 2 - tcy) ** 2
        if d < best_d:
            best, best_d = s, d
    if best is None:
        raise RuntimeError("SAM3 returned no detections for prompt")
    return best["mask"]


def path_continuous_window(masks, seed_frame, jump=70, exit_gap=120) -> dict[int, np.ndarray]:
    """Keep only frames whose mask stays on the continuous backward path.

    Walk back from the seed: DISCARD any frame whose centroid leaps > `jump`px
    off the last kept position (off-path noise, e.g. a corner teleport) using a
    FIXED threshold; short occlusion gaps (empty/tiny masks) are bridged; stop
    once the object has been continuously absent for `exit_gap` frames (it left
    the field of view). Area is never a cutoff by itself, so a small occluded
    mask on-path is kept.
    """
    kept = {seed_frame: masks[seed_frame]}
    last, last_f = mask_centroid(masks[seed_frame]), seed_frame
    for f in range(seed_frame - 1, -1, -1):
        if last_f - f > exit_gap:
            break
        m = masks.get(f)
        if m is None or not m.any() or int(m.sum()) < 20:
            continue  # bridge short absence
        c = mask_centroid(m)
        if c is None:
            continue
        if last is not None and np.linalg.norm(c - last) > jump:
            continue  # off-path jump -> discard this frame, keep looking
        kept[f] = m
        last, last_f = c, f
    return kept


def reliable_run(kept, seed_frame, k=4.0, win=15, frac=0.5) -> dict[int, np.ndarray]:
    """Trim the jittery tail: keep the contiguous run around the seed that stays
    smooth, using a threshold calibrated from THIS track (no fixed pixel value).

    Jitter = frame-to-frame centroid acceleration, measured in OBJECT-SIZE units
    (centroid step / sqrt(mask area)) so a large object's normal motion is not
    mistaken for jitter — scale-invariant across small blocks and big headphones.
    Its robust centre (median + MAD over the window) sets a per-track spike level
    `median + k*MAD`; we walk outward from the seed and stop only where jitter is
    *sustained* — a local window of `win` frames is >`frac` spikes — so an
    isolated blip during a smooth carry does not cut the run. No fixed pixel value.
    """
    order = sorted(kept)
    if len(order) < 2 * win:
        return kept
    cen = {f: mask_centroid(kept[f]) for f in order}
    scale = {f: max(8.0, np.sqrt(int(kept[f].sum()))) for f in order}  # ~object width
    speed = [0.0]
    for a, b in zip(order, order[1:]):
        step = np.linalg.norm(cen[b] - cen[a]) / max(1, b - a)
        speed.append(step / scale[b])  # normalise by object size
    accel = np.abs(np.diff(speed, prepend=speed[0]))
    med = float(np.median(accel))
    mad = float(np.median(np.abs(accel - med))) or 1.0
    spike = accel > (med + k * 1.4826 * mad)
    si = min(range(len(order)), key=lambda i: abs(order[i] - seed_frame))

    def sustained(i):  # jitter is bad only if a local window is mostly spikes
        lo, hi = max(0, i - win), min(len(order), i + win + 1)
        return spike[lo:hi].mean() > frac

    good = {order[si]}
    for j in range(si - 1, -1, -1):
        if sustained(j):
            break
        good.add(order[j])
    for j in range(si + 1, len(order)):
        if sustained(j):
            break
        good.add(order[j])
    return {f: kept[f] for f in good}


def find_origin(video: str, seed_frame: int, seed_box, arm: str,
                episode: str, label: str = "obj", prompt: str = "block",
                out_mp4: str | None = None, port: int = 8791) -> dict:
    """Back-track an object and render a full-length windowed-overlay mp4.

    The overlay (green mask + box + label) is drawn only on the path-continuous
    reliable window; the rest of the episode plays clean. Returns the origin =
    earliest reliable frame that is also outside a grasp of `arm`.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    frames, fps = video_io.read_frames(video)
    mask0 = _seed_mask(frames[seed_frame], seed_box, prompt, port)
    tr = trk.Tracker(video)
    masks = tr.track(seed_frame, mask0, directions=(True,))  # reverse only
    kept = path_continuous_window(masks, seed_frame)

    busy = meta.contact_frames(episode, arm)
    origin = next((f for f in sorted(kept) if f not in busy), min(kept))
    out_mp4 = out_mp4 or os.path.join(OUT_DIR, f"{label}_origin.mp4")
    video_io.write_overlay(frames, kept, out_mp4, fps, label=label)
    return {"origin_frame": origin, "reliable_window": [min(kept), max(kept)],
            "origin_bbox_xywh": video_io._mask_bbox(kept[origin]),
            "seed_frame": seed_frame, "total_frames": len(frames), "mp4": out_mp4}


if __name__ == "__main__":
    # Demo: find the "O" block's visible, no-contact origin in one episode.
    os.makedirs(OUT_DIR, exist_ok=True)
    trk._static_server(OUT_DIR)
    ep = "xdof/019d2c4e-6b30-790b-b841-1e8445c46ce9"
    res = find_origin(meta.camera_video(ep, "right_wrist_0_rgb"),
                      seed_frame=963, seed_box=[240, 181, 284, 239],
                      arm="right", episode=ep, label="O", prompt="block")
    print(json.dumps(res, indent=2))

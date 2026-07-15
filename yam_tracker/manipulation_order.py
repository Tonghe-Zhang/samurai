"""manipulation_order.py — anchor tracked objects to dataset contact intervals.

Proves the pipeline understands *who* is manipulated *when*, not just "segment
everything". Steps:
  1. track every object (SAM3 seed + SAMURAI) and keep raw per-frame masks,
  2. read the episode's `contact_annotations` -> per-arm grasp intervals,
  3. attribute each interval to the tracked object whose mask centroid moves
     most during it (base_0 is a fixed camera, so idle objects stay still),
  4. render base_0 with the ACTIVE object drawn bright + a "picking #k (arm)"
     banner, and idle objects dimmed — so the manipulation order is visible.

Run:  python manipulation_order.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_meta as meta  # noqa: E402
import tracker as trk  # noqa: E402
import video_io  # noqa: E402
from video_io import mask_centroid as centroid  # noqa: E402

EPISODE = "xdof/019cc383-4530-7285-a592-560b5c000c37"
# Per-camera config keyed by dataset camera name. base_0 is a fixed top camera
# and sees both arms; each wrist rides one arm and only that arm's grasps apply.
# "attrib" picks how the manipulated object is chosen per frame:
#   "motion"  -> object that moves most in the interval (fixed camera),
#   "gripper" -> object nearest the gripper region (moving wrist camera).
CAMERAS = {
    "base0":  {"cam": "base_0_rgb", "seed": "last",
               "arms": ("left", "right"), "attrib": "motion"},
    "lwrist": {"cam": "left_wrist_0_rgb", "seed": 1000,
               "arms": ("left",), "attrib": "gripper"},
    "rwrist": {"cam": "right_wrist_0_rgb", "seed": 1500,
               "arms": ("right",), "attrib": "gripper"},
}
PROMPT, THR = "fruit", 0.85
DIM, BRIGHT = (90, 90, 90), (0, 255, 0)  # idle box gray, active box green


def active_by_motion(per_obj, start, end) -> int | None:
    """Fixed-camera: object whose centroid travels farthest over [start, end]."""
    best, best_disp = None, 8.0  # require >8px net travel to count as manipulated
    for oid, masks in per_obj.items():
        pts = [centroid(masks[f]) for f in range(start, end + 1)
               if f in masks and masks[f].any()]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2:
            continue
        disp = float(np.linalg.norm(np.ptp(np.stack(pts), axis=0)))
        if disp > best_disp:
            best, best_disp = oid, disp
    return best


def active_by_gripper(per_obj, start, end, h, w) -> int | None:
    """Moving wrist-camera: object closest to the gripper (bottom-centre) region.

    The wrist gripper enters frame from the bottom-centre, so the held mango is
    the tracked object whose centroid sits nearest that point across the grasp.
    """
    gripper = np.array([w / 2.0, h * 0.9])
    best, best_d = None, 0.35 * w  # must be within ~a third of the frame width
    for oid, masks in per_obj.items():
        ds_ = [np.linalg.norm(centroid(masks[f]) - gripper)
               for f in range(start, end + 1)
               if f in masks and masks[f].any() and centroid(masks[f]) is not None]
        if not ds_:
            continue
        med = float(np.median(ds_))
        if med < best_d:
            best, best_d = oid, med
    return best


def active_at(intervals, per_obj, cfg, h, w) -> dict[int, tuple[int, str]]:
    """Map frame_idx -> (active_obj_id, arm), using this camera's attrib method."""
    frame_active = {}
    for arm, s, e in intervals:
        if arm not in cfg["arms"]:
            continue  # this wrist camera does not ride that arm
        if cfg["attrib"] == "gripper":
            oid = active_by_gripper(per_obj, s, e, h, w)
        else:
            oid = active_by_motion(per_obj, s, e)
        if oid is None:
            continue
        for f in range(s, e + 1):
            frame_active[f] = (oid, arm)
    return frame_active


def render(frames, per_obj, frame_active, fps, out_path):
    """Bright box + banner on the active object; dim thin boxes on the rest."""
    order = sorted(per_obj)
    out = []
    for idx, frame in enumerate(frames):
        img = frame.copy()
        active = frame_active.get(idx)
        for oid in order:
            m = per_obj[oid].get(idx)
            if m is None or not m.any():
                continue
            box = video_io._mask_bbox(m)
            if box is None:
                continue
            x, y, bw, bh = box
            is_active = active is not None and active[0] == oid
            color = BRIGHT if is_active else DIM
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 3 if is_active else 1)
            cv2.putText(img, f"#{oid + 1}", (x, max(11, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2 if is_active else 1, cv2.LINE_AA)
        if active is not None:
            cv2.putText(img, f"picking #{active[0] + 1} ({active[1]} arm)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BRIGHT, 2, cv2.LINE_AA)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    video_io._encode_h264(out, out_path, fps)


def run_camera(name: str, cfg: dict, intervals):
    """Track one camera's objects and render its manipulation-order video."""
    video = meta.camera_video(EPISODE, cfg["cam"])
    frames, fps = video_io.read_frames(video)
    h, w = frames[0].shape[:2]
    seed_idx = len(frames) - 1 if cfg["seed"] == "last" else int(cfg["seed"])
    seed_png = f"/shared/tmp/track_hangers/_mo_{name}.png"
    cv2.imwrite(seed_png, frames[seed_idx])
    seeds = trk.seed_from_sam3(
        f"http://localhost:{trk._SERVE_PORT}/_mo_{name}.png", h, w, PROMPT, THR)
    print(f"[{name}] {len(seeds)} objects seeded at frame {seed_idx}")
    tr = trk.Tracker(video)
    per_obj = {s["id"]: tr.track(seed_idx, s["mask"], obj_id=s["id"]) for s in seeds}
    del tr
    trk.torch.cuda.empty_cache()
    fa = active_at(intervals, per_obj, cfg, h, w)
    for arm, s, e in intervals:
        if arm not in cfg["arms"]:
            continue
        oid = (active_by_gripper(per_obj, s, e, h, w) if cfg["attrib"] == "gripper"
               else active_by_motion(per_obj, s, e))
        print(f"  [{name}] {arm:5s} [{s:4d}-{e:4d}] -> "
              f"{'#' + str(oid + 1) if oid is not None else '??'}")
    out_path = f"/shared/tmp/track_hangers/out_mango/{name}_manipulation_order.mp4"
    render(frames, per_obj, fa, fps, out_path)
    print(f"[{name}] wrote {out_path}")


def main():
    which = sys.argv[1:] or list(CAMERAS)
    trk._static_server("/shared/tmp/track_hangers")
    intervals = meta.grasp_intervals(EPISODE)
    print(f"{len(intervals)} grasp intervals; cameras: {which}")
    for name in which:
        run_camera(name, CAMERAS[name], intervals)


if __name__ == "__main__":
    main()

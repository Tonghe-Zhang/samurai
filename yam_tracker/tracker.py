"""tracker.py — SAM3 seeding + SAMURAI bidirectional tracking, with a CLI.

Pipeline for one video:
  1. seed_from_sam3(): query the local SAM3 adapter (:8727) with a text prompt
     ("fruit", "mango", ...) on the clearest frame and turn each detection into
     a tight full-frame seed mask. SAM 2 / SAMURAI take no text, so SAM3 is what
     converts language -> geometry.
  2. Tracker.track(): seed SAMURAI at that frame and propagate *backward* then
     *forward* in time, yielding a per-frame mask over the object's whole life.
  3. main(): run every camera of an episode, write per-object + merged H.264
     overlays via video_io, and run a rule-based QC that flags any two tracks
     that are really the same object (high mask IoU).

Run:  python tracker.py --episode <id> --prompt fruit --seed-frame last
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import requests
import torch

# Environment-configurable locations (override via env vars; defaults suit the
# dev cluster). SAM2_REPO holds the SAMURAI checkout, TRACKER_CKPT_DIR the SAM2.1
# checkpoints, SAM3_URL the segmentation adapter.
SAM2_REPO = os.environ.get("SAM2_REPO", "/shared/samurai/sam2")
CKPT_DIR = os.environ.get("TRACKER_CKPT_DIR", "/shared/ckpts/tracker")
sys.path.append(SAM2_REPO)
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

import video_io  # noqa: E402

_CFG = "configs/samurai/sam2.1_hiera_l.yaml"
_CKPT = os.path.join(CKPT_DIR, "sam2.1_hiera_large.pt")
SAM3_URL = os.environ.get("SAM3_URL", "http://localhost:8727/segment")
_SERVE_PORT = int(os.environ.get("TRACKER_SERVE_PORT", "8791"))


# ----------------------------------------------------------------------------- SAM3 seeding
def _static_server(root: str, port: int = _SERVE_PORT) -> ThreadingHTTPServer:
    """Start a background HTTP server rooted at `root` for SAM3 image fetches."""
    handler = partial(SimpleHTTPRequestHandler, directory=root)
    srv = ThreadingHTTPServer(("localhost", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _det_to_mask(det: dict, h: int, w: int) -> np.ndarray:
    """Decode a detection's base64 mask PNG onto a full (h, w) bool canvas."""
    png = base64.b64decode(det["mask_png"].split(",")[-1])
    patch = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    x, y, pw, ph = det["mask_rect"]
    canvas = np.zeros((h, w), dtype=bool)
    canvas[y:y + ph, x:x + pw] = patch > 127
    return canvas


def seed_from_sam3(image_url: str, h: int, w: int, prompt: str,
                   thr: float, min_area: int = 80) -> list[dict]:
    """Return seed dicts {id, mask, box_xyxy, score} sorted left-to-right.

    Left-to-right ordering gives each object a stable, human-readable number
    that matches across cameras and the merged video.
    """
    # The SAM3 adapter occasionally returns 0 detections transiently; retry a
    # few times before giving up so a single flaky call doesn't drop a seed.
    dets = []
    for _ in range(4):
        r = requests.post(SAM3_URL, json={"image_url": image_url, "text": prompt,
                                          "score_threshold": thr}, timeout=60)
        r.raise_for_status()
        dets = r.json()["detections"]
        if dets:
            break
    seeds = []
    for det in dets:
        mask = _det_to_mask(det, h, w)
        if int(mask.sum()) < min_area:
            continue
        ys, xs = np.where(mask)
        seeds.append({"mask": mask, "score": float(det["score"]),
                      "box_xyxy": [int(xs.min()), int(ys.min()),
                                   int(xs.max()), int(ys.max())],
                      "cx": float(xs.mean())})
    seeds.sort(key=lambda s: s["cx"])
    for i, s in enumerate(seeds):
        s["id"] = i
    return seeds


# ----------------------------------------------------------------------------- SAMURAI tracker
class Tracker:
    """SAMURAI video predictor over one video; reuses the encoded state."""

    def __init__(self, video_path: str, device: str = "cuda:0"):
        self.predictor = build_sam2_video_predictor(_CFG, _CKPT, device=device)
        self.state = self.predictor.init_state(
            video_path, offload_video_to_cpu=True, offload_state_to_cpu=True)
        self.num_frames = self.state["num_frames"]

    def track(self, seed_frame: int, seed_mask: np.ndarray, obj_id: int = 0,
              directions: tuple = (True, False)) -> dict[int, np.ndarray]:
        """Seed with a tight mask at `seed_frame` and propagate.

        `directions` selects the passes: (True, False) = reverse then forward
        (whole lifetime), (True,) = backward only (origin search), (False,) =
        forward only (shell game).
        """
        self.predictor.reset_state(self.state)
        masks: dict[int, np.ndarray] = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.predictor.add_new_mask(self.state, frame_idx=seed_frame,
                                        obj_id=obj_id,
                                        mask=torch.as_tensor(seed_mask, dtype=torch.bool))
            for reverse in directions:
                for f_idx, _ids, logits in self.predictor.propagate_in_video(
                        self.state, start_frame_idx=seed_frame, reverse=reverse):
                    masks[f_idx] = (logits[0, 0] > 0.0).cpu().numpy()
        return masks


# ----------------------------------------------------------------------------- QC
def qc_duplicate_tracks(per_obj: dict[int, dict[int, np.ndarray]],
                        iou_thresh: float = 0.5) -> list[tuple]:
    """Flag object pairs whose masks overlap heavily over time (same object).

    For every frame both tracks exist, compute mask IoU; report pairs whose
    mean IoU exceeds `iou_thresh` so we never ship two clips of one object.
    """
    ids = sorted(per_obj)
    flags = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ia, ib = ids[a], ids[b]
            ious = []
            common = set(per_obj[ia]) & set(per_obj[ib])
            for f in common:
                ma, mb = per_obj[ia][f], per_obj[ib][f]
                inter = np.logical_and(ma, mb).sum()
                union = np.logical_or(ma, mb).sum()
                if union:
                    ious.append(inter / union)
            mean_iou = float(np.mean(ious)) if ious else 0.0
            if mean_iou > iou_thresh:
                flags.append((ia, ib, round(mean_iou, 3)))
    return flags


# ----------------------------------------------------------------------------- CLI
def _run_camera(name: str, video: str, prompt: str, seed_frame_arg: str,
                thr: float, out_dir: str):
    """Seed + track every object in one camera, write per-object + merged mp4s."""
    frames, fps = video_io.read_frames(video)
    h, w = frames[0].shape[:2]
    seed_idx = len(frames) - 1 if seed_frame_arg == "last" else int(seed_frame_arg)
    # expose the seed frame over HTTP for the SAM3 adapter
    seed_png = os.path.join(out_dir, f"_seed_{name}.png")
    cv2.imwrite(seed_png, frames[seed_idx])
    seeds = seed_from_sam3(f"http://localhost:{_SERVE_PORT}/{os.path.basename(seed_png)}",
                           h, w, prompt, thr)
    print(f"[{name}] {len(seeds)} seed(s) at frame {seed_idx}")
    tr = Tracker(video)
    per_obj = {}
    for s in seeds:
        per_obj[s["id"]] = tr.track(seed_idx, s["mask"], obj_id=s["id"])
        video_io.write_overlay(frames, per_obj[s["id"]],
                               os.path.join(out_dir, f"{name}_obj{s['id']}.mp4"),
                               fps, label=f"#{s['id'] + 1}")
        print(f"  [{name}] obj{s['id']} -> {name}_obj{s['id']}.mp4")
    video_io.write_merged(frames, per_obj, os.path.join(out_dir, f"{name}_merged.mp4"), fps)
    dups = qc_duplicate_tracks(per_obj)
    print(f"  [{name}] QC duplicate-track pairs (mean IoU>0.5): {dups or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", required=True,
                    help="comma list of name=video_path (e.g. base0=/a.mp4,lwrist=/b.mp4)")
    ap.add_argument("--prompt", default="fruit")
    ap.add_argument("--seed-frame", default="last")
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _static_server(args.out_dir)
    for spec in args.cameras.split(","):
        name, video = spec.split("=", 1)
        _run_camera(name, video, args.prompt, args.seed_frame, args.thr, args.out_dir)


if __name__ == "__main__":
    main()

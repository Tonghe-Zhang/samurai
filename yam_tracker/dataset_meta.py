"""dataset_meta.py — read XDOF episode metadata (blobs + contact intervals).

Single source of truth for the two things the tracking apps need from the
dataset parquet: a camera's video blob path, and the per-arm grasp intervals
parsed from `contact_annotations`. Both manipulation_order and relabel_api
import from here instead of re-parsing the parquet.
"""

from __future__ import annotations

import json
import os

import pyarrow.dataset as ds

# Dataset root (override via XDOF_DATASET_ROOT). Contains split=train/ and blobs/.
ROOT = os.environ.get(
    "XDOF_DATASET_ROOT",
    "/shared/datasets/vla/real/xdof_14k_emb_aligned_independent_trimtail")
DATASET = os.path.join(ROOT, "split=train")
BLOBS = os.path.join(ROOT, "blobs") + "/"
CAMERAS = ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _episode_row(episode: str, columns: list[str]) -> dict:
    """Return the requested columns for one episode as a dict (first match)."""
    d = ds.dataset(DATASET, format="parquet")
    for batch in d.to_batches(columns=["episode_id", *columns]):
        t = batch.to_pydict()
        for i, e in enumerate(t["episode_id"]):
            if e == episode:
                return {c: t[c][i] for c in columns}
    raise RuntimeError(f"episode {episode} not found")


def camera_video(episode: str, camera: str) -> str:
    """Absolute mp4 path for one camera (e.g. 'right_wrist_0_rgb') of an episode."""
    rel = _episode_row(episode, [camera])[camera]
    return BLOBS + rel.split("blobs/")[-1]


def grasp_intervals(episode: str, arm: str | None = None) -> list[tuple[str, int, int]]:
    """Per-arm grasp intervals [(arm, start, end)] from contact_annotations.

    Pairs each `contact` event with the next `no contact` on the same arm.
    Filter to one arm with `arm=`; omit for both. Sorted by start frame.
    """
    events = json.loads(_episode_row(episode, ["contact_annotations"])["contact_annotations"])["events"]
    out, open_f = [], {}
    for x in events:
        a = x["arm"]
        if arm and a != arm:
            continue
        if x["type"] == "contact":
            open_f[a] = x["frame_idx"]
        elif a in open_f:
            out.append((a, open_f.pop(a), x["frame_idx"]))
    return sorted(out, key=lambda r: r[1])


def contact_frames(episode: str, arm: str) -> set[int]:
    """Set of frames where `arm` is grasping (union of its grasp intervals)."""
    busy: set[int] = set()
    for _, s, e in grasp_intervals(episode, arm):
        busy.update(range(s, e + 1))
    return busy

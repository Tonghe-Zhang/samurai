"""video_io.py — load frames from an mp4 and render tracking-overlay videos.

This module owns *all* disk / pixel I/O for the hanger-tracking pipeline:
  * `read_frames`   : decode an mp4 into an in-memory list of BGR frames.
  * `write_overlay` : write an mp4 where a tracked object is painted with a
                      translucent green mask plus a green bounding box.
  * `write_merged`  : write one mp4 that overlays *several* tracked objects at
                      once, each with a numbered label, to show the full
                      pick-and-hang story in a single clip.

Kept deliberately small (<200 lines). The tracking maths lives in tracker.py.
"""

from __future__ import annotations

import cv2
import imageio.v2 as imageio
import numpy as np

# Green used for every mask / box / label (BGR). Distinct per-object hues are
# only used in the merged video so overlapping hangers stay separable.
GREEN = (0, 255, 0)
_PALETTE = [  # BGR, high-contrast, colour-blind-friendly-ish
    (0, 255, 0), (0, 165, 255), (255, 128, 0), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 0, 255),
]


def read_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """Decode `video_path` into a list of BGR frames and return (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return frames, fps


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    """Return [x, y, w, h] tightly bounding the True pixels, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0, y1 - y0]


def mask_centroid(mask: np.ndarray):
    """Return the (x, y) centroid of the True pixels, or None if empty."""
    ys, xs = np.where(mask)
    return None if xs.size == 0 else np.array([xs.mean(), ys.mean()])


def _paint(img: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    """Blend `color` into `img` wherever `mask` is True (in place-ish)."""
    if mask is None or not mask.any():
        return img
    layer = np.zeros_like(img)
    layer[mask] = color
    return cv2.addWeighted(img, 1.0, layer, alpha, 0.0)


def _encode_h264(rgb_frames: list[np.ndarray], out_path: str, fps: float) -> None:
    """Write RGB frames to an H.264 (libx264, yuv420p) mp4 via imageio-ffmpeg.

    cv2's default `mp4v` (MPEG-4 Part 2) is not playable in browsers / many
    viewers, so we always re-encode to H.264 for portability.
    """
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for frame in rgb_frames:
        writer.append_data(frame)
    writer.close()


def write_overlay(
    frames: list[np.ndarray],
    masks: dict[int, np.ndarray],
    out_path: str,
    fps: float,
    label: str = "",
    color=GREEN,
) -> None:
    """Render one object: `masks` maps frame_idx -> bool mask (H, W).

    Frames with no mask entry are written unmodified so the clip stays aligned
    with the source video's timeline. Output is H.264.
    """
    rendered: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        img = frame.copy()
        mask = masks.get(idx)
        if mask is not None and mask.any():
            img = _paint(img, mask, color)
            box = _mask_bbox(mask)
            if box is not None:
                x, y, bw, bh = box
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
                if label:
                    cv2.putText(img, label, (x, max(12, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        rendered.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    _encode_h264(rendered, out_path, fps)


def write_merged(
    frames: list[np.ndarray],
    per_object_masks: dict[int, dict[int, np.ndarray]],
    out_path: str,
    fps: float,
) -> None:
    """Render every tracked object into a single H.264 mp4.

    `per_object_masks` maps object_id -> {frame_idx -> bool mask}. Each object
    gets a stable palette colour and a numbered label so the viewer can follow
    hanger No.1, No.2, ... from the table to the rack.
    """
    obj_ids = sorted(per_object_masks)
    rendered: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        img = frame.copy()
        for oid in obj_ids:
            color = _PALETTE[oid % len(_PALETTE)]
            mask = per_object_masks[oid].get(idx)
            if mask is None or not mask.any():
                continue
            img = _paint(img, mask, color, alpha=0.4)
            box = _mask_bbox(mask)
            if box is not None:
                x, y, bw, bh = box
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(img, f"#{oid + 1}", (x, max(12, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        rendered.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    _encode_h264(rendered, out_path, fps)

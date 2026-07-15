# yam_tracker

Open-vocabulary object tracking + hindsight relabeling for XDOF robot episodes,
built on **SAM3** (text→mask seeding) and **SAMURAI** (motion-aware SAM 2.1
propagation). One engine drives real-video multi-object tracking, the 9-tile
shell-game solver, and contact-anchored manipulation labeling.

## Layout
| File | Role |
|---|---|
| `tracker.py` | SAM3 seeding + `Tracker.track(directions=...)` (reverse/forward/both) + duplicate-track QC + CLI |
| `video_io.py` | H.264 frame load, overlay rendering, `_mask_bbox`, `mask_centroid` |
| `dataset_meta.py` | episode camera-blob paths + per-arm grasp intervals from `contact_annotations` |
| `shellgame.py` | autonomous 9-tile shell-game cup tracker (forward) |
| `manipulation_order.py` | anchor tracks to contact intervals; highlight who/when per camera |
| `relabel_api.py` | back-track an object to its visible, no-contact origin (windowed overlay) |
| `accel/` | TensorRT-encoder acceleration (export → build → run) + benchmark/verify |

SAM 2/SAMURAI take **no text** — SAM3 (adapter on `:8727`) converts a prompt to
a seed mask; SAMURAI propagates it. All three apps share `tracker`, `video_io`,
and `dataset_meta`; no logic is duplicated.

## Requirements
- venv with torch+cuda, sam2 (editable), cv2, imageio-ffmpeg, requests, pyarrow.
- SAM2.1 checkpoint at `/shared/ckpts/tracker/sam2.1_hiera_large.pt`.
- SAM3 adapter live at `http://localhost:8727/segment`.
- `sam2/sam2/modeling/sam2_base.py` patched for reverse-time propagation.

## Usage
```bash
# multi-object track of every "fruit" across an episode's cameras
python tracker.py --cameras base0=/path/a.mp4,lwrist=/path/b.mp4 \
  --prompt fruit --seed-frame last --out-dir out/

# hindsight origin of a named object (reverse-only, windowed overlay)
python relabel_api.py            # see __main__ for the API call signature

# manipulation order anchored to contact intervals
python manipulation_order.py base0 lwrist rwrist

# autonomous shell game -> 3x3 grid mp4
python shellgame.py
```

All outputs are H.264 (`yuv420p`) for universal playback. Heavy assets
(checkpoints, mp4s, TRT engines) are gitignored.

# Object tracking with SAM3 + SAMURAI

Track every instance of an object through an XDOF episode video and render
H.264 overlays (green mask + green bounding box) per object, plus one merged
clip. SAM3 (open-vocab, text→mask) supplies the seeds; SAMURAI (motion-aware
SAM2.1) propagates each seed **backward then forward** in time.

## Files
- `tracker.py` — SAM3 seeding + SAMURAI tracking + duplicate-track QC + CLI.
- `video_io.py` — frame loading and H.264 overlay rendering (per-object & merged).

## Requirements (already set up on this node)
- venv: `/shared/samurai/.venv` (torch+cuda, sam2, cv2, imageio-ffmpeg, requests).
- checkpoint: `/shared/ckpts/tracker/sam2.1_hiera_large.pt`.
- SAM3 adapter live at `http://localhost:8727/segment`.

## Run
```bash
cd /shared/tmp/track_hangers
CUDA_VISIBLE_DEVICES=0 /shared/samurai/.venv/bin/python tracker.py \
  --cameras base0=/path/to/base_0.mp4,lwrist=/path/to/left_wrist.mp4 \
  --prompt fruit \        # SAM3 text query (e.g. "fruit", "mango")
  --seed-frame last \     # frame to detect on: "last" or an integer index
  --thr 0.85 \            # SAM3 score threshold
  --out-dir /shared/tmp/track_hangers/out_mango
```
`--cameras` is a comma list of `name=video_path`. Each camera is seeded and
tracked independently. Pick `--seed-frame` where the objects are clearest and
most separated (top camera: `last`; wrist camera: a mid-episode index).

## Outputs (per camera `<name>`)
- `<name>_obj<k>.mp4` — object #k tracked over the whole video.
- `<name>_merged.mp4` — all objects in one clip, numbered #1, #2, ...
- console prints a QC list of object pairs whose masks overlap heavily
  (mean IoU > 0.5), i.e. two tracks that are likely the *same* object.

## Notes
- SAM2/SAMURAI take **no text** — SAM3 converts the prompt to seed masks.
- Thin/tangled objects (wire hangers) seed poorly from the top camera; use the
  wrist camera, where objects are large and high-contrast.
- All mp4s are H.264 (`yuv420p`) so they play in any browser/viewer.

"""verify_accuracy.py — compare an accelerated grid against the ground-truth grid.

For each of the 9 tiles at a set of probe frames, locate the green tracking box
(the pipeline draws a pure-green (0,255,0) rectangle + "track" label, no fill)
inside that tile's region of BOTH grids and compare the box centres. A tile
"passes" if the box centre stays within `--tol` px of ground truth at every
probe frame (i.e. the SAME cup is tracked).

Usage:
  python verify_accuracy.py --accel out_shellgame/shellgame_trt_grid.mp4
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import shellgame as sg
import video_io

GT = os.path.join(sg.OUT_DIR, "shellgame_tracked_grid.mp4")
PROBES = [45, 90, 130, 185]


def _tile_region(grid: np.ndarray, tid: int) -> np.ndarray:
    gw, gh = sg.TILE_W + sg.GUTTER, sg.TILE_H + sg.GUTTER
    r, c = tid // 3, tid % 3
    y0, x0 = sg.GUTTER + r * gh, sg.GUTTER + c * gw
    return grid[y0:y0 + sg.TILE_H, x0:x0 + sg.TILE_W]


def _green_box_center(bgr_tile: np.ndarray):
    """Return (cx, cy) of the pure-green box pixels, or None if absent."""
    b, g, r = bgr_tile[..., 0], bgr_tile[..., 1], bgr_tile[..., 2]
    mask = (g > 180) & (b < 90) & (r < 90)
    ys, xs = np.where(mask)
    if xs.size < 10:
        return None
    return float(xs.mean()), float(ys.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", required=True)
    ap.add_argument("--tol", type=float, default=25.0)
    args = ap.parse_args()

    gt_frames, _ = video_io.read_frames(GT)
    ac_frames, _ = video_io.read_frames(args.accel)
    n = min(len(gt_frames), len(ac_frames))
    probes = [p for p in PROBES if p < n]

    print(f"GT={GT}\nAC={args.accel}\nframes GT={len(gt_frames)} AC={len(ac_frames)} "
          f"probes={probes} tol={args.tol}px\n")
    all_pass = True
    for tid in range(9):
        row = []
        tile_pass = True
        for f in probes:
            gt_c = _green_box_center(_tile_region(gt_frames[f], tid))
            ac_c = _green_box_center(_tile_region(ac_frames[f], tid))
            if gt_c is None or ac_c is None:
                row.append(f"f{f}:GT={gt_c is not None}/AC={ac_c is not None}")
                if (gt_c is None) != (ac_c is None):
                    tile_pass = False
                continue
            d = float(np.hypot(gt_c[0] - ac_c[0], gt_c[1] - ac_c[1]))
            row.append(f"f{f}:d={d:.0f}")
            if d > args.tol:
                tile_pass = False
        verdict = "PASS" if tile_pass else "FAIL"
        all_pass &= tile_pass
        print(f"  tile{tid}: {verdict}  " + "  ".join(row))
    print(f"\nOVERALL: {'PASS — accel tracks the same cups as ground truth' if all_pass else 'FAIL — divergence detected'}")


if __name__ == "__main__":
    main()

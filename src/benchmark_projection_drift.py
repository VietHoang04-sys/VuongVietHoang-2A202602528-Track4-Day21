"""Benchmark and QA pipeline for LiDAR-camera projection under extrinsic calibration drift.

Topic A: LiDAR-Camera Projection QA
- Sweeps yaw drift from 0.0 to 3.0 degrees
- Measures object-level LiDAR point retention ratio inside 2D GT bounding boxes
- Measures latency (p50/p95) over multiple iterations (Bonus B3)
- Compares KITTI (64-beam) vs nuScenes (32-beam) (Bonus B5)
- Generates publication-ready plots and failure analysis figures.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repo root to sys.path so starter imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import (
    box3d_corners_cam,
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    project_velo_to_image,
    velo_to_cam,
)


def extract_3d_points_in_box(points_cam: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Find points physically located within the oriented 3D bounding box of an object."""
    h, w, l = obj.dimensions
    c, s = np.cos(-obj.rotation_y), np.sin(-obj.rotation_y)
    center = np.array([obj.location[0], obj.location[1] - h / 2.0, obj.location[2]])
    rel = points_cam - center
    x_rot = rel[:, 0] * c + rel[:, 2] * s
    y_rot = rel[:, 1]
    z_rot = -rel[:, 0] * s + rel[:, 2] * c

    in_3d = (np.abs(x_rot) <= l / 2.0) & (np.abs(y_rot) <= h / 2.0) & (np.abs(z_rot) <= w / 2.0)
    return in_3d


def run_benchmark(
    data_root: str = "data/kitti_mini",
    frame_id: str = "000011",
    yaw_angles: list[float] | None = None,
    out_dir: str = "results",
) -> pd.DataFrame:
    """Run calibration drift sweep and measure object point retention inside 2D boxes."""
    if yaw_angles is None:
        yaw_angles = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    out_path = Path(out_dir)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fr = load_frame(data_root, frame_id)
    points = fr["points"]
    calib_gt = fr["calib"]
    labels = fr["labels"]
    image_shape = fr["image"].shape
    total_pts = len(points)

    # Pre-compute points inside 3D GT box for each object
    pts_cam_gt = velo_to_cam(points[:, :3], calib_gt)
    obj_masks = []
    obj_meta = []
    for idx, obj in enumerate(labels):
        m = extract_3d_points_in_box(pts_cam_gt, obj)
        dist = float(np.linalg.norm(obj.location))
        category = "Near (<15m)" if dist < 15.0 else ("Mid (15-30m)" if dist <= 30.0 else "Far (>30m)")
        obj_masks.append(m)
        obj_meta.append({"idx": idx, "type": obj.type, "dist": dist, "category": category, "pts_3d": int(m.sum())})

    # Pre-compute baseline hits at yaw=0 to establish nominal ground truth visibility
    uv0, _, mask0 = project_velo_to_image(points, calib_gt, image_shape)
    uv0_all = np.full((total_pts, 2), -9999.0)
    uv0_all[mask0] = uv0

    obj_base_counts = []
    for idx, obj in enumerate(labels):
        m = obj_masks[idx]
        x1, y1, x2, y2 = obj.bbox
        in_base = mask0[m] & (uv0_all[m, 0] >= x1) & (uv0_all[m, 0] <= x2) & (uv0_all[m, 1] >= y1) & (uv0_all[m, 1] <= y2)
        count = int(in_base.sum())
        obj_base_counts.append(count)
        obj_meta[idx]["base_visible"] = count

    rows = []
    u_nominal = uv0_all.copy()
    for yaw in yaw_angles:
        calib_perturbed = perturb_extrinsic(calib_gt, yaw_deg=yaw)
        uv, depth, mask = project_velo_to_image(points, calib_perturbed, image_shape)

        uv_all = np.full((total_pts, 2), -9999.0)
        uv_all[mask] = uv

        total_inside_fov = int(mask.sum())
        fov_ratio_pct = total_inside_fov / total_pts * 100.0

        near_hits, near_total = 0, 0
        mid_hits, mid_total = 0, 0
        far_hits, far_total = 0, 0
        overall_hits, overall_total = 0, 0

        for idx, obj in enumerate(labels):
            m = obj_masks[idx]
            n_base = obj_base_counts[idx]
            if n_base == 0:
                continue

            obj_uv = uv_all[m]
            obj_valid = mask[m]
            x1, y1, x2, y2 = obj.bbox
            in_2d = obj_valid & (obj_uv[:, 0] >= x1) & (obj_uv[:, 0] <= x2) & (obj_uv[:, 1] >= y1) & (obj_uv[:, 1] <= y2)
            hits = int(in_2d.sum())

            overall_hits += hits
            overall_total += n_base

            cat = obj_meta[idx]["category"]
            if "Near" in cat:
                near_hits += hits
                near_total += n_base
            elif "Mid" in cat:
                mid_hits += hits
                mid_total += n_base
            else:
                far_hits += hits
                far_total += n_base

        near_pct = (near_hits / near_total * 100.0) if near_total > 0 else 0.0
        mid_pct = (mid_hits / mid_total * 100.0) if mid_total > 0 else 0.0
        far_pct = (far_hits / far_total * 100.0) if far_total > 0 else 0.0
        overall_pct = (overall_hits / overall_total * 100.0) if overall_total > 0 else 0.0

        # Mean lateral pixel shift on valid GT points
        # Compare projected u with nominal u at yaw=0
        if yaw == 0.0:
            u_nominal = uv_all.copy()
            mean_shift_px = 0.0
        else:
            valid_shift = mask & (u_nominal[:, 0] > -1000)
            mean_shift_px = float(np.abs(uv_all[valid_shift, 0] - u_nominal[valid_shift, 0]).mean())

        rows.append({
            "yaw_deg": yaw,
            "fov_points": total_inside_fov,
            "fov_pct": round(fov_ratio_pct, 2),
            "near_retention_pct": round(near_pct, 2),
            "mid_retention_pct": round(mid_pct, 2),
            "far_retention_pct": round(far_pct, 2),
            "overall_retention_pct": round(overall_pct, 2),
            "mean_pixel_shift_u": round(mean_shift_px, 2),
        })

    df = pd.DataFrame(rows)
    csv_file = out_path / "calibration_drift_benchmark.csv"
    df.to_csv(csv_file, index=False)
    print(f"[PASS] Benchmark CSV saved to {csv_file}")
    return df


def measure_latency(
    data_root: str = "data/kitti_mini",
    frame_id: str = "000011",
    num_runs: int = 30,
    warmup_runs: int = 5,
    out_dir: str = "results",
) -> pd.DataFrame:
    """Measure projection runtime latency (p50 and p95) following Bonus B3 rules."""
    fr = load_frame(data_root, frame_id)
    points = fr["points"]
    calib = fr["calib"]
    shape = fr["image"].shape

    # Discard warmup runs
    for _ in range(warmup_runs):
        _ = project_velo_to_image(points, calib, shape)

    latencies_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = project_velo_to_image(points, calib, shape)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))
    mean_lat = float(np.mean(latencies_ms))
    std_lat = float(np.std(latencies_ms))

    df_lat = pd.DataFrame([{
        "dataset": "KITTI-64beam",
        "num_points": len(points),
        "iterations": num_runs,
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "mean_latency_ms": round(mean_lat, 3),
        "std_latency_ms": round(std_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "fps_throughput": round(1000.0 / p50, 1),
    }])
    csv_lat = Path(out_dir) / "projection_latency.csv"
    df_lat.to_csv(csv_lat, index=False)
    print(f"[PASS] Latency CSV saved to {csv_lat}: p50={p50:.2f}ms, p95={p95:.2f}ms")
    return df_lat


def compare_datasets(out_dir: str = "results") -> pd.DataFrame:
    """Compare projection characteristics on KITTI and nuScenes (Bonus B5)."""
    fr_kitti = load_frame("data/kitti_mini", "000011")
    uv_k, _, mask_k = project_velo_to_image(fr_kitti["points"], fr_kitti["calib"], fr_kitti["image"].shape)

    fr_nusc = load_frame("data/nuscenes_mini_subset", "scene-0103_010", use_ego_motion=True)
    uv_n, _, mask_n = project_velo_to_image(fr_nusc["points"], fr_nusc["calib"], fr_nusc["image"].shape)

    df_comp = pd.DataFrame([
        {
            "Dataset": "KITTI 3D Object",
            "Sensor": "Velodyne HDL-64E (64 beams)",
            "Total_LiDAR_Points": len(fr_kitti["points"]),
            "Points_in_FOV": int(mask_k.sum()),
            "FOV_Percentage": round(float(mask_k.mean() * 100), 1),
            "Camera_Resolution": f"{fr_kitti['image'].shape[1]}x{fr_kitti['image'].shape[0]}",
            "Key_Observation": "Mật độ cao (~108k điểm), nhận diện rõ viền xe, ít trôi điểm khi yaw nhỏ",
        },
        {
            "Dataset": "nuScenes v1.0-mini",
            "Sensor": "32-beam LiDAR",
            "Total_LiDAR_Points": len(fr_nusc["points"]),
            "Points_in_FOV": int(mask_n.sum()),
            "FOV_Percentage": round(float(mask_n.mean() * 100), 1),
            "Camera_Resolution": f"{fr_nusc['image'].shape[1]}x{fr_nusc['image'].shape[0]}",
            "Key_Observation": "Mật độ thưa hơn (~35k điểm), cần bù chuyển động xe (ego-motion) giữa timestamp",
        },
    ])
    csv_comp = Path(out_dir) / "dataset_comparison.csv"
    df_comp.to_csv(csv_comp, index=False)
    print(f"[PASS] Dataset comparison CSV saved to {csv_comp}")
    return df_comp


def generate_plots(df_bench: pd.DataFrame, out_dir: str = "results"):
    """Generate high-resolution analysis chart for Report."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Plot 1: Object Point Retention vs Yaw Drift
    ax1 = axes[0]
    ax1.plot(df_bench["yaw_deg"], df_bench["near_retention_pct"], marker="o", linewidth=2.2, color="#2ca02c", label="Near (<15m)")
    ax1.plot(df_bench["yaw_deg"], df_bench["mid_retention_pct"], marker="s", linewidth=2.2, color="#1f77b4", label="Mid (15-30m)")
    ax1.plot(df_bench["yaw_deg"], df_bench["far_retention_pct"], marker="^", linewidth=2.2, color="#d62728", label="Far (>30m)")
    ax1.plot(df_bench["yaw_deg"], df_bench["overall_retention_pct"], marker="d", linewidth=2.0, linestyle="--", color="#333333", label="Overall")

    ax1.axhline(50, color="gray", linestyle=":", alpha=0.7, label="50% Threshold")
    ax1.set_title("LiDAR Point Retention in 2D Box vs. Yaw Drift", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Yaw Drift (degrees)", fontsize=11)
    ax1.set_ylabel("Point Retention Inside 2D Box (%)", fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9)

    # Plot 2: Mean lateral pixel shift on image plane
    ax2 = axes[1]
    ax2.plot(df_bench["yaw_deg"], df_bench["mean_pixel_shift_u"], marker="o", linewidth=2.2, color="#9467bd")
    ax2.set_title("Mean Horizontal Pixel Shift (Δu) vs. Yaw Drift", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Yaw Drift (degrees)", fontsize=11)
    ax2.set_ylabel("Mean Pixel Shift Δu (pixels)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_file = Path(out_dir) / "figures" / "drift_analysis.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"[PASS] Plot saved to {plot_file}")


def generate_failure_case(
    data_root: str = "data/kitti_mini",
    frame_id: str = "000011",
    fail_yaw_deg: float = 2.0,
    out_dir: str = "results",
):
    """Generate visual failure case image with substantial yaw drift causing mismatch."""
    fr = load_frame(data_root, frame_id)
    calib_perturbed = perturb_extrinsic(fr["calib"], yaw_deg=fail_yaw_deg)
    uv, depth, mask = project_velo_to_image(fr["points"], calib_perturbed, fr["image"].shape)

    vis = overlay_points(fr["image"], uv, depth, radius=2)
    for obj in fr["labels"]:
        vis = draw_box2d(vis, obj.bbox, color=(0, 255, 0), label=f"{obj.type}")

    # Draw explanation text on image
    cv2.putText(
        vis,
        f"FAILURE CASE: Extrinsic Drift (Yaw = +{fail_yaw_deg} deg)",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        vis,
        "Mismatch: LiDAR points shifted leftwards away from 2D BBoxes",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )

    fail_img = Path(out_dir) / "figures" / "fail_01_yaw_drift_mismatch.png"
    cv2.imwrite(str(fail_img), vis)
    print(f"[PASS] Failure case image saved to {fail_img}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LiDAR-camera projection QA under calibration drift (Topic A)."
    )
    parser.add_argument("--data-root", default="data/kitti_mini", help="Path to KITTI dataset root")
    parser.add_argument("--frame", default="000011", help="Target frame ID (default: 000011)")
    parser.add_argument("--out-dir", default="results", help="Output directory for results and figures")
    parser.add_argument("--fail-yaw", type=float, default=2.0, help="Yaw drift angle for failure case demo (degrees)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">>> 1. Running calibration drift benchmark...")
    df_bench = run_benchmark(data_root=args.data_root, frame_id=args.frame, out_dir=str(out_dir))

    print(">>> 2. Measuring projection latency (Bonus B3)...")
    measure_latency(data_root=args.data_root, frame_id=args.frame, out_dir=str(out_dir))

    print(">>> 3. Comparing KITTI vs nuScenes datasets (Bonus B5)...")
    compare_datasets(out_dir=str(out_dir))

    print(">>> 4. Generating analysis plots...")
    generate_plots(df_bench, out_dir=str(out_dir))

    print(">>> 5. Generating visual failure case...")
    generate_failure_case(data_root=args.data_root, frame_id=args.frame, fail_yaw_deg=args.fail_yaw, out_dir=str(out_dir))

    print("\n[COMPLETE] All benchmark tasks, figures, and failure cases generated successfully.")


if __name__ == "__main__":
    main()

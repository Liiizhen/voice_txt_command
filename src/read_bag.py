import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np


def load_depth(npz_path: str) -> Tuple[np.ndarray, float]:
    with np.load(npz_path) as data:
        depth = data["depth"]
        scale = float(data.get("depth_scale", 1.0))
    return depth, scale


def load_rgb_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from video.")
    return frame


def depth_to_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)

    z = depth[grid_y, grid_x].astype(np.float32) * scale
    mask = z > 0
    x = (grid_x.astype(np.float32) - cx) * z / fx
    y = (grid_y.astype(np.float32) - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)[mask]
    colors = rgb[grid_y, grid_x][mask]
    return points, colors


def downsample(points: np.ndarray, colors: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    idx = np.random.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx]


def visualize(points: np.ndarray, colors: np.ndarray) -> None:
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
        o3d.visualization.draw_geometries([pcd])
    except Exception:
        # Fallback to matplotlib if open3d isn't available.
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors.astype(np.float32) / 255.0,
            s=0.2,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors_uint8 = colors.astype(np.uint8)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for (x, y, z), (r, g, b) in zip(points, colors_uint8):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def generate_sequence(
    rgb_path: str,
    depth_all: np.ndarray,
    scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    max_points: int,
    out_dir: str,
    start: int,
    end: Optional[int],
    step: int,
) -> None:
    cap = cv2.VideoCapture(rgb_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {rgb_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None or end > total:
        end = total
    if end > depth_all.shape[0]:
        end = depth_all.shape[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_idx = start
    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start) % step != 0:
            frame_idx += 1
            continue

        depth = depth_all[frame_idx]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points, colors = depth_to_points(
            depth=depth,
            rgb=rgb,
            scale=scale,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            stride=max(1, stride),
        )
        points, colors = downsample(points, colors, max_points)

        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.ply")
        write_ply(out_path, points, colors)
        print(f"[{frame_idx}] points={len(points)} -> {out_path}")
        frame_idx += 1

    cap.release()


def save_frames(
    rgb_path: str,
    out_dir: str,
    start: int,
    end: Optional[int],
    step: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(rgb_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {rgb_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None or end > total:
        end = total

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_idx = start
    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start) % step != 0:
            frame_idx += 1
            continue
        out_path = os.path.join(out_dir, f"frame_{frame_idx}.png")
        cv2.imwrite(out_path, frame)
        frame_idx += 1
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB + depth -> point cloud")
    parser.add_argument("--rgb", default="../data/rgb.mp4")
    parser.add_argument("--depth", default="../data/depth_all.npz")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--save-frames-dir", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=150000)
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    args = parser.parse_args()

    if args.save_frames_dir:
        save_frames(
            rgb_path=args.rgb,
            out_dir=args.save_frames_dir,
            start=max(0, args.start),
            end=args.end,
            step=max(1, args.step),
        )
        return

    depth_all, scale = load_depth(args.depth)

    if args.out_dir:
        generate_sequence(
            rgb_path=args.rgb,
            depth_all=depth_all,
            scale=scale,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            stride=args.stride,
            max_points=args.max_points,
            out_dir=args.out_dir,
            start=max(0, args.start),
            end=args.end,
            step=max(1, args.step),
        )
        return

    if args.frame_index < 0 or args.frame_index >= depth_all.shape[0]:
        raise ValueError("frame-index out of range")
    depth = depth_all[args.frame_index]

    rgb_bgr = load_rgb_frame(args.rgb, args.frame_index)
    print(len(rgb_bgr))
    print(rgb_bgr.shape)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    print(len(rgb))
    print(rgb.shape)
    visualize(rgb_bgr, rgb)


if __name__ == "__main__":
    main()
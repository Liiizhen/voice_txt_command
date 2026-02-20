import os
import re
import time
from glob import glob
import numpy as np
import cv2
from ultralytics import YOLO


def _frame_index(path: str) -> int:
    match = re.search(r"frame_(\d+)\.", os.path.basename(path))
    return int(match.group(1)) if match else -1

def draw_masks_only(result, frame):
    """
    只绘制 mask，不绘制类别和框
    """
    if result.masks is None:
        return frame

    # 每个实例的 mask
    masks = result.masks.data.cpu().numpy()  # shape: [N, H, W]

    # 给每个 mask 随机颜色
    for mask in masks:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        # mask 是二值化的，True/False 或 0/1
        mask_bool = mask.astype(bool)
        frame[mask_bool] = frame[mask_bool] * 0.5 + color * 0.5

    return frame


def main() -> None:
    frames_dir = "/home/user/nelson/voice_txt_command/data/frames"
    out_dir = "/home/user/nelson/voice_txt_command/data/yolo_seg4"
    video_path = "/home/user/nelson/voice_txt_command/data/rgb.mp4"
    out_video = "/home/user/nelson/voice_txt_command/data/yolo_seg4.mp4"
    fps_log = "/home/user/nelson/voice_txt_command/data/yolo_seg_fps.txt"
    os.makedirs(out_dir, exist_ok=True)

    # 加载预训练的 YOLOv8-Seg 模型（可以选择 n/s/m/l/x）
    model = YOLO("yolov8l-seg.pt")  # n=Nano, s=Small, m=Medium, l=Large, x=XL

    frame_paths = sorted(glob(os.path.join(frames_dir, "frame_*.png")), key=_frame_index)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    for path in frame_paths:
        result = model(path, verbose=False)[0]
        frame = cv2.imread(path)
        rendered = draw_masks_only(result, frame)
        out_path = os.path.join(out_dir, os.path.basename(path))
        cv2.imwrite(out_path, rendered)
        print(f"Saved {out_path}")


    # Second pass: process the whole video and save segmentation video + FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (width, height),
    )

    frame_count = 0
    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model(frame, verbose=False)[0]
        rendered = draw_masks_only(result, frame)
        writer.write(rendered)
        frame_count += 1

    end_time = time.time()
    cap.release()
    writer.release()

    elapsed = max(1e-6, end_time - start_time)
    seg_fps = frame_count / elapsed
    with open(fps_log, "w", encoding="utf-8") as handle:
        handle.write(f"{seg_fps:.3f}\n")
    print(f"Saved {out_video}")
    print(f"Seg FPS: {seg_fps:.3f} (logged to {fps_log})")


if __name__ == "__main__":
    main()
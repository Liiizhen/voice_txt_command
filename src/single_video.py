import argparse
import os
import queue
import socket
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from asr_whisper import build_asr, transcribe_audio
from bbox_from_image import load_vlm_bbox, run_vlm_bbox
from command_from_text import run_vlm
from utils import resolve_device

HERE = os.path.dirname(__file__)

HOST = "0.0.0.0"
PORT = 50007
LANGUAGE = "en"
CHUNK = 1024  # bytes
RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16

# VAD / buffering settings
START_WINDOW_SECONDS = 2.0
START_RMS_THRESHOLD = 500.0
STOP_RMS_THRESHOLD = 300.0
STOP_SILENCE_SECONDS = 0.3
STOP_WINDOW_SECONDS = 0.2
MIN_SPEECH_SECONDS = 0.2


class HighPassFilter:
    def __init__(self, cutoff: float = 200.0, fs: float = 16000.0) -> None:
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * np.pi * cutoff)
        self.alpha = rc / (rc + dt)
        self.prev_x = 0.0
        self.prev_y = 0.0

    def process(self, x: float) -> float:
        y = self.alpha * (self.prev_y + x - self.prev_x)
        self.prev_x = x
        self.prev_y = y
        return y

    def process_array(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y = np.empty_like(x, dtype=np.float32)
        prev_x = self.prev_x
        prev_y = self.prev_y
        alpha = self.alpha
        for i, xi in enumerate(x):
            yi = alpha * (prev_y + xi - prev_x)
            y[i] = yi
            prev_x = float(xi)
            prev_y = float(yi)
        self.prev_x = prev_x
        self.prev_y = prev_y
        return y


def _chunk_rms(chunk: bytes, hp_filter: Optional["HighPassFilter"] = None) -> float:
    if len(chunk) < SAMPLE_WIDTH:
        return 0.0
    if len(chunk) % SAMPLE_WIDTH != 0:
        chunk = chunk[: len(chunk) - (len(chunk) % SAMPLE_WIDTH)]
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    if hp_filter is not None:
        samples = hp_filter.process_array(samples)
    return float(np.sqrt(np.mean(samples ** 2)))


def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
    if len(raw_bytes) % SAMPLE_WIDTH != 0:
        raw_bytes = raw_bytes[: len(raw_bytes) - (len(raw_bytes) % SAMPLE_WIDTH)]
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return samples / 32768.0


def _resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _start_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(1)
    print(f"等待客户端连接 {port} ...")
    conn, addr = sock.accept()
    print("客户端已连接:", addr)
    return conn


def _audio_listener(
    conn: socket.socket,
    asr: Any,
    language: str,
    out_queue: "queue.Queue[str]",
    stop_event: threading.Event,
) -> None:
    recording = False
    buffer = b""
    silence_seconds = 0.0

    chunk_duration = (CHUNK / (SAMPLE_WIDTH * CHANNELS)) / RATE
    window_chunks = max(1, int(START_WINDOW_SECONDS / chunk_duration))
    rms_window = deque(maxlen=window_chunks)
    prebuffer = deque(maxlen=window_chunks)
    hp_filter = HighPassFilter(cutoff=200.0, fs=RATE)
    stop_window_chunks = max(1, int(STOP_WINDOW_SECONDS / chunk_duration))
    stop_rms_window = deque(maxlen=stop_window_chunks)

    def _finalize_recording(reason: str) -> None:
        nonlocal recording, buffer, silence_seconds
        duration = len(buffer) / (SAMPLE_WIDTH * CHANNELS * RATE)
        if duration >= MIN_SPEECH_SECONDS:
            audio = _bytes_to_float32(buffer)
            text = transcribe_audio(
                asr=asr,
                audio=audio,
                sample_rate=RATE,
                language=language,
                return_timestamps=True,
            )
            text = text.strip()
            if text:
                out_queue.put(text)
                print(f"[识别结果] {text}")
        buffer = b""
        recording = False
        silence_seconds = 0.0
        rms_window.clear()
        prebuffer.clear()
        print(reason)

    while not stop_event.is_set():
        data = conn.recv(CHUNK)
        if not data:
            break

        rms = _chunk_rms(data, hp_filter)
        rms_window.append(rms)
        prebuffer.append(data)
        stop_rms_window.append(rms)
        stop_rms = float(np.mean(stop_rms_window)) if stop_rms_window else rms

        if not recording:
            if len(rms_window) == rms_window.maxlen:
                window_rms = float(np.mean(rms_window))
                if window_rms >= START_RMS_THRESHOLD:
                    recording = True
                    buffer = b"".join(prebuffer)
                    silence_seconds = 0.0
                    print("🎤 开始录音（VAD触发）...")
            continue

        buffer += data
        if stop_rms < STOP_RMS_THRESHOLD:
            silence_seconds += chunk_duration
        else:
            silence_seconds = 0.0

        if silence_seconds >= STOP_SILENCE_SECONDS:
            _finalize_recording("🛑 停止录音（静音触发）")


def _text_listener(
    conn: socket.socket,
    out_queue: "queue.Queue[str]",
    stop_event: threading.Event,
) -> None:
    buffer = b""
    while not stop_event.is_set():
        data = conn.recv(CHUNK)
        if not data:
            break
        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="ignore").strip()
            if text:
                out_queue.put(text)
                print(f"[文本输入] {text}")


def _yolo_detect(model: YOLO, frame_bgr: np.ndarray) -> Dict[str, Any]:
    result = model(frame_bgr, verbose=False)[0]
    h, w = frame_bgr.shape[:2]

    if result.boxes is None or result.boxes.xyxy is None:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "masks": np.zeros((0, h, w), dtype=np.uint8),
            "classes": np.zeros((0,), dtype=np.int32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.detach().cpu().numpy()
        masks = (masks > 0.5).astype(np.uint8)
        if masks.shape[1] != h or masks.shape[2] != w:
            resized = np.zeros((masks.shape[0], h, w), dtype=np.uint8)
            for i in range(masks.shape[0]):
                resized[i] = cv2.resize(
                    masks[i], (w, h), interpolation=cv2.INTER_NEAREST
                )
            masks = resized
    else:
        masks = np.zeros((0, h, w), dtype=np.uint8)

    keep_indices = []
    for i, cls_id in enumerate(classes):
        cls_name = result.names[int(cls_id)]
        if cls_name not in {"dining table", "table"}:
            keep_indices.append(i)

    boxes = boxes[keep_indices]
    masks = masks[keep_indices]
    classes = classes[keep_indices]
    scores = scores[keep_indices]

    return {"boxes": boxes, "masks": masks, "classes": classes, "scores": scores}



def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [float(x1), float(y1), float(x2), float(y2)]


def _warp_mask(prev_mask: np.ndarray, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = prev_mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(
        prev_mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def _refine_mask(
    warped_mask: np.ndarray, boxes: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return warped_mask
    warped_bbox = _mask_to_bbox(warped_mask)
    if warped_bbox is None:
        return warped_mask
    ious = [_bbox_iou(warped_bbox, box.tolist()) for box in boxes]
    best_idx = int(np.argmax(ious)) if ious else -1
    if best_idx >= 0 and ious[best_idx] > 0:
        return masks[best_idx]
    return warped_mask


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    if mask is None or mask.size == 0:
        return frame
    overlay = frame.copy()
    alpha = 0.5
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    )
    return overlay


_COLOR_PALETTE = [
    (255, 99, 71),
    (30, 144, 255),
    (50, 205, 50),
    (255, 215, 0),
    (138, 43, 226),
    (0, 206, 209),
    (255, 105, 180),
    (160, 82, 45),
    (0, 191, 255),
    (154, 205, 50),
    (255, 140, 0),
    (72, 61, 139),
]


def _color_for_index(idx: int) -> Tuple[int, int, int]:
    return _COLOR_PALETTE[idx % len(_COLOR_PALETTE)]


def _draw_yolo_results(
    frame: np.ndarray, dets: Optional[Dict[str, Any]], score_threshold: float = 0.3
) -> np.ndarray:
    if dets is None:
        return frame
    boxes = dets.get("boxes")
    masks = dets.get("masks")
    classes = dets.get("classes")
    scores = dets.get("scores")
    if boxes is None or len(boxes) == 0:
        return frame

    vis = frame.copy()
    alpha = 0.45
    for idx, box in enumerate(boxes):
        score = float(scores[idx]) if scores is not None and len(scores) > idx else 1.0
        if score < score_threshold:
            continue
        color = _color_for_index(idx)
        if masks is not None and masks.size > 0:
            mask = masks[idx].astype(bool)
            vis[mask] = vis[mask] * (1 - alpha) + np.array(color) * alpha
        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"id{int(classes[idx])} {score:.2f}".strip() if classes is not None else f"{score:.2f}"
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Single video + VLM + YOLO-Seg tracking")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--asr-model", default="../hf_models/whisper-small")
    parser.add_argument("--vlm-model", default="../hf_models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--yolo-model", default="yolov8l-seg.pt")
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--window-name", default="single-video")
    parser.add_argument("--window-x", type=int, default=100)
    parser.add_argument("--window-y", type=int, default=100)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--vlm-4bit", action="store_true", default=True)
    parser.add_argument("--no-vlm-4bit", dest="vlm_4bit", action="store_false")
    parser.add_argument("--input-mode", choices=["audio", "text"], default="audio")
    args = parser.parse_args()

    device = resolve_device(args.device)
    asr = None
    if args.input_mode == "audio":
        asr = build_asr(args.asr_model, device)
    vlm_processor, vlm_tokenizer, vlm_model = load_vlm_bbox(
        model_id=args.vlm_model, device=device, load_4bit=args.vlm_4bit
    )
    yolo_model = YOLO(args.yolo_model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(args.window_name, args.window_x, args.window_y)
    cv2.resizeWindow(args.window_name, 1080, 720)

    conn = _start_socket(args.host, args.port)
    transcript_queue: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()
    if args.input_mode == "audio":
        audio_thread = threading.Thread(
            target=_audio_listener,
            args=(conn, asr, args.language, transcript_queue, stop_event),
            daemon=True,
        )
        audio_thread.start()
    else:
        text_thread = threading.Thread(
            target=_text_listener,
            args=(conn, transcript_queue, stop_event),
            daemon=True,
        )
        text_thread.start()

    latest_frame: Optional[np.ndarray] = None
    latest_dets: Optional[Dict[str, Any]] = None
    frame_index = 0

    recording = False
    recorded: List[Dict[str, Any]] = []
    start_record: Optional[Dict[str, Any]] = None
    vlm_pending = False
    vlm_done = False
    vlm_bbox: Optional[List[float]] = None
    vlm_command: Optional[Dict[str, Any]] = None
    target_mask: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    prev_mask: Optional[np.ndarray] = None

    def _vlm_worker(text: str, image: Image.Image) -> None:
        nonlocal vlm_done, vlm_bbox, vlm_command, vlm_pending
        try:
            vlm_command = run_vlm(
                model=vlm_model,
                tokenizer=vlm_tokenizer,
                text=text,
                config_path=args.config,
            )
            bbox_result = run_vlm_bbox(
                processor=vlm_processor,
                model=vlm_model,
                image=image,
                command=vlm_command,
                config_path=args.config,
            )
            vlm_bbox = bbox_result.get("bbox", [])
        except Exception as exc:
            print(f"[vlm] error: {exc}")
            vlm_bbox = []
            vlm_command = None
        vlm_done = True
        vlm_pending = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            frame = _resize_frame(frame, args.max_side)

            dets = _yolo_detect(yolo_model, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            latest_frame = frame
            latest_dets = dets

            if recording:
                recorded.append(
                    {"frame": frame.copy(), "gray": gray, "boxes": dets["boxes"], "masks": dets["masks"]}
                )

            while not transcript_queue.empty() and not vlm_pending:
                text = transcript_queue.get()
                if latest_frame is None or latest_dets is None:
                    continue
                start_record = {
                    "frame": latest_frame.copy(),
                    "gray": gray.copy(),
                    "boxes": latest_dets["boxes"],
                    "masks": latest_dets["masks"],
                }
                recorded = [start_record]
                recording = True
                vlm_pending = True
                vlm_done = False
                pil_image = Image.fromarray(cv2.cvtColor(start_record["frame"], cv2.COLOR_BGR2RGB))
                threading.Thread(
                    target=_vlm_worker, args=(text, pil_image), daemon=True
                ).start()

            if vlm_done and start_record is not None:
                vlm_done = False
                recording = False
                if vlm_bbox and len(start_record["boxes"]) > 0:
                    ious = [
                        _bbox_iou(vlm_bbox, box.tolist())
                        for box in start_record["boxes"]
                    ]
                    best_idx = int(np.argmax(ious)) if ious else -1
                    if best_idx >= 0 and ious[best_idx] > 0:
                        target_mask = start_record["masks"][best_idx]
                        prev_mask = target_mask
                        prev_gray = start_record["gray"]
                # playback recorded frames
                if target_mask is not None:
                    cur_mask = target_mask
                    cur_gray = start_record["gray"]
                    for rec in recorded:
                        if rec is start_record:
                            cur_mask = target_mask
                            cur_gray = rec["gray"]
                        else:
                            warped = _warp_mask(cur_mask, cur_gray, rec["gray"])
                            cur_mask = _refine_mask(warped, rec["boxes"], rec["masks"])
                            cur_gray = rec["gray"]
                        vis = _overlay_mask(rec["frame"], cur_mask)
                        cv2.imshow(args.window_name, vis)
                        cv2.waitKey(1)
                    prev_mask = cur_mask
                    prev_gray = cur_gray
                recorded = []

            if prev_mask is not None and prev_gray is not None and not recording:
                warped = _warp_mask(prev_mask, prev_gray, gray)
                cur_mask = _refine_mask(warped, dets["boxes"], dets["masks"])
                vis = _overlay_mask(frame, cur_mask)
                cv2.imshow(args.window_name, vis)
                cv2.waitKey(1)
                prev_mask = cur_mask
                prev_gray = gray
            elif not recording:
                vis = _draw_yolo_results(frame, dets)
                cv2.imshow(args.window_name, vis)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("停止接收")
    finally:
        stop_event.set()
        cap.release()
        conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

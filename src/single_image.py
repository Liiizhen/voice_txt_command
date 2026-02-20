import argparse
import socket
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from asr_whisper import build_asr, transcribe_audio
from bbox_from_image import load_vlm_bbox, run_vlm_bbox
from command_from_text import run_vlm
from utils import resolve_device

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
STOP_SILENCE_SECONDS = 1.0
MIN_SPEECH_SECONDS = 0.2


def _chunk_rms(chunk: bytes) -> float:
    if len(chunk) < SAMPLE_WIDTH:
        return 0.0
    if len(chunk) % SAMPLE_WIDTH != 0:
        chunk = chunk[: len(chunk) - (len(chunk) % SAMPLE_WIDTH)]
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
    if len(raw_bytes) % SAMPLE_WIDTH != 0:
        raw_bytes = raw_bytes[: len(raw_bytes) - (len(raw_bytes) % SAMPLE_WIDTH)]
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return samples / 32768.0


def _draw_bbox(image_bgr: np.ndarray, bbox: List[float], label: str) -> np.ndarray:
    canvas = image_bgr.copy()
    if len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            cv2.putText(
                canvas,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return canvas


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


def _open_image(image_path: str, max_side: int) -> Tuple[Image.Image, np.ndarray]:
    pil_image = Image.open(image_path).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    bgr = _resize_frame(bgr, max_side)
    pil_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return pil_image, bgr


def _start_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(1)
    print(f"等待客户端连接 {port} ...")
    conn, addr = sock.accept()
    print("客户端已连接:", addr)
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR -> command -> bbox on one image")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--asr-model", default="../hf_models/whisper-small")
    parser.add_argument("--vlm-model", default="../hf_models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--window-name", default="single-image")
    parser.add_argument("--window-x", type=int, default=100)
    parser.add_argument("--window-y", type=int, default=100)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--vlm-4bit", action="store_true", default=True)
    parser.add_argument("--no-vlm-4bit", dest="vlm_4bit", action="store_false")
    args = parser.parse_args()

    device = resolve_device(args.device)
    asr = build_asr(args.asr_model, device)
    vlm_processor, vlm_tokenizer, vlm_model = load_vlm_bbox(
        model_id=args.vlm_model, device=device, load_4bit=args.vlm_4bit
    )

    pil_image, base_bgr = _open_image(args.image, args.max_side)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(args.window_name, args.window_x, args.window_y)
    cv2.resizeWindow(args.window_name, 1200, 900)
    cv2.imshow(args.window_name, base_bgr)
    cv2.waitKey(1)

    conn = _start_socket(args.host, args.port)

    recording = False
    buffer = b""
    silence_seconds = 0.0

    chunk_duration = (CHUNK / (SAMPLE_WIDTH * CHANNELS)) / RATE
    window_chunks = max(1, int(START_WINDOW_SECONDS / chunk_duration))
    rms_window = deque(maxlen=window_chunks)
    prebuffer = deque(maxlen=window_chunks)

    try:
        while True:
            data = conn.recv(CHUNK)
            if not data:
                break

            rms = _chunk_rms(data)
            rms_window.append(rms)
            prebuffer.append(data)

            if not recording:
                if len(rms_window) == rms_window.maxlen:
                    window_rms = float(np.mean(rms_window))
                    if window_rms >= START_RMS_THRESHOLD:
                        recording = True
                        buffer = b"".join(prebuffer)
                        silence_seconds = 0.0
                        print("🎤 开始录音（VAD触发）...")
                cv2.waitKey(1)
                continue

            buffer += data
            if rms < STOP_RMS_THRESHOLD:
                silence_seconds += chunk_duration
            else:
                silence_seconds = 0.0

            if silence_seconds >= STOP_SILENCE_SECONDS:
                duration = len(buffer) / (SAMPLE_WIDTH * CHANNELS * RATE)
                if duration >= MIN_SPEECH_SECONDS:
                    audio = _bytes_to_float32(buffer)
                    text = transcribe_audio(
                        asr=asr,
                        audio=audio,
                        sample_rate=RATE,
                        language=args.language,
                        return_timestamps=True,
                    )
                    print(f"[识别结果] {text}")
                    if text:
                        command = run_vlm(
                            model=vlm_model,
                            tokenizer=vlm_tokenizer,
                            text=text,
                            config_path=args.config,
                        )
                        bbox_result = run_vlm_bbox(
                            processor=vlm_processor,
                            model=vlm_model,
                            image=pil_image,
                            command=command,
                            config_path=args.config,
                        )
                        label = f"{command.get('action', '')} {command.get('object', '')}".strip()
                        vis = _draw_bbox(base_bgr, bbox_result.get("bbox", []), label)
                        cv2.imshow(args.window_name, vis)
                        cv2.waitKey(1)

                buffer = b""
                recording = False
                silence_seconds = 0.0
                rms_window.clear()
                prebuffer.clear()
                print("🛑 停止录音（静音触发）")

    except KeyboardInterrupt:
        print("停止接收")
    finally:
        conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

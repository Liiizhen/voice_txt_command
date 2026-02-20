import argparse
import json
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

import cv2
from PIL import Image

from asr_whisper import build_asr
from bbox_from_image import load_vlm_bbox, run_vlm_bbox
from command_from_text import load_vlm, run_vlm

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None


HERE = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(HERE, "config.yaml")
DEFAULT_ASR = os.path.abspath(os.path.join(HERE, "..", "hf_models", "whisper-small"))
DEFAULT_LLM = os.path.abspath(os.path.join(HERE, "..", "hf_models", "qwen2.5-7b-instruct"))
DEFAULT_VLM = os.path.abspath(
    os.path.join(HERE, "..", "hf_models", "Qwen2.5-VL-7B-Instruct")
)


def _emit_result(payload: Dict[str, Any], output_json: str) -> None:
    text = json.dumps(payload, ensure_ascii=True)
    print(text, flush=True)
    if output_json:
        with open(output_json, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _record_and_transcribe_loop(
    asr: Any,
    language: str,
    sample_rate: int,
    segment_seconds: float,
    out_queue: "queue.Queue[str]",
    stop_event: threading.Event,
    mic_device: Optional[int],
) -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed. Run: pip install sounddevice"
        )
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")

    generate_kwargs: Dict[str, Any] = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    while not stop_event.is_set():
        frames = int(sample_rate * segment_seconds)
        audio = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=mic_device,
        )
        sd.wait()
        audio = audio.reshape(-1)
        if audio.size == 0:
            continue
        try:
            result = asr(
                {"array": audio, "sampling_rate": sample_rate},
                generate_kwargs=generate_kwargs,
            )
            text = (result.get("text") or "").strip()
            if text:
                out_queue.put(text)
        except Exception as exc:
            print(f"[asr] error: {exc}", flush=True)


def _get_video_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1e-2:
        return float(fps)
    return 30.0


def run_pipeline(args: argparse.Namespace) -> None:
    asr = build_asr(args.asr_model, args.device)
    llm_model, llm_tokenizer = load_vlm(
        model_id=args.llm_model,
        device=args.device,
        load_4bit=args.llm_4bit,
        load_8bit=args.llm_8bit,
    )
    vlm_processor, _, vlm_model = load_vlm_bbox(
        model_id=args.vlm_model,
        device=args.device,
        load_4bit=args.vlm_4bit,
        load_8bit=args.vlm_8bit,
    )

    transcript_queue: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()

    mic_thread = threading.Thread(
        target=_record_and_transcribe_loop,
        args=(
            asr,
            args.language,
            args.sample_rate,
            args.segment_seconds,
            transcript_queue,
            stop_event,
            args.mic_device,
        ),
        daemon=True,
    )
    mic_thread.start()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        stop_event.set()
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = _get_video_fps(cap)
    frame_delay = 1.0 / fps if args.realtime else 0.0

    latest_command: Optional[Dict[str, Any]] = None
    latest_text = ""
    frame_index = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            while True:
                try:
                    text = transcript_queue.get_nowait()
                except queue.Empty:
                    break
                latest_text = text
                latest_command = run_vlm(
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    text=text,
                    config_path=args.command_config,
                    max_new_tokens=args.command_max_tokens,
                )

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                bbox_result = run_vlm_bbox(
                    processor=vlm_processor,
                    model=vlm_model,
                    image=image,
                    command=latest_command,
                    max_new_tokens=args.bbox_max_tokens,
                    config_path=args.bbox_config,
                )
                payload = {
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / fps,
                    "transcript": latest_text,
                    "command": latest_command,
                    "bbox": bbox_result.get("bbox", []),
                }
                _emit_result(payload, args.output_json)

            if frame_delay > 0:
                time.sleep(frame_delay)
    finally:
        stop_event.set()
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASR (mic) -> LLM command -> VLM bbox video pipeline"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--language", default="en", help="ASR language code")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--asr-model", default=DEFAULT_ASR)
    parser.add_argument("--llm-model", default=DEFAULT_LLM)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM)
    parser.add_argument("--llm-4bit", action="store_true", default=True)
    parser.add_argument("--no-llm-4bit", dest="llm_4bit", action="store_false")
    parser.add_argument("--llm-8bit", action="store_true", default=False)
    parser.add_argument("--vlm-4bit", action="store_true", default=True)
    parser.add_argument("--no-vlm-4bit", dest="vlm_4bit", action="store_false")
    parser.add_argument("--vlm-8bit", action="store_true", default=False)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--mic-device", type=int, default=None)
    parser.add_argument("--command-config", default=DEFAULT_CONFIG)
    parser.add_argument("--bbox-config", default=DEFAULT_CONFIG)
    parser.add_argument("--command-max-tokens", type=int, default=128)
    parser.add_argument("--bbox-max-tokens", type=int, default=128)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()

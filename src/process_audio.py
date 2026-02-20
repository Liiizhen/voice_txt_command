import socket
import threading
from collections import deque

import numpy as np

from asr_whisper import build_asr, transcribe_audio
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

def main():
    device = resolve_device("cuda")
    asr = build_asr("../hf_models/whisper-small", device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)

    print(f"等待客户端连接 {PORT} ...")
    conn, addr = sock.accept()
    print("客户端已连接:", addr)

    # VAD state
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
                        language=LANGUAGE,
                        return_timestamps=True,
                    )
                    print(f"[识别结果] {text}")
                buffer = b""
                recording = False
                silence_seconds = 0.0
                rms_window.clear()
                prebuffer.clear()
                print("🛑 停止录音（静音触发）")

    except KeyboardInterrupt:
        print("停止接收")

    conn.close()
    sock.close()


if __name__ == "__main__":
    main()

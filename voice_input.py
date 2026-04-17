"""
Voice input module: socket audio stream → VAD → Whisper ASR → object query
Includes high-pass filter and RMS-based VAD from original single_video.py
"""
import os
import socket
import threading
import numpy as np
from collections import deque
from typing import Optional, Callable

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Audio settings
HOST = "0.0.0.0"
PORT = 50007
RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16
CHUNK = 1024      # bytes per recv

# VAD settings
START_WINDOW_SECONDS = 2.0
START_RMS_THRESHOLD = 500.0
STOP_RMS_THRESHOLD = 300.0
STOP_SILENCE_SECONDS = 0.3
STOP_WINDOW_SECONDS = 0.2
MIN_SPEECH_SECONDS = 0.2


class HighPassFilter:
    """一阶 RC 高通滤波器，去除低频噪声（风噪、空调声等）"""
    def __init__(self, cutoff: float = 200.0, fs: float = 16000.0):
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * np.pi * cutoff)
        self.alpha = rc / (rc + dt)
        self.prev_x = 0.0
        self.prev_y = 0.0

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


def _chunk_rms(chunk: bytes, hp_filter: Optional[HighPassFilter] = None) -> float:
    """计算音频块的 RMS 能量"""
    if len(chunk) < SAMPLE_WIDTH:
        return 0.0
    if len(chunk) % SAMPLE_WIDTH != 0:
        chunk = chunk[:len(chunk) - (len(chunk) % SAMPLE_WIDTH)]
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    if hp_filter is not None:
        samples = hp_filter.process_array(samples)
    return float(np.sqrt(np.mean(samples ** 2)))


def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
    """int16 bytes → float32 [-1, 1]"""
    if len(raw_bytes) % SAMPLE_WIDTH != 0:
        raw_bytes = raw_bytes[:len(raw_bytes) - (len(raw_bytes) % SAMPLE_WIDTH)]
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return samples / 32768.0


def parse_command(text: str) -> dict:
    """
    用 spaCy 从语音文字中提取物体描述
    "pick up the brown cup" → {"action": "pick up", "query": "brown cup"}
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except:
        # fallback: 简单规则提取
        return _parse_simple(text)

    doc = nlp(text.strip().lower())

    # 找动词（action）
    action_tokens = []
    for token in doc:
        if token.pos_ == "VERB":
            action_tokens.append(token.text)
            # 包含动词小品词 (pick up, put down)
            for child in token.children:
                if child.dep_ == "prt":
                    action_tokens.append(child.text)
            break

    # 找名词短语中的直接宾语
    query_parts = []
    for chunk in doc.noun_chunks:
        # 跳过 "me", "I" 等代词
        if chunk.root.pos_ == "PRON":
            continue
        # 提取形容词 + 名词
        for token in chunk:
            if token.pos_ in ("ADJ", "NOUN", "PROPN") and token.dep_ != "det":
                query_parts.append(token.text)

    action = " ".join(action_tokens) if action_tokens else ""
    query = " ".join(query_parts) if query_parts else text.strip()

    return {"action": action, "query": query}


def _parse_simple(text: str) -> dict:
    """简单 fallback 解析：去掉常见动词，剩下的当 query"""
    text = text.strip().lower()
    verbs = ["pick up", "grab", "get", "give me", "take", "move", "put", "place",
             "find", "show me", "point to", "track", "follow"]
    action = ""
    for v in verbs:
        if text.startswith(v):
            action = v
            text = text[len(v):].strip()
            break
    # 去掉 the/a/an
    for article in ["the ", "a ", "an "]:
        if text.startswith(article):
            text = text[len(article):]
    return {"action": action, "query": text.strip()}


def build_asr(model_path: str = "openai/whisper-tiny", device: str = "cuda"):
    """构建 Whisper ASR pipeline"""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    if device == "cuda":
        model = model.to("cuda")

    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device == "cuda" else -1,
        dtype=torch_dtype,
    )


def start_voice_listener(asr, on_command: Callable[[dict], None],
                         language: str = "en", port: int = PORT):
    """
    启动语音监听线程：
    1. 接收 socket 音频流
    2. 高通滤波 + VAD 检测语音段
    3. Whisper 识别
    4. spaCy 解析命令
    5. 回调 on_command({"action": "pick up", "query": "brown cup"})
    """
    def _listener():
        from asr_whisper import transcribe_audio

        hp_filter = HighPassFilter(cutoff=200.0, fs=RATE)
        chunk_duration = (CHUNK / (SAMPLE_WIDTH * CHANNELS)) / RATE
        window_chunks = max(1, int(START_WINDOW_SECONDS / chunk_duration))
        stop_window_chunks = max(1, int(STOP_WINDOW_SECONDS / chunk_duration))

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, port))
        sock.listen(1)
        print(f"[Voice] Listening on port {port}...", flush=True)

        conn, addr = sock.accept()
        print(f"[Voice] Client connected: {addr}", flush=True)

        recording = False
        buffer = b""
        silence_seconds = 0.0
        rms_window = deque(maxlen=window_chunks)
        prebuffer = deque(maxlen=window_chunks)
        stop_rms_window = deque(maxlen=stop_window_chunks)

        try:
            while True:
                data = conn.recv(CHUNK)
                if not data:
                    break

                rms = _chunk_rms(data, hp_filter)
                rms_window.append(rms)
                prebuffer.append(data)

                if not recording:
                    if len(rms_window) == rms_window.maxlen:
                        window_rms = float(np.mean(rms_window))
                        if window_rms >= START_RMS_THRESHOLD:
                            recording = True
                            buffer = b"".join(prebuffer)
                            silence_seconds = 0.0
                            stop_rms_window.clear()
                            print("[Voice] Recording started (VAD)", flush=True)
                    continue

                buffer += data
                stop_rms_window.append(rms)

                if len(stop_rms_window) == stop_rms_window.maxlen:
                    if float(np.mean(stop_rms_window)) < STOP_RMS_THRESHOLD:
                        silence_seconds += chunk_duration * stop_window_chunks
                    else:
                        silence_seconds = 0.0

                if silence_seconds >= STOP_SILENCE_SECONDS:
                    duration = len(buffer) / (SAMPLE_WIDTH * CHANNELS * RATE)
                    if duration >= MIN_SPEECH_SECONDS:
                        audio = _bytes_to_float32(buffer)
                        text = transcribe_audio(
                            asr=asr, audio=audio,
                            sample_rate=RATE, language=language,
                            return_timestamps=False)
                        print(f"[Voice] Recognized: {text}", flush=True)

                        cmd = parse_command(text)
                        print(f"[Voice] Command: {cmd}", flush=True)
                        if cmd["query"]:
                            on_command(cmd)

                    buffer = b""
                    recording = False
                    silence_seconds = 0.0
                    rms_window.clear()
                    prebuffer.clear()
                    stop_rms_window.clear()
                    print("[Voice] Recording stopped (silence)", flush=True)

        except KeyboardInterrupt:
            pass
        finally:
            conn.close()
            sock.close()

    thread = threading.Thread(target=_listener, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    # 测试 parse_command
    tests = [
        "pick up the brown cup",
        "grab the mouse",
        "give me the transparent cup",
        "red bottle",
        "find the laptop on the table",
    ]
    for t in tests:
        result = parse_command(t)
        print(f"  '{t}' → {result}")

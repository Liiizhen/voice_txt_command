# python asr_whisper.py -a ../data/3.wav -o ../data/test3.txt 
# -l en -m ../hf_models/whisper-small -d cuda -o ../data/test3.txt

import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils import resolve_device, resolve_model_id

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def build_asr(model_id: str, device: str):
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    resolved_id, local_only = resolve_model_id(model_id)

    processor = AutoProcessor.from_pretrained(resolved_id, local_files_only=local_only)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        resolved_id,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=local_only,
    )

    if device == "cuda":
        model = model.to("cuda")

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device == "cuda" else -1,
        dtype=torch_dtype,
    )


def transcribe(asr,
               audio_path: str,
               language: str = "en",
               return_timestamps: bool = False) -> str:
    """
    语音转文字：传入已构建的 ASR pipeline
    """
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    result = asr(
        audio_path,
        generate_kwargs=generate_kwargs,
        return_timestamps=return_timestamps,
    )
    return result["text"].strip()


def transcribe_audio(asr,
                     audio,
                     sample_rate: int,
                     language: str = "en",
                     return_timestamps: bool = False) -> str:
    """
    语音转文字：传入内存中的音频数组（float32, -1~1）与采样率
    """
    generate_kwargs = {"task": "transcribe"}

    if language:
        generate_kwargs["language"] = language

    result = asr(
        {"array": audio, "sampling_rate": sample_rate},
        generate_kwargs=generate_kwargs,
        return_timestamps=return_timestamps,
    )

    return result["text"].strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Whisper ASR transcript generator")
    parser.add_argument("-a", required=True, help="Path to audio file")
    parser.add_argument("-l", default="en", help="Language code")
    parser.add_argument("-m", default="./hf_models/whisper-small", help="Model path")
    parser.add_argument("-d", choices=["cuda", "cpu"], default="cuda", help="Device")
    parser.add_argument("-o", default="", help="Output file")
    args = parser.parse_args()

    device = resolve_device(args.d)
    asr = build_asr(args.m, device)
    text = transcribe(
        asr=asr,
        audio_path=args.a,
        language=args.l,
    )

    if args.o:
        with open(args.o, "w", encoding="utf-8") as f:
            f.write(text)

    print(text)


if __name__ == "__main__":
    main()

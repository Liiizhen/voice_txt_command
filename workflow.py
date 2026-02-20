import argparse
import os
import json
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

SYSTEM_PROMPT = (
    "You convert a spoken command into a compact JSON object.\n"
    "Return JSON only, no extra text.\n"
    "Schema:\n"
    '{ "action": "<verb>", "object": "<noun>", "attributes": ["<attr>", ...] }\n'
    "The action must be the verb phrase from the command.\n"
    'If no action is mentioned, use an empty string: "action": "".\n'
    "Action hints (use if present): pick up, grab, place, move, open, close, press, "
    "release, turn, rotate, push, pull, slide, lift, drop, pour, shake.\n"
    "The object must be the noun only (no colors/adjectives).\n"
    "Put every descriptive word in attributes; keep order.\n"
    'If no attributes are mentioned, use an empty list: "attributes": []\n'
    'Example: "pick up the red small bottle" -> '
    '{"action":"pick up","object":"bottle","attributes":["red","small"]}'
)

# Lightweight attribute hints to split "red bottle" -> object/bottle, attrs/red.
ATTRIBUTE_HINTS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "small",
    "large",
    "big",
    "tiny",
    "empty",
    "full",
    "open",
    "closed",
}

# Lightweight action hints to extract verb phrases from transcripts.
ACTION_HINTS = [
    "pick up",
    "put down",
    "pick",
    "place",
    "put",
    "grab",
    "grasp",
    "move",
    "open",
    "close",
    "press",
    "release",
    "turn",
    "rotate",
    "push",
    "pull",
    "slide",
    "lift",
    "drop",
    "pour",
    "shake",
]

# Force offline mode by default. Models are expected to be local directories.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def resolve_model_id(model_id: str) -> tuple[str, bool]:
    expanded = os.path.expanduser(model_id)
    if os.path.isdir(expanded) or os.path.isfile(expanded):
        return expanded, True
    if model_id.startswith("/") and not os.path.exists(expanded):
        raise FileNotFoundError(f"Local model path not found: {expanded}")
    offline = (
        os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
        or os.environ.get("HF_HUB_OFFLINE", "") == "1"
    )
    return model_id, offline


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        # Provide a clearer error so GPU issues don't silently fall back.
        details = []
        details.append(f"torch={torch.__version__} cuda={torch.version.cuda}")
        try:
            details.append(f"device_count={torch.cuda.device_count()}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            details.append(f"device_count_error={exc}")
        raise RuntimeError(
            "CUDA requested but not available. This is usually a driver/GPU visibility issue.\n"
            "Check `nvidia-smi` and ensure the NVIDIA driver is loaded, or run inside a GPU-enabled environment.\n"
            + " | ".join(details)
        )
    return requested


def build_asr(model_id: str, device: str) -> Any:
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    resolved_id, local_only = resolve_model_id(model_id)
    processor = AutoProcessor.from_pretrained(resolved_id, local_files_only=local_only)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        resolved_id,
        torch_dtype=torch_dtype,
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
        torch_dtype=torch_dtype,
    )


def transcribe(
    asr_pipeline: Any, audio_path: str, language: Optional[str]
) -> str:
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    result = asr_pipeline(audio_path, generate_kwargs=generate_kwargs)
    return result["text"].strip()


def build_llm(model_id: str, device: str) -> Any:
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    resolved_id, local_only = resolve_model_id(model_id)
    config = AutoConfig.from_pretrained(resolved_id, local_files_only=local_only)
    model_type = getattr(config, "model_type", "")
    if model_type in {"qwen2_vl", "qwen2_5_vl"}:
        model_cls = AutoModelForVision2Seq
    else:
        model_cls = AutoModelForCausalLM
    if device == "cpu":
        return model_cls.from_pretrained(
            resolved_id,
            torch_dtype=torch_dtype,
            device_map="cpu",
            local_files_only=local_only,
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    return model_cls.from_pretrained(
        resolved_id,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto",
        local_files_only=local_only,
    )


def build_prompt(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Command: {text}\nReturn JSON only."},
    ]


def extract_json_block(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("JSON object not closed in model output.")


def split_object_attributes(obj: str, attrs: List[str]) -> tuple[str, List[str]]:
    if attrs or not obj:
        return obj, attrs
    tokens = obj.split()
    if len(tokens) <= 1:
        return obj, attrs
    extracted = [token for token in tokens if token.lower() in ATTRIBUTE_HINTS]
    if not extracted:
        return obj, attrs
    remaining = [token for token in tokens if token.lower() not in ATTRIBUTE_HINTS]
    if not remaining:
        return obj, attrs
    return " ".join(remaining), attrs + extracted


def find_action_hint(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    best = ""
    best_idx = None
    for phrase in ACTION_HINTS:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        match = re.search(pattern, lowered)
        if not match:
            continue
        idx = match.start()
        if best_idx is None or idx < best_idx or (
            idx == best_idx and len(phrase) > len(best)
        ):
            best_idx = idx
            best = text[match.start() : match.end()]
    return best


def normalize_command(raw: Dict[str, Any], source_text: str = "") -> Dict[str, Any]:
    action: Any = raw.get("action", "")
    obj: Any = raw.get("object", "")
    attrs: Any = raw.get("attributes", [])
    if action is None:
        action = ""
    if not isinstance(action, str):
        action = str(action)
    if attrs is None:
        attrs = []
    if isinstance(attrs, str):
        attrs = [attrs]
    elif not isinstance(attrs, list):
        attrs = [str(attrs)]

    if isinstance(obj, dict):
        obj_name = obj.get("object") or obj.get("name") or obj.get("label") or ""
        obj_attrs = obj.get("attributes") or obj.get("attrs") or []
        obj = obj_name
        if isinstance(obj_attrs, str):
            obj_attrs = [obj_attrs]
        if obj_attrs:
            attrs = attrs + list(obj_attrs)
    elif isinstance(obj, list):
        obj = " ".join([str(item) for item in obj if item])

    if obj is None:
        obj = ""
    if not isinstance(obj, str):
        obj = str(obj)

    action_hint = find_action_hint(source_text)
    if action_hint:
        if not action.strip():
            action = action_hint
        elif action.lower() not in source_text.lower():
            action = action_hint

    obj, attrs = split_object_attributes(obj, attrs)

    cleaned_attrs: List[str] = []
    seen = set()
    for item in attrs:
        if item is None:
            continue
        if isinstance(item, list):
            for nested in item:
                if nested is None:
                    continue
                text = str(nested).strip()
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    cleaned_attrs.append(text)
            continue
        text = str(item).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            cleaned_attrs.append(text)

    return {"action": action.strip(), "object": obj.strip(), "attributes": cleaned_attrs}


def text_to_command(
    tokenizer: Any, model: Any, text: str, max_new_tokens: int
) -> Dict[str, Any]:
    messages = build_prompt(text)
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        prompt_length = input_ids.shape[-1]
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        decoded = tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        ).strip()
    else:
        prompt = (
            SYSTEM_PROMPT
            + "\nUser: "
            + text
            + "\nAssistant:"
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    json_text = extract_json_block(decoded)
    command = json.loads(json_text)
    return normalize_command(command, source_text=text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper-small -> Qwen2.5-7B-Instruct command pipeline"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language code (default: en)",
    )
    parser.add_argument(
        "--whisper-model",
        default="/home/user/nelson/voice_txt_command/hf_models/whisper-small",
        help="Whisper model id",
    )
    parser.add_argument(
        "--qwen-model",
        default="/home/user/nelson/voice_txt_command/hf_models/qwen2.5-7b-instruct",
        help="Qwen model id",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens for command generation",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON file",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    asr = build_asr(args.whisper_model, device)
    transcript = transcribe(asr, args.audio, args.language)
    print(f"Transcript: {transcript}")

    resolved_qwen, local_only = resolve_model_id(args.qwen_model)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_qwen, use_fast=True, local_files_only=local_only
    )
    model = build_llm(resolved_qwen, device)
    command = text_to_command(
        tokenizer, model, transcript, max_new_tokens=args.max_new_tokens
    )

    print("Command JSON:")
    print(json.dumps(command, indent=2, ensure_ascii=True))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(command, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()

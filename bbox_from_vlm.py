import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

sys.path.append(os.path.dirname(__file__))

from config_utils import load_config
from shared_utils import (
    extract_bbox,
    extract_json_block,
    fallback_parse_fields,
    normalize_command,
    safe_json_loads,
    resolve_device,
    resolve_model_id,
)


# Force offline mode by default. Models are expected to be local directories.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def build_model(model_id: str, device: str, load_4bit: bool, load_8bit: bool):
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    resolved_id, local_only = resolve_model_id(model_id)
    processor = AutoProcessor.from_pretrained(resolved_id, local_files_only=local_only)

    model_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "local_files_only": local_only,
    }
    if device == "cpu":
        model_kwargs["device_map"] = "cpu"
    elif load_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    model_cls = AutoModelForVision2Seq
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        try:
            from transformers import Qwen2VLForConditionalGeneration

            model_cls = Qwen2VLForConditionalGeneration
        except Exception:
            model_cls = AutoModelForVision2Seq

    model = model_cls.from_pretrained(resolved_id, **model_kwargs)
    return processor, model


def build_prompt(config: Dict[str, Any], text: str, size: Tuple[int, int]) -> str:
    prompts = config.get("prompts", {})
    system_prompt = prompts.get("vlm_system", "")
    user_template = prompts.get(
        "vlm_user",
        "Transcript: {text}\nImage size: {width}x{height}\nReturn JSON only.",
    )
    action_hints = ", ".join(config.get("action_hints", []))
    attribute_hints = ", ".join(config.get("attribute_hints", []))
    width, height = size
    user_prompt = user_template.format(
        text=text,
        width=width,
        height=height,
        action_hints=action_hints,
        attribute_hints=attribute_hints,
    )
    if system_prompt:
        return system_prompt + "\n" + user_prompt
    return user_prompt


def build_command_prompt(config: Dict[str, Any], text: str) -> str:
    prompts = config.get("prompts", {})
    system_prompt = prompts.get("vlm_command_system", "")
    user_template = prompts.get(
        "vlm_command_user",
        "Transcript: {text}\nReturn JSON only.",
    )
    action_hints = ", ".join(config.get("action_hints", []))
    attribute_hints = ", ".join(config.get("attribute_hints", []))
    user_prompt = user_template.format(
        text=text, action_hints=action_hints, attribute_hints=attribute_hints
    )
    if system_prompt:
        return system_prompt + "\n" + user_prompt
    return user_prompt


def build_bbox_prompt(
    config: Dict[str, Any],
    command: Dict[str, Any],
    size: Tuple[int, int],
) -> str:
    prompts = config.get("prompts", {})
    system_prompt = prompts.get("vlm_bbox_system", "")
    user_template = prompts.get(
        "vlm_bbox_user",
        "Command JSON: {command_json}\nImage size: {width}x{height}\nReturn JSON only.",
    )
    width, height = size
    command_json = json.dumps(command, ensure_ascii=True)
    user_prompt = user_template.format(
        command_json=command_json,
        width=width,
        height=height,
    )
    if system_prompt:
        return system_prompt + "\n" + user_prompt
    return user_prompt


def prepare_inputs(
    processor: Any, image: Optional[Image.Image], prompt: str
) -> Dict[str, Any]:
    if hasattr(processor, "apply_chat_template"):
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image"})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if image is not None:
            return processor(text=[text], images=[image], return_tensors="pt")
        return processor(text=[text], return_tensors="pt")
    if image is not None:
        return processor(text=prompt, images=image, return_tensors="pt")
    return processor(text=prompt, return_tensors="pt")


def decode_output(processor: Any, generated_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "decode"):
        return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return ""


def read_text(text_arg: str, text_file: str) -> str:
    if text_arg and text_file:
        raise ValueError("Use either --text or --text-file, not both.")
    if text_file:
        with open(text_file, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    if text_arg:
        return text_arg.strip()
    raise ValueError("Provide --text or --text-file.")


def should_retry_output(decoded: str, raw: Dict[str, Any]) -> bool:
    if not raw:
        return True
    markers = ["<verb>", "<noun>", "<attr>", "...", "…"]
    if any(marker in decoded for marker in markers):
        return True
    placeholder_re = re.compile(r"<[^>]+>")

    def is_placeholder(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        if value.strip() in ("...", "…"):
            return True
        return placeholder_re.search(value) is not None

    if is_placeholder(raw.get("action", "")):
        return True
    if is_placeholder(raw.get("object", "")):
        return True
    attrs = raw.get("attributes", [])
    if isinstance(attrs, list) and any(is_placeholder(item) for item in attrs):
        return True
    return False


def run_generation(
    processor: Any,
    model: Any,
    image: Optional[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, Dict[str, Any]]:
    inputs = prepare_inputs(processor, image, prompt)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = decode_output(processor, generated_ids).strip()
    try:
        json_text = extract_json_block(decoded)
    except Exception:
        json_text = ""
    raw = safe_json_loads(json_text) if json_text else {}
    if not raw:
        raw = fallback_parse_fields(decoded)
    return decoded, raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate command JSON + bbox using Qwen2.5-VL."
    )
    parser.add_argument("--text", default="", help="Input transcript text")
    parser.add_argument("--text-file", default="", help="Path to transcript file")
    parser.add_argument("--image", required=True, help="Input image file")
    parser.add_argument(
        "--config",
        default="",
        help="Path to YAML config (defaults to config.yaml)",
    )
    parser.add_argument(
        "--qwen-model",
        default="./hf_models/Qwen2.5-VL-7B-Instruct",
        help="Qwen-VL model path or id",
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
        default=192,
        help="Max tokens for bbox generation",
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Run VLM twice: transcript->command, then command+image->bbox",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Enable 4-bit quantization (GPU only)",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Enable 8-bit quantization (GPU only)",
    )
    parser.add_argument("--output", default="", help="Output JSON file")
    args = parser.parse_args()

    text = read_text(args.text, args.text_file)
    config = load_config(args.config or None)
    action_hints = config.get("action_hints", [])
    attribute_hints = config.get("attribute_hints", [])

    if args.load_4bit and args.load_8bit:
        raise ValueError("Choose only one: --load-4bit or --load-8bit.")
    device = resolve_device(args.device)
    if device == "cpu" and (args.load_4bit or args.load_8bit):
        raise ValueError("Quantization flags require CUDA.")
    image = Image.open(args.image).convert("RGB")
    size = image.size

    processor, model = build_model(
        args.qwen_model,
        device,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )

    if args.two_pass:
        command_prompt = build_command_prompt(config, text)
        decoded, raw = run_generation(
            processor, model, None, command_prompt, max_new_tokens=args.max_new_tokens
        )
        if should_retry_output(decoded, raw):
            strict_prompt = (
                command_prompt
                + "\nSTRICT: Do not use placeholders like <verb>, <noun>, <attr>, or .... "
                + 'If unsure, use "action": "", "object": "", "attributes": []. '
                + "Return JSON only."
            )
            decoded, raw = run_generation(
                processor,
                model,
                None,
                strict_prompt,
                max_new_tokens=args.max_new_tokens,
            )
        command = normalize_command(raw, text, action_hints, attribute_hints)

        bbox_prompt = build_bbox_prompt(config, command, size)
        decoded, raw = run_generation(
            processor, model, image, bbox_prompt, max_new_tokens=args.max_new_tokens
        )
        if should_retry_output(decoded, raw):
            strict_prompt = (
                bbox_prompt
                + "\nSTRICT: Do not use placeholders like <verb>, <noun>, <attr>, or .... "
                + 'If unsure, use "bbox": []. Return JSON only.'
            )
            decoded, raw = run_generation(
                processor,
                model,
                image,
                strict_prompt,
                max_new_tokens=args.max_new_tokens,
            )
        bbox = extract_bbox(raw, size)
        output = {**command, "bbox": bbox}
    else:
        prompt = build_prompt(config, text, size)
        decoded, raw = run_generation(
            processor, model, image, prompt, max_new_tokens=args.max_new_tokens
        )
        if should_retry_output(decoded, raw):
            strict_prompt = (
                prompt
                + "\nSTRICT: Do not use placeholders like <verb>, <noun>, <attr>, or .... "
                + 'If unsure, use "action": "", "object": "", "attributes": [], "bbox": []. '
                + "Return JSON only."
            )
            decoded, raw = run_generation(
                processor,
                model,
                image,
                strict_prompt,
                max_new_tokens=args.max_new_tokens,
            )
        command = normalize_command(raw, text, action_hints, attribute_hints)
        bbox = extract_bbox(raw, size)
        output = {**command, "bbox": bbox}

    output_text = json.dumps(output, indent=2, ensure_ascii=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output_text)
    print(output_text)


if __name__ == "__main__":
    main()

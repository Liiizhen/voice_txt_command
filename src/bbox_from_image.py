import os
import re
import json
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from command_from_text import load_vlm, run_vlm
from utils import load_config, extract_json_block, safe_json_loads

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------
# 1. 加载模型（外部调用一次）
# ---------------------------------------------------------
def load_vlm_bbox(
    model_id: str,
    device="cuda",
    load_4bit=False,
    load_8bit=False,
    model=None,
):
    """
    Load or reuse a VLM model, and create the processor.
    If a model is provided, weights won't be reloaded.
    """
    if model is None:
        model, tokenizer = load_vlm(
            model_id=model_id,
            device=device,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False
    )

    return processor, tokenizer, model


# ---------------------------------------------------------
# 2. 构建 prompt（内部使用）
# ---------------------------------------------------------
def build_prompt(
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


# ---------------------------------------------------------
# 3. 解析 bbox（内部使用）
# ---------------------------------------------------------
def _looks_normalized(values: List[float], size: Tuple[int, int]) -> bool:
    width, height = size
    if width <= 1 or height <= 1:
        return False
    return all(0.0 <= value <= 1.0 for value in values)


def normalize_bbox(values: List[float], size: Tuple[int, int]) -> List[Union[int, float]]:
    if len(values) != 4:
        return []
    width, height = size
    x1, y1, x2, y2 = values
    if _looks_normalized([x1, y1, x2, y2], size):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0.0, min(float(x1), float(width)))
    x2 = max(0.0, min(float(x2), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    y2 = max(0.0, min(float(y2), float(height)))
    normalized: List[Union[int, float]] = []
    for value in (x1, y1, x2, y2):
        if float(value).is_integer():
            normalized.append(int(value))
        else:
            normalized.append(float(value))
    return normalized


def _parse_bbox_value(value: Any, size: Tuple[int, int]) -> List[Union[int, float]]:
    if value is None:
        return []
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(v, (int, float, str)) for v in value):
            values = [float(v) for v in value]
            return normalize_bbox(values, size)
        for item in value:
            found = _parse_bbox_value(item, size)
            if found:
                return found
        return []
    if isinstance(value, dict):
        keys = ["x1", "y1", "x2", "y2"]
        if all(key in value for key in keys):
            values = [float(value[key]) for key in keys]
            return normalize_bbox(values, size)
        alt = ["left", "top", "right", "bottom"]
        if all(key in value for key in alt):
            values = [float(value[key]) for key in alt]
            return normalize_bbox(values, size)
        coco = ["x", "y", "width", "height"]
        if all(key in value for key in coco):
            x = float(value["x"])
            y = float(value["y"])
            w = float(value["width"])
            h = float(value["height"])
            return normalize_bbox([x, y, x + w, y + h], size)
        center = ["cx", "cy", "w", "h"]
        if all(key in value for key in center):
            cx = float(value["cx"])
            cy = float(value["cy"])
            w = float(value["w"])
            h = float(value["h"])
            return normalize_bbox([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], size)
        return []
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 4:
            values = [float(num) for num in numbers[:4]]
            return normalize_bbox(values, size)
    return []


def extract_bbox_from_text(
    text: str, size: Tuple[int, int]
) -> List[Union[int, float]]:
    try:
        json_text = extract_json_block(text)
        raw = safe_json_loads(json_text)
    except Exception:
        raw = {}
    if raw:
        candidate = raw.get("bbox") if isinstance(raw, dict) else None
        if candidate is None and isinstance(raw, dict):
            candidate = raw.get("box")
        if candidate is None:
            candidate = raw
        found = _parse_bbox_value(candidate, size)
        if found:
            return found

    lowered = text.lower()
    indices = [lowered.rfind("bbox"), lowered.rfind("bounding box"), lowered.rfind("box")]
    start = max(indices)
    if start == -1:
        return []
    tail = text[start : start + 300]
    labeled = re.findall(
        r"\b(x1|y1|x2|y2)\s*[:=]\s*(-?\d+(?:\.\d+)?)", tail, re.IGNORECASE
    )
    if labeled:
        values_map = {key.lower(): float(value) for key, value in labeled}
        if all(key in values_map for key in ("x1", "y1", "x2", "y2")):
            return normalize_bbox(
                [values_map["x1"], values_map["y1"], values_map["x2"], values_map["y2"]],
                size,
            )
    numbers = re.findall(r"-?\d+(?:\.\d+)?", tail)
    if len(numbers) >= 4:
        values = [float(num) for num in numbers[:4]]
        return normalize_bbox(values, size)
    return []


def prepare_inputs(processor: Any, image: Image.Image, prompt: str) -> Dict[str, Any]:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[text], images=[image], return_tensors="pt")
    return processor(text=prompt, images=image, return_tensors="pt")


def decode_output(processor: Any, output_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "decode"):
        return processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return ""


# ---------------------------------------------------------
# 4. ⭐ 外部调用的核心函数：run_vlm_bbox() ⭐
# ---------------------------------------------------------
def run_vlm_bbox(
    processor: Any,
    model: Any,
    image: Image.Image,
    command: Dict[str, Any],
    max_new_tokens: int = 128,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:

    config = load_config(config_path)
    prompt = build_prompt(config, command, image.size)

    inputs = prepare_inputs(processor, image, prompt)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    text = decode_output(processor, output_ids).strip()

    bbox = extract_bbox_from_text(text, image.size)

    return {**command, "bbox": bbox}

def main():
    from PIL import Image
    # 1) 加载 VLM（或外部传入已加载的 model）
    processor, tokenizer, model = load_vlm_bbox(
        model_id="../hf_models/Qwen2.5-VL-7B-Instruct",
        device="cuda",
        load_4bit=True,
    )
    result = run_vlm(
        model=model,
        tokenizer=tokenizer,
        text="get the blue surfboard",
        config_path="config.yaml"
    ) 
    # 2) 示例：直接给 command + image
    command = result
    print(command)
    image = Image.open("/home/user/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/val2017/000000002261.jpg").convert("RGB")

    result = run_vlm_bbox(
        processor=processor,
        model=model,
        image=image,
        command=command,
        config_path="config.yaml",
    )

    print(result)

if __name__ == "__main__":
    main()
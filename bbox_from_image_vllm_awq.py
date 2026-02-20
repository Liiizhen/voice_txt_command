import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


# Force offline mode by default. Models are expected to be local directories.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def resolve_model_id(model_id: str) -> Tuple[str, bool]:
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


def build_prompt(command: Dict[str, Any], size: Tuple[int, int]) -> str:
    width, height = size
    command_text = json.dumps(command, ensure_ascii=True)
    return (
        "You are given a command JSON and an image. "
        "Find the object described by the command and return a JSON object with the same "
        '"action", "object", "attributes" plus a "bbox" in pixel coordinates [x1, y1, x2, y2]. '
        f"Image size: {width}x{height}. "
        'If the object is not visible, return "bbox": []. '
        "If it is visible, always return a best-effort bbox.\n"
        f"Command JSON: {command_text}\n"
        "Return JSON only."
    )


def prepare_prompt(processor: Any, image: Image.Image, prompt: str) -> str:
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
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def normalize_bbox(values: List[float], size: Tuple[int, int]) -> List[int | float]:
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
    normalized: List[int | float] = []
    for value in (x1, y1, x2, y2):
        if float(value).is_integer():
            normalized.append(int(value))
        else:
            normalized.append(float(value))
    return normalized


def _looks_normalized(values: List[float], size: Tuple[int, int]) -> bool:
    width, height = size
    if width <= 1 or height <= 1:
        return False
    return all(0.0 <= value <= 1.0 for value in values)


def _first_key(data: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _extract_bbox_from_dict(bbox: Dict[str, Any], size: Tuple[int, int]) -> List[int | float]:
    keys = ["x1", "y1", "x2", "y2"]
    if all(key in bbox for key in keys):
        values = [float(bbox[key]) for key in keys]
        return normalize_bbox(values, size)
    alt = ["left", "top", "right", "bottom"]
    if all(key in bbox for key in alt):
        values = [float(bbox[key]) for key in alt]
        return normalize_bbox(values, size)
    coco = ["x", "y", "width", "height"]
    if all(key in bbox for key in coco):
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["width"])
        h = float(bbox["height"])
        return normalize_bbox([x, y, x + w, y + h], size)
    center = ["cx", "cy", "w", "h"]
    if all(key in bbox for key in center):
        cx = float(bbox["cx"])
        cy = float(bbox["cy"])
        w = float(bbox["w"])
        h = float(bbox["h"])
        return normalize_bbox([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], size)
    return []


def extract_bbox(data: Dict[str, Any], size: Tuple[int, int]) -> List[int | float]:
    bbox = _first_key(
        data,
        [
            "bbox",
            "box",
            "bboxes",
            "bbox_2d",
            "box_2d",
            "bboxes_2d",
            "bounding_box",
            "boundingBox",
            "bounding_boxes",
        ],
    )
    if bbox is None:
        return []
    if isinstance(bbox, list):
        if len(bbox) == 4 and all(
            isinstance(value, (int, float, str)) for value in bbox
        ):
            values = [float(value) for value in bbox]
            return normalize_bbox(values, size)
        for item in bbox:
            if isinstance(item, list) and len(item) == 4:
                values = [float(value) for value in item]
                return normalize_bbox(values, size)
            if isinstance(item, dict):
                values = _extract_bbox_from_dict(item, size)
                if values:
                    return values
    if isinstance(bbox, dict):
        values = _extract_bbox_from_dict(bbox, size)
        if values:
            return values
    if isinstance(bbox, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", bbox)
        if len(numbers) >= 4:
            values = [float(num) for num in numbers[:4]]
            return normalize_bbox(values, size)
    return []


def extract_bbox_any(data: Any, size: Tuple[int, int]) -> List[int | float]:
    if isinstance(data, dict):
        direct = extract_bbox(data, size)
        if direct:
            return direct
        for value in data.values():
            found = extract_bbox_any(value, size)
            if found:
                return found
    if isinstance(data, list):
        for item in data:
            found = extract_bbox_any(item, size)
            if found:
                return found
    return []


def extract_bbox_from_text(text: str, size: Tuple[int, int]) -> List[int | float]:
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


def generate_with_vllm(
    llm: LLM,
    prompt: str,
    image: Image.Image,
    max_new_tokens: int,
    temperature: float,
) -> str:
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
        ],
        sampling_params,
    )
    if not outputs or not outputs[0].outputs:
        return ""
    return outputs[0].outputs[0].text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bbox using a Qwen-VL model with vLLM + AWQ."
    )
    parser.add_argument("--input-json", required=True, help="Input command JSON file")
    parser.add_argument("--image", required=True, help="Input image file")
    parser.add_argument(
        "--qwen-model",
        default="./hf_models/Qwen2.5-VL-7B-Instruct-AWQ",
        help="Qwen-VL AWQ model path or id",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens for bbox generation",
    )
    parser.add_argument(
        "--quantization",
        default="awq",
        help="vLLM quantization mode (e.g. awq, awq_marlin)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Model dtype (float16, bfloat16, float32)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory to use for vLLM (0-1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="Override max model length (0 keeps model default)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom model code",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--debug-output",
        default="",
        help="Write raw model output to a text file",
    )
    parser.add_argument("--output", default="", help="Output JSON file")
    args = parser.parse_args()

    resolved_id, local_only = resolve_model_id(args.qwen_model)
    processor = AutoProcessor.from_pretrained(resolved_id, local_files_only=local_only)

    with open(args.input_json, "r", encoding="utf-8") as handle:
        command = json.load(handle)

    image = Image.open(args.image).convert("RGB")
    size = image.size

    prompt = build_prompt(command, size)
    text = prepare_prompt(processor, image, prompt)

    llm_kwargs: Dict[str, Any] = {
        "model": resolved_id,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    decoded = generate_with_vllm(
        llm, text, image, max_new_tokens=args.max_new_tokens, temperature=args.temperature
    ).strip()

    bbox: List[int | float] = []
    json_text = ""
    try:
        json_text = _extract_json_block(decoded)
        parsed = json.loads(json_text)
        bbox = extract_bbox_any(parsed, size)
    except Exception:
        bbox = []
    if not bbox:
        bbox = extract_bbox_from_text(decoded, size)

    output = {**command, "bbox": bbox}
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input_json)
        output_path = f"{base}_bbox{ext or '.json'}"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=True)
    print(output_path)

    if args.debug_output:
        with open(args.debug_output, "w", encoding="utf-8") as handle:
            handle.write(decoded)
            if json_text:
                handle.write("\n\n---- extracted json ----\n")
                handle.write(json_text)


def _extract_json_block(text: str) -> str:
    blocks = _extract_json_blocks(text)
    if not blocks:
        raise ValueError("No JSON object found in model output.")
    for block in reversed(blocks):
        try:
            json.loads(block)
            return block
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in model output.")


def _extract_json_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    start = None
    depth = 0
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : idx + 1])
                start = None
    return blocks


if __name__ == "__main__":
    main()

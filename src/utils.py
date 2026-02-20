import ast
import json
import os
import re
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    target = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(target):
        raise FileNotFoundError(f"Config not found: {target}")
    with open(target, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")
    return data


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


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
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


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    try:
        data = ast.literal_eval(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def fallback_parse_fields(text: str) -> Dict[str, Any]:
    def extract_string(key: str) -> str:
        pattern = rf'["\']{key}["\']\s*:\s*["\']([^"\']*)["\']'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        pattern = rf'["\']{key}["\']\s*:\s*([^,\}}\]]+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().strip('"').strip("'")
        return ""

    def extract_list(key: str) -> List[str]:
        pattern = rf'["\']{key}["\']\s*:\s*\[(.*?)\]'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            value = extract_string(key)
            return [value] if value else []
        raw = match.group(1)
        items = []
        for part in raw.split(","):
            item = part.strip().strip('"').strip("'")
            if item:
                items.append(item)
        return items

    def extract_bbox_list(key: str) -> List[Union[int, float]]:
        pattern = rf'["\']{key}["\']\s*:\s*\[(.*?)\]'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return []
        raw = match.group(1)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", raw)
        return [float(num) for num in numbers]

    return {
        "action": extract_string("action"),
        "object": extract_string("object"),
        "attributes": extract_list("attributes"),
        "bbox": extract_bbox_list("bbox"),
    }


def split_object_attributes(
    obj: str, attrs: List[str], attribute_hints: List[str]
) -> Tuple[str, List[str]]:
    if attrs or not obj:
        return obj, attrs
    tokens = obj.split()
    if len(tokens) <= 1:
        return obj, attrs
    hints = {item.lower() for item in attribute_hints}
    extracted = [token for token in tokens if token.lower() in hints]
    if not extracted:
        return obj, attrs
    remaining = [token for token in tokens if token.lower() not in hints]
    if not remaining:
        return obj, attrs
    return " ".join(remaining), attrs + extracted


def find_action_hint(text: str, action_hints: List[str]) -> str:
    if not text:
        return ""
    lowered = text.lower()
    best = ""
    best_idx = None
    for phrase in action_hints:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
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


PLACEHOLDER_RE = re.compile(r"<[^>]+>")
PLACEHOLDER_LITERALS = {"<verb>", "<noun>", "<attr>", "...", "…"}


def is_placeholder_text(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in PLACEHOLDER_LITERALS:
        return True
    if PLACEHOLDER_RE.search(text):
        return True
    return False


def normalize_command(
    raw: Dict[str, Any],
    source_text: str,
    action_hints: List[str],
    attribute_hints: List[str],
) -> Dict[str, Any]:
    action: Any = raw.get("action", "")
    obj: Any = raw.get("object", "")
    attrs: Any = raw.get("attributes", [])
    if action is None:
        action = ""
    if not isinstance(action, str):
        action = str(action)
    if is_placeholder_text(action):
        action = ""

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
    if is_placeholder_text(obj):
        obj = ""

    action_hint = find_action_hint(source_text, action_hints)
    if action_hint:
        if not action.strip():
            action = action_hint
        elif action.lower() not in source_text.lower():
            action = action_hint

    obj, attrs = split_object_attributes(obj, attrs, attribute_hints)

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
        if text and not is_placeholder_text(text) and text.lower() not in seen:
            seen.add(text.lower())
            cleaned_attrs.append(text)

    return {"action": action.strip(), "object": obj.strip(), "attributes": cleaned_attrs}


def normalize_bbox(
    values: List[float], size: Tuple[int, int]
) -> List[Union[int, float]]:
    if len(values) != 4:
        return []
    width, height = size
    x1, y1, x2, y2 = values
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


def extract_bbox(
    data: Dict[str, Any], size: Tuple[int, int]
) -> List[Union[int, float]]:
    bbox = data.get("bbox") or data.get("box") or data.get("bboxes")
    if bbox is None:
        return []
    if isinstance(bbox, list):
        if len(bbox) == 4 and all(
            isinstance(value, (int, float, str)) for value in bbox
        ):
            values = [float(value) for value in bbox]
            return normalize_bbox(values, size)
        if len(bbox) == 1 and isinstance(bbox[0], list) and len(bbox[0]) == 4:
            values = [float(value) for value in bbox[0]]
            return normalize_bbox(values, size)
    if isinstance(bbox, dict):
        keys = ["x1", "y1", "x2", "y2"]
        if all(key in bbox for key in keys):
            values = [float(bbox[key]) for key in keys]
            return normalize_bbox(values, size)
        alt = ["left", "top", "right", "bottom"]
        if all(key in bbox for key in alt):
            values = [float(bbox[key]) for key in alt]
            return normalize_bbox(values, size)
    if isinstance(bbox, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", bbox)
        if len(numbers) >= 4:
            values = [float(num) for num in numbers[:4]]
            return normalize_bbox(values, size)
    return []

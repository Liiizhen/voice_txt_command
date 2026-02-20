import os
import torch
from typing import Any, Dict, List

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils import *

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------
# 1. 加载模型（外部调用一次即可）
# ---------------------------------------------------------

def load_vlm(model_id: str, device="cuda", load_4bit=False, load_8bit=False):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type in {"qwen2_vl", "qwen2_5_vl"}:
        model_cls = AutoModelForImageTextToText
    else:
        model_cls = AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    quant = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True)
    elif load_8bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_8bit=True)

    model = model_cls.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else "cpu",
        quantization_config=quant,
        trust_remote_code=True
    )

    return model, tokenizer


# ---------------------------------------------------------
# 2. 构建 prompt（内部使用）
# ---------------------------------------------------------
def build_prompt(config: Dict[str, Any], text: str):
    prompts = config.get("prompts", {})
    system_prompt = prompts.get("command_system", "")
    user_template = prompts.get("command_user", "Command: {text}\nReturn JSON only.")

    action_hints = ", ".join(config.get("action_hints", []))
    attribute_hints = ", ".join(config.get("attribute_hints", []))

    user_prompt = user_template.format(
        text=text,
        action_hints=action_hints,
        attribute_hints=attribute_hints,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


# ---------------------------------------------------------
# 3. ⭐⭐ 外部调用的核心函数：run_vlm() ⭐⭐
# ---------------------------------------------------------
def run_vlm(
    model: Any,
    tokenizer: Any,
    text: str,
    config_path: str = "config.yaml",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    输入文本 → 输出 JSON dict
    模型和 tokenizer 由外部传入（不会重复加载）
    """

    # 1. 加载配置
    config = load_config(config_path)
    action_hints = config.get("action_hints", [])
    attribute_hints = config.get("attribute_hints", [])

    # 2. 构建 prompt
    messages = build_prompt(config, text)

    # 3. 生成
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        prompt_len = input_ids.shape[-1]

        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    else:
        prompt = "\n".join([
            messages[0]["content"] if messages else "",
            "User: " + messages[-1]["content"],
            "Assistant:",
        ])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 4. 提取 JSON
    json_text = extract_json_block(decoded)
    raw = safe_json_loads(json_text)
    if not raw:
        raw = fallback_parse_fields(decoded)

    # 5. 归一化
    command = normalize_command(raw, text, action_hints, attribute_hints)
    return command

def main():
    model, tokenizer = load_vlm(
        model_id="../hf_models/Qwen2.5-VL-7B-Instruct",
        device="cuda",
        load_4bit=True
    ) 
    
    result = run_vlm(
        model=model,
        tokenizer=tokenizer,
        text="把桌子上的红色杯子拿起来",
        config_path="config.yaml"
    ) 
    print(result) # dict

if __name__ == "__main__":
    main()
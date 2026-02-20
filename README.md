# Voice -> Text -> Command Workflow

This folder provides a minimal offline workflow:
1) Whisper-small transcribes an audio file into text.
2) Qwen2.5-7B-Instruct converts text into a JSON command.
3) Qwen2.5-VL-7B-Instruct can generate a bbox from transcript + image.

## Files
- `workflow.py`: end-to-end audio -> command JSON
- `asr_whisper.py`: audio -> transcript
- `command_from_text.py`: transcript -> command JSON
- `bbox_from_vlm.py`: transcript + image -> command JSON + bbox
- `config.yaml`: prompts + action/attribute hints
- `requirements.txt`: Python dependencies

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```
python workflow.py --audio /path/to/command.wav
```

Optional:
```
python workflow.py --audio /path/to/command.wav --output command.json
```

Split pipeline (decoupled):
```
python asr_whisper.py --audio /path/to/command.wav --output transcript.txt
python command_from_text.py --text-file transcript.txt --output command.json
```

Direct VLM pipeline (skip LLM):
```
python asr_whisper.py --audio /path/to/command.wav --output transcript.txt
python bbox_from_vlm.py --text-file transcript.txt --image /path/to/image.png --output command_bbox.json
```

## Output Format
```json
{
  "action": "grasp",
  "object": "cup",
  "attributes": ["red"]
}
```

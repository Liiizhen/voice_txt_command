# Open-Vocabulary Real-Time Object Tracker

End-to-end voice-guided object tracking system.
Voice input → ASR → NLP → Open-vocabulary detection → Real-time tracking.

## Architecture
```
Voice Input (socket audio stream)
    ↓ High-pass filter + VAD
    ↓ Whisper ASR (~50ms)
    ↓ spaCy NLP (~3ms)
    ↓ "brown cup"
C++ main process (KCF tracker, ~3ms/frame)
    ↕ pipe + shared memory
Python subprocess (Grounding DINO, ~70ms async)
```

## Directory Structure
```
├── kcf_main.cpp              # C++ KCF tracker
├── gdino_pipe_server.py      # Grounding DINO detection server
├── voice_input.py            # Voice input: ASR + VAD + NLP
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-click install
├── run.sh                    # Launch tracker
└── models/                   # Created by setup.sh
    ├── bert-base-uncased/    # BERT (~420MB)
    ├── groundingdino_swinb_cogcoor.pth  # GDino (~895MB)
    └── GroundingDINO_SwinB.cfg.py
```

## Quick Start
```bash
# 1. Setup (one-time)
bash setup.sh

# 2. Run with text query
bash run.sh --video input.mp4 --query "brown cup"

# 3. Run with voice input (start audio client separately)
bash run.sh --video input.mp4 --voice
```

## Demo

Tracking "brown cup" on 640p video (RTX 5090):

https://github.com/Liiizhen/voice_txt_command/raw/main/demo_brown_cup_640p.mp4

## Performance (RTX 5090)

| Resolution | KCF Tracker | Grounding DINO (async) | Skip |
|-----------|------------|----------------------|------|
| 640p | ~3ms/frame | ~70ms | 0 |
| 1080p | ~3ms/frame | ~70ms | ~4% |

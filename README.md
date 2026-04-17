# Open-Vocabulary Real-Time Object Tracker

C++ KCF tracker + Grounding DINO open-vocabulary detector.
Track any object described in natural language in real-time.

## Architecture
```
C++ main process (KCF tracker, ~3ms/frame)
    ↕ pipe (stdin/stdout) + shared memory (zero-copy frame)
Python subprocess (Grounding DINO, ~155ms async)
```

## Directory Structure
```
deploy/
├── kcf_main.cpp              # C++ tracker
├── gdino_pipe_server.py      # GDino detection server
├── requirements.txt          # Python dependencies
├── setup.sh                  # Install + compile
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

# 2. Run
bash run.sh --video input.mp4 --query "brown cup"
bash run.sh --video input.mp4 --query "mouse" --output result.mp4

# 3. Camera mode (future)
bash run.sh --camera 0 --query "red bottle"
```

## Performance

| GPU | KCF Tracker | Grounding DINO (async) |
|-----|------------|----------------------|
| RTX 5090 | ~3ms/frame | ~70ms |
| RTX 3090 | ~3ms/frame | ~155ms |

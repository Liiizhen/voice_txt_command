#!/bin/bash
set -e
PYTHON="${PYTHON:-python3}"

echo "=== Open-Vocabulary Tracker Setup ==="

# 1. Python packages
echo "[1/3] Installing Python packages..."
$PYTHON -m pip install -r requirements.txt 2>&1 | tail -3

# 2. Compile C++
echo "[2/3] Compiling C++ tracker..."
g++ -O2 -std=c++17 kcf_main.cpp -o kcf_main \
    $(pkg-config --cflags --libs opencv4) -lpthread -lrt
echo "  Compiled: ./kcf_main"

# 3. Download models
echo "[3/3] Checking models..."
mkdir -p models

if [ ! -f models/groundingdino_swinb_cogcoor.pth ]; then
    echo "  Downloading GDino weights (895MB)..."
    wget -q --show-progress -O models/groundingdino_swinb_cogcoor.pth \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
fi

if [ ! -f models/GroundingDINO_SwinB.cfg.py ]; then
    echo "  Downloading GDino config..."
    wget -q -O models/GroundingDINO_SwinB.cfg.py \
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py"
fi

if [ ! -d models/bert-base-uncased ] || [ ! -f models/bert-base-uncased/pytorch_model.bin ]; then
    echo "  Downloading BERT (420MB)..."
    mkdir -p models/bert-base-uncased
    cd models/bert-base-uncased
    for f in config.json tokenizer_config.json tokenizer.json vocab.txt pytorch_model.bin; do
        wget -q --show-progress "https://huggingface.co/bert-base-uncased/resolve/main/$f" -O "$f"
    done
    cd ../..
fi


# 4. spaCy model
echo "[4/4] Checking spaCy model..."
$PYTHON -m spacy download en_core_web_sm 2>&1 | tail -2

echo ""
echo "=== Setup complete ==="
echo "Total model size: $(du -sh models/ | cut -f1)"
echo ""
echo "Run: bash run.sh --video input.mp4 --query 'brown cup'"

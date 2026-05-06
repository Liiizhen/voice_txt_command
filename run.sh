#!/bin/bash
set -e

VIDEO=""
QUERY=""
OUTPUT="output.mp4"
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --video)   VIDEO="$2"; shift 2;;
        --camera)
            # 纯数字 → 自动补全为 /dev/videoN
            if [[ "$2" =~ ^[0-9]+$ ]]; then
                VIDEO="/dev/video$2"
            else
                VIDEO="$2"
            fi
            shift 2;;
        --query)   QUERY="$2"; shift 2;;
        --output)  OUTPUT="$2"; shift 2;;
        --python)  PYTHON="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

if [ -z "$VIDEO" ] || [ -z "$QUERY" ]; then
    echo "Usage:"
    echo "  bash run.sh --video input.mp4  --query 'brown cup' [--output result.mp4]"
    echo "  bash run.sh --camera 0         --query 'brown cup' [--output result.mp4]"
    echo ""
    echo "Options:"
    echo "  --video    Input video file path"
    echo "  --camera   Camera index (0, 1, …) for live mode"
    echo "  --query    Object description (e.g. 'brown cup', 'mouse')"
    echo "  --output   Output video path (default: output.mp4)"
    echo "  --python   Python path (default: python3)"
    exit 1
fi

if [ ! -f ./kcf_main ]; then
    echo "Error: ./kcf_main not found. Run 'bash setup.sh' first."
    exit 1
fi

export VIDEO QUERY OUTPUT PYTHON
echo "Tracking '$QUERY' in $VIDEO → $OUTPUT"

# live 模式（摄像头）用 Python；file 模式用 C++
if [[ "$VIDEO" =~ ^/dev/video || "$VIDEO" =~ ^[0-9]+$ ]]; then
    $PYTHON live_tracker.py
else
    ./kcf_main
fi

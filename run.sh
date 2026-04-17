#!/bin/bash
set -e

VIDEO=""
QUERY=""
OUTPUT="output.mp4"
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --video) VIDEO="$2"; shift 2;;
        --query) QUERY="$2"; shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        --python) PYTHON="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

if [ -z "$VIDEO" ] || [ -z "$QUERY" ]; then
    echo "Usage: bash run.sh --video input.mp4 --query 'brown cup' [--output result.mp4]"
    echo ""
    echo "Options:"
    echo "  --video    Input video path"
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
./kcf_main

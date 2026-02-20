cd /home/user/nelson/voice_txt_command
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python workflow.py --audio /path/to/command.wav
python workflow.py --audio ./data/1.wav --output ./data/1.json
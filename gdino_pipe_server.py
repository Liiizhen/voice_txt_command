"""GDino pipe server: stdin 读命令, stdout 写结果, 共享内存读帧"""
import os, sys, time, torch, numpy as np, cv2
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
import groundingdino.util.get_tokenlizer as _gt
from PIL import Image
from multiprocessing import shared_memory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GDINO_CFG = os.path.join(SCRIPT_DIR, 'models', 'GroundingDINO_SwinB.cfg.py')
GDINO_CKPT = os.path.join(SCRIPT_DIR, 'models', 'groundingdino_swinb_cogcoor.pth')
BERT_PATH = os.path.join(SCRIPT_DIR, 'models', 'bert-base-uncased')
SHM_NAME = sys.argv[1]
SHM_H, SHM_W = int(sys.argv[2]), int(sys.argv[3])  # 从 C++ 传入实际分辨率
SHM_C = 3
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

_gt.get_tokenlizer = lambda t: _gt.BertTokenizer.from_pretrained(BERT_PATH)
_gt.get_pretrained_language_model = lambda t: _gt.BertModel.from_pretrained(BERT_PATH)

model = load_model(GDINO_CFG, GDINO_CKPT, device='cuda')
transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# Warmup
dummy = np.zeros((SHM_H, SHM_W, SHM_C), dtype=np.uint8)
pil = Image.fromarray(dummy)
img, _ = transform(pil, None)
with torch.amp.autocast('cuda', dtype=torch.float16):
    predict(model=model, image=img, caption='test',
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device='cuda')

# 连接共享内存
shm = shared_memory.SharedMemory(name=SHM_NAME)
frame_buf = np.ndarray((SHM_H, SHM_W, SHM_C), dtype=np.uint8, buffer=shm.buf)

print("READY", flush=True)

for line in sys.stdin:
    line = line.strip()
    if line == "STOP":
        break
    parts = line.split(None, 4)
    if parts[0] == "DETECT":
        fidx = int(parts[1])
        orig_h = int(parts[2])
        orig_w = int(parts[3])
        query = parts[4]
        t0 = time.perf_counter()

        frame_orig = frame_buf.copy()
        pil = Image.fromarray(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB))
        img, _ = transform(pil, None)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            boxes, logits, phrases = predict(model=model, image=img, caption=query,
                box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device='cuda')
        dt = (time.perf_counter()-t0)*1000

        if len(boxes) == 0:
            print(f"RESULT {fidx} NONE 0.0 {dt:.0f}", flush=True)
        else:
            best = logits.argmax().item()
            score = logits[best].item()
            cx,cy,bw,bh = boxes[best].tolist()
            # bbox 直接在原图坐标上
            x1 = max(0, int((cx-bw/2)*orig_w))
            y1 = max(0, int((cy-bh/2)*orig_h))
            print(f"RESULT {fidx} {x1},{y1},{int(bw*orig_w)},{int(bh*orig_h)} {score:.3f} {dt:.0f}", flush=True)

shm.close()

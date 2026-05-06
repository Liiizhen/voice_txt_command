"""Live camera tracker: Python OpenCV (FFMPEG) + GDino async + momentum prediction."""
import os, sys, subprocess, select, time, signal
import numpy as np, cv2
from multiprocessing import shared_memory

_STOP = False
def _on_signal(signum, frame):
    global _STOP
    _STOP = True
signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO  = os.environ.get('VIDEO', '1')
QUERY  = os.environ.get('QUERY', 'object')
OUT    = os.environ.get('OUTPUT', 'output.mp4')
PYTHON = os.environ.get('PYTHON', sys.executable)

CONF_THRESHOLD = 0.35
MAX_MISS       = 3
TARGET_FPS     = 30
SHM_NAME       = 'live_gdino_shm'


class Momentum:
    """EMA velocity + acceleration predictor, mirrors C++ predict_next()."""
    def __init__(self):
        self.rect = None
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]
        self._pv  = [0.0, 0.0]

    def update(self, rect):
        x, y, w, h = rect
        if self.rect:
            rvx = x - self.rect[0];  rvy = y - self.rect[1]
            self.vel[0] = 0.6*rvx + 0.4*self.vel[0]
            self.vel[1] = 0.6*rvy + 0.4*self.vel[1]
            self.acc[0] = 0.4*(self.vel[0]-self._pv[0]) + 0.6*self.acc[0]
            self.acc[1] = 0.4*(self.vel[1]-self._pv[1]) + 0.6*self.acc[1]
            self._pv[:] = self.vel
        self.rect = (x, y, w, h)

    def predict(self):
        if self.rect is None:
            return None
        x, y, w, h = self.rect
        return (max(0, x + int(self.vel[0] + 0.5*self.acc[0])),
                max(0, y + int(self.vel[1] + 0.5*self.acc[1])),
                w, h)


def main():
    # ── Open camera ────────────────────────────────────────────────────────
    # /dev/videoN → extract integer index (cv2 FFMPEG opens by index, not path)
    import re
    m = re.match(r'/dev/video(\d+)$', VIDEO)
    if m:
        cam_arg = int(m.group(1))
    elif VIDEO.isdigit():
        cam_arg = int(VIDEO)
    else:
        cam_arg = VIDEO  # file path
    cap = cv2.VideoCapture(cam_arg)
    if not cap.isOpened():
        print(f"Cannot open camera: {VIDEO}"); sys.exit(1)

    ok, first = cap.read()
    if not ok:
        print("Cannot read first frame"); sys.exit(1)

    fh, fw = first.shape[:2]
    print(f"Camera {VIDEO}: {fw}x{fh}  fps={cap.get(cv2.CAP_PROP_FPS):.0f}")

    # ── Shared memory ──────────────────────────────────────────────────────
    shm_size = fh * fw * 3
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=shm_size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False, size=shm_size)
    shm_arr = np.ndarray((fh, fw, 3), dtype=np.uint8, buffer=shm.buf)

    # ── Launch GDino subprocess ────────────────────────────────────────────
    gdino_path = os.path.join(SCRIPT_DIR, 'gdino_pipe_server.py')
    proc = subprocess.Popen(
        [PYTHON, '-u', gdino_path, SHM_NAME, str(fh), str(fw)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, cwd=SCRIPT_DIR
    )

    print("Waiting for GDino...", flush=True)
    for line in proc.stdout:
        sys.stderr.write(f"  [GDino] {line}")
        if 'READY' in line:
            break
    print("GDino ready", flush=True)

    # ── Helpers ────────────────────────────────────────────────────────────
    det_pending = False

    def send_detect(frame, idx):
        nonlocal det_pending
        shm_arr[:] = frame
        proc.stdin.write(f"DETECT {idx} {fh} {fw} {QUERY}\n")
        proc.stdin.flush()
        det_pending = True

    def poll_result():
        nonlocal det_pending
        r, _, _ = select.select([proc.stdout], [], [], 0)
        if not r:
            return None
        line = proc.stdout.readline().strip()
        if not line:
            return None
        parts = line.split(None, 4)
        if parts[0] != 'RESULT':
            return None
        det_pending = False
        fidx, bbox_str, score = int(parts[1]), parts[2], float(parts[3])
        if bbox_str == 'NONE':
            return (fidx, None, score)
        x, y, bw, bh = map(int, bbox_str.split(','))
        return (fidx, (x, y, bw, bh), score)

    # ── Tracking state ─────────────────────────────────────────────────────
    momentum  = Momentum()
    tracking  = False
    rect      = None
    miss_count = 0
    tag       = "WAIT"
    corrections = det_count = 0
    i = 0

    # ── Output ────────────────────────────────────────────────────────────
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (fw, fh))
    win = f"Tracker: {QUERY}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print(f"Recording → {OUT}   Press 'q' or ESC to stop", flush=True)

    frame = first

    try:
        while True:
            if _STOP:
                break
            # Submit detect for current frame
            if not det_pending:
                send_detect(frame, i)

            # Poll GDino result
            res = poll_result()
            if res is not None:
                _, rbbox, rscore = res
                det_count += 1
                if rbbox is not None and rscore >= CONF_THRESHOLD:
                    rect = rbbox
                    tracking  = True
                    miss_count = 0
                    momentum.update(rbbox)
                    corrections += 1
                    tag = "DET"
                else:
                    miss_count += 1
                    tag = "LOW" if rbbox else "NO-DET"
                    if miss_count >= MAX_MISS:
                        tracking = False; tag = "GONE"
            elif tracking and rect:
                predicted = momentum.predict()
                if predicted:
                    rect = predicted
                    momentum.update(rect)
                    tag = "PRED"

            if not tracking and det_count == 0:
                tag = "WAIT"

            # ── Draw ────────────────────────────────────────────────────────
            vis = frame.copy()
            if tracking and rect:
                x, y, w, h = rect
                cv2.rectangle(vis, (max(0,x), max(0,y)),
                              (min(fw, x+w), min(fh, y+h)), (0,255,255), 2)
            cv2.putText(vis, f"F{i} {tag}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(vis, QUERY, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow(win, vis)
            writer.write(vis)

            if i % 30 == 0:
                print(f"F{i} {tag}  det={det_count} corr={corrections}", flush=True)

            key = cv2.waitKey(1)
            if key in (ord('q'), 27):
                break

            i += 1
            tag = ""

            ok, frame = cap.read()
            if not ok:
                break
    finally:
        # ── Cleanup (always finalize MP4) ─────────────────────────────────
        try:
            proc.stdin.write("STOP\n"); proc.stdin.flush()
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try: proc.wait(timeout=5)
        except Exception: proc.kill()
        try: cap.release()
        except Exception: pass
        try: writer.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: shm.close()
        except Exception: pass
        try: shm.unlink()
        except Exception: pass

        print(f"\nLive: det={det_count} corr={corrections} frames={i}")
        print(f"saved → {OUT}")


if __name__ == '__main__':
    main()

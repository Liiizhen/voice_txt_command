"""Play output.mp4 with OpenCV. Press q to quit, Space to pause."""
import cv2, sys, time

path = sys.argv[1] if len(sys.argv) > 1 else "output.mp4"
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print(f"Cannot open: {path}"); sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000 / fps)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{path}  {int(cap.get(3))}x{int(cap.get(4))}  {fps:.1f}fps  {total}frames")
print("Space=pause  q=quit  ←/→=±10frames")

paused = False
while True:
    if not paused:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

    cv2.imshow("preview", frame)
    key = cv2.waitKey(1 if paused else delay) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == 81:  # left arrow
        pos = max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 10)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
    elif key == 83:  # right arrow
        pos = min(total - 1, cap.get(cv2.CAP_PROP_POS_FRAMES) + 10)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

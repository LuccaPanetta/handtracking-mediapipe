from time import time
import cv2
from mhp.capture import VideoCaptureThread
from mhp.detector import HandDetector


def main():
    with VideoCaptureThread(0, width=640, height=480).start() as cap:
        det = HandDetector(model_complexity=0)
        t0 = time()
        frames = 0
        try:
            while frames < 300:
                ok, frame = cap.read()
                if not ok:
                    continue
                _ = det.process(frame, draw=False)
                frames += 1
            dt = time() - t0
            print(f"Processed {frames} frames in {dt:.2f}s -> {frames/dt:.2f} FPS")
        finally:
            det.close()


if __name__ == "__main__":
    main()

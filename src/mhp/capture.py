import cv2
import threading
import time
from collections import deque
from typing import Optional, Tuple


class VideoCaptureThread:
    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        buffer_size: int = 2,
        api_preference: Optional[int] = None,
        use_mjpeg: bool = True,
        target_fps: int | None = 30,
        set_small_buffer: bool = True,
    ):
        """
        On Windows, passing api_preference=cv2.CAP_DSHOW often reduces startup latency.
        Setting MJPG, FPS and a small buffer helps keep latency low.
        """
        self.source = source
        self.width = width
        self.height = height
        self.api_preference = api_preference
        self.use_mjpeg = use_mjpeg
        self.target_fps = target_fps
        self.set_small_buffer = set_small_buffer
        self.buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        # Try to open with preferred backend (e.g., CAP_DSHOW on Windows), fallback to default
        if self.api_preference is not None:
            self.cap = cv2.VideoCapture(self.source, self.api_preference)
        else:
            self.cap = cv2.VideoCapture(self.source)

        # Try to set capture properties for performance
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.use_mjpeg:
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
        if self.target_fps:
            try:
                self.cap.set(cv2.CAP_PROP_FPS, int(self.target_fps))
            except Exception:
                pass
        if self.set_small_buffer:
            try:
                # Not all backends honor this, but it's safe to attempt
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self._stopped.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stopped.is_set():
            if not self.cap:
                break
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            # Keep only the most recent frame
            with self._lock:
                self.buffer.append(frame)

    def read(self) -> Tuple[bool, Optional[any]]:
        with self._lock:
            if not self.buffer:
                return False, None
            return True, self.buffer[-1]

    def stop(self):
        self._stopped.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        self.buffer.clear()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

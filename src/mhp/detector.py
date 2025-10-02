from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def landmarks_to_vector(hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> np.ndarray:
    arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    return arr.flatten() 


class HandDetector:
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5, model_complexity: int = 0):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> tuple[Optional[np.ndarray], np.ndarray]:
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmarks_vec: Optional[np.ndarray] = None
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            if draw:
                mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_vec = landmarks_to_vector(hand_landmarks)

        return landmarks_vec, frame_bgr

    def close(self):
        self.hands.close()

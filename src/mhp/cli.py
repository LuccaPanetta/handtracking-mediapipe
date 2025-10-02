from __future__ import annotations
import argparse
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from time import time

from .capture import VideoCaptureThread
from .detector import HandDetector
from .cache import append_sample, load_dataset
from .gestures import load_gestures


def collect_interactive(
    width: int = 640,
    height: int = 480,
    backend: str = "auto",
    process_every_n: int = 2,
    movement_threshold: float = 0.01,
    draw: bool = True,
    manual: bool = False,
    warmup_frames: int = 10,
):
    gestures = load_gestures()
    current_idx = 0
    samples_for_current = 0

    print("Presioná 'n' para cambiar gesto, 'q' para entrenar, 'ESC' para salir" + (", 'c' para capturar manual" if manual else ""))

    api_pref = None
    if backend.lower() == "dshow":
        api_pref = cv2.CAP_DSHOW
    elif backend.lower() == "msmf":
        api_pref = cv2.CAP_MSMF

    with VideoCaptureThread(0, width=width, height=height, api_preference=api_pref).start() as cap:
        detector = HandDetector(max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)
        frame_count = 0
        last_landmarks = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                frame_count += 1

                # Warmup
                if frame_count <= warmup_frames:
                    cv2.putText(
                        frame,
                        f"Warming up... {frame_count}/{warmup_frames}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("Recolección de Gestos", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        return None
                    continue

                do_process = (frame_count % process_every_n == 0)
                if do_process:
                    landmarks_vec, frame = detector.process(frame, draw=draw)
                    status = f"Recolectando: {gestures[current_idx]}  muestras:{samples_for_current}"
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if landmarks_vec is not None:
                        if (last_landmarks is None) or (np.linalg.norm(landmarks_vec - last_landmarks) > movement_threshold):
                                append_sample(gestures[current_idx], landmarks_vec)
                                last_landmarks = landmarks_vec
                                samples_for_current += 1

                cv2.imshow("Recolección de Gestos", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('n'):
                    current_idx = (current_idx + 1) % len(gestures)
                    last_landmarks = None
                    samples_for_current = 0
                elif key == ord('q'):
                    break
                elif key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return None
                elif manual and key == ord('c') and 'landmarks_vec' in locals() and landmarks_vec is not None:
                    append_sample(gestures[current_idx], landmarks_vec)
                    samples_for_current += 1

        finally:
            detector.close()
            cv2.destroyAllWindows()

    return True


def train_model():
    X, y = load_dataset()
    if X.size == 0:
        raise RuntimeError("No hay datos en cache/ para entrenar. Ejecuta la recolección primero.")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    model.fit(X, y_enc)
    return model, le


def predict_realtime(model, le, width: int = 640, height: int = 480, backend: str = "auto", draw: bool = True):
    api_pref = None
    if backend.lower() == "dshow":
        api_pref = cv2.CAP_DSHOW
    elif backend.lower() == "msmf":
        api_pref = cv2.CAP_MSMF

    with VideoCaptureThread(0, width=width, height=height, api_preference=api_pref).start() as cap:
        detector = HandDetector(max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)
        fps_t0 = time()
        frames = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame = cv2.flip(frame, 1)

                landmarks_vec, frame = detector.process(frame, draw=draw)
                if landmarks_vec is not None:
                    pred = model.predict(landmarks_vec.reshape(1, -1))[0]
                    label = le.inverse_transform([pred])[0]
                    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frames += 1
                if frames % 30 == 0:
                    fps = frames / (time() - fps_t0)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Predicción en Tiempo Real", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            detector.close()
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Mediapipe Handtracking Plus")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--backend", choices=["auto", "dshow", "msmf"], default="dshow")
    p.add_argument("--process-every-n", type=int, default=2)
    p.add_argument("--movement-threshold", type=float, default=0.01)
    p.add_argument("--draw", action="store_true", help="Dibujar landmarks en la imagen")
    p.add_argument("--no-draw", dest="draw", action="store_false")
    p.set_defaults(draw=True)
    p.add_argument("--warmup-frames", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    ok = collect_interactive(
        width=args.width,
        height=args.height,
        backend=args.backend,
        process_every_n=args.process_every_n,
        movement_threshold=args.movement_threshold,
        draw=args.draw,
        warmup_frames=args.warmup_frames,
    )
    if ok:
        print("Entrenando modelo con datos en cache/...")
        model, le = train_model()
        print("Modelo entrenado. Iniciando predicción en tiempo real...")
        predict_realtime(model, le, width=args.width, height=args.height, backend=args.backend, draw=args.draw)


if __name__ == "__main__":
    main()

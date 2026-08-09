"""
recognize_live.py

Live webcam sign recognition using the trained classifier, with
correctly-working text-to-speech and light prediction smoothing.

TTS FIX EXPLAINED:
pyttsx3 on Windows uses the SAPI5 driver, which goes through
Microsoft's COM system. COM requires every THREAD that uses it to
explicitly call pythoncom.CoInitialize() before use - our earlier
attempts skipped this, which is why speech worked inconsistently
(sometimes only the first utterance, sometimes nothing). Each speech
request now runs in its own short-lived thread that properly
initializes and tears down COM, with a lock ensuring only one
utterance plays at a time (so rapid sign changes don't overlap into
garbled audio).

PREDICTION SMOOTHING:
Instead of either "predict every single frame" (jittery) or "require
8 identical frames" (sluggish, as you found), this uses a lighter
3-frame majority vote: the displayed/spoken label is whatever won at
least 2 of the last 3 predictions. This filters out single-frame
flicker while staying responsive.

Controls:
    c - clear the accumulated sentence
    q - quit

Usage:
    python src/recognize_live.py
"""

import json
import os
import threading
from collections import deque, Counter

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmark_utils import extract_feature_vector
from dataset_io import SEQUENCE_LENGTH

MODEL_PATH_HAND = "models/hand_landmarker.task"
MODEL_PATH_CLASSIFIER = "models/sign_classifier.keras"
LABEL_MAP_PATH = "models/label_map.json"

CONFIDENCE_THRESHOLD = 0.6
VOTE_WINDOW = 3  # majority vote over this many recent frames

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_tts_lock = threading.Lock()


def _configure_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)
    for voice in engine.getProperty("voices") or []:
        voice_id = str(getattr(voice, "id", "")).lower()
        languages = str(getattr(voice, "languages", "")).lower()
        if "english" in voice_id or "en_" in voice_id or "en-" in languages:
            engine.setProperty("voice", voice.id)
            break
    return engine


def speak_async(text):
    """
    Speak `text` in its own short-lived thread, with proper COM
    lifecycle management (required by pyttsx3's Windows SAPI5 driver).
    A lock ensures only one utterance plays at a time, so rapid sign
    changes queue up cleanly instead of overlapping into garbled audio.
    """

    def _run():
        pythoncom = None
        com_initialized = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_initialized = True
        except ImportError:
            pass  # non-Windows platform, COM is not applicable

        try:
            if not text.strip():
                return
            with _tts_lock:
                engine = _configure_tts_engine()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print(f"[TTS ERROR] Could not speak '{text}': {e}")
        finally:
            if com_initialized:
                pythoncom.CoUninitialize()

    threading.Thread(target=_run, daemon=True).start()


def draw_landmarks(frame, hand_landmarks_list, frame_width, frame_height):
    for hand_landmarks in hand_landmarks_list:
        points = [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in hand_landmarks
        ]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def label_to_display(label):
    if label.startswith("letter_"):
        return label.replace("letter_", "").upper()
    return label.replace("_", " ")


def label_to_sentence_fragment(label):
    if label.startswith("letter_"):
        return label.replace("letter_", "").upper()
    return " " + label.replace("_", " ")


def majority_vote(recent_labels):
    """
    Return whichever label appears most often in recent_labels, or
    None if there isn't a clear pick yet (window not full) or the
    winner doesn't have at least 2 votes out of VOTE_WINDOW=3.
    """
    if len(recent_labels) < VOTE_WINDOW:
        return None
    counts = Counter(recent_labels)
    label, count = counts.most_common(1)[0]
    if label is None or count < 2:
        return None
    return label


def main():
    if not os.path.isfile(MODEL_PATH_CLASSIFIER):
        print(f"[ERROR] No trained model found at {MODEL_PATH_CLASSIFIER}")
        print("        Run train_classifier.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH_CLASSIFIER)
    with open(LABEL_MAP_PATH) as f:
        label_names = json.load(f)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH_HAND)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Could not open webcam.")
        return

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    recent_labels = deque(maxlen=VOTE_WINDOW)
    last_spoken_label = None
    sentence = ""

    frame_timestamp_ms = 0
    fps_estimate = 30

    print("Live recognition running. 'c' clears sentence, 'q' quits.")
    print("Testing speech output...")
    speak_async("Sign speak ready")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += int(1000 / fps_estimate)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        h, w, _ = frame.shape
        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks, w, h)

        feature_vector = extract_feature_vector(result)
        frame_buffer.append(feature_vector)

        raw_label = None
        current_confidence = 0.0

        if len(frame_buffer) == SEQUENCE_LENGTH:
            window = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
            predictions = model.predict(window, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            current_confidence = float(predictions[predicted_idx])
            if current_confidence >= CONFIDENCE_THRESHOLD:
                raw_label = label_names[predicted_idx]

        recent_labels.append(raw_label)
        voted_label = majority_vote(recent_labels)

        if voted_label is not None:
            if voted_label != last_spoken_label:
                sentence += label_to_sentence_fragment(voted_label)
                speak_async(label_to_display(voted_label))
                last_spoken_label = voted_label
        elif raw_label is None:
            last_spoken_label = None

        # --- UI overlay ---
        if raw_label:
            status = f"Prediction: {label_to_display(raw_label)} ({current_confidence:.0%})"
            color = (0, 255, 0) if voted_label else (0, 165, 255)
        else:
            status = "..." if len(frame_buffer) == SEQUENCE_LENGTH else \
                f"Buffering... {len(frame_buffer)}/{SEQUENCE_LENGTH}"
            color = (255, 255, 0)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            frame, f"Text: {sentence[-40:]}",
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        cv2.imshow("SignSpeak - Live Recognition (c=clear, q=quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
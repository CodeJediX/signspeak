"""
sign_to_speech.py

Step 7: Full recognize -> text -> speech pipeline.

Builds on recognize_live.py's live prediction, adding:
    1. Debouncing: a sign only counts as "confirmed" once predicted
       stably (same label, high confidence) for STABILITY_FRAMES in a
       row - prevents one held sign from spamming dozens of duplicate
       confirmations.
    2. Text accumulation: confirmed letters (labels starting with
       "letter_") chain together into a spelled word with no spaces;
       confirmed whole-word signs insert as standalone words with
       spaces around them.
    3. Speech output: press 's' to speak the accumulated text aloud
       using pyttsx3 (offline TTS), running in a background thread so
       it doesn't freeze the camera feed while speaking.

Controls:
    s     - speak the accumulated text aloud
    c     - clear the accumulated text
    SPACE - manually insert a space (e.g. to finish a fingerspelled
            word before starting the next one)
    q     - quit

Usage:
    python src/sign_to_speech.py
"""

import json
import os
import threading

import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque

from landmark_utils import extract_feature_vector
from dataset_io import SEQUENCE_LENGTH

MODEL_PATH_HAND = "models/hand_landmarker.task"
MODEL_PATH_CLASSIFIER = "models/sign_classifier.keras"
LABEL_MAP_PATH = "models/label_map.json"

CONFIDENCE_THRESHOLD = 0.6
STABILITY_FRAMES = 12  # ~0.4s at 30fps - how long a sign must hold steady

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


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


def label_to_text_fragment(label):
    """
    Convert a raw class label into the text fragment it should
    contribute. Letters (e.g. "letter_a") become a single character
    "a". Whole-word signs (e.g. "thank_you") become the word with
    underscores replaced by spaces, plus surrounding spaces so it
    reads as a standalone word.
    """
    if label.startswith("letter_"):
        return label.replace("letter_", "")
    return f" {label.replace('_', ' ')} "


def speak_async(text):
    """
    Run TTS in a background thread so it doesn't block the camera loop
    while speaking (pyttsx3's runAndWait() is blocking by design).
    """
    def _speak():
        pythoncom = None
        com_initialized = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_initialized = True
        except ImportError:
            pass
        try:
            if not text.strip():
                return
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as error:
            print(f"[TTS ERROR] Could not speak '{text}': {error}")
        finally:
            if com_initialized:
                pythoncom.CoUninitialize()

    if text.strip():
        threading.Thread(target=_speak, daemon=True).start()


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
    frame_timestamp_ms = 0
    fps_estimate = 30

    # Debounce state
    stable_label = None
    stable_count = 0
    last_confirmed_label = None  # prevents re-confirming the same held sign

    accumulated_text = ""

    print("Sign-to-speech running. s=speak, c=clear, SPACE=insert space, q=quit")

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

        current_prediction = None
        current_confidence = 0.0

        if len(frame_buffer) == SEQUENCE_LENGTH:
            window = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
            predictions = model.predict(window, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            current_confidence = float(predictions[predicted_idx])
            if current_confidence >= CONFIDENCE_THRESHOLD:
                current_prediction = label_names[predicted_idx]

        # --- Debounce logic ---
        if current_prediction == stable_label and current_prediction is not None:
            stable_count += 1
        else:
            stable_label = current_prediction
            stable_count = 1 if current_prediction is not None else 0

        if (
            stable_label is not None
            and stable_count >= STABILITY_FRAMES
            and stable_label != last_confirmed_label
        ):
            accumulated_text += label_to_text_fragment(stable_label)
            last_confirmed_label = stable_label
            print(f"[CONFIRMED] {stable_label} -> \"{accumulated_text}\"")

        # Once the hand's raw prediction moves away from what we last
        # confirmed (a different sign, or no confident prediction at
        # all), clear last_confirmed_label so the same sign can be
        # confirmed again later if the user repeats it (e.g. a double
        # letter like the two L's in "hello").
        if current_prediction != last_confirmed_label:
            last_confirmed_label = None

        # --- UI ---
        pred_display = stable_label if stable_label else "..."
        cv2.putText(
            frame, f"Prediction: {pred_display} ({current_confidence:.0%})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, f"Text: {accumulated_text}",
            (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )
        cv2.putText(
            frame, "s=speak  c=clear  SPACE=space  q=quit",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("SignSpeak - Sign to Speech", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            accumulated_text = ""
            last_confirmed_label = None
        elif key == ord(" "):
            accumulated_text += " "
        elif key == ord("s"):
            speak_async(accumulated_text)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
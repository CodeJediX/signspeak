"""
collect_data.py

Step 4: Data collection tool for building the SignSpeak training set.

Scope: 26-letter ASL fingerspelling alphabet + 10-word core vocabulary
(36 classes total). All classes are recorded the same way - as 30-frame
sequences - even static letters, since holding a pose for 1 second costs
almost nothing extra during recording and keeps the data format uniform
for a single classifier (rather than needing separate static/motion
models). Letters J and Z are naturally motion-based (traced in the air)
and will use this same format without any special-casing.

Workflow per sample:
    1. Choose a label (cycle through with n/p, includes letters + words).
    2. Press SPACE to begin.
    3. A 3-2-1 countdown plays on screen.
    4. The script auto-records 30 frames of your hand landmarks.
    5. The sequence is saved as data/raw/<label>/sample_XXX.npy

Each saved file is a NumPy array of shape (30, 126):
    30 frames x 126 features (63 left hand + 63 right hand, normalized)

Reference sources for correct handshapes/signs (since fluency isn't
required, just accurate imitation):
    - Fingerspelling alphabet: any standard ASL fingerspelling chart
      (search "ASL fingerspelling chart") - all 26 letters shown together
    - Words: Lifeprint (ASL University), HandSpeak, or SigningSavvy -
      search each word individually

Controls:
    SPACE - start a countdown + recording for the current label
    n     - move to the next label in the vocabulary list
    p     - move to the previous label
    q     - quit

Usage:
    python src/collect_data.py
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmark_utils import extract_feature_vector, TOTAL_FEATURES
from dataset_io import DATA_DIR, SEQUENCE_LENGTH, save_sample

MODEL_PATH = "models/hand_landmarker.task"
COUNTDOWN_SECONDS = 3

# Locked target vocabulary (49 classes): 26 letters + 10 core words +
# 13 additional common words. Letters are labeled "letter_a".."letter_z"
# to keep folder names unambiguous and filesystem-safe.
ALPHABET = [f"letter_{chr(c)}" for c in range(ord("a"), ord("z") + 1)]
CORE_WORDS = [
    "hello",
    "thank_you",
    "please",
    "yes",
    "no",
    "i_love_you",
    "sorry",
    "help",
    "more",
    "friend",
]
EXTRA_WORDS = [
    "book",
    "bye",
    "computer",
    "eat",
    "excuse_me",
    "how",
    "i",
    "the",
    "what",
    "when",
    "where",
    "why",
    "you",
]
VOCABULARY = ALPHABET + CORE_WORDS + EXTRA_WORDS

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


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
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

    label_idx = 0
    frame_timestamp_ms = 0
    fps_estimate = 30

    # State machine: "idle" -> "countdown" -> "recording" -> back to "idle"
    state = "idle"
    countdown_start_time = None
    recording_buffer = []  # list of feature vectors while recording

    print("Data collection tool ready.")
    print("SPACE = record, n/p = change label, q = quit")

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

        current_label = VOCABULARY[label_idx]

        # --- State machine handling ---
        if state == "countdown":
            elapsed = time.time() - countdown_start_time
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining > 0:
                cv2.putText(
                    frame, str(int(remaining) + 1),
                    (w // 2 - 30, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 0, 255), 5,
                )
            else:
                state = "recording"
                recording_buffer = []

        elif state == "recording":
            feature_vector = extract_feature_vector(result)
            recording_buffer.append(feature_vector)
            cv2.putText(
                frame, f"RECORDING {len(recording_buffer)}/{SEQUENCE_LENGTH}",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            if len(recording_buffer) >= SEQUENCE_LENGTH:
                # Save the completed sample.
                sample_array = np.array(recording_buffer, dtype=np.float32)
                save_path = save_sample(current_label, sample_array)
                print(f"[SAVED] {save_path}  shape={sample_array.shape}")
                state = "idle"

        # --- UI overlay (always shown) ---
        label_dir = os.path.join(DATA_DIR, current_label)
        if os.path.isdir(label_dir):
            existing_count = len(
                [f for f in os.listdir(label_dir) if f.endswith(".npy")]
            )
        else:
            existing_count = 0

        cv2.putText(
            frame, f"Label: {current_label} ({label_idx + 1}/{len(VOCABULARY)})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )
        cv2.putText(
            frame, f"Samples saved: {existing_count}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
        if state == "idle":
            cv2.putText(
                frame, "SPACE=record | n/p=cycle | a=letters w=words e=extra | q=quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        cv2.imshow("SignSpeak - Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n") and state == "idle":
            label_idx = (label_idx + 1) % len(VOCABULARY)
        elif key == ord("p") and state == "idle":
            label_idx = (label_idx - 1) % len(VOCABULARY)
        elif key == ord("a") and state == "idle":
            label_idx = 0  # jump to start of alphabet (letter_a)
        elif key == ord("w") and state == "idle":
            label_idx = len(ALPHABET)  # jump to start of core word list
        elif key == ord("e") and state == "idle":
            label_idx = len(ALPHABET) + len(CORE_WORDS)  # jump to extra words
        elif key == ord(" ") and state == "idle":
            state = "countdown"
            countdown_start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
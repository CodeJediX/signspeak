"""
landmark_extraction_demo.py

Step 3: Live webcam feed showing hand tracking AND the extracted,
normalized feature vector (126 values) that our classifier will
eventually train on.

This is a verification tool - confirms extraction/normalization is
working correctly before we build the actual data collection script.

Press 'q' to quit.

Usage:
    python src/landmark_extraction_demo.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmark_utils import extract_feature_vector, TOTAL_FEATURES

MODEL_PATH = "models/hand_landmarker.task"

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

    print(f"Feature vector size: {TOTAL_FEATURES} values (63 left hand + 63 right hand)")
    print("Press 'q' in the video window to quit.")

    frame_timestamp_ms = 0
    fps_estimate = 30

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

        # This is the key part: extract the normalized feature vector
        # exactly as our future data-collection and inference code will.
        feature_vector = extract_feature_vector(result)

        # Report which hands are non-zero (i.e. actually detected).
        left_active = bool(feature_vector[0:63].any())
        right_active = bool(feature_vector[63:126].any())

        cv2.putText(
            frame, f"Left hand: {'YES' if left_active else 'no'}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if left_active else (0, 0, 255), 2,
        )
        cv2.putText(
            frame, f"Right hand: {'YES' if right_active else 'no'}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if right_active else (0, 0, 255), 2,
        )
        cv2.putText(
            frame, f"Feature vector shape: {feature_vector.shape}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
        # Show a handful of actual normalized values so you can see them
        # change as you move your hand - useful for sanity-checking.
        # Display from whichever hand slice is actually active, since
        # showing a zeroed-out inactive slice is not a useful check.
        y_offset = 120
        if left_active:
            left_samples = ", ".join(f"{v:.2f}" for v in feature_vector[0:6])
            cv2.putText(
                frame, f"Left samples: [{left_samples}...]",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )
            y_offset += 25
        if right_active:
            right_samples = ", ".join(f"{v:.2f}" for v in feature_vector[63:69])
            cv2.putText(
                frame, f"Right samples: [{right_samples}...]",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

        cv2.imshow("SignSpeak - Landmark Extraction (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
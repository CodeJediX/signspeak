"""
hand_tracking_demo.py

Step 2: Live webcam feed with MediaPipe hand landmark detection,
using the current MediaPipe Tasks API (HandLandmarker).

Older MediaPipe versions exposed a simpler `mp.solutions.hands` API,
but recent releases (0.10.3x+) require the Tasks API instead, which
needs a downloaded .task model file. See README / chat for the
download command.

Draws 21 keypoints + connections per detected hand, in real time.
Press 'q' to quit the window.

Usage:
    python src/hand_tracking_demo.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Path to the downloaded model file. See setup instructions to fetch this.
MODEL_PATH = "models/hand_landmarker.task"

# Hand connections: pairs of landmark indices that should be joined by a
# line when drawing (matches MediaPipe's standard 21-point hand skeleton).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def draw_landmarks(frame, hand_landmarks_list, frame_width, frame_height):
    """
    Draw keypoints and connecting lines for each detected hand onto `frame`.

    hand_landmarks_list: list of hands, each a list of 21 landmarks with
    normalized (0-1) x, y coordinates as returned by HandLandmarker.
    """
    for hand_landmarks in hand_landmarks_list:
        # Convert normalized coordinates to pixel coordinates for drawing.
        points = [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in hand_landmarks
        ]

        # Draw the skeleton connections first (so dots render on top).
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)

        # Draw each landmark as a filled circle.
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def main():
    # Build the HandLandmarker options:
    #   base_options -> points at our downloaded model file
    #   running_mode -> VIDEO mode is optimized for processing a sequence
    #       of frames (vs IMAGE mode for single still images)
    #   num_hands -> detect up to 2 hands (covers two-handed signs)
    #   min_hand_detection_confidence -> how confident the model must be
    #       to report "yes, a hand is here" on first detection
    #   min_tracking_confidence -> how confident it must be to keep
    #       following an already-detected hand across frames
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

    print("Hand tracking running. Press 'q' in the video window to quit.")

    frame_timestamp_ms = 0
    fps_estimate = 30  # used to fake a timestamp increment per frame

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARN] Failed to read frame from webcam, stopping.")
            break

        # Mirror the frame for a natural "looking in a mirror" feel.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the frame in MediaPipe's Image type.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires a monotonically increasing timestamp (ms).
        frame_timestamp_ms += int(1000 / fps_estimate)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        num_hands = 0
        if result.hand_landmarks:
            num_hands = len(result.hand_landmarks)
            h, w, _ = frame.shape
            draw_landmarks(frame, result.hand_landmarks, w, h)

        cv2.putText(
            frame,
            f"Hands detected: {num_hands}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("SignSpeak - Hand Tracking (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
"""
test_environment.py

Quick sanity check that all core libraries are installed correctly
and that the webcam can be accessed. Run this once after setup,
before writing any real project code.

Usage:
    python src/test_environment.py
"""

import sys


def check_imports():
    """Try importing every core library and report versions."""
    print("=== Checking library imports ===")
    try:
        import cv2
        print(f"[OK] OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"[FAIL] OpenCV not installed: {e}")
        sys.exit(1)

    try:
        import mediapipe as mp
        print(f"[OK] MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"[FAIL] MediaPipe not installed: {e}")
        sys.exit(1)

    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"[FAIL] TensorFlow not installed: {e}")
        sys.exit(1)

    try:
        import numpy as np
        print(f"[OK] NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy not installed: {e}")
        sys.exit(1)

    try:
        import sklearn
        print(f"[OK] scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"[FAIL] scikit-learn not installed: {e}")
        sys.exit(1)

    print("All libraries imported successfully.\n")


def check_webcam():
    """Try to open the default webcam and grab a single frame."""
    import cv2

    print("=== Checking webcam access ===")
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[FAIL] Could not open webcam (device index 0).")
        print("       Check that no other app is using the camera,")
        print("       and that camera permissions are granted to Python/Terminal.")
        return

    ret, frame = cap.read()
    if ret:
        h, w, c = frame.shape
        print(f"[OK] Webcam frame captured successfully: {w}x{h}, {c} channels")
    else:
        print("[FAIL] Webcam opened but failed to read a frame.")

    cap.release()


if __name__ == "__main__":
    check_imports()
    check_webcam()
    print("\nEnvironment check complete.")

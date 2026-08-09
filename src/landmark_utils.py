"""
landmark_utils.py

Reusable functions for turning raw MediaPipe hand landmarks into
normalized, fixed-size numeric feature vectors suitable for training
a classifier.

Feature vector layout (126 values total):
    [0:63]   -> Left hand, 21 landmarks x (x, y, z), normalized
    [63:126] -> Right hand, 21 landmarks x (x, y, z), normalized
If a hand is not detected in a given frame, its 63 values are all zero.

Normalization applied per hand:
    1. Translate so the wrist (landmark 0) is at the origin.
       -> makes the features invariant to WHERE the hand is in the frame.
    2. Scale by the wrist-to-middle-fingertip distance.
       -> makes the features invariant to hand size / distance from camera.
"""

import numpy as np

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126 (left + right)

MIDDLE_FINGERTIP_IDX = 12
WRIST_IDX = 0


def normalize_single_hand(hand_landmarks):
    """
    Convert one hand's 21 MediaPipe landmarks into a normalized,
    flattened 63-value numpy array.

    Args:
        hand_landmarks: list of 21 landmark objects, each with .x, .y, .z
                         (normalized 0-1 coordinates as given by MediaPipe)

    Returns:
        np.ndarray of shape (63,), dtype float32
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
    )  # shape (21, 3)

    # Step 1: make position-invariant by subtracting the wrist coordinate
    # from every landmark. After this, the wrist itself is always (0,0,0).
    wrist = coords[WRIST_IDX].copy()
    coords -= wrist

    # Step 2: make scale-invariant by dividing by a reference distance.
    # We use the (already wrist-relative) distance to the middle fingertip,
    # since that's a stable "hand span" reference regardless of which sign
    # is being made.
    ref_distance = np.linalg.norm(coords[MIDDLE_FINGERTIP_IDX])
    if ref_distance > 1e-6:  # avoid division by zero on degenerate detections
        coords /= ref_distance

    return coords.flatten()  # shape (63,)


def extract_feature_vector(detection_result):
    """
    Build the full 126-value feature vector for a single frame's
    HandLandmarker detection result, correctly slotting detected hands
    into fixed Left/Right positions.

    Args:
        detection_result: the object returned by
            HandLandmarker.detect_for_video(...), with
            .hand_landmarks (list of hands, each a list of 21 landmarks)
            and .handedness (list of classification results per hand)

    Returns:
        np.ndarray of shape (126,), dtype float32
    """
    features = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    if not detection_result.hand_landmarks:
        return features  # no hands detected this frame -> all zeros

    for hand_landmarks, handedness in zip(
        detection_result.hand_landmarks, detection_result.handedness
    ):
        # handedness is a list containing one Category object; its
        # category_name is "Left" or "Right" as seen from the camera's
        # point of view (after our horizontal flip, this matches what
        # the person doing the signing perceives as their own left/right
        # hand -- good enough for consistent, repeatable labeling).
        hand_label = handedness[0].category_name

        normalized = normalize_single_hand(hand_landmarks)

        if hand_label == "Left":
            features[0:FEATURES_PER_HAND] = normalized
        elif hand_label == "Right":
            features[FEATURES_PER_HAND:TOTAL_FEATURES] = normalized
        # If MediaPipe ever returns an unexpected label, we silently skip
        # that hand rather than crash -- acceptable for now, revisit if
        # this happens often during data collection.

    return features
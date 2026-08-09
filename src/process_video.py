"""
process_video.py

Alternative data-collection method: extract a labeled training sample
from a pre-recorded reference video file (e.g. a downloaded ASL sign
demonstration from SignASL.org) instead of live webcam capture.

Given a video showing one sign performed once, this script:
    1. Runs hand-landmark detection across every frame of the video.
    2. Automatically trims to the span where a hand is actually visible,
       skipping any idle/resting frames before and after the sign
       (common in reference clips where the signer's hands start and
       end at rest).
    3. Resamples that active span to exactly SEQUENCE_LENGTH (30)
       frames - picking evenly-spaced frames if the span is long, or
       evenly duplicating frames if it's short - so every saved sample
       has the same shape as webcam-recorded samples and can be mixed
       into the same training set without any special-casing later.
    4. Saves the result as data/raw/<label>/sample_XXX.npy

This is useful for bulk-importing reference clips without manually
re-performing every sign yourself on webcam.

Usage (interactive):
    python src/process_video.py

You'll be prompted for a video file path and a label, repeatedly.
Type 'q' at the video path prompt to quit.

Tip: you can process the SAME video multiple times under the same
label if you trim/crop slightly different versions of it, or process
several different reference videos of the same word under one label
to build up more samples per class.
"""

import math
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmark_utils import extract_feature_vector
from dataset_io import SEQUENCE_LENGTH, save_sample

MODEL_PATH = "models/hand_landmarker.task"


def extract_all_frame_features(video_path, landmarker):
    """
    Run hand detection across every frame of a video file.

    Returns:
        List of (feature_vector, hand_detected) tuples, one per frame,
        in the video's original order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Some downloaded/converted video files report unreliable fps
    # metadata (0, NaN, or absurdly high values from variable-frame-rate
    # encodes). Fall back to a sane default whenever that's detected,
    # rather than letting bad metadata corrupt our timestamps.
    if fps <= 0 or math.isnan(fps) or fps > 240:
        fps = 30

    frame_results = []
    frame_idx = 0
    last_timestamp_ms = -1  # ensures the very first frame is >= 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Compute an ideal timestamp from frame index and fps, but then
        # explicitly enforce that it's strictly greater than the last
        # one we sent. This guarantees MediaPipe's monotonic-timestamp
        # requirement is met even if the video's fps metadata is
        # inaccurate or causes rounding collisions between frames.
        candidate_ms = int((frame_idx / fps) * 1000)
        timestamp_ms = max(candidate_ms, last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        feature_vector = extract_feature_vector(result)
        hand_detected = bool(feature_vector.any())
        frame_results.append((feature_vector, hand_detected))

        frame_idx += 1

    cap.release()
    return frame_results


def trim_to_active_span(frame_results):
    """
    Trim leading/trailing frames where no hand was detected, focusing
    on the actual signing motion rather than idle frames before/after it.

    Returns:
        List of feature vectors for just the active span. If no hand
        was ever detected, returns the full (all-zero) sequence so the
        caller can decide to warn/skip.
    """
    active_indices = [i for i, (_, detected) in enumerate(frame_results) if detected]

    if not active_indices:
        return [fv for fv, _ in frame_results]

    start, end = active_indices[0], active_indices[-1]
    return [fv for fv, _ in frame_results[start:end + 1]]


def resample_to_fixed_length(feature_sequence, target_length=SEQUENCE_LENGTH):
    """
    Resample a variable-length sequence of feature vectors to exactly
    `target_length` frames using evenly-spaced index selection.

    Works whether the input is longer than target_length (downsampling,
    picking a representative subset) or shorter (upsampling, repeating
    frames to stretch it out) - so short and long reference clips both
    end up in the same fixed shape our classifier expects.
    """
    original_length = len(feature_sequence)
    if original_length == target_length:
        return np.array(feature_sequence, dtype=np.float32)

    indices = np.linspace(0, original_length - 1, target_length)
    indices = np.round(indices).astype(int)

    resampled = [feature_sequence[i] for i in indices]
    return np.array(resampled, dtype=np.float32)


def build_landmarker():
    """
    Create a fresh HandLandmarker instance.

    IMPORTANT: in VIDEO running mode, a landmarker tracks a running
    timestamp across its ENTIRE lifetime, not per video file. If we
    reused one landmarker across multiple videos, the second video's
    timestamps (starting again near 0ms) would appear to go backwards
    relative to the first video's already-processed timestamps, and
    MediaPipe would reject them. Creating a new landmarker per video
    gives each one a clean timestamp timeline and avoids this entirely.
    """
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def process_single_video(video_path, label):
    print(f"Processing '{video_path}' as label '{label}'...")

    landmarker = build_landmarker()
    try:
        frame_results = extract_all_frame_features(video_path, landmarker)
    finally:
        landmarker.close()

    total_frames = len(frame_results)
    active_frames = sum(1 for _, detected in frame_results if detected)

    if active_frames == 0:
        print("[WARN] No hand detected in any frame of this video - skipping.")
        print("       Check the video shows a clear, well-lit hand, and that")
        print("       the file path/format is correct (mp4 works reliably).")
        return

    trimmed = trim_to_active_span(frame_results)
    print(
        f"  Total frames: {total_frames}, active (hand visible): {active_frames}, "
        f"trimmed span: {len(trimmed)}"
    )

    resampled = resample_to_fixed_length(trimmed)
    save_path = save_sample(label, resampled)
    print(f"[SAVED] {save_path}  shape={resampled.shape}")


def main():
    print("SignSpeak video importer.")
    print("Enter a video file path and a label to extract a training sample.")
    print("Type 'q' at the video path prompt to quit.\n")

    while True:
        video_path = input("Video file path: ").strip().strip('"')
        if video_path.lower() == "q":
            break
        if not os.path.isfile(video_path):
            print(f"[ERROR] File not found: {video_path}\n")
            continue

        label = input("Label for this sign (e.g. hello): ").strip().lower().replace(" ", "_")
        if not label:
            print("[ERROR] Label cannot be empty.\n")
            continue

        try:
            process_single_video(video_path, label)
        except Exception as e:
            print(f"[ERROR] Failed to process video: {e}")

        print()  # blank line before next prompt

    print("Done.")


if __name__ == "__main__":
    main()
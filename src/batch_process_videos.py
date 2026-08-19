"""
batch_process_videos.py

Batch version of process_video.py: automatically processes an entire
folder of reference videos organized as:

    root_folder/
        hello/
            video1.mp4
            video2.mp4
        thank_you/
            clip_a.mp4
            clip_b.mov
        ...

Each SUBFOLDER name becomes the label (matching the data/raw/<label>/
convention), and every video file inside it is processed automatically
- no need to type a label for each video one at a time.

This imports and reuses process_single_video() from process_video.py
directly rather than duplicating its logic, so results are identical
to processing videos one at a time through that script - this is
purely an automation layer on top of it. process_video.py itself is
untouched.

Usage:
    python src/batch_process_videos.py

You'll be prompted once for the root folder path, then everything is
processed automatically with a summary printed at the end.
"""

import os

from process_video import process_single_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def sanitize_label(folder_name):
    """
    Convert a folder name into a label matching the existing naming
    convention: lowercase, spaces/hyphens replaced with underscores.
    e.g. 'Thank You' -> 'thank_you', 'excuse-me' -> 'excuse_me'
    """
    label = folder_name.strip().lower()
    label = label.replace(" ", "_").replace("-", "_")
    return label


def find_video_files(folder_path):
    """Return a sorted list of video file paths directly inside folder_path."""
    files = []
    for fname in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, fname)
        if os.path.isfile(full_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                files.append(full_path)
    return files


def main():
    root_folder = input(
        "Path to the folder containing one subfolder per word: "
    ).strip().strip('"')

    if not os.path.isdir(root_folder):
        print(f"[ERROR] Not a valid folder: {root_folder}")
        return

    word_folders = sorted(
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    )

    if not word_folders:
        print(f"[ERROR] No subfolders found inside {root_folder}")
        print("        Expected structure: root_folder/word_name/video1.mp4 ...")
        return

    print(f"Found {len(word_folders)} word folders: {word_folders}\n")

    summary = {}  # label -> (processed_count, failed_count)

    for folder_name in word_folders:
        label = sanitize_label(folder_name)
        folder_path = os.path.join(root_folder, folder_name)
        video_files = find_video_files(folder_path)

        if not video_files:
            print(f"[SKIP] '{folder_name}' has no video files, skipping folder.\n")
            summary[label] = (0, 0)
            continue

        print(f"=== Processing '{folder_name}' -> label '{label}' ({len(video_files)} videos) ===")

        processed = 0
        failed = 0
        for video_path in video_files:
            try:
                process_single_video(video_path, label)
                processed += 1
            except Exception as e:
                print(f"[ERROR] Failed on '{video_path}': {e}")
                failed += 1

        summary[label] = (processed, failed)
        print()  # blank line between word folders

    # --- Final summary ---
    print("=== Batch import summary ===")
    total_processed = 0
    total_failed = 0
    for label, (processed, failed) in summary.items():
        print(f"  {label:15s}: {processed} saved, {failed} failed")
        total_processed += processed
        total_failed += failed
    print(f"\nTotal: {total_processed} samples saved, {total_failed} failed, across {len(summary)} labels.")


if __name__ == "__main__":
    main()
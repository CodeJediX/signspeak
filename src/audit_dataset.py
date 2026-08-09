"""
audit_dataset.py

Quick utility: print how many samples exist per class in data/raw/,
and flag classes below the recommended minimum before retraining.

Usage:
    python src/audit_dataset.py
"""

import os

from dataset_io import DATA_DIR

RECOMMENDED_MIN = 15

# The current locked target vocabulary (49 classes). Anything else found
# in data/raw (e.g. earlier one-off test words) is reported separately,
# not counted against this target.
ALPHABET = [f"letter_{chr(c)}" for c in range(ord("a"), ord("z") + 1)]
CORE_WORDS = [
    "hello", "thank_you", "please", "yes", "no",
    "i_love_you", "sorry", "help", "more", "friend",
]
EXTRA_WORDS = [
    "book", "bye", "computer", "eat", "excuse_me", "how", "i",
    "the", "what", "when", "where", "why", "you",
]
TARGET_VOCABULARY = ALPHABET + CORE_WORDS + EXTRA_WORDS


def count_samples(label):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.endswith(".npy")])


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No data directory found at {DATA_DIR}")
        return

    existing_folders = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    print("=== Target vocabulary status ===")
    needs_more = []
    for label in TARGET_VOCABULARY:
        count = count_samples(label)
        status = "OK" if count >= RECOMMENDED_MIN else "NEEDS MORE"
        if count < RECOMMENDED_MIN:
            needs_more.append((label, count))
        print(f"  {label:15s}: {count:3d} samples  [{status}]")

    other_folders = [f for f in existing_folders if f not in TARGET_VOCABULARY]
    if other_folders:
        print("\n=== Other folders (not in current target vocabulary) ===")
        for label in other_folders:
            print(f"  {label:15s}: {count_samples(label):3d} samples")

    print(f"\n{len(needs_more)} of {len(TARGET_VOCABULARY)} target classes "
          f"are below the recommended minimum of {RECOMMENDED_MIN} samples.")
    if needs_more:
        print("Classes needing more samples:")
        print(", ".join(f"{label} ({count})" for label, count in needs_more))


if __name__ == "__main__":
    main()
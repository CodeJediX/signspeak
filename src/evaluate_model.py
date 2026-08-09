"""
evaluate_model.py

Step 7b: Detailed evaluation of the trained classifier - a confusion
matrix and per-class report, so we can see WHICH signs are being
confused with each other, not just an overall accuracy number.

This re-loads the full dataset, re-does the same train/val split style
as training (for a fair held-out check), and reports:
    - Per-class precision/recall/F1 (from sklearn's classification_report)
    - A confusion matrix saved as an image, showing the most-confused
      pairs of signs at a glance

Usage:
    python src/evaluate_model.py
"""

import json
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset_io import DATA_DIR, SEQUENCE_LENGTH
from landmark_utils import TOTAL_FEATURES

MODEL_PATH = "models/sign_classifier.keras"
LABEL_MAP_PATH = "models/label_map.json"
CONFUSION_MATRIX_IMAGE_PATH = "docs/confusion_matrix.png"


def load_dataset():
    """Same loading logic as train_classifier.py, kept independent here
    so this script can be run standalone without importing training
    internals."""
    label_names = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    X, y = [], []
    for class_idx, label in enumerate(label_names):
        label_dir = os.path.join(DATA_DIR, label)
        for fname in os.listdir(label_dir):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(label_dir, fname))
            if arr.shape != (SEQUENCE_LENGTH, TOTAL_FEATURES):
                continue
            X.append(arr)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_names


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] No trained model at {MODEL_PATH}. Run train_classifier.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH) as f:
        label_names = json.load(f)

    X, y, dataset_labels = load_dataset()

    # Sanity check: the label ordering in the saved model must match
    # the current dataset's folder ordering, or predictions will be
    # silently misaligned. This would happen if you added/removed
    # classes and evaluated without retraining.
    if dataset_labels != label_names:
        print("[WARN] Dataset folders don't match the model's saved label map.")
        print("       This usually means classes were added/removed since training.")
        print("       Re-run train_classifier.py before trusting this evaluation.")
        return

    counts = np.bincount(y, minlength=len(label_names))
    can_stratify = all(c >= 2 for c in counts)
    split_kwargs = {"test_size": 0.2, "random_state": 42}
    if can_stratify:
        split_kwargs["stratify"] = y
    _, X_val, _, y_val = train_test_split(X, y, **split_kwargs)

    print(f"Evaluating on {len(X_val)} held-out validation samples...\n")

    predictions = model.predict(X_val, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Only include classes that actually appear in this validation
    # split (small/rare classes may not show up at all in a random 20%
    # slice), otherwise sklearn's report includes confusing all-zero rows.
    present_classes = sorted(set(y_val.tolist()) | set(y_pred.tolist()))
    present_label_names = [label_names[i] for i in present_classes]

    print("=== Per-class report (on validation split) ===")
    print(
        classification_report(
            y_val, y_pred,
            labels=present_classes,
            target_names=present_label_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_val, y_pred, labels=present_classes)

    os.makedirs(os.path.dirname(CONFUSION_MATRIX_IMAGE_PATH), exist_ok=True)
    fig_size = max(8, len(present_classes) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(present_label_names)))
    ax.set_yticks(range(len(present_label_names)))
    ax.set_xticklabels(present_label_names, rotation=90, fontsize=7)
    ax.set_yticklabels(present_label_names, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("SignSpeak Confusion Matrix (validation split)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_IMAGE_PATH, dpi=150)
    print(f"[SAVED] Confusion matrix image -> {CONFUSION_MATRIX_IMAGE_PATH}")

    # Print the top confused pairs as a quick text summary too, since
    # scanning a large image for small numbers is tedious.
    print("\n=== Most confused pairs (true -> predicted, count) ===")
    confusions = []
    for i, true_label in enumerate(present_label_names):
        for j, pred_label in enumerate(present_label_names):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], true_label, pred_label))
    confusions.sort(reverse=True)
    for count, true_label, pred_label in confusions[:15]:
        print(f"  {true_label:15s} -> {pred_label:15s}  ({count} times)")

    if not confusions:
        print("  No confusions in this validation split (or too few samples to tell).")


if __name__ == "__main__":
    main()
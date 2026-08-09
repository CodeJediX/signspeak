"""
train_classifier.py

Improved training script:
    - Data augmentation: each recorded sample generates several
      slightly-varied copies (small position jitter + timing
      variation), multiplying effective training data without
      recording more. This is the single biggest lever available
      right now, since dataset size per class is still the main
      accuracy ceiling.
    - Class weighting: compensates for classes that still have fewer
      samples than others, so the model doesn't just learn to favor
      whichever classes are best represented.
    - Deeper, regularized architecture: two LSTM layers with dropout
      between them, tuned against overfitting (previous runs showed a
      real gap between training and validation accuracy).
    - Early stopping with best-weights restoration: training stops
      once validation accuracy stops improving, and the FINAL saved
      model is whichever epoch had the best validation accuracy, not
      just whatever epoch training happened to end on.

Usage:
    python src/train_classifier.py
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from dataset_io import DATA_DIR, SEQUENCE_LENGTH
from landmark_utils import TOTAL_FEATURES

MODELS_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "sign_classifier.keras")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")

EPOCHS = 150
BATCH_SIZE = 16
AUGMENTATIONS_PER_SAMPLE = 4  # each real sample produces this many extra variants
NOISE_STD = 0.02              # how much random jitter to add to landmark positions
TIME_WARP_MAX_FRAMES = 3      # how many frames of timing variation to simulate


def load_raw_dataset():
    """
    Walk data/raw/ and load every sample_*.npy file as-is (no
    augmentation yet). Returns (X, y, label_names) same as before.
    """
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"No data directory found at {DATA_DIR}")

    label_names = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    if not label_names:
        raise ValueError(f"No label folders found under {DATA_DIR}")

    X, y = [], []
    print("Class sample counts (before augmentation):")
    for class_idx, label in enumerate(label_names):
        label_dir = os.path.join(DATA_DIR, label)
        sample_files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
        print(f"  {label:15s}: {len(sample_files)} samples")

        for fname in sample_files:
            arr = np.load(os.path.join(label_dir, fname))
            if arr.shape != (SEQUENCE_LENGTH, TOTAL_FEATURES):
                print(f"  [WARN] Skipping {fname} in '{label}': unexpected shape {arr.shape}")
                continue
            X.append(arr)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_names


def augment_sequence(sequence, rng):
    """
    Produce one slightly-varied copy of a (SEQUENCE_LENGTH, TOTAL_FEATURES)
    sample, simulating natural variation a real repeat performance of
    the same sign would have:

    1. Small Gaussian noise on every coordinate - simulates natural
       hand-position jitter and tracking noise (a real repeat of the
       same sign is never pixel-identical to a previous one).
    2. Light time-warping - randomly stretches or compresses the
       sequence slightly then resamples back to SEQUENCE_LENGTH,
       simulating performing the sign a bit faster or slower.

    Zero-valued (inactive hand) frames are left as zero, since adding
    noise to "no hand detected" would fabricate a fake hand presence.
    """
    seq = sequence.copy()

    # Only add noise where a hand was actually detected (non-zero),
    # so we don't invent hand presence in frames where none existed.
    active_mask = seq != 0
    noise = rng.normal(0, NOISE_STD, seq.shape).astype(np.float32)
    seq = seq + noise * active_mask

    # Light time-warp: stretch/compress the frame count slightly, then
    # resample back to exactly SEQUENCE_LENGTH so shapes stay consistent.
    warp = rng.integers(-TIME_WARP_MAX_FRAMES, TIME_WARP_MAX_FRAMES + 1)
    if warp != 0:
        n = seq.shape[0]
        stretch_indices = np.linspace(0, n - 1, n + warp)
        stretch_indices = np.clip(np.round(stretch_indices).astype(int), 0, n - 1)
        stretched = seq[stretch_indices]
        resample_indices = np.linspace(0, len(stretched) - 1, SEQUENCE_LENGTH)
        resample_indices = np.round(resample_indices).astype(int)
        seq = stretched[resample_indices]

    return seq.astype(np.float32)


def build_augmented_dataset(X_raw, y_raw, augmentations_per_sample, seed=42):
    """
    Expand the raw dataset by generating `augmentations_per_sample`
    extra variants of every real sample. Applying the same multiplier
    to every class keeps relative class balance the same as the raw
    data (class_weight still handles whatever imbalance remains).
    """
    rng = np.random.default_rng(seed)
    X_aug, y_aug = [X_raw], [y_raw]

    for _ in range(augmentations_per_sample):
        variants = np.array([augment_sequence(s, rng) for s in X_raw], dtype=np.float32)
        X_aug.append(variants)
        y_aug.append(y_raw.copy())

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


def build_model(num_classes):
    """
    Two-layer LSTM with dropout between layers:
        Input (30, 126)
        -> LSTM(96, return_sequences=True) - first pass over the
           temporal sequence, kept full-length so the second LSTM
           layer can look across the whole processed sequence too
        -> Dropout(0.4)
        -> LSTM(48) - condenses the sequence into a single summary vector
        -> Dropout(0.4)
        -> Dense(32, relu, L2 regularization) - reduces overfitting risk
        -> Dense(num_classes, softmax)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, TOTAL_FEATURES)),
        tf.keras.layers.LSTM(96, return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(48),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X_raw, y_raw, label_names = load_raw_dataset()
    num_classes = len(label_names)
    print(f"\nLoaded {len(X_raw)} raw samples across {num_classes} classes.")

    counts = np.bincount(y_raw, minlength=num_classes)
    sparse_classes = [label_names[i] for i in range(num_classes) if counts[i] < 2]
    if sparse_classes:
        print(
            f"[NOTE] These classes have fewer than 2 raw samples: {sparse_classes}\n"
            f"       Augmentation multiplies existing samples but can't substitute "
            f"for genuinely new examples - keep recording these when you can.\n"
        )

    # Split BEFORE augmenting, so augmented copies of a validation
    # sample never leak into training (that would inflate validation
    # accuracy artificially, since the model would have "seen" a near
    # duplicate of the test example during training).
    can_stratify = all(c >= 2 for c in counts)
    split_kwargs = {"test_size": 0.2, "random_state": 42}
    if can_stratify:
        split_kwargs["stratify"] = y_raw
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(X_raw, y_raw, **split_kwargs)

    print(f"Train (raw): {len(X_train_raw)}, Validation: {len(X_val)}")

    X_train, y_train = build_augmented_dataset(X_train_raw, y_train_raw, AUGMENTATIONS_PER_SAMPLE)
    print(f"Train (after {AUGMENTATIONS_PER_SAMPLE}x augmentation): {len(X_train)}\n")

    # Class weights computed on the ORIGINAL training distribution
    # (augmentation preserves relative ratios, so this still applies
    # correctly to the augmented set).
    unique_classes = np.unique(y_train_raw)
    class_weight_values = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train_raw
    )
    class_weight_dict = dict(zip(unique_classes.tolist(), class_weight_values.tolist()))

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    # After restore_best_weights=True, the model already holds the
    # weights from its best validation-accuracy epoch, not whatever
    # epoch training happened to stop on.
    best_val_acc = max(history.history["val_accuracy"])
    final_train_acc = history.history["accuracy"][-1]
    print(f"\nBest validation accuracy achieved: {best_val_acc:.2%}")
    print(f"Training accuracy at stopping point: {final_train_acc:.2%}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_names, f, indent=2)

    print(f"\n[SAVED] Model -> {MODEL_SAVE_PATH}")
    print(f"[SAVED] Label map -> {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
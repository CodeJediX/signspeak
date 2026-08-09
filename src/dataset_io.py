"""
dataset_io.py

Shared constants and helper functions for reading/writing the
SignSpeak training dataset. Used by both collect_data.py (live webcam
capture) and process_video.py (importing from reference video files),
so both tools save samples in exactly the same format and folder
structure.
"""

import os

import numpy as np

DATA_DIR = "data/raw"
SEQUENCE_LENGTH = 30  # every saved sample is (SEQUENCE_LENGTH, 126)


def get_next_sample_index(label):
    """
    Find the next free sample_XXX.npy filename index for this label,
    so repeated runs (or different tools) never overwrite existing
    samples.
    """
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    existing = [f for f in os.listdir(label_dir) if f.startswith("sample_")]
    return len(existing) + 1


def save_sample(label, sequence_array):
    """
    Save a (SEQUENCE_LENGTH, 126) feature array as the next available
    sample file for the given label.

    Returns the path it was saved to.
    """
    sample_idx = get_next_sample_index(label)
    save_path = os.path.join(DATA_DIR, label, f"sample_{sample_idx:03d}.npy")
    np.save(save_path, sequence_array)
    return save_path
"""
config.py
---------
Central configuration file for the training pipeline.
Edit the paths and hyperparameters here — no other file needs to change.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────

# Root directory of your training dataset (folder containing A/, B/, ... Z/, space/, etc.)
DATASET_DIR = os.path.join("asl_alphabet_train", "asl_alphabet_train")

# Directory where trained .pkl artifacts will be saved
MODEL_OUTPUT_DIR = "models"

# ── MediaPipe Settings ─────────────────────────────────────────────────────────

MEDIAPIPE_CONFIG = {
    "static_image_mode": True,       # True for dataset images (not live video)
    "max_num_hands": 1,              # Detect only one hand per image
    "min_detection_confidence": 0.5  # Minimum confidence to accept a detection
}

# ── Train / Test Split ─────────────────────────────────────────────────────────

TEST_SIZE = 0.2        # 20% of data held out for testing
RANDOM_STATE = 42      # Seed for reproducibility

# ── Random Forest Hyperparameters ──────────────────────────────────────────────

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,    # Number of decision trees in the forest
    "max_depth": None,      # Trees grow until all leaves are pure (no limit)
    "min_samples_split": 2, # Minimum samples required to split an internal node
    "min_samples_leaf": 1,  # Minimum samples required to be at a leaf node
    "n_jobs": -1,           # Use all available CPU cores in parallel
    "random_state": RANDOM_STATE,
    "verbose": 1            # Print progress during training
}

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"   # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

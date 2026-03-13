"""
config.py
---------
Central configuration file for the training pipeline.
Edit the paths and hyperparameters here — no other file needs to change.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join("asl_alphabet_train", "asl_alphabet_train")
MODEL_OUTPUT_DIR = "models"

# ── MediaPipe Settings ─────────────────────────────────────────────────────────
MEDIAPIPE_CONFIG = {
    "static_image_mode"       : True,
    "max_num_hands"           : 1,
    "min_detection_confidence": 0.5,
}

# ── Train / Test Split ─────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Random Forest Base Hyperparameters ─────────────────────────────────────────
# Used only if ENABLE_TUNING = False
RANDOM_FOREST_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : None,
    "min_samples_split": 2,
    "min_samples_leaf" : 1,
    "max_features"     : "sqrt",
    "class_weight"     : "balanced",
    "n_jobs"           : -1,
    "random_state"     : 42,
    "verbose"          : 1,
}

# ── Hyperparameter Tuning ──────────────────────────────────────────────────────
ENABLE_TUNING = False      # Keep False — True takes 11+ hours and Colab crashes!
N_ITER_SEARCH = 30
CV_FOLDS      = 5

TUNING_PARAM_DIST = {
    "n_estimators"     : [100, 200, 300, 400, 500],
    "max_depth"        : [None, 20, 30, 40],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf" : [1, 2, 3],
    "max_features"     : ["sqrt", "log2", 0.3],
    "class_weight"     : ["balanced", None],
    "bootstrap"        : [True, False],
}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"
LOG_FORMAT      = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

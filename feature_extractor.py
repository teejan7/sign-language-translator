"""
feature_extractor.py
--------------------
Module responsible for converting a raw image into a 68-dimensional
geometric feature vector using Google MediaPipe hand landmarks.

Feature Vector Breakdown (68 features total):
  - 63 normalized landmark coordinates (21 landmarks × 3 axes: x, y, z)
  - 5  Euclidean distances from wrist to each fingertip
"""

import logging
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional

logger = logging.getLogger(__name__)

# ── MediaPipe setup ────────────────────────────────────────────────────────────
_mp_hands = mp.solutions.hands

# Fingertip landmark indices (MediaPipe hand model)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
WRIST_INDEX = 0
NUM_LANDMARKS = 21
FEATURE_SIZE = NUM_LANDMARKS * 3 + len(FINGERTIP_INDICES)   # 63 + 5 = 68


def _normalize_landmarks(landmarks) -> np.ndarray:
    """
    Step 1 — Origin Normalization:
        Sets the wrist (landmark 0) as the origin (0, 0, 0) and
        recalculates all 21 landmarks relative to it.

    Returns:
        Flattened array of 63 floats [x0, y0, z0, x1, y1, z1, ...].
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = coords[WRIST_INDEX]          # anchor point
    coords -= wrist                      # shift everything so wrist = (0,0,0)
    return coords.flatten()              # shape: (63,)


def _compute_fingertip_distances(landmarks) -> np.ndarray:
    """
    Step 2 — Euclidean Distance Features:
        Calculates 3-D distance from wrist to each of the 5 fingertips.
        These capture whether each finger is extended or curled.

    Returns:
        Array of 5 floats — one distance per fingertip.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = coords[WRIST_INDEX]
    distances = np.linalg.norm(coords[FINGERTIP_INDICES] - wrist, axis=1)
    return distances                     # shape: (5,)


def extract_features_from_image(
    image_path: str,
    hands_detector
) -> Optional[np.ndarray]:
    """
    Full pipeline for a single image:
      1. Load image from disk.
      2. Run MediaPipe to detect hand landmarks.
      3. Compute 63 normalized coords + 5 fingertip distances.
      4. Return combined 68-feature vector, or None if no hand detected.

    Args:
        image_path:     Absolute or relative path to the image file.
        hands_detector: An active mediapipe Hands() context object.

    Returns:
        np.ndarray of shape (68,) or None.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.debug(f"Could not read image: {image_path}")
        return None

    # MediaPipe expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None   # No hand detected in this image

    # Use only the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0].landmark

    normalized_coords = _normalize_landmarks(hand_landmarks)       # (63,)
    fingertip_distances = _compute_fingertip_distances(hand_landmarks)  # (5,)

    feature_vector = np.concatenate([normalized_coords, fingertip_distances])  # (68,)
    return feature_vector


def build_feature_matrix(
    image_paths: list,
    string_labels: list,
    static_image_mode: bool = True,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5
):
    """
    Processes every image in image_paths and builds the (X, y) matrices
    ready for Scikit-Learn model training.

    Args:
        image_paths:             List of paths returned by data_loader.
        string_labels:           Parallel list of class-name strings.
        static_image_mode:       True for still images (not video stream).
        max_num_hands:           Detect only 1 hand per image.
        min_detection_confidence: MediaPipe confidence threshold.

    Returns:
        X: np.ndarray of shape (N, 68)   — feature matrix
        y: list of str of length N       — filtered labels
        skipped: int                     — images with no hand detected
    """
    X, y = [], []
    skipped = 0
    total = len(image_paths)

    with _mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence
    ) as hands:
        for idx, (path, label) in enumerate(zip(image_paths, string_labels)):
            features = extract_features_from_image(path, hands)
            if features is not None:
                X.append(features)
                y.append(label)
            else:
                skipped += 1

            if (idx + 1) % 1000 == 0:
                logger.info(f"  Processed {idx + 1}/{total} images ...")

    logger.info(
        f"Feature extraction complete: {len(X)} valid, {skipped} skipped (no hand)"
    )
    return np.array(X, dtype=np.float32), y, skipped

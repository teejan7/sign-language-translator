"""
feature_extractor.py
--------------------
Converts a raw image into a 91-dimensional geometric feature vector
using Google MediaPipe hand landmarks.

Feature Vector Breakdown (91 features total):
  - 63: Normalized (x,y,z) for all 21 landmarks (wrist-centered + scale-normalized)
  - 5:  Euclidean distances from wrist to each fingertip
  - 5:  Finger extension: tip-to-base distance per finger
  - 10: Inter-fingertip pairwise distances
  - 8:  Finger joint bend angles + thumb/index spread
"""

import logging
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional

logger = logging.getLogger(__name__)

_mp_hands = mp.solutions.hands

WRIST_INDEX     = 0
NUM_LANDMARKS   = 21
FINGERTIP_IDS   = [4, 8, 12, 16, 20]   # Thumb → Pinky tips
FINGER_BASE_IDS = [2, 5, 9, 13, 17]    # Thumb → Pinky bases
KNUCKLE_IDS     = [3, 6, 10, 14, 18]   # Thumb → Pinky knuckles

FEATURE_SIZE = 91   # 63 + 5 + 5 + 10 + 8


def _build_coords(landmarks) -> np.ndarray:
    """
    Converts MediaPipe landmarks to a (21, 3) numpy array,
    centered on the wrist and scaled by hand size for invariance.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
    )
    coords -= coords[WRIST_INDEX]                                        # origin at wrist
    scale = np.max(np.linalg.norm(coords, axis=1)) + 1e-6               # hand-size scale
    coords /= scale
    return coords


def _normalized_coords(coords: np.ndarray) -> np.ndarray:
    """63 flattened, wrist-centered, scale-normalized landmark coords."""
    return coords.flatten()


def _fingertip_distances(coords: np.ndarray) -> np.ndarray:
    """5 distances: wrist → each fingertip. Captures finger extension."""
    return np.linalg.norm(coords[FINGERTIP_IDS], axis=1)


def _finger_extension(coords: np.ndarray) -> np.ndarray:
    """5 distances: fingertip → base joint. Better extension measure than wrist dist."""
    return np.linalg.norm(
        coords[FINGERTIP_IDS] - coords[FINGER_BASE_IDS], axis=1
    )


def _inter_fingertip_distances(coords: np.ndarray) -> np.ndarray:
    """10 pairwise distances between all fingertip pairs. Captures finger spread."""
    dists = []
    for i in range(len(FINGERTIP_IDS)):
        for j in range(i + 1, len(FINGERTIP_IDS)):
            d = np.linalg.norm(coords[FINGERTIP_IDS[i]] - coords[FINGERTIP_IDS[j]])
            dists.append(d)
    return np.array(dists, dtype=np.float32)


def _joint_angles(coords: np.ndarray) -> np.ndarray:
    """
    8 angle features:
      - 5 finger bend angles (tip–knuckle–base for each finger)
      - 1 thumb–index spread distance
      - 1 wrist-to-middle-finger orientation angle
      - 1 palm normal direction (cross product z component)
    """
    angles = []

    # Finger bend angles (cosine of angle at knuckle)
    for tip, knuckle, base in zip(FINGERTIP_IDS, KNUCKLE_IDS, FINGER_BASE_IDS):
        v1 = coords[tip] - coords[knuckle]
        v2 = coords[base] - coords[knuckle]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(float(np.clip(cos_a, -1.0, 1.0)))

    # Thumb–index spread
    angles.append(float(np.linalg.norm(coords[4] - coords[8])))

    # Wrist-to-middle-finger orientation (2D angle in x-y plane)
    v_mid = coords[9] - coords[0]
    angles.append(float(np.arctan2(v_mid[1], v_mid[0])))

    # Palm normal: z-component of cross product (index_base - wrist) × (pinky_base - wrist)
    v1 = coords[5] - coords[0]
    v2 = coords[17] - coords[0]
    normal_z = v1[0] * v2[1] - v1[1] * v2[0]
    angles.append(float(normal_z))

    return np.array(angles, dtype=np.float32)


def extract_features_from_image(
    image_path: str,
    hands_detector
) -> Optional[np.ndarray]:
    """
    Full pipeline for a single image:
      1. Load image from disk.
      2. Run MediaPipe to detect hand landmarks.
      3. Compute 91-feature vector.
      4. Return feature vector, or None if no hand detected.

    Args:
        image_path:     Path to the image file.
        hands_detector: Active mediapipe Hands() context object.

    Returns:
        np.ndarray of shape (91,) or None.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.debug(f"Could not read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        # Try horizontally flipped image (catches mirror issues in training data)
        img_flipped = cv2.flip(img, 1)
        results = hands_detector.process(cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None

    hand_landmarks = results.multi_hand_landmarks[0].landmark
    coords = _build_coords(hand_landmarks)

    feature_vector = np.concatenate([
        _normalized_coords(coords),         # 63
        _fingertip_distances(coords),       # 5
        _finger_extension(coords),          # 5
        _inter_fingertip_distances(coords), # 10
        _joint_angles(coords),              # 8
    ])                                      # = 91

    return feature_vector.astype(np.float32)


def build_feature_matrix(
    image_paths: list,
    string_labels: list,
    static_image_mode: bool = True,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5
):
    """
    Processes every image and builds the (X, y) matrices for training.

    Returns:
        X:       np.ndarray of shape (N, 91)
        y:       list of str of length N
        skipped: int — images where no hand was detected
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

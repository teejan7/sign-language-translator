"""
model_io.py
-----------
Module responsible for persisting and loading trained model artifacts.

Saves two files:
  - rf_model_68.pkl       : the trained RandomForestClassifier
  - label_encoder.pkl     : the fitted LabelEncoder (maps int ↔ class name)
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODEL_FILENAME = "rf_model_68.pkl"
ENCODER_FILENAME = "label_encoder.pkl"


def save_artifacts(
    model: RandomForestClassifier,
    encoder: LabelEncoder,
    output_dir: str = "."
) -> Tuple[str, str]:
    """
    Serializes the trained model and label encoder to disk using pickle.

    Args:
        model:      Trained RandomForestClassifier.
        encoder:    Fitted LabelEncoder.
        output_dir: Directory where .pkl files will be saved.

    Returns:
        Tuple of (model_path, encoder_path) — the absolute file paths saved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / MODEL_FILENAME
    encoder_path = out / ENCODER_FILENAME

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {model_path}")

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    logger.info(f"LabelEncoder saved → {encoder_path}")

    return str(model_path), str(encoder_path)


def load_artifacts(
    model_dir: str = "."
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """
    Loads a previously saved model and label encoder from disk.
    Used at inference time by main.py.

    Args:
        model_dir: Directory containing the .pkl files.

    Returns:
        Tuple of (model, encoder).

    Raises:
        FileNotFoundError: If either .pkl file is missing.
    """
    model_path = Path(model_dir) / MODEL_FILENAME
    encoder_path = Path(model_dir) / ENCODER_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded ← {model_path}")

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    logger.info(f"LabelEncoder loaded ← {encoder_path}")

    return model, encoder

"""
model_trainer.py
----------------
Module responsible for:
  1. Encoding string labels → integer indices (LabelEncoder)
  2. Splitting data into train / test sets
  3. Training the Random Forest Classifier
  4. Evaluating accuracy and generating a classification report
"""

import logging
import numpy as np
from typing import Tuple, List, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)


# ── Label Encoding ─────────────────────────────────────────────────────────────

def encode_labels(string_labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Converts string class names to integer indices that Scikit-Learn expects.

    Args:
        string_labels: e.g. ['A', 'A', 'B', 'space', ...]

    Returns:
        y_encoded: np.ndarray of integer class indices.
        encoder:   Fitted LabelEncoder (needed to decode predictions later).
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(string_labels)
    logger.info(f"Classes encoded: {list(encoder.classes_)}")
    return y_encoded, encoder


# ── Train / Test Split ─────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits (X, y) into stratified train and test sets.

    Args:
        X:            Feature matrix, shape (N, 68).
        y:            Integer-encoded label array, shape (N,).
        test_size:    Fraction of data reserved for testing (default 20%).
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    logger.info(
        f"Data split — Train: {len(X_train)} samples | Test: {len(X_test)} samples"
    )
    return X_train, X_test, y_train, y_test


# ── Model Training ─────────────────────────────────────────────────────────────

def build_model(hyperparams: Dict[str, Any]) -> RandomForestClassifier:
    """
    Instantiates a Random Forest Classifier with given hyperparameters.

    Args:
        hyperparams: Dictionary of RF parameters, e.g.:
                     {
                       "n_estimators": 100,
                       "max_depth": None,
                       "min_samples_split": 2,
                       "n_jobs": -1,
                       "random_state": 42
                     }

    Returns:
        Untrained RandomForestClassifier instance.
    """
    model = RandomForestClassifier(**hyperparams)
    logger.info(f"Model instantiated with params: {hyperparams}")
    return model


def train_model(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> RandomForestClassifier:
    """
    Fits the Random Forest on the training data.

    Args:
        model:   Untrained RandomForestClassifier.
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Fitted model.
    """
    logger.info("Training Random Forest Classifier ...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder
) -> Dict[str, Any]:
    """
    Runs inference on the test set and prints accuracy + per-class metrics.

    Args:
        model:   Trained RandomForestClassifier.
        X_test:  Test feature matrix.
        y_test:  True integer labels for test set.
        encoder: LabelEncoder to map integer predictions back to class names.

    Returns:
        Dictionary with 'accuracy' (float) and 'report' (str).
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Test Accuracy: {acc * 100:.2f}%")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"{'='*50}")

    return {"accuracy": acc, "report": report}

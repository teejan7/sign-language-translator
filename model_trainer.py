"""
model_trainer.py
----------------
Module responsible for:
  1. Encoding string labels → integer indices (LabelEncoder)
  2. Splitting data into train / test sets
  3. Training the Random Forest (with optional hyperparameter tuning)
  4. Evaluating accuracy and generating a classification report
"""

import logging
import numpy as np
from typing import Tuple, List, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)


# ── Label Encoding ─────────────────────────────────────────────────────────────

def encode_labels(string_labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    """Converts string class names to integer indices."""
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
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Data split — Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── Model Building ─────────────────────────────────────────────────────────────

def build_model(hyperparams: Dict[str, Any]) -> RandomForestClassifier:
    """Instantiates a Random Forest with given hyperparameters."""
    model = RandomForestClassifier(**hyperparams)
    logger.info(f"Model instantiated with params: {hyperparams}")
    return model


# ── Hyperparameter Tuning ──────────────────────────────────────────────────────

def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_dist: Dict[str, Any],
    n_iter: int = 30,
    cv_folds: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Runs RandomizedSearchCV to find the best Random Forest hyperparameters.

    Args:
        X_train:      Training feature matrix.
        y_train:      Training labels.
        param_dist:   Dict of hyperparameter distributions to sample from.
        n_iter:       Number of random combinations to try.
        cv_folds:     Number of stratified CV folds.
        random_state: Seed for reproducibility.

    Returns:
        Best fitted RandomForestClassifier found by search.
    """
    logger.info(
        f"Starting RandomizedSearchCV — {n_iter} iterations × {cv_folds}-fold CV ..."
    )
    logger.info("  ⏳ This typically takes 5–15 minutes depending on dataset size.")

    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring="accuracy",
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        refit=True   # Automatically refit best model on full training set
    )

    search.fit(X_train, y_train)

    logger.info(f"Best CV Accuracy : {search.best_score_ * 100:.2f}%")
    logger.info(f"Best Params      : {search.best_params_}")

    return search.best_estimator_


# ── Training (no tuning) ───────────────────────────────────────────────────────

def train_model(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> RandomForestClassifier:
    """Fits the Random Forest on training data."""
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
    """Evaluates on test set and logs accuracy + per-class metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  Test Accuracy : {acc * 100:.2f}%")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"{'='*60}")

    return {"accuracy": acc, "report": report}

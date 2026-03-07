"""
train.py
--------
Main training orchestrator — ties all modules together.

Run this script to train the ASL Random Forest model end-to-end:

    python train.py

Pipeline Steps:
  1. Load dataset paths & labels          [data_loader]
  2. Extract 91-D geometric features      [feature_extractor]
  3. Encode labels & split data           [model_trainer]
  4. Tune OR train Random Forest          [model_trainer]
  5. Evaluate on test set                 [model_trainer]
  6. Save model + encoder to disk         [model_io]
"""

import logging
import time

import config
from data_loader import load_dataset
from feature_extractor import build_feature_matrix
from model_trainer import (
    encode_labels,
    split_data,
    build_model,
    train_model,
    tune_model,
    evaluate_model,
)
from model_io import save_artifacts


# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  ASL Sign Language — Training Pipeline v2")
    logger.info("=" * 60)

    # ── Step 1: Load Dataset ───────────────────────────────────────────────────
    logger.info("\n[STEP 1/6] Loading dataset ...")
    image_paths, string_labels, class_labels = load_dataset(config.DATASET_DIR)
    logger.info(f"  → {len(image_paths)} images across {len(class_labels)} classes")

    # ── Step 2: Extract Features ───────────────────────────────────────────────
    logger.info("\n[STEP 2/6] Extracting 91-D geometric features via MediaPipe ...")
    logger.info("  (This is the slowest step — grab a coffee ☕)")
    X, y_strings, skipped = build_feature_matrix(
        image_paths,
        string_labels,
        static_image_mode=config.MEDIAPIPE_CONFIG["static_image_mode"],
        max_num_hands=config.MEDIAPIPE_CONFIG["max_num_hands"],
        min_detection_confidence=config.MEDIAPIPE_CONFIG["min_detection_confidence"],
    )
    logger.info(f"  → Feature matrix shape: {X.shape}  |  Skipped: {skipped} images")

    # ── Step 3: Encode Labels & Split ─────────────────────────────────────────
    logger.info("\n[STEP 3/6] Encoding labels and splitting into train/test sets ...")
    y_encoded, encoder = encode_labels(y_strings)
    X_train, X_test, y_train, y_test = split_data(
        X, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # ── Step 4: Train / Tune Model ─────────────────────────────────────────────
    if config.ENABLE_TUNING:
        logger.info("\n[STEP 4/6] Tuning Random Forest with RandomizedSearchCV ...")
        model = tune_model(
            X_train, y_train,
            param_dist=config.TUNING_PARAM_DIST,
            n_iter=config.N_ITER_SEARCH,
            cv_folds=config.CV_FOLDS,
            random_state=config.RANDOM_STATE,
        )
    else:
        logger.info("\n[STEP 4/6] Training Random Forest (no tuning) ...")
        model = build_model(config.RANDOM_FOREST_PARAMS)
        model = train_model(model, X_train, y_train)

    # ── Step 5: Evaluate ───────────────────────────────────────────────────────
    logger.info("\n[STEP 5/6] Evaluating on test set ...")
    results = evaluate_model(model, X_test, y_test, encoder)
    logger.info(f"  → Final Test Accuracy: {results['accuracy'] * 100:.2f}%")

    # ── Step 6: Save Artifacts ─────────────────────────────────────────────────
    logger.info("\n[STEP 6/6] Saving model and encoder to disk ...")
    model_path, encoder_path = save_artifacts(
        model, encoder,
        output_dir=config.MODEL_OUTPUT_DIR
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("  Training Pipeline Complete!")
    logger.info(f"  Total time   : {elapsed / 60:.1f} minutes")
    logger.info(f"  Accuracy     : {results['accuracy'] * 100:.2f}%")
    logger.info(f"  Model saved  : {model_path}")
    logger.info(f"  Encoder saved: {encoder_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

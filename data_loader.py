"""
data_loader.py
--------------
Module responsible for scanning the dataset directory and loading
image file paths with their corresponding class labels.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger(__name__)


def get_class_labels(dataset_dir: str) -> List[str]:
    """
    Scans the dataset directory and returns a sorted list of class labels.
    Each subdirectory name is treated as a class label (e.g., 'A', 'B', ..., 'space').

    Args:
        dataset_dir: Path to the root dataset directory (e.g., asl_alphabet_train/).

    Returns:
        Sorted list of class label strings.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    labels = sorted([
        d.name for d in dataset_path.iterdir()
        if d.is_dir()
    ])

    if not labels:
        raise ValueError(f"No subdirectories (class folders) found in: {dataset_dir}")

    logger.info(f"Found {len(labels)} classes: {labels}")
    return labels


def load_image_paths(
    dataset_dir: str,
    class_labels: List[str],
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> Tuple[List[str], List[str]]:
    """
    Walks through each class folder and collects all valid image file paths
    along with their string labels.

    Args:
        dataset_dir:      Root directory of the dataset.
        class_labels:     List of class names (subfolder names).
        valid_extensions: Tuple of accepted image file extensions.

    Returns:
        A tuple of (image_paths, labels) where both are aligned lists.
    """
    dataset_path = Path(dataset_dir)
    image_paths: List[str] = []
    labels: List[str] = []

    for label in class_labels:
        class_dir = dataset_path / label
        if not class_dir.exists():
            logger.warning(f"Class folder not found, skipping: {class_dir}")
            continue

        count = 0
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(label)
                count += 1

        logger.info(f"  [{label}]: {count} images loaded")

    logger.info(f"Total images collected: {len(image_paths)}")
    return image_paths, labels


def load_dataset(dataset_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    High-level convenience function: discovers classes and loads all image paths.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Tuple of (image_paths, string_labels, class_labels_list).
    """
    class_labels = get_class_labels(dataset_dir)
    image_paths, labels = load_image_paths(dataset_dir, class_labels)
    return image_paths, labels, class_labels

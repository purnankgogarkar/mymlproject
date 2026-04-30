"""Model training and prediction."""

from .trainer import (
    load_cleaned_data,
    train_content_based_model,
    train_collaborative_filtering_model,
    save_model,
    load_model,
    print_training_summary,
)
from .baseline import (
    load_data,
    prepare_data,
    train_baseline,
    evaluate_classification,
    evaluate_regression,
    detect_task_and_target,
)

__all__ = [
    'load_cleaned_data',
    'train_content_based_model',
    'train_collaborative_filtering_model',
    'save_model',
    'load_model',
    'print_training_summary',
    'load_data',
    'prepare_data',
    'train_baseline',
    'evaluate_classification',
    'evaluate_regression',
    'detect_task_and_target',
]
